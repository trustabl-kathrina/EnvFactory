"""Conservative semantic audit of Cycle-3 multi-tool first divergences.

The original episodes and FD dataset are read-only. A new output directory is
created once. Only exact, read-only gold-prefix parallel candidates are replayed;
everything else needs positive evidence of failure or remains uncertain.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import scalar_paths, typed_equal
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import RUN, sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, decode, query_from_row

OUT = RUN / "semantic_audit_v1"
REPORT_ROOT = RUN.parents[1] / "reports"
FORMAT_TYPES = {"PARSER_OR_FORMAT_ERROR", "FORMAT_ERROR", "MULTI_TOOL_OUTPUT"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(stable(row) + "\n" for row in rows))


def parse_calls(step: dict) -> tuple[list[dict], str | None]:
    raw = step.get("model_message") or {}
    parsed = []
    for call in raw.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name")
        try:
            args = json.loads(function.get("arguments", ""))
        except (TypeError, ValueError):
            return parsed, "MALFORMED_JSON"
        if not isinstance(name, str) or not isinstance(args, dict):
            return parsed, "MALFORMED_JSON"
        parsed.append({"name": name, "arguments": args})
    return parsed, None


def schema_errors(call: dict, schemas: list[dict]) -> list[str]:
    spec = next((s["function"] for s in schemas
                 if s["function"]["name"] == call["name"]), None)
    if spec is None:
        return ["UNKNOWN_TOOL_SCHEMA"]
    params = spec.get("parameters") or {}
    args = call["arguments"]
    errors = [f"MISSING_REQUIRED_ARG:{key}" for key in params.get("required", [])
              if key not in args]
    if params.get("additionalProperties") is False:
        errors.extend(f"EXTRA_ARG:{key}" for key in args
                      if key not in (params.get("properties") or {}))
    types = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": dict, "null": type(None)}
    for key, value in args.items():
        prop = (params.get("properties") or {}).get(key) or {}
        expected = prop.get("type")
        if isinstance(expected, str) and expected in types:
            valid = isinstance(value, types[expected])
            if expected in ("integer", "number") and isinstance(value, bool):
                valid = False
            if not valid:
                errors.append(f"TYPE_OR_FORMAT_ERROR:{key}")
        if "enum" in prop and value not in prop["enum"]:
            errors.append(f"ENUM_VALUE_ERROR:{key}")
    return errors


def _source_value_available(value: Any, query: str, state: dict,
                            previous_observations: list[Any]) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, str) and len(value) < 4:
        return True
    if isinstance(value, (int, float)) and abs(value) < 100:
        return True
    if isinstance(value, (str, int, float)) and str(value).casefold() in query.casefold():
        return True
    return any(typed_equal(v, value) for _, v in scalar_paths(state)) or any(
        typed_equal(v, value) for observation in previous_observations
        for _, v in scalar_paths(observation))


def future_dependency(calls: list[dict], matched: dict[int, int],
                      gold: dict, step_index: int, query: str) -> list[dict]:
    """Require concrete future-observation provenance not present at the turn."""
    results = []
    before = gold["initial_state"] if step_index == 0 else gold["states_after"][step_index - 1]
    previous = gold["observations"][:step_index]
    for call_pos, gold_index in matched.items():
        for path, value in scalar_paths(calls[call_pos]["arguments"]):
            if _source_value_available(value, query, before, previous):
                continue
            producers = [
                {"gold_step": prior + 1, "output_field": source_path,
                 "consumer_argument": path}
                for prior in range(step_index, gold_index)
                for source_path, observed in scalar_paths(gold["observations"][prior])
                if typed_equal(observed, value)
            ]
            if producers:
                results.append({"call_position": call_pos, "producers": producers})
    return results


def distinct_gold_matches(calls: list[dict], gold: list[dict], start: int) -> dict[int, int]:
    used: set[int] = set()
    found = {}
    for position, call in enumerate(calls):
        match = next((i for i in range(start, len(gold))
                      if i not in used and typed_equal(call, gold[i])), None)
        if match is not None:
            used.add(match)
            found[position] = match
    return found


def classify_static(fd: dict, episode: dict, schemas: list[dict], query: str) -> dict:
    step_index = fd["student_step"] - 1
    step = episode["student"]["steps"][step_index]
    calls, parse_error = parse_calls(step)
    gold = episode["gold"]
    result = {
        "task_id": fd["task_id"], "source_task_id": fd["task_id"],
        "old_failure_type": fd["failure_type"], "gold_step": fd["gold_step"],
        "student_step": fd["student_step"], "tool_call_count": len(calls),
        "tool_calls": calls, "raw_model_message": step.get("model_message"),
        "gold_next_action": fd["gold_next_action"],
        "environment": fd["environment"], "query_hash": fd["query_hash"],
        "case_subtype": "OTHER_FORMAT", "semantic_validity": "UNCERTAIN",
        "new_failure_type": "UNCERTAIN", "disposition": "UNCERTAIN_EXCLUDE",
        "reason": "Insufficient evidence to judge a non-gold tool sequence.",
        "schema_errors": [], "dependency_evidence": [],
        "exact_gold_matches": {}, "replay_candidate": False,
    }
    if parse_error:
        result.update(case_subtype=parse_error, semantic_validity="TRUE_FAILURE",
                      new_failure_type="TRUE_PROTOCOL_FORMAT_ERROR",
                      disposition="KEEP_AS_FD", reason="Tool-call arguments are not JSON objects.")
        return result
    if not calls:
        reason = (fd.get("student_action") or {}).get("reason")
        subtype = "GENERATION_LENGTH_TRUNCATED" if reason == "generation_length_truncated" else "OTHER_FORMAT"
        result.update(case_subtype=subtype, semantic_validity="TRUE_FAILURE",
                      new_failure_type="TRUE_PROTOCOL_FORMAT_ERROR",
                      disposition="KEEP_AS_FD", reason="No executable call; generation was truncated or malformed.")
        return result
    if len(calls) == 1:
        result["case_subtype"] = "TOOL_CALL_PARSE_FAILURE"
        result["reason"] = "One structured call exists but original miner marked it invalid; raw token stream unavailable."
        return result
    result["case_subtype"] = "MULTIPLE_TOOL_CALLS_ONE_TURN"
    if len({stable(call) for call in calls}) < len(calls):
        result["secondary_subtypes"] = ["DUPLICATE_TOOL_CALL"]
    errors = [dict(call_position=i, error=error)
              for i, call in enumerate(calls) for error in schema_errors(call, schemas)]
    result["schema_errors"] = errors
    matches = distinct_gold_matches(calls, gold["actions"], step_index)
    result["exact_gold_matches"] = {str(i): step + 1 for i, step in matches.items()}
    if errors:
        # The correction target is the *local next* gold action. A malformed
        # call for some later gold tool does not establish a wrong argument at
        # this state; it may instead be a premature/wrong-tool proposal.
        relevant_names = {fd["gold_next_action"]["name"]}
        error_names = {calls[item["call_position"]]["name"] for item in errors}
        mechanism = ("WRONG_ARGUMENT_IN_MULTI_CALL" if error_names <= relevant_names
                     and not any(item["error"] == "UNKNOWN_TOOL_SCHEMA" for item in errors)
                     else "TOOL_SCHEMA_INVALID_IN_MULTI_CALL")
        result.update(semantic_validity="PARTIALLY_VALID" if matches else "TRUE_FAILURE",
                      new_failure_type=mechanism, disposition="RECLASSIFY_AND_KEEP",
                      reason="At least one proposed call violates the actual tool schema.")
        return result
    dependency = future_dependency(calls, matches, gold, step_index, query)
    result["dependency_evidence"] = dependency
    if dependency:
        result.update(semantic_validity="TRUE_FAILURE", new_failure_type="DEPENDENCY_VIOLATION",
                      disposition="RECLASSIFY_AND_KEEP",
                      reason="An argument matches a future gold observation but is absent from current query/state/prefix observations.")
        return result
    indices = sorted(matches.values())
    prefix = list(range(step_index, step_index + len(calls)))
    if len(matches) == len(calls) and indices == prefix:
        before = gold["initial_state"] if step_index == 0 else gold["states_after"][step_index - 1]
        read_only = all(typed_equal(gold["states_after"][i], before) for i in prefix)
        result["read_only_gold_prefix"] = read_only
        if read_only:
            result["replay_candidate"] = True
            result["reason"] = "All calls exactly match distinct read-only gold-prefix actions; pending independent parallel replay."
        else:
            result["reason"] = "Exact gold-prefix calls mutate state; concurrency/commutativity not proven."
        return result
    if result.get("secondary_subtypes"):
        result["reason"] = "Duplicate calls without proven idempotent, goal-compatible behavior."
    return result


async def replay_candidate(record: dict, episode: dict, source_row: dict) -> dict:
    factory = source_row["extra_info"]["mcp_factory_kwargs"]
    servers = decode(factory["mcp_servers"])
    initial = decode(factory["initial_config"])
    gold = episode["gold"]
    start = record["gold_step"] - 1
    calls = record["tool_calls"]
    env = IsolatedEpisode(record["task_id"], servers, initial)
    await env.open()
    try:
        for action in gold["actions"][:start]:
            _, success, _ = await env.call(action["name"], action["arguments"])
            if not success:
                return {"passed": False, "reason": "prefix_gold_replay_failed"}
        before = await env.save()
        expected_before = gold["initial_state"] if start == 0 else gold["states_after"][start - 1]
        if not typed_equal(before, expected_before):
            return {"passed": False, "reason": "prefix_state_mismatch"}
        results = await asyncio.gather(*(env.call(c["name"], c["arguments"]) for c in calls))
        after = await env.save()
        all_success = all(result[1] is True for result in results)
        states_match = typed_equal(after, gold["states_after"][start + len(calls) - 1])
        matches = {int(k): v - 1 for k, v in record["exact_gold_matches"].items()}
        responses_match = all(typed_equal(results[i][0], gold["observations"][matches[i]])
                              for i in range(len(calls)))
        suffix_success = False
        final_match = False
        if all_success and states_match and responses_match:
            suffix_success = True
            for action in gold["actions"][start + len(calls):]:
                _, success, _ = await env.call(action["name"], action["arguments"])
                if not success:
                    suffix_success = False
                    break
            if suffix_success:
                final_match = typed_equal(await env.save(), gold["final_state"])
        return {
            "passed": bool(all_success and states_match and responses_match
                           and suffix_success and final_match),
            "all_calls_success": all_success, "state_matches_gold_progress": states_match,
            "responses_match_gold": responses_match,
            "gold_suffix_success": suffix_success, "final_state_match": final_match,
            "errors": [result[2] for result in results if result[1] is not True],
        }
    finally:
        await env.close()


def summarize_depth(rows: list[dict]) -> dict:
    steps = sorted(row["gold_step"] for row in rows)
    if not steps:
        return {"count": 0, "step1": 0, "step2": 0, "step3": 0, "step4plus": 0,
                "depth_ge2": 0, "mean": None, "median": None, "p75": None}
    def quantile(q: float) -> float:
        pos = (len(steps) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return round(steps[lo] + (steps[hi] - steps[lo]) * (pos - lo), 3)
    return {"count": len(steps), "step1": sum(v == 1 for v in steps),
            "step2": sum(v == 2 for v in steps), "step3": sum(v == 3 for v in steps),
            "step4plus": sum(v >= 4 for v in steps), "depth_ge2": sum(v >= 2 for v in steps),
            "mean": round(sum(steps) / len(steps), 3), "median": quantile(.5), "p75": quantile(.75)}


def stratified_sample(rows: list[dict]) -> list[dict]:
    quotas = {"TRUE_FAILURE": 20, "SEMANTICALLY_VALID_PARALLEL": 20,
              "PARTIALLY_VALID": 20, "PARSER_ARTIFACT": 10}
    selected = []
    for category, quota in quotas.items():
        group = [r for r in rows if r["semantic_validity"] == category]
        group.sort(key=lambda r: hashlib.sha256(r["task_id"].encode()).hexdigest())
        selected.extend(group[:quota])
    return selected


def audit(run: Path = RUN, output: Path = OUT, report_root: Path = REPORT_ROOT,
          replay_cache: Path | None = None) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite semantic audit")
    source = json.loads(SOURCE.read_text())
    old = read_jsonl(run / "audit_v1/first_divergence_all.jsonl")
    baseline = json.loads((run / "audit_v1/cycle3_dataset_manifest.json").read_text())
    plan_path = run / "rollout_manifest.json"
    if (sha256(plan_path) != baseline["plan_sha256"] or len(old) != 1936
            or baseline["failure_taxonomy"]["PARSER_OR_FORMAT_ERROR"] != 1051):
        raise RuntimeError("Cycle-3 source/audit lock changed")
    fmt = [row for row in old if row["failure_type"] in FORMAT_TYPES]
    records = []
    for row in fmt:
        episode = json.loads((run / "episodes" / (row["task_id"] + ".json")).read_text())
        schema_sha = episode["schema_sha256"]
        schema_path = run / "schemas" / (schema_sha + ".json")
        schemas = json.loads(schema_path.read_text())
        if hashlib.sha256(stable(schemas).encode()).hexdigest() != schema_sha:
            raise RuntimeError("schema hash mismatch")
        query = query_from_row(source[row["official_row_index"]])
        records.append(classify_static(row, episode, schemas, query))
    replay_rows = []
    cached = {}
    if replay_cache is not None:
        cache_rows = read_jsonl(replay_cache)
        cached = {item["task_id"]: item for item in cache_rows}
        if len(cached) != len(cache_rows):
            raise RuntimeError("duplicate cached replay identity")
        expected = {item["task_id"] for item in records if item["replay_candidate"]}
        if set(cached) != expected:
            raise RuntimeError("cached replay candidate identities differ")
    for i, record in enumerate(records):
        if not record["replay_candidate"]:
            continue
        if replay_cache is not None:
            replay = {k: v for k, v in cached[record["task_id"]].items() if k != "task_id"}
        else:
            episode = json.loads((run / "episodes" / (record["task_id"] + ".json")).read_text())
            row = source[episode["source_row_index"]]
            try:
                replay = asyncio.run(asyncio.wait_for(replay_candidate(record, episode, row), timeout=120))
            except Exception as exc:
                replay = {"passed": False, "reason": "replay_exception",
                          "error_type": type(exc).__name__, "error": str(exc)[:200]}
        replay_rows.append({"task_id": record["task_id"], **replay})
        record["counterfactual_replay"] = replay
        if replay["passed"]:
            record.update(semantic_validity="SEMANTICALLY_VALID_PARALLEL",
                          new_failure_type="VALID_PARALLEL_ALTERNATIVE",
                          disposition="DROP_FROM_FD",
                          reason="Independent concurrent replay exactly matched gold responses/progress; remaining gold suffix reached verified final state.")
        else:
            record["reason"] = "Parallel replay did not establish safe gold-compatible progress; not assumed to be model failure."
        if (i + 1) % 20 == 0:
            print(stable({"replay_candidates_seen": i + 1,
                          "replay_completed": len(replay_rows),
                          "valid_parallel": sum(r["passed"] for r in replay_rows)}), flush=True)
    corrected = []
    fmt_index = {r["task_id"]: r for r in records}
    for row in old:
        replacement = fmt_index.get(row["task_id"])
        corrected.append({"task_id": row["task_id"], "source_task_id": row["source_task_id"],
                          "old_failure_type": row["failure_type"],
                          "new_failure_type": replacement["new_failure_type"] if replacement else row["failure_type"],
                          "semantic_validity": replacement["semantic_validity"] if replacement else "NOT_IN_FORMAT_AUDIT",
                          "disposition": replacement["disposition"] if replacement else "KEEP_AS_FD",
                          "gold_step": row["gold_step"], "query_hash": row["query_hash"],
                          "reason": replacement["reason"] if replacement else "Original non-format FD unchanged."})
    kept = [r for r in corrected if r["disposition"] in ("KEEP_AS_FD", "RECLASSIFY_AND_KEEP")]
    counts = collections.Counter(r["semantic_validity"] for r in records)
    disposition = collections.Counter(r["disposition"] for r in records)
    subtypes = collections.Counter(r["case_subtype"] for r in records)
    new_taxonomy = collections.Counter(r["new_failure_type"] for r in kept)
    summary = {
        "verdict": "FORMAT-AUDIT-UNCERTAIN" if counts["UNCERTAIN"] else "CLEAN-FRONTIER-READY",
        "old_fd_total": len(old), "old_format_fd": len(fmt),
        "old_format_all_attempted": baseline["failure_taxonomy"]["PARSER_OR_FORMAT_ERROR"],
        "old_wrong_argument": baseline["failure_taxonomy"]["WRONG_ARGUMENT"],
        "format_semantic_validity": dict(counts), "format_subtypes": dict(subtypes),
        "format_disposition": dict(disposition), "replay_candidates": len(replay_rows),
        "replay_passed": sum(r["passed"] for r in replay_rows),
        "corrected_fd_total": len(kept), "new_wrong_argument":
            new_taxonomy["WRONG_ARGUMENT"] + new_taxonomy["WRONG_ARGUMENT_IN_MULTI_CALL"],
        "wrong_argument_added_from_format": new_taxonomy["WRONG_ARGUMENT_IN_MULTI_CALL"],
        "old_depth": summarize_depth(old), "corrected_depth": summarize_depth(kept),
        "corrected_failure_taxonomy": dict(new_taxonomy),
        "source_plan_sha256": sha256(plan_path),
        "source_fd_sha256": sha256(run / "audit_v1/first_divergence_all.jsonl"),
        "replay_cache_sha256": sha256(replay_cache) if replay_cache else None,
        "protocol": {
            "cycle3_collector": "rejects len(tool_calls)>1 as parallel_tool_calls and executes none",
            "verl_sglang_rollout": "asyncio.gather executes parsed calls concurrently; no max_parallel_calls guard found in inspected block",
            "raw_model_tokens_available": False,
        },
    }
    output.mkdir(parents=True)
    write_jsonl(output / "format_semantic_audit.jsonl", records)
    write_jsonl(output / "counterfactual_replays.jsonl", replay_rows)
    write_jsonl(output / "corrected_first_divergence_manifest.jsonl", corrected)
    write_jsonl(output / "sanity_sample.jsonl", stratified_sample(records))
    (output / "corrected_failure_taxonomy.json").write_text(
        json.dumps(dict(new_taxonomy), indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_root.mkdir(parents=True, exist_ok=True)
    report_name = ("cycle3_format_parser_semantic_audit_v2"
                   if output.name.endswith("_v2") else "cycle3_format_parser_semantic_audit")
    (report_root / (report_name + ".json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (report_root / (report_name + ".md")).write_text(markdown(summary))
    return summary


def markdown(summary: dict) -> str:
    count = summary["format_semantic_validity"]
    disp = summary["format_disposition"]
    depth = summary["corrected_depth"]
    return "\n".join([
        "VERDICT = " + summary["verdict"], "", "# Cycle-3 format/parser semantic validity audit", "",
        "## FORMAT AUDIT", "",
        f"Old format/parser in FD = {summary['old_format_fd']}; all-task primary count = {summary['old_format_all_attempted']}.",
        f"TRUE_FAILURE = {count.get('TRUE_FAILURE',0)}; SEMANTICALLY_VALID_PARALLEL = {count.get('SEMANTICALLY_VALID_PARALLEL',0)}; PARTIALLY_VALID = {count.get('PARTIALLY_VALID',0)}; PARSER_ARTIFACT = {count.get('PARSER_ARTIFACT',0)}; UNCERTAIN = {count.get('UNCERTAIN',0)}.",
        f"KEEP_AS_FD = {disp.get('KEEP_AS_FD',0)}; DROP_FROM_FD = {disp.get('DROP_FROM_FD',0)}; RECLASSIFY_AND_KEEP = {disp.get('RECLASSIFY_AND_KEEP',0)}; UNCERTAIN_EXCLUDE = {disp.get('UNCERTAIN_EXCLUDE',0)}.",
        f"Independent counterfactual replays passed = {summary['replay_passed']}/{summary['replay_candidates']}.", "",
        "## DATASET EFFECT", "",
        f"Old FD total = {summary['old_fd_total']}; corrected conservative FD total = {summary['corrected_fd_total']}.",
        f"Old WRONG_ARGUMENT = {summary['old_wrong_argument']}; corrected = {summary['new_wrong_argument']}; added from format = {summary['wrong_argument_added_from_format']}.",
        f"Corrected step1/2/3/4+ = {depth['step1']}/{depth['step2']}/{depth['step3']}/{depth['step4plus']}; depth>=2 = {depth['depth_ge2']}; mean/median/p75 = {depth['mean']}/{depth['median']}/{depth['p75']}.", "",
        "## PROTOCOL AND LIMITATIONS", "",
        "The actual Cycle-3 collector rejected all multi-call turns; the installed verl/SGLang rollout executes parsed calls concurrently with asyncio.gather. The audit therefore separates collector behavior from semantic validity.",
        "Positive parallel-valid labels require exact distinct read-only gold-prefix actions, no detected future-observation dependency, successful independent concurrent replay, matching responses/progress, and a gold-compatible suffix reaching the verified final state.",
        "Other paths remain UNCERTAIN_EXCLUDE unless tool-schema or provenance evidence proves an error. This is a lower bound on valid parallel alternatives, not an estimate of all such paths.",
        "Only server-parsed tool calls were saved, not raw token text; parser extraction artifacts cannot be proven or ruled out from these artifacts. No new model inference or training was run.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--reuse-replay-from", type=Path)
    args = parser.parse_args()
    print(stable(audit(output=args.output, replay_cache=args.reuse_replay_from)), flush=True)


if __name__ == "__main__":
    main()


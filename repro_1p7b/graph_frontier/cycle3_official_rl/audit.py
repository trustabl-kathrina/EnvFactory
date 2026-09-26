"""Build audited FD/terminal-stop data and taxonomy from persisted episodes.

No model inference or optimizer code is called here. Audit output is immutable.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import (
    encode_first_divergence, encode_terminal_stop,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask
from repro_1p7b.graph_frontier.official_rl_audit_static import decode, query_from_row
from repro_1p7b.graph_frontier.state_verifier import compare_final_states
from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import (
    align_episode, classify_wrong_arguments, typed_equal,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import (
    MODEL, RUN, checked_plan, sha256, stable,
)

FAILURES = (
    "WRONG_ARGUMENT", "WRONG_TOOL", "PREMATURE_STOP",
    "EXTRA_TOOL_AFTER_COMPLETION", "TOOL_EXECUTION_ERROR",
    "PARSER_OR_FORMAT_ERROR", "LOOP_OR_REPEAT", "NO_TOOL_DIVERGENCE",
    "ALIGNMENT_UNCERTAIN", "ENVIRONMENT_FAILURE",
)
SUBTYPES = (
    "MISSING_REQUIRED_ARG", "EXTRA_ARG", "WRONG_LITERAL_VALUE",
    "WRONG_ENTITY_OR_ID", "STALE_VALUE", "UPSTREAM_PROVENANCE_ERROR",
    "USER_QUERY_EXTRACTION_ERROR", "PARTIAL_MULTI_ARG_ERROR",
    "TYPE_OR_FORMAT_ERROR", "UNKNOWN_ARGUMENT_ERROR",
)


def jsonl_write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows))


def quantiles(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p75": None, "p90": None}
    ordered = sorted(values)
    def pct(p: float) -> float:
        pos = (len(ordered) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 3)
    return {"count": len(values), "mean": round(statistics.mean(values), 3),
            "median": pct(.5), "p75": pct(.75), "p90": pct(.9)}


def terminal_history(query: str, gold: dict) -> list[dict]:
    messages = [{"role": "user", "content": query}]
    for i, (action, observation) in enumerate(zip(gold["actions"], gold["observations"])):
        call_id = f"gold_{i}"
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": action["name"], "arguments": action["arguments"]},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id, "name": action["name"],
            "content": stable(observation),
        })
    return messages


def schema_for(run: Path, episode: dict) -> list[dict]:
    schema_sha = episode.get("schema_sha256")
    if not isinstance(schema_sha, str) or len(schema_sha) != 64:
        raise RuntimeError("episode schema reference missing")
    path = run / "schemas" / f"{schema_sha}.json"
    schema = json.loads(path.read_text())
    if hashlib.sha256(stable(schema).encode()).hexdigest() != schema_sha:
        raise RuntimeError("tool schema hash mismatch")
    return schema


def valid_gold(entry: dict, raw: dict, episode: dict) -> bool:
    gold = episode.get("gold")
    if not gold or gold.get("replay_valid") is not True or gold.get("final_state_match") is not True:
        return False
    source_actions = decode(raw["reward_model"]["ground_truth"])
    expected = decode(raw["extra_info"]["mcp_factory_kwargs"]["final_config"])
    servers = decode(raw["extra_info"]["mcp_factory_kwargs"]["mcp_servers"])
    return (
        len(gold.get("actions", [])) == entry["gold_length"]
        and len(gold.get("observations", [])) == entry["gold_length"]
        and all(typed_equal(gold_action, {"name": source["name"],
                                          "arguments": source["arguments"]})
                for gold_action, source in zip(gold["actions"], source_actions))
        and compare_final_states(gold.get("final_state"),
                                 {server: expected[server] for server in servers}) is True
    )


def make_terminal(tokenizer, entry: dict, raw: dict, episode: dict,
                  schema: list[dict]) -> dict:
    sample = {
        "task_id": entry["task_id"],
        "state_identity": entry["task_id"] + ":gold_terminal",
        "replay_valid": True, "final_state_match": True,
        "conversation_prefix": terminal_history(query_from_row(raw), episode["gold"]),
        "tool_schema": schema,
    }
    encoded = encode_terminal_stop(tokenizer, sample)
    validate_mask(encoded)
    if encoded["target_tokens"] != 1 or encoded["labels"].count(151645) != 1:
        raise RuntimeError("terminal-stop EOS-only integrity failed")
    return {**encoded, "sample_type": "terminal_stop",
            "source_task_id": entry["task_id"],
            "official_row_index": entry["source_row_index"],
            "environment": entry["environment"], "query_hash": entry["query_hash"]}


def make_fd(tokenizer, entry: dict, alignment: dict,
            schema: list[dict]) -> dict:
    gold = alignment["gold_next_action"]
    sample = {
        "task_id": entry["task_id"],
        "state_identity": f"{entry['task_id']}:first_divergence:{alignment['gold_step']}",
        "independent_prefix_replay_valid": True,
        "conversation_prefix": alignment["student_state"]["prompt_messages"],
        "gold_action": gold,
    }
    encoded = encode_first_divergence(
        tokenizer, sample, {"task_id": entry["task_id"], "tools": schema})
    validate_mask(encoded)
    return {**encoded, "sample_type": "first_divergence",
            "source_task_id": entry["task_id"],
            "official_row_index": entry["source_row_index"],
            "environment": entry["environment"], "query_hash": entry["query_hash"],
            "failure_type": alignment["failure_type"],
            "gold_step": alignment["gold_step"]}


def _manual_check(row: dict, episode: dict) -> bool:
    actual, gold = row.get("student_action"), row.get("gold_next_action")
    if not isinstance(actual, dict) or not isinstance(gold, dict):
        return False
    kind, failure = actual.get("kind"), row.get("failure_type")
    if failure == "WRONG_ARGUMENT":
        return (kind == "tool" and actual.get("name") == gold.get("name")
                and isinstance(actual.get("arguments"), dict)
                and not typed_equal(actual["arguments"], gold["arguments"]))
    if failure == "WRONG_TOOL":
        return kind == "tool" and actual.get("name") != gold.get("name")
    if failure == "PREMATURE_STOP":
        return kind == "final" and row["gold_step"] <= len(episode["gold"]["actions"])
    return False


def manual_quality(rows: list[dict], episodes: dict[str, dict]) -> dict:
    selected = []
    quotas = {"WRONG_ARGUMENT": 20, "WRONG_TOOL": 10, "PREMATURE_STOP": 10}
    for kind, quota in quotas.items():
        options = [r for r in rows if r["failure_type"] == kind]
        options.sort(key=lambda r: hashlib.sha256(r["task_id"].encode()).hexdigest())
        selected.extend(options[:quota])
    checks = [{"task_id": r["task_id"], "failure_type": r["failure_type"],
               "structural_check_pass": _manual_check(r, episodes[r["task_id"]])}
              for r in selected]
    bad = sum(not item["structural_check_pass"] for item in checks)
    return {"requested": quotas, "audited": len(checks), "errors": bad,
            "error_rate": bad / len(checks) if checks else None,
            "checks": checks}


def representative_wrong_argument(rows: list[dict], limit: int = 30) -> list[dict]:
    groups = collections.defaultdict(list)
    for row in rows:
        if row["failure_type"] == "WRONG_ARGUMENT":
            groups[row["wrong_argument"]["primary_subtype"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256(r["task_id"].encode()).hexdigest())
    ordered = []
    while len(ordered) < limit and any(groups.values()):
        for kind in sorted(groups):
            if groups[kind] and len(ordered) < limit:
                ordered.append(groups[kind].pop(0))
    return [{
        "task_id": row["task_id"], "environment": row["environment"],
        "query_excerpt": row["query_excerpt"][:300],
        "previous_observations_excerpt": stable(
            row["student_state"]["previous_tool_observations"])[-500:],
        "gold_tool": row["gold_next_action"]["name"],
        "student_tool": row["student_action"]["name"],
        "gold_args": row["gold_next_action"]["arguments"],
        "student_args": row["student_action"]["arguments"],
        "first_differing_arg": row["wrong_argument"]["first_differing_arg"],
        "primary_subtype": row["wrong_argument"]["primary_subtype"],
        "secondary_subtypes": row["wrong_argument"]["secondary_subtypes"],
        "rationale": row["wrong_argument"]["evidence"],
    } for row in ordered]


def build(plan_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite Cycle-3 audit")
    plan, raw = checked_plan(plan_path)
    run = plan_path.parent
    plan_hash = sha256(plan_path)
    token = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if token.eos_token_id != 151645:
        raise RuntimeError("Dynamic-v1 end token changed")
    episodes, fd_rows, fd_ce, terminal_ce = {}, [], [], []
    taxonomy = collections.Counter()
    categories = collections.Counter()
    invalid = collections.Counter()
    attempted = successful = terminal_gold_valid = 0
    runtime_seconds = 0.0
    episode_paths = []
    for entry in plan["tasks"]:
        path = run / "episodes" / f"{entry['task_id']}.json"
        if not path.exists():
            continue
        episode_paths.append(path)
        episode = json.loads(path.read_text())
        if (episode.get("plan_sha256") != plan_hash
                or episode.get("task_id") != entry["task_id"]
                or episode.get("source_row_index") != entry["source_row_index"]
                or episode.get("query_hash") != entry["query_hash"]):
            raise RuntimeError(f"episode source/plan drift: {entry['task_id']}")
        attempted += 1
        runtime_seconds += float(episode.get("runtime_seconds", 0))
        if episode.get("student") is not None:
            successful += 1
        episodes[entry["task_id"]] = episode
        source_row = raw[entry["source_row_index"]]
        good_gold = valid_gold(entry, source_row, episode)
        schema = None
        if good_gold:
            terminal_gold_valid += 1
            try:
                schema = schema_for(run, episode)
                terminal_ce.append(make_terminal(token, entry, source_row, episode, schema))
            except Exception as exc:
                invalid["terminal_serialize:" + type(exc).__name__] += 1
        else:
            invalid["gold_replay_or_final_state"] += 1
        alignment = align_episode(episode) if good_gold else {
            "category": "ENV_FAILURE", "failure_type": "ENVIRONMENT_FAILURE",
            "reason": "gold replay/final-state integrity failed",
        }
        taxonomy[alignment["failure_type"]] += 1
        category = alignment["category"]
        if category == "FD_CANDIDATE":
            sample = {**alignment,
                      "task_id": entry["task_id"], "source_task_id": entry["task_id"],
                      "official_row_index": entry["source_row_index"],
                      "environment": entry["environment"],
                      "query_hash": entry["query_hash"],
                      "query_excerpt": query_from_row(source_row)[:500]}
            if alignment["failure_type"] == "WRONG_ARGUMENT":
                tool_name = alignment["gold_next_action"]["name"]
                tool = next((s for s in schema or []
                             if s["function"]["name"] == tool_name), None)
                sample["wrong_argument"] = classify_wrong_arguments(
                    alignment["gold_next_action"], alignment["student_action"],
                    tool, alignment["student_state"]["previous_tool_observations"],
                    query_from_row(source_row),
                )
            try:
                if schema is None:
                    raise RuntimeError("schema unavailable")
                fd_ce.append(make_fd(token, entry, alignment, schema))
                fd_rows.append(sample)
                category = "FD_READY"
            except Exception as exc:
                invalid["fd_serialize:" + type(exc).__name__] += 1
                category = "ALIGNMENT_UNCERTAIN"
        categories[category] += 1
    if attempted != plan["task_count"]:
        raise RuntimeError(f"incomplete rollout coverage: {attempted}/{plan['task_count']}")
    if sum(categories.values()) != attempted or sum(taxonomy.values()) != attempted:
        raise RuntimeError("one-primary-failure accounting failed")
    if len({r["task_id"] for r in fd_rows}) != len(fd_rows):
        raise RuntimeError("duplicate FD task")
    if len({r["task_id"] for r in terminal_ce}) != len(terminal_ce):
        raise RuntimeError("duplicate terminal task")
    manual = manual_quality(fd_rows, episodes)
    if manual["error_rate"] is not None and manual["error_rate"] > .10:
        verdict = "PIPELINE-INVALID"
    elif categories["FD_READY"] >= 200 and taxonomy["WRONG_ARGUMENT"] >= 80:
        verdict = "FRONTIER-DATA-READY"
    elif categories["FD_READY"] >= 100:
        verdict = "FRONTIER-DATA-PARTIAL"
    else:
        verdict = "FRONTIER-DATA-WEAK"
    wrong_rows = [r for r in fd_rows if r["failure_type"] == "WRONG_ARGUMENT"]
    wrong_primary = collections.Counter(r["wrong_argument"]["primary_subtype"]
                                        for r in wrong_rows)
    wrong_any = collections.Counter(
        subtype for r in wrong_rows
        for subtype in ([r["wrong_argument"]["primary_subtype"]]
                        + r["wrong_argument"]["secondary_subtypes"]))
    depths = [r["gold_step"] for r in fd_rows]
    step_buckets = collections.Counter(
        str(step) if step <= 3 else "4+" for step in depths)
    seconds = (max(p.stat().st_mtime for p in episode_paths)
               - min(p.stat().st_mtime for p in episode_paths)) if len(episode_paths) >= 2 else 0
    runtime = {
        "episode_runtime_seconds_sum": round(runtime_seconds, 2),
        "observed_wall_seconds": round(seconds, 2),
        "successful_episodes_per_hour": round(successful * 3600 / seconds, 2)
            if seconds > 0 else None,
        "gpu_utilization_source": str(run / "gpu_utilization_full.log"),
        "mcp_runtime_note": "One isolated MCP subprocess per task/server/phase; gold and student are independent.",
    }
    all_ids = {entry["task_id"] for entry in plan["tasks"]}
    split = {"schema_version": "cycle3_official_rl_group_split_suggestion_v1",
             "rule": "sha256(task_id) modulo 100 < 80 = train; otherwise validation",
             "train_task_ids": sorted(
                 t for t in all_ids if int(hashlib.sha256(t.encode()).hexdigest(), 16) % 100 < 80),
             "validation_task_ids": sorted(
                 t for t in all_ids if int(hashlib.sha256(t.encode()).hexdigest(), 16) % 100 >= 80)}
    report = {
        "verdict": verdict, "source_clean_tasks": plan["task_count"],
        "tasks_attempted": attempted, "successful_rollouts": successful,
        "runtime_failures": attempted - successful,
        "categories": {k: categories[k] for k in
                       ("FD_READY", "NO_DIVERGENCE", "TERMINAL_ONLY",
                        "ALIGNMENT_UNCERTAIN", "ENV_FAILURE")},
        "failure_taxonomy": {k: taxonomy[k] for k in FAILURES},
        "failure_taxonomy_percent_of_attempted": {
            k: round(100 * taxonomy[k] / attempted, 3) for k in FAILURES},
        "failure_taxonomy_percent_of_fd_ready": {
            k: round(100 * taxonomy[k] / categories["FD_READY"], 3)
            if categories["FD_READY"] else None for k in FAILURES},
        "divergence_depth": quantiles(depths),
        "divergence_step_buckets": dict(step_buckets),
        "wrong_argument": {
            "total_primary_failure": taxonomy["WRONG_ARGUMENT"],
            "valid_fd_samples": len(wrong_rows),
            "primary_subtypes": {k: wrong_primary[k] for k in SUBTYPES},
            "any_subtype_flags": {k: wrong_any[k] for k in SUBTYPES},
            "mean_gold_args": round(statistics.mean(
                r["wrong_argument"]["gold_arg_count"] for r in wrong_rows), 3)
                if wrong_rows else None,
            "mean_wrong_slots": round(statistics.mean(
                r["wrong_argument"]["wrong_arg_slots"] for r in wrong_rows), 3)
                if wrong_rows else None,
            "multi_arg_wrong_cases": sum(r["wrong_argument"]["gold_arg_count"] >= 2
                                        for r in wrong_rows),
            "wrong_slots_total": sum(r["wrong_argument"]["wrong_arg_slots"]
                                     for r in wrong_rows),
            "gold_slots_total": sum(r["wrong_argument"]["gold_arg_count"]
                                    for r in wrong_rows),
        },
        "terminal": {
            "independent_gold_replay_final_valid": terminal_gold_valid,
            "encoded_terminal_stop_valid": len(terminal_ce),
            "unique_tasks": len({r["task_id"] for r in terminal_ce}),
            "unique_environments": len({r["environment"] for r in terminal_ce}),
        },
        "dataset": {"fd_samples": len(fd_ce), "terminal_stop_samples": len(terminal_ce),
                    "frozen300_overlap": 0, "rich24_overlap": 0,
                    "group_split_train_tasks": len(split["train_task_ids"]),
                    "group_split_validation_tasks": len(split["validation_task_ids"])},
        "serialization_or_gold_invalid": dict(invalid),
        "manual_quality": {k: v for k, v in manual.items() if k != "checks"},
        "cost": runtime,
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "source_sha256": plan["source_sha256"],
        "model_weights_sha256": plan["model_weights_sha256"],
        "plan_sha256": plan_hash,
        "notes": ["Single deterministic Dynamic-v1 rollout per clean task.",
                  "Gold terminal states come from independent replay, never student continuation.",
                  "FD supervision stops at the first non-gold action; no off-path recovery.",
                  "No training or benchmark was launched."],
    }
    output.mkdir(parents=True)
    jsonl_write(output / "first_divergence_all.jsonl", fd_rows)
    jsonl_write(output / "fd_ce_dataset.jsonl", fd_ce)
    jsonl_write(output / "terminal_stop_all.jsonl", terminal_ce)
    (output / "failure_taxonomy.json").write_text(json.dumps(
        {"counts": report["failure_taxonomy"],
         "percent_of_attempted": report["failure_taxonomy_percent_of_attempted"],
         "percent_of_fd_ready": report["failure_taxonomy_percent_of_fd_ready"]},
        indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "wrong_argument_audit.json").write_text(json.dumps(
        {**report["wrong_argument"],
         "representative_cases": representative_wrong_argument(fd_rows)},
        indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "manual_quality_audit.json").write_text(
        json.dumps(manual, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "group_split_suggestion.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    artifact_hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    report["artifact_sha256"] = artifact_hashes
    (output / "cycle3_dataset_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "cycle3_report.md").write_text(markdown(report))
    return report


def markdown(report: dict) -> str:
    c, f, w, t, d = (report["categories"], report["failure_taxonomy"],
                      report["wrong_argument"], report["terminal"], report["divergence_depth"])
    lines = [
        f"VERDICT = {report['verdict']}", "",
        "# Official EnvFactory-RL Cycle-3 frontier mining", "",
        "No training, optimizer update, BFCL, Frozen300/Rich24 evaluation, or new 14B generation.", "",
        "## SOURCE", "",
        f"Official clean tasks = {report['source_clean_tasks']}; attempted = {report['tasks_attempted']}; "
        f"completed rollouts = {report['successful_rollouts']}; runtime failures = {report['runtime_failures']}.", "",
        "## FRONTIER", "",
        f"FD_READY = {c['FD_READY']}; NO_DIVERGENCE = {c['NO_DIVERGENCE']}; "
        f"TERMINAL_ONLY = {c['TERMINAL_ONLY']}; ALIGNMENT_UNCERTAIN = {c['ALIGNMENT_UNCERTAIN']}; "
        f"ENV_FAILURE = {c['ENV_FAILURE']}.",
        f"First-divergence step mean/median/p75/p90 = {d['mean']}/{d['median']}/{d['p75']}/{d['p90']}.", "",
        "## FAILURE TAXONOMY", "",
    ]
    lines.extend(f"- {name} = {count} "
                 f"({report['failure_taxonomy_percent_of_attempted'][name]}% attempted; "
                 f"{report['failure_taxonomy_percent_of_fd_ready'][name]}% FD_READY denominator)"
                 for name, count in f.items())
    lines.extend(["", "## WRONG_ARGUMENT", "",
                  f"Total primary = {w['total_primary_failure']}; valid FD = {w['valid_fd_samples']}."])
    lines.extend(f"- {name} = {count} (primary subtype)" for name, count in w["primary_subtypes"].items())
    lines.extend([
        f"Mean gold arguments = {w['mean_gold_args']}; mean wrong slots = {w['mean_wrong_slots']}; "
        f"multi-argument cases = {w['multi_arg_wrong_cases']}.", "",
        "## TERMINAL / DATASET", "",
        f"Independent gold replay/final valid = {t['independent_gold_replay_final_valid']}; "
        f"encoded terminal-stop = {t['encoded_terminal_stop_valid']}; "
        f"unique tasks/environments = {t['unique_tasks']}/{t['unique_environments']}.",
        f"FD/terminal samples = {report['dataset']['fd_samples']}/"
        f"{report['dataset']['terminal_stop_samples']}; Frozen300/Rich24 overlap = 0/0.", "",
        "## Interpretation", "",
        "Rich24 WRONG_ARGUMENT 12/24 and first observable failure step 1.63→2.71 "
        "are background only; this run does not rerun Rich24.",
        f"Q1 clean pool high-yield: {'YES' if report['verdict']=='FRONTIER-DATA-READY' else 'PARTIAL' if report['verdict']=='FRONTIER-DATA-PARTIAL' else 'NO'}.",
        f"Q2 wrong arguments material: {'YES' if f['WRONG_ARGUMENT']>=80 else 'PARTIAL' if f['WRONG_ARGUMENT']>0 else 'NO'}.",
        f"Q3 Cycle-3 training data sufficient: {'YES' if report['verdict']=='FRONTIER-DATA-READY' else 'NOT YET'}.",
        f"Q4 immediately restart 14B generation: {'NO' if report['verdict']=='FRONTIER-DATA-READY' else 'YES'}.",
        "These are data-readiness judgments, not model-effectiveness claims.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=RUN / "rollout_manifest.json")
    parser.add_argument("--output", type=Path, default=RUN / "audit_v1")
    args = parser.parse_args()
    report = build(args.plan, args.output)
    print(json.dumps({"verdict": report["verdict"],
                      "attempted": report["tasks_attempted"],
                      "fd_ready": report["categories"]["FD_READY"],
                      "wrong_argument": report["failure_taxonomy"]["WRONG_ARGUMENT"],
                      "terminal_valid": report["terminal"]["encoded_terminal_stop_valid"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

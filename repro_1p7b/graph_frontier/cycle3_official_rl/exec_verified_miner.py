"""Offline execution-verified FD v2; never generates a new model response.

The candidate JSONL is append-only during replay and can be resumed after a
remote refresh. Legacy FD and terminal-stop datasets are strictly read-only.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import typed_equal
from repro_1p7b.graph_frontier.cycle3_official_rl.exec_verified import validate_multicall_transition
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import RUN, sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.semantic_audit import (
    read_jsonl, summarize_depth, write_jsonl,
)
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, query_from_row

OUT = RUN / "exec_verified_v2"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def candidate(record: dict, result: dict, original: dict) -> dict | None:
    """Return a correction only when the first semantic failure is proven."""
    status = result["status"]
    if status in {"VALID_PARALLEL", "VALID_ALTERNATIVE", "UNCERTAIN"}:
        return None
    failure = {
        "DEPENDENCY_VIOLATION": "DEPENDENCY_VIOLATION",
        "RECLASSIFY_WRONG_ARGUMENT": "WRONG_ARGUMENT",
        "RECLASSIFY_WRONG_TOOL": "WRONG_TOOL",
        "EXECUTION_FAILURE": "TOOL_EXECUTION_ERROR",
    }.get(status)
    if failure is None:
        return None
    return {**original, "failure_type": failure,
            "original_failure_type": original["failure_type"],
            "new_failure_type": failure, "alignment_version": "exec_verified_v2",
            "validation_path": result["validation_path"],
            "gold_progress_consumed": result["gold_progress_consumed"],
            "student_call_count": len(record["tool_calls"]),
            "state_equivalent": result["state_equivalent"],
            "suffix_verified": result["suffix_verified"],
            "keep_for_training": True, "reason": result["reason"],
            "exec_validation": result}


def resume_after_parallel(episode: dict, student_step: int, gold_pointer: int) -> dict:
    """Continue exact fast-path alignment after a verified multi-call progress.

    Saved Cycle-3 multi-call turns are terminal in the collector, but this
    explicit pointer logic supports future traces containing later turns.
    """
    gold, student = episode["gold"], episode["student"]
    for step in student["steps"][student_step + 1:]:
        expected_state = (gold["initial_state"] if gold_pointer == 0
                          else gold["states_after"][gold_pointer - 1])
        if not typed_equal(step["state_before_action"], expected_state):
            return {"status": "UNCERTAIN", "reason": "post_parallel_state_mismatch"}
        action = step.get("parsed_action") or {}
        if gold_pointer >= len(gold["actions"]):
            return {"status": "TERMINAL", "reason": "gold_path_complete"}
        expected = gold["actions"][gold_pointer]
        if len((step.get("model_message") or {}).get("tool_calls") or []) > 1:
            return {"status": "UNCERTAIN", "reason": "later_multi_call_needs_separate_execution_validation"}
        if action.get("kind") != "tool":
            return {"status": "DEEPER_FD", "failure_type": "PREMATURE_STOP",
                    "gold_step": gold_pointer + 1, "student_step": step["step_index"] + 1}
        if action.get("name") != expected["name"]:
            return {"status": "DEEPER_FD", "failure_type": "WRONG_TOOL",
                    "gold_step": gold_pointer + 1, "student_step": step["step_index"] + 1}
        if not typed_equal(action.get("arguments"), expected["arguments"]):
            return {"status": "DEEPER_FD", "failure_type": "WRONG_ARGUMENT",
                    "gold_step": gold_pointer + 1, "student_step": step["step_index"] + 1}
        event = step.get("typed_event") or {}
        if (event.get("execution_success") is not True
                or not typed_equal(event.get("state_after"), gold["states_after"][gold_pointer])
                or not typed_equal(event.get("tool_response"), gold["observations"][gold_pointer])):
            return {"status": "UNCERTAIN", "reason": "post_parallel_exact_execution_mismatch"}
        gold_pointer += 1
    return {"status": "TRUNCATED_AFTER_PARALLEL", "gold_pointer": gold_pointer,
            "reason": "no_saved_student_turn_after_parallel"}


def _checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    result = {r["task_id"]: r for r in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate checkpoint identity: {path}")
    return result


def run_regression(run: Path = RUN, out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    source = _read(SOURCE)
    static = read_jsonl(run / "semantic_audit_v2/format_semantic_audit.jsonl")
    known = {r["task_id"]: r for r in read_jsonl(
        run / "semantic_audit_v2/counterfactual_replays.jsonl")}
    records = [r for r in static if r.get("replay_candidate")]
    if len(records) != 125 or set(known) != {r["task_id"] for r in records}:
        raise RuntimeError("125-candidate regression identity lock failed")
    path = out / "regression_125.jsonl"
    done = _checkpoint(path)
    with path.open("a", encoding="utf-8") as stream:
        for i, record in enumerate(records):
            task_id = record["task_id"]
            if task_id in done:
                continue
            episode = _read(run / "episodes" / (task_id + ".json"))
            row = source[episode["source_row_index"]]
            schemas = _read(run / "schemas" / (episode["schema_sha256"] + ".json"))
            try:
                verdict = asyncio.run(asyncio.wait_for(validate_multicall_transition(
                    record, episode, row, schemas, query_from_row(row)), timeout=120))
            except Exception as exc:
                verdict = {"status": "UNCERTAIN", "reason": "replay_exception",
                           "error_type": type(exc).__name__, "error": str(exc)[:240]}
            result = {"task_id": task_id, "known_passed": known[task_id]["passed"],
                      "validator": verdict}
            stream.write(stable(result) + "\n")
            stream.flush()
            done[task_id] = result
            if (i + 1) % 20 == 0:
                print(stable({"regression_completed": len(done), "of": 125}), flush=True)
    disagree = [r for r in done.values() if r["known_passed"] !=
                (r["validator"]["status"] == "VALID_PARALLEL")]
    summary = {"known_candidates": 125, "known_valid_parallel": 123,
               "validator_reproduced": 125 - len(disagree), "disagreements": len(disagree),
               "disagreement_task_ids": sorted(r["task_id"] for r in disagree),
               "gate_pass": len(disagree) <= 6,
               "source_replay_sha256": sha256(run / "semantic_audit_v2/counterfactual_replays.jsonl")}
    (out / "regression_summary.json").write_text(stable(summary) + "\n")
    return summary


def run_full(run: Path = RUN, out: Path = OUT) -> dict:
    regression = _read(out / "regression_summary.json")
    if not regression["gate_pass"]:
        raise RuntimeError("regression gate failed; refusing full remine")
    source = _read(SOURCE)
    old = read_jsonl(run / "audit_v1/first_divergence_all.jsonl")
    old_index = {r["task_id"]: r for r in old}
    static_index = {r["task_id"]: r for r in read_jsonl(
        run / "semantic_audit_v2/format_semantic_audit.jsonl")}
    regression_cache = _checkpoint(out / "regression_125.jsonl")
    episodes = sorted((run / "episodes").glob("*.json"))
    if len(episodes) != 2158 or len(old) != 1936:
        raise RuntimeError("saved Cycle-3 source count changed")
    path = out / "multicall_replay_checkpoint.jsonl"
    done = _checkpoint(path)
    with path.open("a", encoding="utf-8") as stream:
        for i, episode_path in enumerate(episodes):
            task_id = episode_path.stem
            record = static_index.get(task_id)
            if not record or record["tool_call_count"] < 2 or task_id in done:
                continue
            episode = _read(episode_path)
            row = source[episode["source_row_index"]]
            schemas = _read(run / "schemas" / (episode["schema_sha256"] + ".json"))
            if task_id in regression_cache:
                verdict = regression_cache[task_id]["validator"]
            else:
                try:
                    verdict = asyncio.run(asyncio.wait_for(validate_multicall_transition(
                        record, episode, row, schemas, query_from_row(row)), timeout=120))
                except Exception as exc:
                    verdict = {"status": "UNCERTAIN", "reason": "replay_exception",
                               "error_type": type(exc).__name__, "error": str(exc)[:240]}
            result = {"task_id": task_id, "validator": verdict}
            stream.write(stable(result) + "\n")
            stream.flush()
            done[task_id] = result
            if (i + 1) % 100 == 0:
                print(stable({"episodes_seen": i + 1, "multicall_checked": len(done)}), flush=True)
    if len(done) != sum(r["tool_call_count"] >= 2 for r in static_index.values()):
        raise RuntimeError("multi-call checkpoint incomplete")
    corrected, alignment_rows = [], []
    new_deeper = []
    for episode_path in episodes:
        task_id = episode_path.stem
        old_row = old_index.get(task_id)
        record = static_index.get(task_id)
        if record and record["tool_call_count"] >= 2:
            verdict = done[task_id]["validator"]
            alignment = {"task_id": task_id, "old_failure_type": old_row["failure_type"],
                         "student_step": record["student_step"], "gold_step": record["gold_step"],
                         "student_call_count": record["tool_call_count"], **verdict}
            if verdict["status"] in {"VALID_PARALLEL", "VALID_ALTERNATIVE"}:
                episode = _read(episode_path)
                next_step = resume_after_parallel(episode, record["student_step"] - 1,
                                                  record["gold_step"] - 1 + verdict["gold_progress_consumed"])
                alignment["realignment"] = next_step
                if next_step["status"] == "DEEPER_FD":
                    new_deeper.append({"task_id": task_id, **next_step})
                    deeper_step = episode["student"]["steps"][next_step["student_step"] - 1]
                    prompt_count = deeper_step["prompt_message_count"]
                    corrected.append({**old_row,
                        "failure_type": next_step["failure_type"],
                        "original_failure_type": old_row["failure_type"],
                        "new_failure_type": next_step["failure_type"],
                        "gold_step": next_step["gold_step"],
                        "student_step": next_step["student_step"],
                        "gold_next_action": episode["gold"]["actions"][next_step["gold_step"] - 1],
                        "student_action": deeper_step.get("parsed_action"),
                        "student_state": {
                            "prompt_messages": episode["student"]["messages"][:prompt_count],
                            "environment_state": deeper_step["state_before_action"],
                            "previous_tool_observations": episode["gold"]["observations"][:next_step["gold_step"] - 1],
                        },
                        "alignment_version": "exec_verified_v2",
                        "validation_path": "state_equivalence",
                        "gold_progress_consumed": verdict["gold_progress_consumed"],
                        "student_call_count": 1, "state_equivalent": verdict["state_equivalent"],
                        "suffix_verified": verdict["suffix_verified"],
                        "keep_for_training": True,
                        "reason": "deeper_semantic_fd_after_verified_parallel_realign"})
            else:
                new_row = candidate(record, verdict, old_row)
                if new_row:
                    corrected.append(new_row)
            alignment_rows.append(alignment)
        elif old_row:
            is_budget_truncation = old_row["failure_type"] == "PARSER_OR_FORMAT_ERROR"
            new_type = ("GENERATION_LENGTH_TRUNCATED" if is_budget_truncation
                        else old_row["failure_type"])
            corrected.append({**old_row, "failure_type": new_type,
                              "original_failure_type": old_row["failure_type"],
                              "new_failure_type": new_type,
                              "alignment_version": "exec_verified_v2",
                              "validation_path": "exact_match", "gold_progress_consumed": 0,
                              "student_call_count": 1, "state_equivalent": False,
                              "suffix_verified": False,
                              "keep_for_training": not is_budget_truncation,
                              "reason": ("generation_length_budget_truncation_excluded"
                                         if is_budget_truncation else
                                         "unchanged_exact_single_action_fast_path")})
            alignment_rows.append({"task_id": task_id, "old_failure_type": old_row["failure_type"],
                                   "status": "UNCHANGED_EXACT_FAST_PATH", "validation_path": "exact_match"})
        else:
            alignment_rows.append({"task_id": task_id, "status": "NO_ORIGINAL_FD"})
    taxonomy = collections.Counter(r["new_failure_type"] for r in corrected)
    taxonomy.update({"TRUE_FORMAT_ERROR": 0, "PARSER_ARTIFACT": 0})
    statuses = collections.Counter(r["status"] for r in alignment_rows)
    previous = read_jsonl(run / "semantic_audit_v2/format_semantic_audit.jsonl")
    previous_uncertain = {r["task_id"] for r in previous if r["semantic_validity"] == "UNCERTAIN"}
    resolved = collections.Counter(done[t]["validator"]["status"] for t in previous_uncertain
                                   if t in done)
    summary = {
        "alignment_version": "exec_verified_v2", "old_fd_total": len(old),
        "new_fd_total": len(corrected), "old_conservative_usable_fd": 918,
        "new_conservative_usable_fd": sum(r["keep_for_training"] for r in corrected),
        "regression": regression, "old_depth": summarize_depth(old),
        "new_depth": summarize_depth(corrected), "taxonomy": dict(taxonomy),
        "multicall_status": dict(collections.Counter(r["validator"]["status"] for r in done.values())),
        "previous_uncertain": len(previous_uncertain),
        "previous_uncertain_resolution": dict(resolved),
        "newly_exposed_deeper_fd": len(new_deeper),
        "newly_exposed_deeper_types": dict(collections.Counter(r["failure_type"] for r in new_deeper)),
        "no_saved_student_turn_after_multicall": True,
        "alignment_status": dict(statuses),
        "source_episode_count": len(episodes),
        "source_fd_sha256": sha256(run / "audit_v1/first_divergence_all.jsonl"),
        "source_terminal_sha256": sha256(run / "audit_v1/terminal_stop_all.jsonl"),
    }
    write_jsonl(out / "first_divergence_exec_verified_v2.jsonl", corrected)
    write_jsonl(out / "exec_verified_alignment_audit.jsonl", alignment_rows)
    write_jsonl(out / "newly_exposed_deeper_fd.jsonl", new_deeper)
    (out / "failure_taxonomy_exec_verified_v2.json").write_text(stable(dict(taxonomy)) + "\n")
    (out / "summary.json").write_text(stable(summary) + "\n")
    return summary


def materialize_report(run: Path = RUN, out: Path = OUT) -> dict:
    """Finalize a typed Parquet audit and human-readable report after mining."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    summary = _read(out / "summary.json")
    audit = read_jsonl(out / "exec_verified_alignment_audit.jsonl")
    corrected = read_jsonl(out / "first_divergence_exec_verified_v2.jsonl")
    replay = read_jsonl(out / "multicall_replay_checkpoint.jsonl")
    if (len(audit) != 2158 or len(replay) != 1040
            or len(corrected) != summary["new_fd_total"]
            or sha256(run / "audit_v1/terminal_stop_all.jsonl") != summary["source_terminal_sha256"]
            or sha256(run / "audit_v1/first_divergence_all.jsonl") != summary["source_fd_sha256"]):
        raise RuntimeError("final source/output integrity gate failed")
    columns = {
        "task_id": [r["task_id"] for r in audit],
        "status": [r.get("status") for r in audit],
        "validation_path": [r.get("validation_path") for r in audit],
        "old_failure_type": [r.get("old_failure_type") for r in audit],
        "gold_step": [r.get("gold_step") for r in audit],
        "student_step": [r.get("student_step") for r in audit],
        "student_call_count": [r.get("student_call_count") for r in audit],
        "gold_progress_consumed": [r.get("gold_progress_consumed") for r in audit],
        "state_equivalent": [r.get("state_equivalent") for r in audit],
        "suffix_verified": [r.get("suffix_verified") for r in audit],
        "reason": [r.get("reason") for r in audit],
        "record_json": [stable(r) for r in audit],
    }
    pq.write_table(pa.Table.from_pydict(columns), out / "exec_verified_alignment_audit.parquet",
                   compression="zstd")
    status = summary["multicall_status"]
    taxonomy = summary["taxonomy"]
    direct_wrong_argument = sum(r["new_failure_type"] == "WRONG_ARGUMENT"
                                and r["validation_path"] == "exact_match" for r in corrected)
    multi_wrong_argument = taxonomy.get("WRONG_ARGUMENT", 0) - direct_wrong_argument
    harmless = sum(r["validator"].get("redundant_read_calls", 0) > 0 for r in replay)
    still_uncertain = status.get("UNCERTAIN", 0)
    verdict = "EXEC-VERIFIED-PARTIAL" if still_uncertain else "EXEC-VERIFIED-FRONTIER-READY"
    old_depth, new_depth = summary["old_depth"], summary["new_depth"]
    prior = summary["previous_uncertain_resolution"]
    lines = [
        f"VERDICT = {verdict}", "", "# Cycle-3 Execution-Verified FD v2", "",
        "## REGRESSION", "",
        "known candidates = 125; known valid parallel = 123; validator reproduced = "
        f"{summary['regression']['validator_reproduced']}; disagreements = "
        f"{summary['regression']['disagreements']}.", "",
        "## OLD VS NEW", "",
        "| Metric | Old | Exec-Verified v2 |", "|---|---:|---:|",
        f"| FD total | 1936 | {summary['new_fd_total']} |",
        f"| Conservative usable FD | 918 | {summary['new_conservative_usable_fd']} |",
        f"| step1 | {old_depth['step1']} | {new_depth['step1']} |",
        f"| step2 | {old_depth['step2']} | {new_depth['step2']} |",
        f"| step3 | {old_depth['step3']} | {new_depth['step3']} |",
        f"| step4+ | {old_depth['step4plus']} | {new_depth['step4plus']} |",
        f"| depth>=2 | {old_depth['depth_ge2']} | {new_depth['depth_ge2']} |",
        f"| WRONG_ARGUMENT | 407 | {taxonomy.get('WRONG_ARGUMENT', 0)} |",
        f"| uncertain multi-call | 895 previous semantic audit | {still_uncertain} |",
        "", f"Mean/median gold divergence step = {new_depth['mean']}/{new_depth['median']}.",
        f"New WRONG_ARGUMENT = {direct_wrong_argument} unchanged direct + "
        f"{multi_wrong_argument} multi-call reclassified.", "",
        "## MULTI-CALL", "",
        f"Candidates = {len(replay)}; valid parallel = {status.get('VALID_PARALLEL', 0)}; "
        f"valid fixed-suffix alternatives = {status.get('VALID_ALTERNATIVE', 0)}; "
        f"jointly state-equivalent extra-call cases = {harmless} "
        "(individual read-only effects not separately proven).",
        f"Dependency violations = {status.get('DEPENDENCY_VIOLATION', 0)}; "
        f"wrong arguments = {status.get('RECLASSIFY_WRONG_ARGUMENT', 0)}; "
        f"wrong tools = {status.get('RECLASSIFY_WRONG_TOOL', 0)}; "
        f"execution failures = {status.get('EXECUTION_FAILURE', 0)}; "
        f"still uncertain = {still_uncertain}.",
        f"Previous 895 uncertain resolved as valid parallel = {prior.get('VALID_PARALLEL', 0)}, "
        f"valid fixed-suffix alternative = {prior.get('VALID_ALTERNATIVE', 0)}, "
        f"execution failure = {prior.get('EXECUTION_FAILURE', 0)}, "
        f"wrong argument = {prior.get('RECLASSIFY_WRONG_ARGUMENT', 0)}, "
        f"wrong tool = {prior.get('RECLASSIFY_WRONG_TOOL', 0)}, "
        f"still uncertain = {prior.get('UNCERTAIN', 0)}.", "",
        "## FRONTIER MIGRATION", "",
        f"Newly exposed deeper FD = {summary['newly_exposed_deeper_fd']}.",
        "The saved Cycle-3 collector rejects multi-call output and stops immediately: "
        "zero multi-call episodes contain a later saved student turn. "
        "Re-alignment is implemented and CPU-tested, but deeper model failures "
        "cannot be observed from these fixed artifacts without new inference.", "",
        "## LIMITATIONS AND ANSWERS", "",
        "Q1: YES. Independent replay removes many exact-sequence false positives.",
        "Q2: PARTIAL. Same-state gold-compatible cases are validated without search; "
        "unproved alternative paths remain excluded.",
        "Q3: NO observable deeper FD in these saved traces, because the collector "
        "terminated every multi-call turn.",
        "Q4: NOT YET for a complete Cycle-3 training composition; the high-confidence "
        "FD subset may inform a later design, but uncertainty and truncated continuations remain.",
        "Raw generation tokens were not saved, so parser artifacts cannot be conclusively "
        "excluded even though none was confirmed. Nine finish_reason=length budget "
        "truncations remain in the FD taxonomy but are excluded from training. "
        "No model inference, training, or search was run.",
        "Terminal-stop and original FD SHA256 values remained unchanged: "
        f"{summary['source_terminal_sha256']} and {summary['source_fd_sha256']}.", "",
    ]
    # Human-readable reports are authored locally, not written to the remote.
    result = {"verdict": verdict, "audit_parquet_rows": len(audit),
              "corrected_fd_rows": len(corrected), "multicall_rows": len(replay),
              "terminal_sha256": summary["source_terminal_sha256"],
              "parquet_sha256": sha256(out / "exec_verified_alignment_audit.parquet")}
    (out / "materialization_check.json").write_text(stable(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("regression", "full", "report"))
    args = parser.parse_args()
    result = (run_regression() if args.phase == "regression" else
              run_full() if args.phase == "full" else materialize_report())
    print(stable(result), flush=True)


if __name__ == "__main__":
    main()


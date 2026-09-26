"""Execution-verified, one-first-error-per-task Recursive OPD miner.

Non-exact multi-call decisions use the established Cycle-3 FD-v2 replay
validator. Ambiguous alternatives and any off-gold state are excluded rather
than converted to a training target. Mining never controls rollout length.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
from pathlib import Path

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_first_divergence, encode_terminal_stop
from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import typed_equal
from repro_1p7b.graph_frontier.cycle3_official_rl.exec_verified import validate_multicall_transition
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode, atomic_json
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, decode, query_from_row
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN
from repro_1p7b.graph_frontier.state_verifier import compare_final_states
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


def result(status: str, depth: int, length: int, *, regime: str | None = None,
           reason: str = "", step: dict | None = None, gold_action: dict | None = None,
           **evidence) -> dict:
    if length < 1 or not 0 <= depth <= length:
        raise RuntimeError("frontier depth out of range")
    return {"status": status, "frontier_depth": depth, "gold_tool_count": length,
            "normalized_frontier": round(depth / length, 8), "error_regime": regime,
            "reason": reason, "student_step": None if step is None else step["step_index"],
            "gold_next_action": gold_action, **evidence}


def prefix_for(student: dict, step: dict) -> list[dict]:
    count = step["prompt_message_count"]
    messages = student["messages"]
    if not isinstance(count, int) or count < 1 or count > len(messages):
        raise RuntimeError("student prompt boundary missing")
    return messages[:count]


def exact_parallel_progress(step: dict, gold: dict, pointer: int) -> int:
    calls = step.get("structured_calls") or []
    if not calls or pointer + len(calls) > len(gold["actions"]):
        return 0
    if step.get("parse_errors") or len(step.get("typed_events") or []) != len(calls):
        return 0
    for offset, (call, event) in enumerate(zip(calls, step["typed_events"])):
        action = gold["actions"][pointer + offset]
        if (not typed_equal({"name": call["name"], "arguments": call["arguments"]}, action)
                or event.get("execution_success") is not True
                or not typed_equal(event.get("tool_response"), gold["observations"][pointer + offset])):
            return 0
    if not typed_equal(step["state_after_action"], gold["states_after"][pointer + len(calls) - 1]):
        return 0
    return len(calls)


async def suffix_valid_from_state(entry: dict, row: dict, state: dict,
                                  gold: dict, pointer: int) -> bool | None:
    factory = row["extra_info"]["mcp_factory_kwargs"]
    env = IsolatedEpisode(entry["audit_id"], decode(factory["mcp_servers"]), state)
    try:
        await env.open()
        for action in gold["actions"][pointer:]:
            _, ok, _ = await env.call(action["name"], action["arguments"])
            if not ok:
                return None
        return typed_equal(await env.save(), gold["final_state"])
    except Exception:
        return None
    finally:
        await env.close()


async def classify(episode: dict, entry: dict, row: dict, schema: list[dict]) -> dict:
    gold, student = episode["gold"], episode["student"]
    length = len(gold["actions"])
    if length < 1 or not student or not student.get("steps"):
        return result("UNCERTAIN", 0, length or 1, reason="missing trajectory")
    expected_final = decode(row["extra_info"]["mcp_factory_kwargs"]["final_config"])
    servers = decode(row["extra_info"]["mcp_factory_kwargs"]["mcp_servers"])
    final_ok = compare_final_states(student["final_state"],
                                    {server: expected_final[server] for server in servers}) is True
    pointer = 0
    for step in student["steps"]:
        before = gold["initial_state"] if pointer == 0 else gold["states_after"][pointer - 1]
        if not typed_equal(step["state_before_action"], before):
            return result("UNCERTAIN", pointer, length, reason="off-gold state after alternative")
        calls = step.get("structured_calls") or []
        raw_calls = (step.get("model_message") or {}).get("tool_calls") or []
        if pointer == length:
            if raw_calls:
                if not typed_equal(step["state_before_action"], gold["final_state"]):
                    return result("UNCERTAIN", pointer, length, reason="terminal state not exact")
                return result("FIRST_ERROR", pointer, length, regime="TERMINATE",
                              reason="tool call after verified terminal state", step=step,
                              prompt_messages=prefix_for(student, step))
            if student.get("termination_reason") == "MODEL_FINAL" and final_ok:
                return result("SUCCESS", length, length, reason="gold progress and termination verified")
            return result("UNCERTAIN", pointer, length, reason="terminal format or final state uncertain")
        if not raw_calls:
            if student.get("termination_reason") != "MODEL_FINAL":
                return result("UNCERTAIN", pointer, length, reason="generation-length/format truncation")
            return result("FIRST_ERROR", pointer, length,
                          regime="START" if pointer == 0 else "ADVANCE",
                          reason="premature model stop", step=step,
                          gold_action=gold["actions"][pointer],
                          prompt_messages=prefix_for(student, step))
        if step.get("parse_errors"):
            return result("UNCERTAIN", pointer, length, reason="malformed structured tool call")
        consumed = exact_parallel_progress(step, gold, pointer)
        if consumed:
            pointer += consumed
            continue
        if len(calls) > 1:
            record = {"task_id": entry["audit_id"], "gold_step": pointer + 1,
                      "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls],
                      "replay_candidate": False}
            verdict = await validate_multicall_transition(
                record, {"gold": gold}, row, schema, query_from_row(row))
            if verdict["status"] in {"VALID_PARALLEL", "VALID_ALTERNATIVE"}:
                consumed = verdict["gold_progress_consumed"]
                if consumed < 1 or pointer + consumed > length:
                    return result("UNCERTAIN", pointer, length, reason="invalid FD-v2 pointer advancement")
                pointer += consumed
                continue
            if verdict["status"] in {"DEPENDENCY_VIOLATION", "EXECUTION_FAILURE",
                                     "RECLASSIFY_WRONG_ARGUMENT", "RECLASSIFY_WRONG_TOOL"}:
                if any(typed_equal({"name": c["name"], "arguments": c["arguments"]},
                                   gold["actions"][pointer]) for c in calls):
                    return result("UNCERTAIN", pointer, length,
                                  reason="partial multi-call progress cannot target pre-call state",
                                  fd_v2_status=verdict["status"])
                return result("FIRST_ERROR", pointer, length,
                              regime="START" if pointer == 0 else "ADVANCE",
                              reason="FD-v2 execution-confirmed multi-call failure", step=step,
                              gold_action=gold["actions"][pointer],
                              prompt_messages=prefix_for(student, step),
                              fd_v2_status=verdict["status"])
            return result("UNCERTAIN", pointer, length, reason="FD-v2 multi-call uncertain",
                          fd_v2_status=verdict["status"])
        if len(calls) != 1 or len(step.get("typed_events") or []) != 1:
            return result("UNCERTAIN", pointer, length, reason="untyped single call")
        call, event = calls[0], step["typed_events"][0]
        if event["execution_success"] is not True:
            return result("FIRST_ERROR", pointer, length,
                          regime="START" if pointer == 0 else "ADVANCE",
                          reason="single tool execution failed", step=step,
                          gold_action=gold["actions"][pointer],
                          prompt_messages=prefix_for(student, step))
        if (typed_equal(step["state_after_action"], gold["states_after"][pointer])
                and typed_equal(event["tool_response"], gold["observations"][pointer])):
            pointer += 1
            continue
        # A harmless read at the same state can be followed by the gold path.
        if typed_equal(step["state_after_action"], step["state_before_action"]):
            if await suffix_valid_from_state(entry, row, step["state_after_action"], gold, pointer):
                continue
        # A changed state is a correction target only when executing the
        # fixed gold suffix from that state cannot reach the verified final.
        elif await suffix_valid_from_state(entry, row, step["state_after_action"], gold, pointer) is False:
            return result("FIRST_ERROR", pointer, length,
                          regime="START" if pointer == 0 else "ADVANCE",
                          reason="executed non-gold action makes gold suffix fail", step=step,
                          gold_action=gold["actions"][pointer],
                          prompt_messages=prefix_for(student, step))
        return result("UNCERTAIN", pointer, length,
                      reason="nonexact successful action may be a legal alternative")
    if pointer == length and student.get("termination_reason") == "MODEL_FINAL" and final_ok:
        return result("SUCCESS", length, length, reason="full verified success")
    return result("UNCERTAIN", pointer, length,
                  reason="max-turn exhaustion or no verifiable next decision")


def mine(cycle: int, *, limit: int | None = None) -> dict:
    if cycle not in range(4):
        raise ValueError("cycle must be 0..3")
    protocol = check()
    root = RUN / f"cycle{cycle}"
    manifest = json.loads((root / "rollout_manifest.json").read_text())
    if manifest["protocol_sha256"] != sha256(PROTOCOL) or manifest["cycle"] != cycle:
        raise RuntimeError("rollout manifest drift")
    source = json.loads(SOURCE.read_text())
    by_id = {r["audit_id"]: r for r in json.loads(
        (RUN.parent / "official_rl_audit/opd_source_manifest.json").read_text())["rows"]}
    ids = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    if limit is not None:
        ids = ids[:limit]
    records = []
    for position, task_id in enumerate(ids):
        path = root / "rollouts" / f"{task_id}.json"
        if not path.exists():
            raise RuntimeError(f"missing rollout: {task_id}")
        episode = json.loads(path.read_text())
        entry = by_id[task_id]
        gold_source = ((RUN / "gold_cache" / f"{task_id}.json")
                       if episode.get("gold_source_kind") == "independent_replay_cache"
                       else (GOLD_RUN / "episodes" / f"{task_id}.json"))
        if (episode["task_id"] != task_id or episode["model_sha256"] != manifest["model_sha256"]
                or episode["manifest_sha256"] != sha256(root / "rollout_manifest.json")
                or episode["pool"] != ("train" if task_id in set(protocol["train_task_ids"]) else "heldout")
                or episode["gold_source_episode_sha256"] != sha256(gold_source)
                or not valid_gold(entry, source[entry["source_row_index"]], episode)):
            raise RuntimeError(f"rollout/gold/source identity invalid: {task_id}")
        schema = schema_for(RUN if episode.get("gold_source_kind") == "independent_replay_cache"
                            else GOLD_RUN, episode)
        if episode.get("student_error") or not episode.get("student"):
            record = result("UNCERTAIN", 0, len(episode["gold"]["actions"]),
                            reason="student runtime failure")
        else:
            try:
                record = asyncio.run(classify(episode, entry, source[entry["source_row_index"]], schema))
            except Exception as exc:
                record = result("UNCERTAIN", 0, len(episode["gold"]["actions"]),
                                reason=f"verifier exception:{type(exc).__name__}:{str(exc)[:200]}")
        record.update(task_id=task_id, cycle=cycle, pool=episode["pool"],
                      environment=entry["environment"], query_hash=entry["query_hash"],
                      termination_reason=(episode.get("student") or {}).get("termination_reason"))
        records.append(record)
        if (position + 1) % 100 == 0:
            print(stable({"mined": position + 1, "of": len(ids)}), flush=True)
    if limit is None and len(records) != 2158:
        raise RuntimeError("full mining incomplete")
    train = [r for r in records if r["pool"] == "train"]
    heldout = [r for r in records if r["pool"] == "heldout"]
    corrections = [r for r in train if r["status"] == "FIRST_ERROR"]
    if len({r["task_id"] for r in corrections}) != len(corrections):
        raise RuntimeError("more than one correction per task")
    metrics = {}
    for name, pool in (("train", train), ("heldout", heldout)):
        depth = [r["frontier_depth"] for r in pool]
        metrics[name] = {
            "tasks": len(pool), "success": sum(r["status"] == "SUCCESS" for r in pool),
            "first_action_accuracy": (sum(r["frontier_depth"] > 0 for r in pool) / len(pool)
                                      if pool else None),
            "start_error": sum(r["error_regime"] == "START" for r in pool),
            "advance_error": sum(r["error_regime"] == "ADVANCE" for r in pool),
            "terminate_error": sum(r["error_regime"] == "TERMINATE" for r in pool),
            "uncertain": sum(r["status"] == "UNCERTAIN" for r in pool),
            "mean_frontier_depth": statistics.mean(depth) if pool else None,
            "median_frontier_depth": statistics.median(depth) if pool else None,
            "mean_normalized_frontier": (statistics.mean(r["normalized_frontier"] for r in pool)
                                         if pool else None),
            "verified_terminal_reached": sum(r["frontier_depth"] == r["gold_tool_count"] for r in pool),
            "correct_termination": sum(r["status"] == "SUCCESS" for r in pool),
            "over_continue": sum(r["error_regime"] == "TERMINATE" for r in pool),
            "max_turn_exhaustion": sum(r["termination_reason"] == "MAX_TURNS" for r in pool),
        }
    if limit is None and (metrics["train"]["tasks"] != 1726 or metrics["heldout"]["tasks"] != 432):
        raise RuntimeError("split count drift")
    output = root / ("smoke_mining" if limit is not None else "metrics")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite mining outputs: {output}")
    output.mkdir(parents=True)
    (output / "per_task.jsonl").write_text("".join(stable(r) + "\n" for r in records))
    (output / "summary.json").write_text(stable({"cycle": cycle, "metrics": metrics,
        "corrections": len(corrections),
        "natural_regimes": dict(collections.Counter(r["error_regime"] for r in corrections)),
        "rollout_manifest_sha256": sha256(root / "rollout_manifest.json"),
        "per_task_sha256": sha256(output / "per_task.jsonl")}) + "\n")
    return metrics


def build_dataset(cycle: int) -> dict:
    from transformers import AutoTokenizer
    if cycle not in range(3):
        raise ValueError("only D0,D1,D2 are trained")
    protocol = check()
    root = RUN / f"cycle{cycle}"
    metrics = json.loads((root / "metrics/summary.json").read_text())
    rows = [json.loads(x) for x in (root / "metrics/per_task.jsonl").read_text().splitlines() if x]
    if (len(rows) != 2158 or metrics["per_task_sha256"] != sha256(root / "metrics/per_task.jsonl")
            or metrics["metrics"]["train"]["tasks"] != 1726
            or metrics["metrics"]["heldout"]["tasks"] != 432):
        raise RuntimeError("mining metrics incomplete")
    output = root / "first_error_dataset"
    if output.exists():
        raise RuntimeError("refusing to overwrite first-error dataset")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("Qwen3 EOS token drift")
    encoded, skipped = [], []
    for record in rows:
        if record["status"] != "FIRST_ERROR" or record["pool"] != "train":
            continue
        episode = json.loads((root / "rollouts" / f"{record['task_id']}.json").read_text())
        schema = schema_for(RUN if episode.get("gold_source_kind") == "independent_replay_cache"
                            else GOLD_RUN, episode)
        if record["error_regime"] == "TERMINATE":
            sample = {"task_id": record["task_id"],
                      "state_identity": f"{record['task_id']}:cycle{cycle}:terminal",
                      "replay_valid": True, "final_state_match": True,
                      "conversation_prefix": record["prompt_messages"], "tool_schema": schema}
            try:
                item = encode_terminal_stop(tokenizer, sample)
            except ValueError as exc:
                skipped.append({"task_id": record["task_id"], "reason": str(exc)})
                continue
        else:
            sample = {"task_id": record["task_id"],
                      "state_identity": f"{record['task_id']}:cycle{cycle}:first:{record['frontier_depth']}",
                      "independent_prefix_replay_valid": True,
                      "conversation_prefix": record["prompt_messages"],
                      "gold_action": record["gold_next_action"]}
            try:
                item = encode_first_divergence(tokenizer, sample,
                                               {"task_id": record["task_id"], "tools": schema})
            except ValueError as exc:
                skipped.append({"task_id": record["task_id"], "reason": str(exc)})
                continue
        validate_mask(item)
        item.update(cycle=cycle, error_regime=record["error_regime"],
                    frontier_depth=record["frontier_depth"], source_task_id=record["task_id"])
        encoded.append(item)
    if len({x["task_id"] for x in encoded}) != len(encoded):
        raise RuntimeError("dataset task duplication")
    if len(encoded) < 64 or skipped:
        raise RuntimeError(f"data gate failed: encoded={len(encoded)}, skipped={len(skipped)}")
    output.mkdir(parents=True)
    (output / "dataset.jsonl").write_text("".join(stable(x) + "\n" for x in encoded))
    manifest = {"schema_version": "recursive_opd_first_error_dataset_v1", "cycle": cycle,
                "source_model_sha256": json.loads((root / "rollout_manifest.json").read_text())["model_sha256"],
                "rollout_manifest_sha256": sha256(root / "rollout_manifest.json"),
                "metrics_sha256": sha256(root / "metrics/per_task.jsonl"),
                "dataset_sha256": sha256(output / "dataset.jsonl"),
                "task_count": len(encoded), "unique_task_count": len(encoded),
                "natural_regimes": dict(collections.Counter(x["error_regime"] for x in encoded)),
                "heldout_overlap": len({x["task_id"] for x in encoded} & set(protocol["heldout_task_ids"])),
                "observation_tokens_in_loss": sum(x["tool_observation_tokens_in_loss"] for x in encoded),
                "skipped": skipped, "status": "DATA_READY"}
    if manifest["heldout_overlap"] or manifest["observation_tokens_in_loss"]:
        raise RuntimeError("train/heldout or loss-mask gate failed")
    (output / "manifest.json").write_text(stable(manifest) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("mine", "build"))
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    answer = mine(args.cycle, limit=args.limit) if args.command == "mine" else build_dataset(args.cycle)
    print(stable(answer), flush=True)


if __name__ == "__main__":
    main()

"""Run the exact v2 independent eval protocol on a v3 checkpoint.

The old 29/10/12/21 inputs, seeds, generation settings and scoring functions
remain those of eval_balanced_preference_v2. Only checkpoint/output locations
and an additional, task-disjoint larger isolated panel change.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier import eval_balanced_preference_v2 as previous
from repro_1p7b.graph_frontier.active_frontier_v2 import classify_sample
from repro_1p7b.graph_frontier.balanced_preference_v3_pairs import GF, OUT, TYPES
from repro_1p7b.graph_frontier.balanced_preference_v3_source import read, rows, write
from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action, stable
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256


RESULT = OUT / "evaluation"
PILOT = GF.parents[1] / "repro_1p7b/checkpoints/balanced_graph_dpo_v3_replication"


def lock_v3() -> dict[str, Any]:
    old = read(previous.OLD / "quick_effect_manifest.json")
    pref = read(OUT / "manifest.json")
    data = read(OUT / "audit.json")
    training = read(PILOT / "run_manifest.json")
    reservation = read(OUT / "heldout_manifest.json")
    if pref["verdict"] != "DATA_READY" or data["verdict"] != "DATA_READY" or \
            training["max_steps"] != 64 or training["preference_manifest_sha256"] != file_sha256(OUT / "manifest.json"):
        raise ValueError("v3 data/training identity mismatch")
    train_ids = {r["task_id"] for r in rows(OUT / "pairs_train.jsonl")}
    val_ids = {r["task_id"] for r in rows(OUT / "pairs_val.jsonl")}
    historical_ids = set(old["task_ids"]["heldout_eval"])
    extended_ids = set(reservation["heldout_task_ids"])
    if train_ids & val_ids or (train_ids | val_ids) & (historical_ids | extended_ids):
        raise ValueError("v3 eval/train task leakage")
    return old


def _configure_old_protocol() -> None:
    previous.RESULT = RESULT
    previous.lock = lock_v3


def historical_isolated(endpoint: str, model: str) -> dict[str, Any]:
    _configure_old_protocol()
    return previous.isolated(endpoint, model)


def historical_full(endpoint: str, model: str) -> dict[str, Any]:
    _configure_old_protocol()
    return previous.full(endpoint, model)


def larger_isolated(endpoint: str, model: str) -> dict[str, Any]:
    if model not in {"dynamic-v1", "old-dpo", "balanced-v2", "balanced-v3"}:
        raise ValueError("unrecognized comparison model identity")
    lock_v3()
    heldout = set(read(OUT / "heldout_manifest.json")["heldout_task_ids"])
    stops = [s for s in rows(OUT / "stop_states.jsonl") if s["task_id"] in heldout]
    downstream = [s for s in rows(OUT / "frontier_states.jsonl") if s["task_id"] in heldout and
                  s["state_type"] == "downstream_continue"]
    found_stop = []
    for index, state in enumerate(stops):
        path = RESULT / "large_isolated" / model / "stop" / f"{state['state_id']}.json"
        seed = previous.SEED + index * 1009
        if path.exists():
            item = read(path)
        else:
            message, action, usage, finish = generate_action(
                endpoint, model, state["prompt_messages"], state["tools"],
                seed=seed, temperature=0.0, max_tokens=512)
            item = {"task_id": state["task_id"], "state_id": state["state_id"],
                    "sampling_seed": seed, "prompt_signature_sha256":
                    previous.hashlib.sha256(stable(state["prompt_messages"]).encode()).hexdigest(),
                    "parsed_action": action, "model_message": message, "usage": usage,
                    "finish_reason": finish, "correct_stop": action["kind"] == "final" and
                    bool(str(action.get("content") or "").strip()),
                    "extra_tool": action["kind"] == "tool",
                    "repeat_tool": action["kind"] == "tool" and
                    action.get("name") == state["last_gold_action"]["name"],
                    "state_damaging": "unknown"}
            write(path, item)
        if item["task_id"] != state["task_id"] or item["state_id"] != state["state_id"] or \
                item["sampling_seed"] != seed:
            raise ValueError("large stop eval cache mismatch")
        found_stop.append(item)
        print(f"LARGE_STOP {len(found_stop)}/{len(stops)}", flush=True)
    found_down = []
    for index, state in enumerate(downstream):
        path = RESULT / "large_isolated" / model / "downstream" / f"{state['state_id']}.json"
        seed = previous.SEED + index * 1009
        if path.exists():
            item = read(path)
        else:
            message, action, usage, finish = generate_action(
                endpoint, model, state["prompt_messages"], state["tools"],
                seed=seed, temperature=0.0, max_tokens=512)
            item = {"task_id": state["task_id"], "state_id": state["state_id"],
                    "sampling_seed": seed, "classification": classify_sample(
                        {"model_message": message, "parsed_action": action}, state),
                    "parsed_action": action, "model_message": message,
                    "usage": usage, "finish_reason": finish}
            write(path, item)
        if item["task_id"] != state["task_id"] or item["state_id"] != state["state_id"] or \
                item["sampling_seed"] != seed:
            raise ValueError("large downstream eval cache mismatch")
        found_down.append(item)
        print(f"LARGE_DOWNSTREAM {len(found_down)}/{len(downstream)}", flush=True)
    summary = {"schema_version": "balanced_preference_v3_large_isolated_v1", "model": model,
               "stop_states": len(stops), "stop_unique_tasks": len({s["task_id"] for s in stops}),
               "stop_correct": sum(x["correct_stop"] for x in found_stop),
               "stop_extra_tool": sum(x["extra_tool"] for x in found_stop),
               "stop_repeat_tool": sum(x["repeat_tool"] for x in found_stop),
               "stop_invalid": sum(x["parsed_action"]["kind"] == "invalid" for x in found_stop),
               "stop_state_damaging": "unknown_not_executed",
               "downstream_states": len(downstream),
               "downstream_unique_tasks": len({s["task_id"] for s in downstream}),
               "downstream_exact": sum(x["classification"] == "correct" for x in found_down),
               "downstream_flow": sum(x["classification"] in {"correct", "ambiguous_same_tool"}
                                      for x in found_down),
               "downstream_classes": dict(Counter(x["classification"] for x in found_down)),
               "large_stop_target_met": len(stops) >= 30,
               "large_downstream_target_met": len(downstream) >= 20}
    write(RESULT / "large_isolated" / model / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("historical-isolated", "historical-full", "larger-isolated"))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="balanced-v3")
    args = parser.parse_args()
    result = (historical_isolated(args.endpoint, args.model) if args.command == "historical-isolated" else
              historical_full(args.endpoint, args.model) if args.command == "historical-full" else
              larger_isolated(args.endpoint, args.model))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

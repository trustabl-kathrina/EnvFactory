"""Targeted additional Dynamic-v1 samples for a missing val-task negative.

Preserves the first-round samples verbatim in an .initial.json backup. It
never changes the fixed task split, chosen actions, or pair-quality gates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import FAILURE_ORDER, classify_sample
from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action


def load_states(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "preference_val/frontier_states.jsonl").read_text().splitlines() if line.strip()]


def state_samples(root: Path, state: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = root / "preference_val/sampled_actions" / f"{state['state_id']}.json"
    return path, json.loads(path.read_text())


def counts(root: Path) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for state in load_states(root):
        _, payload = state_samples(root, state)
        for sample in payload["samples"]:
            result[state["task_id"]][classify_sample(sample, state)] += 1
    return {task: dict(counter) for task, counter in result.items()}


def topup(root: Path, endpoint: str, model: str, max_extra_per_state: int) -> dict[str, Any]:
    original = counts(root)
    target_tasks = {
        task for task, labels in original.items()
        if not any(labels.get(kind, 0) for kind in FAILURE_ORDER)
    }
    if not target_tasks:
        return {"original": original, "target_tasks": [], "final": original, "added": 0}
    added = 0
    for state in load_states(root):
        if state["task_id"] not in target_tasks:
            continue
        if any(counts(root)[state["task_id"]].get(kind, 0) for kind in FAILURE_ORDER):
            continue
        path, payload = state_samples(root, state)
        backup = path.with_suffix(".initial.json")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        existing = {sample["sampling_seed"] for sample in payload["samples"]}
        for index in range(max_extra_per_state):
            seed = 20260923 + 500000 + int(state["state_id"].split("-")[-1][:8], 16) % 100000 + index * 9176
            if seed in existing:
                continue
            try:
                message, action, usage, finish_reason = generate_action(
                    endpoint, model, state["prompt_messages"], state["tools"],
                    seed=seed, temperature=0.9, max_tokens=512,
                )
                sample = {
                    "sample_index": len(payload["samples"]),
                    "sampling_seed": seed,
                    "model_message": message,
                    "parsed_action": action,
                    "usage": usage,
                    "finish_reason": finish_reason,
                }
            except Exception as exc:
                sample = {
                    "sample_index": len(payload["samples"]),
                    "sampling_seed": seed,
                    "model_message": {"role": "assistant", "content": ""},
                    "parsed_action": {"kind": "invalid", "reason": f"request_error:{type(exc).__name__}:{exc}"},
                    "usage": {},
                    "finish_reason": "request_error",
                }
            payload["samples"].append(sample)
            existing.add(seed)
            added += 1
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            temporary.replace(path)
            print(f"VAL_TOPUP_PROGRESS task={state['task_id']} state={state['state_id']} added={added} label={classify_sample(sample, state)}", flush=True)
            if any(classify_sample(item, state) in FAILURE_ORDER for item in payload["samples"]):
                break
    report = {"original": original, "target_tasks": sorted(target_tasks), "final": counts(root), "added": added}
    (root / "val_topup_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "topup"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default="dynamic-v1")
    parser.add_argument("--max-extra-per-state", type=int, default=12)
    args = parser.parse_args()
    if args.command == "inspect":
        report = counts(args.root)
    else:
        if not args.endpoint:
            parser.error("topup requires --endpoint")
        report = topup(args.root, args.endpoint, args.model, args.max_extra_per_state)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

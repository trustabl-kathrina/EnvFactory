"""Build a gated, task-disjoint Rich-v3 Graph DPO micro-pilot.

Input states and observed Dynamic-v1 samples come only from the fixed quick
effect pilot. No source smoke or historical preference artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import (
    FAILURE_ORDER,
    classify_sample,
    stable,
    write_jsonl,
)


SCHEMA = "graph_frontier_rich_v3_quick_pairs_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_samples(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "quick_effect_manifest.json").read_text())
    replay = json.loads((root / "replay_summary.json").read_text())
    if manifest["frozen300_exact_overlap"] != 0 or replay["frozen300_exact_overlap"] != 0:
        raise ValueError("Frozen300 contamination")
    if replay["chosen_executable"] != replay["replay_valid_states"]:
        raise ValueError("replay chosen-action executable gate failed")
    split_ids = {name: set(ids) for name, ids in manifest["task_ids"].items()}
    if len(set.union(*split_ids.values())) != sum(map(len, split_ids.values())):
        raise ValueError("task split leakage")
    classifications: Counter[str] = Counter()
    observed: dict[str, list[dict[str, Any]]] = {}
    missing: Counter[str] = Counter()
    for split in ("train", "preference_val"):
        states = read_jsonl(root / split / "frontier_states.jsonl")
        pairs: list[dict[str, Any]] = []
        for state in states:
            if state["task_id"] not in split_ids[split]:
                raise ValueError(f"state escaped {split} task split")
            if state.get("chosen_executable") is not True:
                raise ValueError("non-executable chosen action")
            sample_path = root / split / "sampled_actions" / f"{state['state_id']}.json"
            if not sample_path.exists():
                missing[split] += 1
                continue
            samples = json.loads(sample_path.read_text())["samples"]
            candidates = []
            seen = set()
            for sample in samples:
                label = classify_sample(sample, state)
                classifications[label] += 1
                if label not in FAILURE_ORDER:
                    continue
                signature = stable(sample.get("parsed_action"))
                if signature in seen:
                    classifications["duplicate_rejected_action"] += 1
                    continue
                seen.add(signature)
                candidates.append((FAILURE_ORDER[label], label, sample))
            candidates.sort(key=lambda item: (item[0], item[2]["sample_index"]))
            kept = []
            used_labels = set()
            for item in candidates:
                if item[1] not in used_labels:
                    kept.append(item)
                    used_labels.add(item[1])
                if len(kept) == 2:
                    break
            for item in candidates:
                if len(kept) == 2:
                    break
                if item not in kept:
                    kept.append(item)
            for _, label, sample in kept:
                pairs.append({
                    "schema_version": "graph_targeted_rich_v3_quick_pair_v1",
                    "pair_id": f"rich-v3-{state['state_id']}-{sample['sample_index']}",
                    "task_id": state["task_id"],
                    "state_id": state["state_id"],
                    "state_signature": state["state_signature"],
                    "edge_id": state["edge_id"],
                    "prompt": state["prompt_messages"],
                    "tools": state["tools"],
                    "chosen": [state["chosen_message"]],
                    "rejected": [sample["model_message"]],
                    "chosen_action": state["chosen_action"],
                    "rejected_action": sample["parsed_action"],
                    "failure_type": label,
                    "producer_tool": state["producer_tool"],
                    "consumer_tool": state["consumer_tool"],
                    "internal_value_source": state["internal_value_source"],
                    "dependency_depth": state["dependency_depth"],
                    "environment_identifiers": state["environment_identifiers"],
                    "chosen_executable": True,
                    "rejected_is_observed_dynamic_v1_output": True,
                    "sampling_seed": sample["sampling_seed"],
                    "frozen_300_overlap": False,
                })
        observed[split] = sorted(pairs, key=lambda pair: pair["pair_id"])
    all_train = observed["train"]
    all_val = observed["preference_val"]
    train_tasks = {pair["task_id"] for pair in all_train}
    val_tasks = {pair["task_id"] for pair in all_val}
    eligible = (
        not missing
        and len(all_train) >= 16
        and len(train_tasks) >= 10
        and len(all_val) >= 2
        and len(val_tasks) >= 2
        and len({pair["failure_type"] for pair in all_train}) >= 2
    )
    report = {
        "schema_version": SCHEMA,
        "verdict": "PAIR_READY" if eligible else "PAIR_GATE_FAILED",
        "train_pairs": len(all_train),
        "train_tasks": len(train_tasks),
        "preference_val_pairs": len(all_val),
        "preference_val_tasks": len(val_tasks),
        "missing_sample_states": dict(missing),
        "sample_classifications": dict(classifications),
        "train_failure_types": dict(Counter(pair["failure_type"] for pair in all_train)),
        "val_failure_types": dict(Counter(pair["failure_type"] for pair in all_val)),
        "frozen300_overlap": 0,
        "chosen_executable_rate": 1.0,
        "required": {
            "train_pairs": 16,
            "train_tasks": 10,
            "preference_val_pairs": 2,
            "preference_val_tasks": 2,
            "train_failure_types": 2,
            "missing_sample_states": 0,
        },
    }
    (root / "quick_pair_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not eligible:
        return report
    # Spread the tiny training budget across independent tasks first.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in all_train:
        buckets[pair["task_id"]].append(pair)
    task_order = sorted(buckets, key=lambda task: hashlib.sha256(f"20260923:{task}".encode()).hexdigest())
    selected = []
    while len(selected) < 16 and task_order:
        next_order = []
        for task in task_order:
            if buckets[task] and len(selected) < 16:
                selected.append(buckets[task].pop(0))
            if buckets[task]:
                next_order.append(task)
        task_order = next_order
    if len({pair["task_id"] for pair in selected}) < 10:
        raise ValueError("selected 16 lost task diversity")
    write_jsonl(root / "graph_pairs_train16.jsonl", selected)
    write_jsonl(root / "graph_pairs_preference_val.jsonl", all_val)
    return report


def materialize(root: Path, model: str, max_length: int) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from repro_1p7b.graph_frontier.materialize_preference_dpo import serialize_pair

    audit = json.loads((root / "quick_pair_audit.json").read_text())
    if audit["verdict"] != "PAIR_READY":
        raise ValueError("preference gate did not pass")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    drops: Counter[str] = Counter()
    serialized = {}
    for name, path in {
        "train": root / "graph_pairs_train16.jsonl",
        "preference_val": root / "graph_pairs_preference_val.jsonl",
    }.items():
        rows = []
        for pair in read_jsonl(path):
            item, reason = serialize_pair(tokenizer, pair, max_length)
            if item is None:
                drops[f"{name}:{reason}"] += 1
            else:
                rows.append(item)
        serialized[name] = rows
    ready = len(serialized["train"]) == 16 and len(serialized["preference_val"]) >= 2 and not drops
    report = {
        "schema_version": SCHEMA,
        "verdict": "DPO_READY" if ready else "SERIALIZATION_GATE_FAILED",
        "train_rows": len(serialized["train"]),
        "val_rows": len(serialized["preference_val"]),
        "drop_reasons": dict(drops),
        "model": model,
        "max_length": max_length,
    }
    (root / "quick_dpo_materialization_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if ready:
        write_jsonl(root / "dpo_train16.jsonl", serialized["train"])
        write_jsonl(root / "dpo_val.jsonl", serialized["preference_val"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-samples", "materialize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "materialize":
        if not args.model:
            parser.error("--model is required for materialize")
        report = materialize(args.root, args.model, args.max_length)
    else:
        report = audit_samples(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

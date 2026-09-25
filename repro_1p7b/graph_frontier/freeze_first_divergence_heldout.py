"""Freeze task-disjoint Rich-v3 heldout tasks before Cycle-2 training."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import (
    MODEL, VALID, reference_for, source_bundle,
)

COUNTS = {"1": 8, "2": 8, "3": 8}
SEED = 20260926


def freeze(train_plan: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite heldout plan")
    manifest, rows, gold, ledger, raw_by_seed = source_bundle()
    train = json.loads(train_plan.read_text())
    excluded = set(train["task_ids"])
    rng = random.Random(SEED)
    selected = []
    for depth, count in COUNTS.items():
        candidates = []
        for task_id in sorted(set(rows) - excluded):
            row = rows[task_id]
            if str(row["extra_info"]["graph_frontier"]["dependency_depth"]) != depth:
                continue
            reference, _ = reference_for(row, ledger[task_id], raw_by_seed)
            if reference is not None and row["extra_info"]["graph_frontier"] == gold[task_id]:
                candidates.append(task_id)
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(f"depth {depth} eligible heldout {len(candidates)} < {count}")
        selected.extend(candidates[:count])
    payload = {
        "schema_version": "first_divergence_heldout_plan_v1",
        "source_manifest_sha256": sha256(VALID / "manifest.json"),
        "valid_data_sha256": manifest["artifacts"]["valid_data_sha256"],
        "valid_gold_sha256": manifest["artifacts"]["valid_gold_sha256"],
        "frozen_manifest_sha256": train["frozen_manifest_sha256"],
        "model_path": str(MODEL),
        "model_weights_sha256": sha256(MODEL / "model.safetensors"),
        "selection_seed": SEED,
        "target_depth_counts": COUNTS,
        "task_ids": selected,
        "task_count": len(selected),
        "train_plan_sha256": sha256(train_plan),
        "train_task_overlap": len(set(selected) & excluded),
        "frozen_exact_id_overlap": 0,
        "sampling_seed": SEED,
        "temperature": 0.7,
        "max_steps": 8,
        "max_tokens": 1024,
        "context_length": 32768,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.train_plan, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

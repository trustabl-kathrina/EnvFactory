"""Freeze new weak-bucket Rich tasks disjoint from train and heldout."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import (
    MODEL, VALID, reference_for, source_bundle,
)

STUDENT = Path(__file__).resolve().parents[2] / "repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle2_smoke"
COUNTS = {"1": 12, "2": 20, "3": 16}
SEED = 20260927


def freeze(train_plan: Path, heldout_plan: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite Cycle-3 plan")
    manifest, rows, gold, ledger, raw_by_seed = source_bundle()
    train = json.loads(train_plan.read_text())
    heldout = json.loads(heldout_plan.read_text())
    if heldout.get("train_plan_sha256") != sha256(train_plan):
        raise RuntimeError("heldout/train plan linkage broken")
    excluded = set(train["task_ids"]) | set(heldout["task_ids"])
    rng = random.Random(SEED)
    selected = []
    available = {}
    for depth, count in COUNTS.items():
        candidates = []
        for task_id in sorted(set(rows) - excluded):
            row = rows[task_id]
            if str(row["extra_info"]["graph_frontier"]["dependency_depth"]) != depth:
                continue
            reference, _ = reference_for(row, ledger[task_id], raw_by_seed)
            if reference is not None and row["extra_info"]["graph_frontier"] == gold[task_id]:
                candidates.append(task_id)
        available[depth] = len(candidates)
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(f"depth {depth}: {len(candidates)} < {count} eligible")
        selected.extend(candidates[:count])
    result = {
        "schema_version": "first_divergence_rich_cycle3_plan_v1",
        "source_manifest_sha256": sha256(VALID / "manifest.json"),
        "valid_data_sha256": manifest["artifacts"]["valid_data_sha256"],
        "valid_gold_sha256": manifest["artifacts"]["valid_gold_sha256"],
        "frozen_manifest_sha256": train["frozen_manifest_sha256"],
        "model_path": str(MODEL),
        "model_weights_sha256": sha256(MODEL / "model.safetensors"),
        "student_model_path": str(STUDENT),
        "student_model_weights_sha256": sha256(STUDENT / "model.safetensors"),
        "served_model_name": "first-divergence-v1",
        "policy_version": "first-divergence-cycle2",
        "sampling_seed": SEED,
        "temperature": 0.7, "max_steps": 8, "max_tokens": 1024,
        "context_length": 32768,
        "selection_seed": SEED, "target_depth_counts": COUNTS,
        "available_by_depth": available,
        "task_ids": selected, "task_count": len(selected),
        "train_plan_sha256": sha256(train_plan),
        "heldout_plan_sha256": sha256(heldout_plan),
        "prior_task_overlap": len(set(selected) & excluded),
        "frozen_exact_id_overlap": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-plan", type=Path, required=True)
    parser.add_argument("--heldout-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.train_plan, args.heldout_plan, args.output),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

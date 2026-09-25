"""Freeze an auditable first-divergence dataset from actual student rollouts."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path

from repro_1p7b.graph_frontier.first_divergence_v1 import (
    align_first_divergence, canonical, reference_from_raw,
)

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
TRAIN = ROOT / "repro_1p7b/data/graph_frontier_rl_v1_frozen_seed20260920/train.json"
AUDIT = TRAIN.parent / "audit.jsonl"
EXPECTED_FROZEN_SHA = "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"
EXPECTED_TRAIN_SHA = "1faabac69216efd36b4bb02822cec641fd623a6fefa5d581aa123fb441e66eeb"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def build(rollout_dir: Path, output_dir: Path, checkpoint: str, policy_version: str) -> dict:
    if sha256(FROZEN) != EXPECTED_FROZEN_SHA or sha256(TRAIN) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("frozen or training source hash mismatch")
    rows = {r["task_id"]: r for r in json.loads(TRAIN.read_text())}
    frozen_ids = {r["task_id"] for r in jsonl(FROZEN)}
    overlap = len(set(rows) & frozen_ids)
    if overlap:
        raise RuntimeError(f"Frozen300 exact task ID overlap: {overlap}")
    audits = {r["task_id"]: r for r in jsonl(AUDIT)}
    if any(audits.get(task_id, {}).get("status") != "REPLAY_PASS" for task_id in rows):
        raise RuntimeError("train row missing REPLAY_PASS gold certificate")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_cache = {}
    ref_cache = {}
    counts = collections.Counter()
    reasons = collections.Counter()
    depths = collections.Counter()
    buckets = collections.Counter()
    samples = []
    seen_states = set()
    used_tasks = set()
    for path in sorted(rollout_dir.glob("*.r*.json")):
        counts["rollouts"] += 1
        rollout = json.loads(path.read_text())
        task_id = rollout.get("task_id")
        if task_id not in rows:
            reasons["task_not_in_train"] += 1
            continue
        row, audit = rows[task_id], audits[task_id]
        if task_id not in ref_cache:
            raw_path = audit["raw_file"]
            if raw_path not in raw_cache:
                raw_cache[raw_path] = json.loads(Path(raw_path).read_text())
            ref_cache[task_id] = reference_from_raw(row, audit, raw_cache[raw_path])
        reference, why = ref_cache[task_id]
        if reference is None:
            reasons[why] += 1
            counts["gold_ineligible_rollouts"] += 1
            continue
        counts["eligible_gold_rollouts"] += 1
        result = align_first_divergence(
            row, rollout, reference, student_checkpoint=checkpoint,
            policy_version=policy_version)
        reasons[result["reason"]] += 1
        status = result["status"]
        if result["alignment_valid"] and result["prefix_replay_valid"]:
            counts["valid_rollouts"] += 1
        if status == "perfect":
            counts["perfect_rollouts"] += 1
        if status != "first_divergence":
            continue
        sample = result["sample"]
        if not sample.get("conversation_prefix") or sample.get("current_observation") is None:
            reasons["missing_onpolicy_state"] += 1
            continue
        identity = hashlib.sha256(canonical(
            [task_id, sample["conversation_prefix"], sample["current_observation"],
             sample["gold_action"]]).encode()).hexdigest()
        if identity in seen_states:
            counts["duplicate_state_skipped"] += 1
            continue
        seen_states.add(identity)
        sample["state_identity"] = identity
        samples.append(sample)
        used_tasks.add(task_id)
        counts["first_divergence_samples_total"] += 1
        counts["divergence_" + sample["divergence_type"]] += 1
        depths[str(sample["graph_depth"])] += 1
        buckets[str(sample["frontier_bucket"])] += 1
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "first_divergence_onpolicy_v1_dataset_audit",
        "source_rollout_dir": str(rollout_dir),
        "student_checkpoint": checkpoint,
        "student_policy_version": policy_version,
        "frozen_manifest_sha256": EXPECTED_FROZEN_SHA,
        "train_sha256": EXPECTED_TRAIN_SHA,
        "frozen_exact_id_overlap": overlap,
        "train_tasks": len(rows), "gold_replay_valid_tasks": len(rows),
        "reference_eligible_tasks": sum(ref is not None for ref, _ in ref_cache.values()),
        "counts": dict(sorted(counts.items())),
        "reasons": dict(sorted(reasons.items())),
        "sample_unique_tasks": len(used_tasks),
        "depth_counts": dict(sorted(depths.items())),
        "bucket_counts": dict(sorted(buckets.items())),
        "first_divergence_yield": counts["first_divergence_samples_total"] / max(1, counts["valid_rollouts"]),
        "usable_supervision_coverage": counts["first_divergence_samples_total"] / max(1, counts["rollouts"]),
        "stop_target_samples": 0,
        "training_ready": False,
        "training_ready_reason": "cycle0 is coverage only; independent prefix replay and minimum data gate pending",
    }
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy-version", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.rollouts, args.output, args.checkpoint, args.policy_version),
                     ensure_ascii=False, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

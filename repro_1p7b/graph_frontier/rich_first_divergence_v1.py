"""Frozen Rich-v3 source, balanced selection, student collection, and v1 audit."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import (
    EXPECTED_FROZEN_SHA, FROZEN, jsonl, sha256,
)
from repro_1p7b.graph_frontier.first_divergence_v1 import (
    align_first_divergence, canonical, reference_from_raw,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repro_1p7b/graph_frontier/preference_generation_v3/pilot_balanced_24chunks_v3"
VALID = SOURCE / "valid"
MODEL = ROOT / "repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b"
DEPTH_COUNTS = {"1": 12, "2": 16, "3": 20}

def source_bundle():
    manifest = json.loads((VALID / "manifest.json").read_text())
    if manifest.get("verdict") != "GRAPH_FRONTIER_RICH_POOL_READY" or manifest.get("task_count") != 289:
        raise RuntimeError("Rich source manifest not ready")
    if manifest.get("frozen300_exact_overlap") != 0 or sha256(FROZEN) != EXPECTED_FROZEN_SHA:
        raise RuntimeError("Frozen300 lock failed")
    artifacts = manifest["artifacts"]
    data_path = VALID / "graph_frontier_rich_train_v1.jsonl"
    gold_path = VALID / "graph_frontier_rich_gold_v1.jsonl"
    if sha256(data_path) != artifacts["valid_data_sha256"] or sha256(gold_path) != artifacts["valid_gold_sha256"]:
        raise RuntimeError("Rich source data or gold hash changed")
    rows = {r["task_id"]: r for r in jsonl(data_path)}
    gold = {r["task_id"]: r for r in jsonl(gold_path)}
    ledger = {r["task_id"]: r for r in jsonl(SOURCE / "audit/final_ledger.jsonl")
              if r.get("status") == "FINAL_VALID"}
    frozen_ids = {r["task_id"] for r in jsonl(FROZEN)}
    if len(rows) != 289 or set(rows) != set(gold) or set(rows) != set(ledger) or set(rows) & frozen_ids:
        raise RuntimeError("Rich row/gold/FINAL_VALID/Frozen identity mismatch")
    raw_by_seed = collections.defaultdict(list)
    for path in (SOURCE / "raw").glob("*.json"):
        match = re.search(r"-(\d+)-sglang", path.name)
        if match:
            raw_by_seed[int(match.group(1))].append(path)
    return manifest, rows, gold, ledger, raw_by_seed

def reference_for(row, ledger, raw_by_seed):
    seed = int(row["extra_info"]["generation_seed"])
    paths = raw_by_seed[seed]
    if len(paths) != 1:
        return None, "raw_seed_not_unique"
    path = paths[0]
    raw = json.loads(path.read_text())
    if len(raw.get("nodes", [])) != 1:
        return None, "raw_node_not_unique"
    cert = {**ledger, "raw_file": str(path), "node_index": 0, "frozen_overlap": False}
    return reference_from_raw(row, cert, raw)

def create_plan(output: Path) -> dict:
    manifest, rows, gold, ledger, raw_by_seed = source_bundle()
    eligible = collections.defaultdict(list)
    reasons = collections.Counter()
    for task_id, row in rows.items():
        if row["extra_info"]["graph_frontier"] != gold[task_id]:
            reasons["sidecar_mismatch"] += 1
            continue
        reference, reason = reference_for(row, ledger[task_id], raw_by_seed)
        reasons[reason] += 1
        if reference is not None:
            depth = str(row["extra_info"]["graph_frontier"]["dependency_depth"])
            eligible[depth].append(task_id)
    rng = random.Random(20260925)
    selected = []
    for depth, count in DEPTH_COUNTS.items():
        tasks = sorted(eligible[depth])
        rng.shuffle(tasks)
        if len(tasks) < count:
            raise RuntimeError(f"depth{depth} has {len(tasks)} eligible, needs {count}")
        selected.extend(tasks[:count])
    if output.exists():
        raise RuntimeError("refusing to overwrite cycle plan")
    payload = {
        "schema_version": "first_divergence_rich_cycle1_plan_v1",
        "source_manifest_sha256": sha256(VALID / "manifest.json"),
        "valid_data_sha256": manifest["artifacts"]["valid_data_sha256"],
        "valid_gold_sha256": manifest["artifacts"]["valid_gold_sha256"],
        "frozen_manifest_sha256": EXPECTED_FROZEN_SHA,
        "model_path": str(MODEL),
        "model_weights_sha256": sha256(MODEL / "model.safetensors"),
        "selection_seed": 20260925,
        "target_depth_counts": DEPTH_COUNTS,
        "eligible_by_depth": {key: len(value) for key, value in sorted(eligible.items())},
        "eligibility_reasons": dict(sorted(reasons.items())),
        "task_ids": selected,
        "task_count": len(selected),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return payload

def checked_plan(path: Path):
    manifest, rows, gold, ledger, raw_by_seed = source_bundle()
    plan = json.loads(path.read_text())
    if plan.get("source_manifest_sha256") != sha256(VALID / "manifest.json"):
        raise RuntimeError("cycle plan source manifest changed")
    if plan.get("model_weights_sha256") != sha256(MODEL / "model.safetensors"):
        raise RuntimeError("Dynamic-v1 model weights changed")
    student_model = Path(plan.get("student_model_path", str(MODEL)))
    student_sha = plan.get("student_model_weights_sha256", plan["model_weights_sha256"])
    if sha256(student_model / "model.safetensors") != student_sha:
        raise RuntimeError("student checkpoint weights changed")
    if len(plan["task_ids"]) != len(set(plan["task_ids"])) or any(t not in rows for t in plan["task_ids"]):
        raise RuntimeError("cycle plan task identity invalid")
    return plan, rows, ledger, raw_by_seed

def collect(plan_path: Path, output: Path, endpoint: str, shard_index: int, shard_count: int):
    from repro_1p7b.graph_frontier.collect_preference_rollouts import collect_one
    plan, rows, _, _ = checked_plan(plan_path)
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for index, task_id in enumerate(plan["task_ids"]):
        if index % shard_count != shard_index:
            continue
        row = rows[task_id]
        try:
            sample = collect_one(
                row, endpoint=endpoint, model=plan.get("served_model_name", "dynamic-v1"), rollout_index=0,
                seed=plan.get("sampling_seed", 20260925) + index * 1009, max_steps=plan.get("max_steps", 8),
                temperature=plan.get("temperature", 0.7), max_tokens=plan.get("max_tokens", 1024), output=output)
        except Exception as exc:
            response = getattr(exc, "response", None)
            detail = getattr(response, "text", "") if response is not None else ""
            print(json.dumps({"task_id": task_id, "shard": shard_index,
                              "error": repr(exc), "server_response": detail[:2000]},
                             ensure_ascii=False), flush=True)
            raise
        completed += 1
        print(json.dumps({"shard": shard_index, "completed": completed,
                          "task_id": task_id, "steps": len(sample.get("steps", [])),
                          "semantic_success": sample.get("semantic_success")}), flush=True)
    return {"completed": completed, "shard": shard_index}

def build(plan_path: Path, rollouts: Path, output: Path) -> dict:
    plan, rows, ledger, raw_by_seed = checked_plan(plan_path)
    if output.exists():
        raise RuntimeError("refusing to overwrite built cycle dataset")
    expected = set(plan["task_ids"])
    files = {p.name.split(".r0.json")[0]: p for p in rollouts.glob("*.r0.json")}
    if set(files) != expected:
        raise RuntimeError(f"rollout coverage mismatch: missing={len(expected-set(files))} extra={len(set(files)-expected)}")
    output.mkdir(parents=True)
    counts = collections.Counter()
    reasons = collections.Counter()
    depth = collections.Counter()
    buckets = collections.Counter()
    samples = []
    used_tasks = set()
    seen_states = set()
    for task_id in plan["task_ids"]:
        row = rows[task_id]
        reference, why = reference_for(row, ledger[task_id], raw_by_seed)
        if reference is None:
            reasons[why] += 1
            continue
        rollout = json.loads(files[task_id].read_text())
        result = align_first_divergence(
            row, rollout, reference, student_checkpoint=plan.get("student_model_path", str(MODEL)),
            policy_version=plan.get("policy_version", "dynamic-v1-cycle1"))
        counts["rollouts"] += 1
        reasons[result["reason"]] += 1
        if result["alignment_valid"] and result["prefix_replay_valid"]:
            counts["valid_rollouts"] += 1
        if result["status"] == "perfect":
            counts["perfect_rollouts"] += 1
        if result["status"] != "first_divergence":
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
        depth[str(sample["graph_depth"])] += 1
        buckets[str(sample["frontier_bucket"])] += 1
    with (output / "samples.jsonl").open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "first_divergence_rich_cycle_audit_v1",
        "plan_sha256": sha256(plan_path), "rollout_tasks": len(expected),
        "gold_replay_valid": len(expected), "counts": dict(sorted(counts.items())),
        "reasons": dict(sorted(reasons.items())), "sample_unique_tasks": len(used_tasks),
        "depth_counts": dict(sorted(depth.items())),
        "frontier_bucket_counts": dict(sorted(buckets.items())),
        "first_divergence_yield": counts["first_divergence_samples_total"] / max(1, counts["valid_rollouts"]),
        "usable_supervision_coverage": counts["first_divergence_samples_total"] / len(expected),
        "stop_target_samples": 0, "frozen_exact_id_overlap": 0,
        "training_ready": False,
        "training_ready_reason": "requires independent prefix replay and minimum distinct-task gate",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--endpoint", required=True)
    c.add_argument("--shard-index", type=int, required=True)
    c.add_argument("--shard-count", type=int, default=2)
    b = sub.add_parser("build")
    b.add_argument("--plan", type=Path, required=True)
    b.add_argument("--rollouts", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = create_plan(args.output)
    elif args.command == "collect":
        result = collect(args.plan, args.output, args.endpoint, args.shard_index, args.shard_count)
    else:
        result = build(args.plan, args.rollouts, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()

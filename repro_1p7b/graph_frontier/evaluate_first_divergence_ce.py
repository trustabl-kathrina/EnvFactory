"""Matched, task-disjoint executable heldout evaluation for CE smoke."""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256
from repro_1p7b.graph_frontier.first_divergence_v1 import align_first_divergence
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import (
    checked_plan, reference_for,
)


def collect(plan_path: Path, model_path: Path, served_name: str,
            endpoint: str, output: Path) -> dict:
    from repro_1p7b.graph_frontier.collect_preference_rollouts import collect_one

    plan, rows, _, _ = checked_plan(plan_path)
    if plan.get("schema_version") != "first_divergence_heldout_plan_v1":
        raise RuntimeError("not a frozen heldout plan")
    if plan.get("task_count") != 24 or plan.get("train_task_overlap") != 0:
        raise RuntimeError("heldout isolation gate failed")
    if output.exists() and (output / "collection_manifest.json").exists():
        raise RuntimeError("heldout collection already complete")
    model_sha = sha256(model_path / "model.safetensors")
    output.mkdir(parents=True, exist_ok=True)
    for index, task_id in enumerate(plan["task_ids"]):
        row = rows[task_id]
        rollout = collect_one(
            row, endpoint=endpoint, model=served_name, rollout_index=0,
            seed=plan["sampling_seed"] + index * 1009,
            max_steps=plan["max_steps"], temperature=plan["temperature"],
            max_tokens=plan["max_tokens"], output=output,
        )
        print(json.dumps({"completed": index + 1, "task_id": task_id,
                          "semantic_success": rollout["semantic_success"],
                          "steps": len(rollout["steps"])}, sort_keys=True), flush=True)
    result = {
        "schema_version": "first_divergence_heldout_collection_v1",
        "plan_sha256": sha256(plan_path),
        "model_path": str(model_path),
        "model_weights_sha256": model_sha,
        "served_model_name": served_name,
        "task_count": len(plan["task_ids"]),
        "all_rollout_files_present": all(
            (output / (task_id + ".r0.json")).exists() for task_id in plan["task_ids"]
        ),
    }
    (output / "collection_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def paired_ci(differences: list[float], seed: int = 20260926) -> list[float]:
    rng = random.Random(seed)
    n = len(differences)
    draws = []
    for _ in range(5000):
        draws.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def summarize(plan_path: Path, base_dir: Path, trained_dir: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite heldout report")
    plan, rows, ledger, raw_by_seed = checked_plan(plan_path)
    if plan.get("schema_version") != "first_divergence_heldout_plan_v1":
        raise RuntimeError("heldout plan mismatch")
    for directory in (base_dir, trained_dir):
        manifest = json.loads((directory / "collection_manifest.json").read_text())
        if manifest.get("plan_sha256") != sha256(plan_path) or manifest.get("task_count") != 24:
            raise RuntimeError("collection provenance mismatch")
    model_info = {
        "base": json.loads((base_dir / "collection_manifest.json").read_text()),
        "trained": json.loads((trained_dir / "collection_manifest.json").read_text()),
    }
    if model_info["base"]["model_weights_sha256"] == model_info["trained"]["model_weights_sha256"]:
        raise RuntimeError("base and trained checkpoint hashes identical")
    per_task = []
    aggregates = {name: collections.Counter() for name in ("base", "trained")}
    by_depth = {name: collections.defaultdict(collections.Counter)
                for name in ("base", "trained")}
    differences = []
    reward_differences = []
    for task_id in plan["task_ids"]:
        reference, why = reference_for(rows[task_id], ledger[task_id], raw_by_seed)
        if reference is None:
            raise RuntimeError("heldout gold invalid: " + why)
        depth = str(rows[task_id]["extra_info"]["graph_frontier"]["dependency_depth"])
        task = {"task_id": task_id, "depth": depth}
        for name, directory in (("base", base_dir), ("trained", trained_dir)):
            rollout = json.loads((directory / (task_id + ".r0.json")).read_text())
            result = align_first_divergence(
                rows[task_id], rollout, reference,
                student_checkpoint=model_info[name]["model_path"],
                policy_version=name + "-heldout",
            )
            success = bool(rollout.get("semantic_success"))
            official_reward = rollout.get("official_reward")
            events = [step.get("typed_event") for step in rollout["steps"]]
            tool_events = [event for event in events if isinstance(event, dict)]
            value = {
                "semantic_success": success,
                "official_reward": official_reward,
                "alignment_status": result["status"],
                "alignment_reason": result["reason"],
                "first_divergence_step": result.get("first_divergence_step"),
                "gold_step_count": len(reference["actions"]),
                "tool_events": len(tool_events),
                "tool_execution_success": sum(e.get("execution_success") is True
                                              for e in tool_events),
                "trajectory_steps": len(rollout["steps"]),
            }
            task[name] = value
            counter = aggregates[name]
            counter["tasks"] += 1
            counter["semantic_success"] += success
            counter["perfect"] += result["status"] == "perfect"
            counter["first_divergence"] += result["status"] == "first_divergence"
            counter["invalid_alignment"] += result["reason"] == "invalid_alignment"
            counter["prefix_mismatch"] += result["reason"] == "student_prefix_observation_mismatch"
            counter["tool_events"] += value["tool_events"]
            counter["tool_execution_success"] += value["tool_execution_success"]
            counter["divergence_" + str(result["reason"])] += 1
            by_depth[name][depth]["tasks"] += 1
            by_depth[name][depth]["semantic_success"] += success
            by_depth[name][depth]["perfect"] += result["status"] == "perfect"
        differences.append(
            float(task["trained"]["semantic_success"]) - float(task["base"]["semantic_success"])
        )
        reward_differences.append(
            float(task["trained"]["official_reward"]) - float(task["base"]["official_reward"])
        )
        per_task.append(task)
    output.mkdir(parents=True)
    with (output / "per_task.jsonl").open("w", encoding="utf-8") as out:
        for task in per_task:
            out.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "schema_version": "first_divergence_heldout_comparison_v1",
        "plan_sha256": sha256(plan_path),
        "base_model_weights_sha256": model_info["base"]["model_weights_sha256"],
        "trained_model_weights_sha256": model_info["trained"]["model_weights_sha256"],
        "task_count": len(per_task),
        "train_task_overlap": 0,
        "frozen_exact_id_overlap": 0,
        "base": dict(aggregates["base"]),
        "trained": dict(aggregates["trained"]),
        "by_depth": {
            name: {depth: dict(counter) for depth, counter in sorted(by_depth[name].items())}
            for name in ("base", "trained")
        },
        "paired_semantic_success_delta": statistics.mean(differences),
        "paired_semantic_success_delta_ci95": paired_ci(differences),
        "base_mean_official_reward": statistics.mean(float(x["base"]["official_reward"]) for x in per_task),
        "trained_mean_official_reward": statistics.mean(float(x["trained"]["official_reward"]) for x in per_task),
        "paired_official_reward_delta": statistics.mean(reward_differences),
        "paired_official_reward_delta_ci95": paired_ci(reward_differences),
        "mean_first_divergence_step_aligned": {
            name: statistics.mean(x[name]["first_divergence_step"] for x in per_task
                                  if x[name]["alignment_status"] == "first_divergence")
            for name in ("base", "trained")
        },
        "paired_success_wins": sum(x > 0 for x in differences),
        "paired_success_losses": sum(x < 0 for x in differences),
        "paired_success_ties": sum(x == 0 for x in differences),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--model-path", type=Path, required=True)
    c.add_argument("--served-name", required=True)
    c.add_argument("--endpoint", required=True)
    c.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--trained", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.plan, args.model_path, args.served_name,
                         args.endpoint, args.output)
    else:
        result = summarize(args.plan, args.base, args.trained, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

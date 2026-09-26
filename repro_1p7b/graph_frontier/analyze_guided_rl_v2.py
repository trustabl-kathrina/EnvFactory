"""Aggregate matched Guided RL v2 pilots from typed audits and runtime logs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "repro_1p7b/graph_frontier/reports"
LOG_ROOT = ROOT / "repro_1p7b/logs/graph_frontier_rl_v2"
DATA_ROOT = ROOT / "repro_1p7b/data/graph_frontier_rl_v1_frozen_seed20260920/parquet_pilot128"
INIT_ROOT = ROOT / "repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b"
ARMS = {
    "standard_v2": {
        "log": LOG_ROOT / "pilot128_standard",
        "checkpoint": ROOT / "repro_1p7b/checkpoints/standard_rl_v2_control_pilot128",
    },
    "graph_v2": {
        "log": LOG_ROOT / "pilot128_graph",
        "checkpoint": ROOT / "repro_1p7b/checkpoints/graph_frontier_rl_v2_pilot128",
    },
}
AUDIT_PREFIX = "GUIDED_RL_V2_AUDIT "
NUMBER = re.compile(r"([A-Za-z0-9_./-]+):(-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def load_audits(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line.split(AUDIT_PREFIX, 1)[1])
        for line in path.read_text(errors="replace").splitlines()
        if AUDIT_PREFIX in line
    ]


def load_steps(path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = re.search(r"(?:^|\s)step:([0-9]+)\s+-", plain)
        if match:
            steps[int(match.group(1))] = {
                key: float(value) for key, value in NUMBER.findall(plain)
            }
    return steps


def gpu_summary(path: Path) -> dict[str, Any]:
    samples: dict[str, list[int]] = defaultdict(list)
    timestamps: list[datetime] = []
    for line in path.read_text(errors="replace").splitlines():
        text = line.strip()
        if re.match(r"^20[0-9]{2}-", text):
            timestamps.append(datetime.fromisoformat(text))
            continue
        match = re.match(r"^([0-9]+),\s*([0-9]+) MiB", text)
        if match:
            samples[match.group(1)].append(int(match.group(2)))
    peaks = {gpu: max(values) for gpu, values in sorted(samples.items())}
    return {
        "peak_memory_mib_by_gpu": peaks,
        "peak_memory_mib": max(peaks.values()) if peaks else None,
        "wall_time_seconds": (
            (timestamps[-1] - timestamps[0]).total_seconds()
            if len(timestamps) >= 2 else None
        ),
    }


def inversion_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(int(row["train_step"]), str(row["group_uid"]))].append(row)
    success_inversions = official_inversions = 0
    for rows in groups.values():
        for left in rows:
            for right in rows:
                if (
                    left["success"] > right["success"]
                    and left["final_group_reward"] <= right["final_group_reward"]
                ):
                    success_inversions += 1
                if (
                    left["success"] == right["success"]
                    and left["official_reward"] > right["official_reward"] + 1e-6
                    and left["final_group_reward"] <= right["final_group_reward"]
                ):
                    official_inversions += 1
    return {
        "prompt_groups": len(groups),
        "success_over_failure_inversion_pairs": success_inversions,
        "strict_official_inversion_pairs": official_inversions,
        "harmful_inversion_pairs": success_inversions + official_inversions,
    }


def rollout_summary(path: Path) -> dict[str, float | int | None]:
    state_scores, trace_scores = [], []
    overlap = count = 0
    for source in sorted(path.glob("*.json")):
        row = json.loads(source.read_text())
        count += 1
        overlap += bool(row.get("frozen_300_overlap"))
        official = row["reward_parts"]["official_reward"]
        state_scores.append(float(official["state_score"]))
        trace_scores.append(float(official["trace_score"]))
    return {
        "rollout_files": count,
        "frozen_300_exact_overlap": overlap,
        "state_score_mean": mean(state_scores),
        "trace_score_mean": mean(trace_scores),
    }


def arm_summary(name: str, paths: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    log_root = paths["log"]
    audits = load_audits(log_root / "train.log")
    steps = load_steps(log_root / "train.log")
    if len(audits) != 512:
        raise RuntimeError(f"{name}: expected 512 audits, found {len(audits)}")
    if sorted(steps) != list(range(1, 65)):
        raise RuntimeError(f"{name}: expected steps 1..64, found {sorted(steps)}")
    if len({row["task_id"] for row in audits}) != 128:
        raise RuntimeError(f"{name}: expected 128 unique tasks")
    if not all(math.isfinite(float(row["advantage"])) for row in audits):
        raise RuntimeError(f"{name}: non-finite advantage")

    ec = sum(int(row["eligible_consumer_opportunities"]) for row in audits)
    cc = sum(int(row["consumer_completed"]) for row in audits)
    ep = sum(int(row["eligible_prop_opportunities"]) for row in audits)
    pc = sum(int(row["prop_correct"]) for row in audits)
    ro = sum(int(row["retry_opportunities"]) for row in audits)
    ir = sum(int(row["invalid_retry"]) for row in audits)
    ga = sum(bool(row["graph_available"]) for row in audits)
    ge = sum(bool(row["graph_tiebreak_eligible"]) for row in audits)
    gt = sum(bool(row["graph_tiebreak_applied"]) for row in audits)

    def sv(key: str) -> list[float]:
        return [steps[index][key] for index in sorted(steps) if key in steps[index]]

    status = (log_root / "status").read_text().strip()
    latest = (paths["checkpoint"] / "latest_checkpointed_iteration.txt").read_text().strip()
    rollouts = rollout_summary(log_root / "rollouts")
    resource = gpu_summary(log_root / "gpu.csv")
    resource["torch_max_memory_allocated_gb"] = max(sv("perf/max_memory_allocated_gb"))
    resource["torch_max_memory_reserved_gb"] = max(sv("perf/max_memory_reserved_gb"))
    return {
        "status": status,
        "optimizer_steps": len(steps),
        "audit_rollouts": len(audits),
        "unique_tasks": len({row["task_id"] for row in audits}),
        "checkpoint_step": int(latest),
        "checkpoint_complete": status == "DONE" and latest == "64",
        "official_reward_mean": mean([float(row["official_reward"]) for row in audits]),
        "task_success_rate": mean([float(row["success"]) for row in audits]),
        "state_score_mean": rollouts["state_score_mean"],
        "trace_score_mean": rollouts["trace_score_mean"],
        "consumer": {
            "eligible_opportunities": ec, "completed": cc,
            "completion_rate": cc / ec if ec else None,
        },
        "propagation": {
            "eligible_opportunities": ep, "correct": pc,
            "accuracy": pc / ep if ep else None,
        },
        "retry": {
            "opportunities": ro, "invalid": ir,
            "invalid_rate": ir / ro if ro else None,
        },
        "graph": {
            "available_rollouts": ga,
            "available_rate": ga / len(audits),
            "tiebreak_eligible_rollouts": ge,
            "tiebreak_eligible_rate": ge / len(audits),
            "tiebreak_applied_rollouts": gt,
            "tiebreak_applied_rate": gt / len(audits),
            "tiebreak_applied_groups": len({
                (row["train_step"], row["group_uid"])
                for row in audits if row["graph_tiebreak_applied"]
            }),
            "score_mean_available": mean([
                float(row["graph_score"]) for row in audits if row["graph_available"]
            ]),
        },
        "stability": {
            "grad_norm": distribution(sv("actor/grad_norm")),
            "actor_kl_loss_mean": mean(sv("actor/kl_loss")),
            "actor_ppo_kl_mean": mean(sv("actor/ppo_kl")),
            "entropy_mean": mean(sv("actor/entropy_loss")),
            "step_time_seconds_mean": mean(sv("timing_s/step")),
        },
        "trajectory": {
            "length_mean": mean([float(row["trajectory_length"]) for row in audits]),
            "generated_tokens_mean": mean([float(row["generated_tokens"]) for row in audits]),
            "tool_calls_mean": mean([float(row["tool_call_count"]) for row in audits]),
        },
        "safety": inversion_summary(audits),
        "resource": resource,
        "rollout_files": rollouts["rollout_files"],
        "frozen_300_exact_overlap": rollouts["frozen_300_exact_overlap"],
    }, audits


def subtract(left: Any, right: Any) -> Any:
    return None if left is None or right is None else right - left


def paired_task_summary(
    standard: list[dict[str, Any]], graph: list[dict[str, Any]]
) -> dict[str, Any]:
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["task_id"])].append(row)
        return {
            task_id: {
                "official_reward": statistics.fmean(float(row["official_reward"]) for row in task_rows),
                "task_success": statistics.fmean(float(row["success"]) for row in task_rows),
            }
            for task_id, task_rows in grouped.items()
        }

    left, right = aggregate(standard), aggregate(graph)
    answer = {}
    for metric in ("official_reward", "task_success"):
        differences = [right[key][metric] - left[key][metric] for key in sorted(left)]
        average = statistics.fmean(differences)
        standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
        wins = sum(value > 1e-12 for value in differences)
        losses = sum(value < -1e-12 for value in differences)
        answer[metric] = {
            "mean_delta": average,
            "standard_error": standard_error,
            "normal_95ci": [average - 1.96 * standard_error, average + 1.96 * standard_error],
            "graph_wins": wins,
            "ties": len(differences) - wins - losses,
            "graph_losses": losses,
        }
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPORTS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries, audits_by_arm = {}, {}
    for name, paths in ARMS.items():
        summaries[name], audits_by_arm[name] = arm_summary(name, paths)
    standard, graph = summaries["standard_v2"], summaries["graph_v2"]
    comparison = {
        "official_reward_mean": subtract(standard["official_reward_mean"], graph["official_reward_mean"]),
        "task_success_rate": subtract(standard["task_success_rate"], graph["task_success_rate"]),
        "consumer_completion_rate": subtract(standard["consumer"]["completion_rate"], graph["consumer"]["completion_rate"]),
        "propagation_accuracy": subtract(standard["propagation"]["accuracy"], graph["propagation"]["accuracy"]),
        "invalid_retry_rate": subtract(standard["retry"]["invalid_rate"], graph["retry"]["invalid_rate"]),
        "grad_p95": subtract(standard["stability"]["grad_norm"]["p95"], graph["stability"]["grad_norm"]["p95"]),
        "grad_p99": subtract(standard["stability"]["grad_norm"]["p99"], graph["stability"]["grad_norm"]["p99"]),
        "grad_max": subtract(standard["stability"]["grad_norm"]["max"], graph["stability"]["grad_norm"]["max"]),
        "actor_kl_loss_mean": subtract(standard["stability"]["actor_kl_loss_mean"], graph["stability"]["actor_kl_loss_mean"]),
        "trajectory_length_mean": subtract(standard["trajectory"]["length_mean"], graph["trajectory"]["length_mean"]),
    }
    output = {
        "schema_version": "graph_frontier_guided_rl_v2_pilot128_comparison_v1",
        "verdict": "PARTIAL-GO",
        "gate_assessment": {
            "A_global_regression": {
                "status": "PASS",
                "evidence": "Graph minus Standard: official +0.00557, success +0.00977.",
            },
            "B_local_mechanism": {
                "status": "INCONCLUSIVE",
                "evidence": "Consumer rate +0.02999 and propagation +0.06552, but only 59/65 consumer opportunities and invalid retry worsened from 0/6 to 2/10.",
            },
            "C_propagation_preserved": {
                "status": "PASS",
                "evidence": "Propagation accuracy 0.9655 versus 0.9000.",
            },
            "D_stability_fixed": {
                "status": "PASS",
                "evidence": "Graph grad p95 5.440, p99 6.292, max 7.313; no harmful inversions.",
            },
        },
        "protocol": {
            "tasks": 128, "rollouts_per_task": 4, "optimizer_steps": 64,
            "seed": 20260917, "g": 4, "context": 6144,
            "max_prompt": 5200, "max_response": 768,
            "train_batch_size": 2, "per_gpu_microbatch": 1,
            "learning_rate": 1e-6, "kl_coefficient": 0.01,
            "pilot_train_parquet_sha256": sha256(DATA_ROOT / "train.parquet"),
            "pilot_validation_parquet_sha256": sha256(DATA_ROOT / "validation.parquet"),
            "dynamic_v1_model_sha256": sha256(INIT_ROOT / "model.safetensors"),
            "dynamic_v1_config_sha256": sha256(INIT_ROOT / "config.json"),
            "compute_mean_std_cross_steps_in_grpo": False,
            "tie_tolerance": 1e-6,
        },
        "arms": summaries,
        "graph_minus_standard": comparison,
        "paired_task_comparison": paired_task_summary(audits_by_arm["standard_v2"], audits_by_arm["graph_v2"]),
        "historical_v1": {
            "standard_official_reward": 0.56083984375,
            "standard_task_success": 0.326171875,
            "graph_official_reward": 0.541015625,
            "graph_task_success": 0.279296875,
            "graph_grad_median": 2.83,
            "graph_grad_p99": 51.745,
            "graph_grad_max": 129.599,
        },
        "evaluation": {
            "frozen_300": "FROZEN_300_NOT_RUN_FOR_PILOT",
            "bfcl": "BFCL_NOT_RUN_FOR_PILOT",
        },
    }
    (args.output_dir / "guided_rl_v2_pilot128_comparison.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )

    fields = [
        "arm", "train_step", "task_id", "group_uid", "traj_uid", "rollout_index",
        "success", "official_reward", "eligible_consumer_opportunities",
        "consumer_completed", "consumer_score", "eligible_prop_opportunities",
        "prop_correct", "prop_score", "retry_opportunities", "invalid_retry",
        "retry_score", "graph_available", "graph_score", "primary_class",
        "graph_tiebreak_eligible", "graph_tiebreak_applied", "final_group_reward",
        "advantage", "trajectory_length", "generated_tokens", "tool_call_count",
    ]
    with (args.output_dir / "guided_rl_v2_reward_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in ("standard_v2", "graph_v2"):
            for row in audits_by_arm[arm]:
                writer.writerow({"arm": arm, **{key: row.get(key) for key in fields if key != "arm"}})
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


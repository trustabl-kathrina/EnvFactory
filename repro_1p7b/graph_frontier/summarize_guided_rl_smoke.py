"""Summarize auditable gates from a guided-RL smoke output directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_RE = re.compile(r"(?:^| - )([A-Za-z0-9_./-]+):(-?[0-9]+(?:\.[0-9]+)?)")
GPU_RE = re.compile(r"^(\d+),\s*(\d+) MiB,\s*(\d+) MiB,\s*(\d+) %$")


def parse_rewards(text: str) -> list[dict[str, Any]]:
    rewards = []
    for line in text.splitlines():
        marker = "ENVFACTORY_REWARD "
        if marker not in line:
            continue
        payload = ANSI_RE.sub("", line.split(marker, 1)[1]).strip()
        rewards.append(json.loads(payload))
    return rewards


def parse_metrics(text: str) -> dict[str, float]:
    clean = ANSI_RE.sub("", text).replace("\r", "\n")
    lines = [line for line in clean.splitlines() if "training/global_step:" in line]
    if not lines:
        return {}
    return {key: float(value) for key, value in METRIC_RE.findall(lines[-1])}


def parse_gpu(path: Path) -> dict[str, Any]:
    peaks: dict[str, int] = {}
    totals: dict[str, int] = {}
    if not path.exists():
        return {"peak_memory_mib": peaks, "total_memory_mib": totals}
    for line in path.read_text(errors="replace").splitlines():
        match = GPU_RE.match(line.strip())
        if not match:
            continue
        gpu, used, total, _ = match.groups()
        peaks[gpu] = max(peaks.get(gpu, 0), int(used))
        totals[gpu] = int(total)
    return {
        "peak_memory_mib": peaks,
        "total_memory_mib": totals,
        "headroom_mib": {gpu: totals[gpu] - peaks.get(gpu, 0) for gpu in totals},
    }


def parse_rollouts(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    event_count = typed_arguments = typed_responses = 0
    task_ids = []
    for path in files:
        payload = json.loads(path.read_text())
        task_ids.append(str(payload.get("task_id", "unknown")))
        events = payload.get("trace", {}).get("events", []) or []
        event_count += len(events)
        typed_arguments += sum(isinstance(event.get("tool_arguments"), dict) for event in events)
        typed_responses += sum(
            event.get("returned_fields") not in (None, "unknown") for event in events
        )
    return {
        "files": len(files),
        "task_ids": task_ids,
        "typed_events": event_count,
        "typed_argument_events": typed_arguments,
        "typed_response_events": typed_responses,
    }


def summarize(directory: Path, expected_mode: str) -> dict[str, Any]:
    log_path = directory / "train.log"
    text = log_path.read_text(errors="replace")
    rewards = parse_rewards(text)
    metrics = parse_metrics(text)
    scores = [float(item["score"]) for item in rewards]
    graph = [item.get("graph_frontier", {}) for item in rewards]
    modes = sorted({item.get("reward_mode", "unknown") for item in rewards})
    result = {
        "schema_version": "graph_frontier_guided_rl_v1_smoke_summary",
        "expected_reward_mode": expected_mode,
        "observed_reward_modes": modes,
        "optimizer": {
            "global_step": metrics.get("training/global_step", 0.0),
            "grad_norm": metrics.get("actor/grad_norm"),
            "kl_loss": metrics.get("actor/kl_loss"),
            "pg_loss": metrics.get("actor/pg_loss"),
            "update_actor_seconds": metrics.get("timing_s/update_actor"),
        },
        "reward": {
            "count": len(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": sum(scores) / len(scores) if scores else None,
            "has_variance": len(set(scores)) > 1,
            "nonzero_propagation_rollouts": sum(
                float(item.get("propagation_fraction", 0.0)) > 0.0 for item in graph
            ),
            "nonzero_completion_rollouts": sum(
                float(item.get("consumer_completion_fraction", 0.0)) > 0.0 for item in graph
            ),
            "retry_detected_rollouts": sum(
                int(item.get("invalid_retry_count", 0)) > 0 for item in graph
            ),
            "decompositions": rewards,
        },
        "gpu": parse_gpu(directory / "gpu.csv"),
        "rollouts": parse_rollouts(directory / "rollouts"),
        "log_has_job_error": "Error executing job with overrides" in text,
        "training_progress_complete": "Training Progress: 100%" in ANSI_RE.sub("", text),
    }
    optimizer_ok = (
        result["optimizer"]["global_step"] >= 1
        and result["optimizer"]["grad_norm"] is not None
        and math.isfinite(result["optimizer"]["grad_norm"])
    )
    common = (
        modes == [expected_mode]
        and optimizer_ok
        and result["reward"]["has_variance"]
        and not result["log_has_job_error"]
    )
    graph_runtime = True
    if expected_mode == "graph_frontier":
        graph_runtime = (
            result["reward"]["nonzero_propagation_rollouts"] > 0
            and result["reward"]["nonzero_completion_rollouts"] > 0
        )
    result["gate"] = {
        "optimizer_update": optimizer_ok,
        "common_runtime": common,
        "graph_runtime_components": graph_runtime,
        "pass": common and graph_runtime,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("official", "graph_frontier"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.log_dir, args.mode)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()

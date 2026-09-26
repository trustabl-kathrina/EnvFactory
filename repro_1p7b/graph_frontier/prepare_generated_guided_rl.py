"""Convert the newly generated EnvFactory pool to verl-agent parquet rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "EnvFactory-generated-guided-rl-v1"


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(stable(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def prompt_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return decode(row["prompt"])


def env_kwargs(row: Mapping[str, Any]) -> dict[str, Any]:
    factory = row["extra_info"]["mcp_factory_kwargs"]
    sidecar = row["extra_info"]["graph_frontier"]
    result = {key: decode(value) for key, value in factory.items()}
    result.update(
        {
            "ground_truth": decode(row["reward_model"]["ground_truth"]),
            "query": prompt_messages(row)[-1]["content"],
            "task_id": row["task_id"],
            "graph_frontier": sidecar,
            "source": SOURCE,
            "frozen_300_overlap": False,
            "reward_scheduler": {"mode": "fix", "default_ratio": 0.5},
        }
    )
    if result["task_id"] != sidecar.get("task_id"):
        raise RuntimeError("task_id/sidecar mismatch")
    return result


def structural(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    edges = sidecar.get("dependency_edges", []) or []
    depth = sidecar.get("dependency_depth", "unknown")
    internal = sum(edge.get("internal_parameter") is True for edge in edges)
    terminal = bool(edges) and edges[-1].get("consumer_tool") is not None
    eligibility = sidecar.get("probe_eligibility", {}) or {}
    graph_eligible = eligibility.get("eligible") is True
    # A structural edge is not sufficient evidence that it represents the
    # selected-reference semantics.  Cross-turn/alternate-reference edges are
    # useful broad-pool examples, but must never enter the Graph-Frontier
    # oversampling bucket.
    targeted = graph_eligible and (
        internal > 0 or (isinstance(depth, int) and depth >= 2) or terminal
    )
    return {
        "dependency_depth_bucket": "3+" if isinstance(depth, int) and depth >= 3 else str(depth),
        "internal_edge_count": internal,
        "multi_hop_internal_dependency": internal > 0 and isinstance(depth, int) and depth >= 2,
        "terminal_consumer_action": terminal,
        "tool_chain_length": len(sidecar.get("gold_tool_sequence", []) or []),
        "graph_reward_eligible": graph_eligible,
        "graph_reward_skip_reasons": list(eligibility.get("reasons", []) or []),
        "frontier_targeted": targeted,
        "sampling_weight": 1.65 if targeted else 0.35,
    }


def export(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        kwargs = env_kwargs(row)
        metadata = structural(kwargs["graph_frontier"])
        result.append(
            {
                "data_source": "EnvFactory",
                "prompt": prompt_messages(row),
                "agent_name": "tool_agent",
                "ability": "tool_use",
                "reward_model": row["reward_model"],
                "extra_info": {
                    "split": split,
                    "task_id": row["task_id"],
                    "generation_seed": str(
                        row["extra_info"].get("generation_seed", "unknown")
                    ),
                    "dependency_depth_bucket": metadata["dependency_depth_bucket"],
                    "internal_edge_count": int(metadata["internal_edge_count"]),
                    "frontier_targeted": bool(metadata["frontier_targeted"]),
                    "sampling_weight": float(metadata["sampling_weight"]),
                    "structural_metadata_json": stable(metadata),
                },
                "env_kwargs": stable(kwargs),
            }
        )
    return result


def materialize_sampling_schedule(
    rows: list[dict[str, Any]],
    *,
    total_samples: int,
    targeted_fraction: float = 0.65,
    seed: int = 20260917,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the one shared, deterministic Standard/Graph prompt schedule."""

    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if not 0.0 < targeted_fraction < 1.0:
        raise ValueError("targeted_fraction must be between zero and one")
    buckets = {
        True: [row for row in rows if row["extra_info"]["frontier_targeted"]],
        False: [row for row in rows if not row["extra_info"]["frontier_targeted"]],
    }
    if not buckets[True] or not buckets[False]:
        raise RuntimeError("both frontier-targeted and broad pools are required")
    targeted_count = round(total_samples * targeted_fraction)
    counts = {True: targeted_count, False: total_samples - targeted_count}
    rng = random.Random(seed)
    for pool in buckets.values():
        rng.shuffle(pool)
    labels = [True] * counts[True] + [False] * counts[False]
    rng.shuffle(labels)
    cursors = {True: 0, False: 0}
    schedule = []
    reuse: Counter[str] = Counter()
    for targeted in labels:
        pool = buckets[targeted]
        cursor = cursors[targeted]
        row = pool[cursor % len(pool)]
        cursors[targeted] += 1
        copy = dict(row)
        copy["extra_info"] = dict(row["extra_info"])
        copy["extra_info"]["schedule_index"] = len(schedule)
        copy["extra_info"]["sampling_seed"] = seed
        schedule.append(copy)
        reuse[str(copy["extra_info"]["task_id"])] += 1
    audit = {
        "sampling_seed": seed,
        "total_prompt_samples": total_samples,
        "targeted_prompt_samples": counts[True],
        "broad_prompt_samples": counts[False],
        "targeted_fraction": counts[True] / total_samples,
        "unique_prompt_tasks": len(reuse),
        "unique_prompt_ratio": len(reuse) / total_samples,
        "max_task_reuse": max(reuse.values()),
        "task_reuse_distribution": dict(sorted(Counter(reuse.values()).items())),
        "task_reuse_counts": dict(sorted(reuse.items())),
    }
    return schedule, audit


def prepare(
    source_dir: Path,
    output_dir: Path,
    *,
    sampled_train_size: int | None = None,
    sampling_seed: int = 20260917,
    targeted_fraction: float = 0.65,
    train_json: str = "train.json",
    val_json: str = "val.json",
) -> dict[str, Any]:
    from datasets import Dataset

    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = source_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source") != SOURCE:
        raise RuntimeError("dataset source mismatch")
    overlap = manifest.get(
        "FROZEN_300_EXACT_TRAIN_OVERLAP",
        manifest.get("frozen_300_train_overlap"),
    )
    if overlap != 0:
        raise RuntimeError("frozen-300 overlap gate failed")
    train_raw = json.loads((source_dir / train_json).read_text())
    val_raw = json.loads((source_dir / val_json).read_text())
    train, val = export(train_raw, "train"), export(val_raw, "rl_val")
    schedule_audit = None
    if sampled_train_size is not None:
        train, schedule_audit = materialize_sampling_schedule(
            train,
            total_samples=sampled_train_size,
            targeted_fraction=targeted_fraction,
            seed=sampling_seed,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).to_parquet(output_dir / "train.parquet")
    Dataset.from_list(val).to_parquet(output_dir / "validation.parquet")
    for row in train[:4] + val[:4]:
        parsed = json.loads(row["env_kwargs"])
        assert parsed["source"] == SOURCE
        assert parsed["frozen_300_overlap"] is False
        assert parsed["task_id"] == parsed["graph_frontier"]["task_id"]
    report = {
        "schema_version": "graph_frontier_guided_rl_v1_generated_parquet_audit",
        "source": SOURCE,
        "source_manifest_sha256": digest(manifest),
        "train_count": len(train),
        "validation_count": len(val),
        "formal_released_envfactory_rl_rows": 0,
        "frozen_300_train_overlap": 0,
        "depth_train": dict(Counter(row["extra_info"]["dependency_depth_bucket"] for row in train)),
        "targeted_train": sum(row["extra_info"]["frontier_targeted"] for row in train),
        "sampling_policy": (
            "natural frozen distribution; no frontier-aware sampling"
            if schedule_audit is None else "materialized frontier-aware schedule"
        ),
        "materialized_sampling_schedule": schedule_audit,
        "prompt_schedule_sha256": digest([
            row["extra_info"]["task_id"] for row in train
        ]),
        "train_parquet_sha256": file_digest(output_dir / "train.parquet"),
        "validation_parquet_sha256": file_digest(output_dir / "validation.parquet"),
    }
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampled-train-size", type=int)
    parser.add_argument("--sampling-seed", type=int, default=20260917)
    parser.add_argument("--targeted-fraction", type=float, default=0.65)
    parser.add_argument("--train-json", default="train.json")
    parser.add_argument("--val-json", default="val.json")
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.source_dir,
        args.output_dir,
        sampled_train_size=args.sampled_train_size,
        sampling_seed=args.sampling_seed,
        targeted_fraction=args.targeted_fraction,
        train_json=args.train_json,
        val_json=args.val_json,
    ), indent=2))


if __name__ == "__main__":
    main()

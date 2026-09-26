"""Prepare official EnvFactory-RL for Graph-Frontier Guided RL v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260916


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(stable(value).encode()).hexdigest()


def prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row["prompt"]
    return json.loads(value) if isinstance(value, str) else value


def query_text(row: dict[str, Any]) -> str:
    messages = prompt_messages(row)
    return str(messages[-1]["content"]).strip()


def kwargs(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["extra_info"]["mcp_factory_kwargs"]
    result = {
        key: json.loads(value) if isinstance(value, str) else value
        for key, value in raw.items()
    }
    result["ground_truth"] = json.loads(row["reward_model"]["ground_truth"])
    result["query"] = query_text(row)
    result["task_id"] = "envfactory-rl-" + digest(
        {
            "prompt": prompt_messages(row),
            "servers": result["mcp_servers"],
            "initial": result["initial_config"],
        }
    )[:20]
    return result


def structural(ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    length = len(ground_truth)
    depth = 1 if length <= 2 else 2 if length <= 4 else 3
    repeated_fields = 0
    prior_values: set[str] = set()
    for call in ground_truth:
        arguments = call.get("arguments", {})
        values = {stable(value) for value in arguments.values()}
        repeated_fields += len(values & prior_values)
        prior_values |= values
    terminal_consumer = length >= 3
    consumer = min(1.0, max(0.0, (length - 2) / 4))
    propagation = min(1.0, repeated_fields / max(1, length - 1))
    semantic = 1.0 / max(1, length)
    frontier = 0.50 * consumer + 0.30 * propagation + 0.20 * semantic
    return {
        "dependency_depth_bucket": "3+" if depth >= 3 else str(depth),
        "internal_edge_proxy_count": repeated_fields,
        "multi_hop_internal_dependency": depth >= 3 and repeated_fields > 0,
        "terminal_consumer_action": terminal_consumer,
        "tool_chain_length": length,
        "frontier_score": frontier,
        "sampling_weight": min(2.0, max(0.25, 0.25 + 1.75 * frontier)),
    }


def frozen_queries(manifest: Path) -> set[str]:
    return {
        digest(str(json.loads(line)["query"]).strip())
        for line in manifest.read_text().splitlines()
        if line.strip()
    }


def prepare(source: Path, manifest: Path, output: Path) -> dict[str, Any]:
    rows = json.loads(source.read_text())
    frozen = frozen_queries(manifest)
    prepared = []
    overlap = []
    local_servers = {path.stem for path in (ROOT / "envs/tools").glob("*.py")}
    missing_servers = Counter()
    for row in rows:
        item = kwargs(row)
        qhash = digest(item["query"])
        if qhash in frozen:
            overlap.append(item["task_id"])
            continue
        absent = set(item["mcp_servers"]) - local_servers
        if absent:
            missing_servers.update(absent)
            continue
        item["structural_metadata"] = structural(item["ground_truth"])
        item["source"] = "LARK-Lab/EnvFactory-RL"
        item["frozen_300_overlap"] = False
        prepared.append((row, item))

    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in prepared:
        item = pair[1]
        key = digest(
            {
                "servers": item["mcp_servers"],
                "initial_config": item["initial_config"],
            }
        )
        groups.setdefault(key, []).append(pair)
    ordered_groups = sorted(
        groups,
        key=lambda key: digest({"seed": SEED, "group": key}),
    )
    val_target = max(1, round(len(prepared) * 0.05))
    val_groups: set[str] = set()
    val_count = 0
    for key in ordered_groups:
        if val_count >= val_target:
            break
        val_groups.add(key)
        val_count += len(groups[key])

    rng = random.Random(SEED)
    train_pairs = []
    val_pairs = []
    for key, pairs in groups.items():
        if key in val_groups:
            val_pairs.extend(pairs)
        else:
            train_pairs.extend(pairs)
    train_pairs.sort(
        key=lambda pair: rng.random()
        ** (1.0 / pair[1]["structural_metadata"]["sampling_weight"])
    )
    val_pairs.sort(key=lambda pair: pair[1]["task_id"])

    def export(pairs, split):
        output_rows = []
        for row, item in pairs:
            output_rows.append(
                {
                    "data_source": "EnvFactory",
                    "prompt": prompt_messages(row),
                    "agent_name": "tool_agent",
                    "ability": "tool_use",
                    "reward_model": row["reward_model"],
                    "extra_info": {
                        **row["extra_info"],
                        "split": split,
                        "task_id": item["task_id"],
                        "structural_metadata": item["structural_metadata"],
                    },
                    # Keep Arrow schema stable across heterogeneous MCP states.
                    # The adapter parses this lossless JSON string at reset time.
                    "env_kwargs": stable(item),
                }
            )
        return output_rows

    train = export(train_pairs, "train")
    val = export(val_pairs, "validation")
    output.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).to_parquet(output / "train.parquet")
    Dataset.from_list(val).to_parquet(output / "validation.parquet")
    for row in (train[:3] + val[:3]):
        assert json.loads(stable(row)) == row
        assert json.loads(row["env_kwargs"])["frozen_300_overlap"] is False
    report = {
        "schema_version": "graph_frontier_guided_rl_v1_data_audit",
        "source": str(source),
        "source_count": len(rows),
        "eligible_count": len(prepared),
        "train_count": len(train),
        "validation_count": len(val),
        "task_group_count": len(groups),
        "frozen_300_query_overlap_removed": len(overlap),
        "frozen_300_overlap_task_ids": overlap,
        "missing_server_counts": dict(missing_servers),
        "depth_train": dict(Counter(
            json.loads(row["env_kwargs"])["structural_metadata"]["dependency_depth_bucket"]
            for row in train
        )),
        "split_seed": SEED,
        "split_rule": "task-disjoint by servers plus initial_config hash",
        "frontier_formula": "0.50 consumer + 0.30 propagation + 0.20 semantic",
        "exploration_floor": 0.25,
        "sampling_weight_cap": 2.0,
        "frozen_300_not_used_for_training": True,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "repro_1p7b/data/envfactory_rl_official/env_factory_rl.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / (
            "repro_1p7b/results/graph_frontier/confirm_300/frozen/"
            "confirm_300_seed_20260914.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "repro_1p7b/data/graph_frontier_rl_v1",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()

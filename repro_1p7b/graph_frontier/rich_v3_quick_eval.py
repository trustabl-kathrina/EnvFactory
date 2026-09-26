"""Task-disjoint short heldout evaluation for Rich-v3's quick DPO pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.collect_preference_rollouts import collect_one
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256


SCHEMA = "graph_frontier_rich_v3_quick_effect_eval_v1"


def source_rows(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((root / "quick_effect_manifest.json").read_text())
    if manifest["frozen300_exact_overlap"] != 0:
        raise ValueError("Frozen300 contamination")
    source = Path(manifest["source_run"]) / "valid/graph_frontier_rich_train_v1.jsonl"
    if file_sha256(source) != manifest["source_data_sha256"]:
        raise ValueError("source file changed")
    path = root / "heldout_eval/source_tasks.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 10 or [row["task_id"] for row in rows] != manifest["task_ids"]["heldout_eval"]:
        raise ValueError("heldout split changed")
    if any(row.get("frozen_300_overlap") is not False for row in rows):
        raise ValueError("heldout Frozen300 overlap flag")
    return rows


def collect(root: Path, label: str, endpoint: str, model: str, seed: int, max_steps: int, max_tokens: int) -> dict[str, Any]:
    if label not in ("base", "graph_dpo"):
        raise ValueError("unexpected model label")
    rows = source_rows(root)
    output = root / "heldout_eval" / label
    output.mkdir(exist_ok=True)
    config = {
        "schema_version": SCHEMA,
        "label": label,
        "endpoint": endpoint,
        "model": model,
        "seed": seed,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "task_ids": [row["task_id"] for row in rows],
        "source_sha256": json.loads((root / "quick_effect_manifest.json").read_text())["source_data_sha256"],
    }
    config_path = output / "eval_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("evaluation config changed on resume")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    results = []
    for index, row in enumerate(rows):
        rollout_seed = seed + index * 1009
        result = collect_one(
            row,
            endpoint=endpoint,
            model=model,
            rollout_index=0,
            seed=rollout_seed,
            max_steps=max_steps,
            temperature=0.0,
            max_tokens=max_tokens,
            output=output,
        )
        if result["task_id"] != row["task_id"] or result["sampling_seed"] != rollout_seed:
            raise ValueError("cached rollout identity mismatch")
        results.append(result)
        print(f"QUICK_EVAL_PROGRESS {label} {index + 1}/{len(rows)} success={sum(x['semantic_success'] for x in results)}", flush=True)
    rewards = [x["official_reward"] for x in results if isinstance(x.get("official_reward"), (int, float))]
    summary = {
        "schema_version": SCHEMA,
        "label": label,
        "tasks": len(rows),
        "semantic_success": sum(x["semantic_success"] for x in results),
        "official_reward_known": len(rewards),
        "official_reward_mean": sum(rewards) / len(rewards) if rewards else "unknown",
        "total_tool_calls": sum(sum(x.get("typed_event") is not None for x in r["steps"]) for r in results),
        "task_ids": [r["task_id"] for r in results],
    }
    (output / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def compare(root: Path) -> dict[str, Any]:
    rows = source_rows(root)
    paired = []
    for row in rows:
        task = row["task_id"]
        path_base = root / "heldout_eval/base" / f"{task}.r0.json"
        path_graph = root / "heldout_eval/graph_dpo" / f"{task}.r0.json"
        if not path_base.exists() or not path_graph.exists():
            raise ValueError(f"missing paired heldout result: {task}")
        base = json.loads(path_base.read_text())
        graph = json.loads(path_graph.read_text())
        if base["sampling_seed"] != graph["sampling_seed"]:
            raise ValueError(f"unequal evaluation seed: {task}")
        paired.append({
            "task_id": task,
            "base_semantic_success": bool(base["semantic_success"]),
            "graph_dpo_semantic_success": bool(graph["semantic_success"]),
            "base_official_reward": base.get("official_reward", "unknown"),
            "graph_dpo_official_reward": graph.get("official_reward", "unknown"),
            "base_tool_calls": sum(s.get("typed_event") is not None for s in base["steps"]),
            "graph_dpo_tool_calls": sum(s.get("typed_event") is not None for s in graph["steps"]),
        })
    wins = sum(x["graph_dpo_semantic_success"] and not x["base_semantic_success"] for x in paired)
    losses = sum(x["base_semantic_success"] and not x["graph_dpo_semantic_success"] for x in paired)
    report = {
        "schema_version": SCHEMA,
        "task_count": len(paired),
        "base_success": sum(x["base_semantic_success"] for x in paired),
        "graph_dpo_success": sum(x["graph_dpo_semantic_success"] for x in paired),
        "graph_dpo_wins": wins,
        "graph_dpo_losses": losses,
        "ties": len(paired) - wins - losses,
        "base_reward_mean": sum(x["base_official_reward"] for x in paired if isinstance(x["base_official_reward"], (int, float))) / max(1, sum(isinstance(x["base_official_reward"], (int, float)) for x in paired)),
        "graph_dpo_reward_mean": sum(x["graph_dpo_official_reward"] for x in paired if isinstance(x["graph_dpo_official_reward"], (int, float))) / max(1, sum(isinstance(x["graph_dpo_official_reward"], (int, float)) for x in paired)),
        "paired": paired,
        "limitations": "Only 10 heldout tasks and one deterministic rollout each; this compares Graph DPO to Dynamic-v1, not Graph DPO to matched Standard DPO.",
    }
    (root / "quick_effect_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(args.root)
    else:
        if not all((args.label, args.endpoint, args.model)):
            parser.error("collect requires --label, --endpoint and --model")
        result = collect(args.root, args.label, args.endpoint, args.model, args.seed, args.max_steps, args.max_tokens)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

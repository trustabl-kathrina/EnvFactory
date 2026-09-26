"""Paired, heldout next-action diagnosis after the Rich-v3 quick DPO smoke."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import classify_sample
from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action


def states(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((root / "quick_effect_manifest.json").read_text())
    ids = set(manifest["task_ids"]["heldout_eval"])
    if manifest["frozen300_exact_overlap"] != 0:
        raise ValueError("Frozen300 overlap")
    path = root / "heldout_eval/frontier_states.jsonl"
    result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(result) != 20 or any(s["task_id"] not in ids or s["chosen_executable"] is not True for s in result):
        raise ValueError("heldout replay states changed")
    return result


def collect(root: Path, label: str, endpoint: str, model: str, seed: int, max_tokens: int) -> dict[str, Any]:
    if label not in ("base", "graph_dpo"):
        raise ValueError("invalid label")
    result_dir = root / "heldout_eval" / f"{label}_frontier"
    result_dir.mkdir(exist_ok=True)
    config = {
        "label": label, "model": model, "seed": seed, "max_tokens": max_tokens,
        "temperature": 0.0, "task_ids": json.loads((root / "quick_effect_manifest.json").read_text())["task_ids"]["heldout_eval"],
    }
    config_path = result_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("frontier eval config changed")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    found = []
    for index, state in enumerate(states(root)):
        path = result_dir / f"{state['state_id']}.json"
        state_seed = seed + index * 1009
        if path.exists():
            item = json.loads(path.read_text())
        else:
            try:
                message, action, usage, finish_reason = generate_action(
                    endpoint, model, state["prompt_messages"], state["tools"],
                    seed=state_seed, temperature=0.0, max_tokens=max_tokens,
                )
                sample = {"parsed_action": action, "model_message": message}
                classification = classify_sample(sample, state)
                item = {
                    "state_id": state["state_id"], "task_id": state["task_id"],
                    "sampling_seed": state_seed, "classification": classification,
                    "parsed_action": action, "model_message": message,
                    "usage": usage, "finish_reason": finish_reason,
                }
            except Exception as exc:
                item = {
                    "state_id": state["state_id"], "task_id": state["task_id"],
                    "sampling_seed": state_seed, "classification": "request_error",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            temporary.replace(path)
        if item["state_id"] != state["state_id"] or item["sampling_seed"] != state_seed:
            raise ValueError("cached frontier identity changed")
        found.append(item)
        print(f"FRONTIER_EVAL_PROGRESS {label} {index + 1}/{len(states(root))} correct={sum(x['classification']=='correct' for x in found)}", flush=True)
    categories = dict(Counter(x["classification"] for x in found))
    report = {
        "schema_version": "graph_frontier_rich_v3_quick_frontier_eval_v1",
        "label": label, "states": len(found), "tasks": len({x["task_id"] for x in found}),
        "classifications": categories, "exact_correct": categories.get("correct", 0),
        "flow_correct": categories.get("correct", 0) + categories.get("ambiguous_same_tool", 0),
        "request_errors": categories.get("request_error", 0),
    }
    (result_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def compare(root: Path) -> dict[str, Any]:
    paired = []
    for state in states(root):
        sid = state["state_id"]
        base = json.loads((root / "heldout_eval/base_frontier" / f"{sid}.json").read_text())
        graph = json.loads((root / "heldout_eval/graph_dpo_frontier" / f"{sid}.json").read_text())
        if base["sampling_seed"] != graph["sampling_seed"]:
            raise ValueError("unequal frontier seeds")
        paired.append({"task_id": state["task_id"], "state_id": sid,
                       "base": base["classification"], "graph_dpo": graph["classification"]})
    valid = [p for p in paired if p["base"] != "request_error" and p["graph_dpo"] != "request_error"]
    is_flow = lambda label: label in ("correct", "ambiguous_same_tool")
    report = {
        "schema_version": "graph_frontier_rich_v3_quick_frontier_comparison_v1",
        "states": len(paired), "valid_paired_states": len(valid),
        "base_exact_correct": sum(p["base"] == "correct" for p in valid),
        "graph_dpo_exact_correct": sum(p["graph_dpo"] == "correct" for p in valid),
        "base_flow_correct": sum(is_flow(p["base"]) for p in valid),
        "graph_dpo_flow_correct": sum(is_flow(p["graph_dpo"]) for p in valid),
        "graph_dpo_flow_wins": sum(is_flow(p["graph_dpo"]) and not is_flow(p["base"]) for p in valid),
        "graph_dpo_flow_losses": sum(is_flow(p["base"]) and not is_flow(p["graph_dpo"]) for p in valid),
        "base_classes": dict(Counter(p["base"] for p in valid)),
        "graph_dpo_classes": dict(Counter(p["graph_dpo"] for p in valid)),
        "paired": paired,
        "limitations": "This is gold-prefix next-action accuracy, not a full executable trajectory. Only 20 frontiers from 10 heldout tasks.",
    }
    (root / "quick_frontier_effect_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(args.root)
    else:
        if not all((args.label, args.endpoint, args.model)):
            parser.error("collect requires --label, --endpoint, --model")
        result = collect(args.root, args.label, args.endpoint, args.model, args.seed, args.max_tokens)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

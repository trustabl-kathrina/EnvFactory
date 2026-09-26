"""One-time deterministic replacement of a premature-stop pair by a real wrong-value pair."""
from __future__ import annotations
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from repro_1p7b.graph_frontier.active_frontier_v2 import classify_sample, write_jsonl
from repro_1p7b.graph_frontier.materialize_preference_dpo import serialize_pair
from repro_1p7b.graph_frontier.rich_v3_quick_pairs import materialize, read_jsonl

def run(root: Path, model: str) -> dict:
    audit = json.loads((root / "quick_pair_audit.json").read_text())
    if audit["verdict"] != "PAIR_READY":
        raise ValueError("pair gate failed")
    selected_path = root / "graph_pairs_train16.jsonl"
    selected = read_jsonl(selected_path)
    if len(selected) != 16:
        raise ValueError("expected 16 selected pairs")
    if (root / "graph_pairs_train16.initial.jsonl").exists():
        raise FileExistsError("already rebalanced")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    candidates = []
    for state in read_jsonl(root / "train/frontier_states.jsonl"):
        path = root / "train/sampled_actions" / f"{state['state_id']}.json"
        for sample in json.loads(path.read_text())["samples"]:
            if classify_sample(sample, state) != "wrong_value":
                continue
            pair = {
                "schema_version": "graph_targeted_rich_v3_quick_pair_v1",
                "pair_id": f"rich-v3-{state['state_id']}-{sample['sample_index']}",
                "task_id": state["task_id"], "state_id": state["state_id"],
                "state_signature": state["state_signature"], "edge_id": state["edge_id"],
                "prompt": state["prompt_messages"], "tools": state["tools"],
                "chosen": [state["chosen_message"]], "rejected": [sample["model_message"]],
                "chosen_action": state["chosen_action"], "rejected_action": sample["parsed_action"],
                "failure_type": "wrong_value", "producer_tool": state["producer_tool"],
                "consumer_tool": state["consumer_tool"],
                "internal_value_source": state["internal_value_source"],
                "dependency_depth": state["dependency_depth"],
                "environment_identifiers": state["environment_identifiers"],
                "chosen_executable": True, "rejected_is_observed_dynamic_v1_output": True,
                "sampling_seed": sample["sampling_seed"], "frozen_300_overlap": False,
            }
            serialized, reason = serialize_pair(tokenizer, pair, 12288)
            if serialized is not None:
                candidates.append(pair)
    candidates.sort(key=lambda p: (p["task_id"], p["pair_id"]))
    if not candidates:
        raise ValueError("no serializable observed wrong_value pair")
    original_mix = dict(Counter(p["failure_type"] for p in selected))
    replacement = candidates[0]
    index = max(i for i,p in enumerate(selected) if p["failure_type"] == "premature_stop")
    selected[index] = replacement
    if len({p["task_id"] for p in selected}) < 10:
        raise ValueError("task diversity would drop below 10")
    shutil.copy2(selected_path, root / "graph_pairs_train16.initial.jsonl")
    shutil.copy2(root / "dpo_train16.jsonl", root / "dpo_train16.initial.jsonl")
    write_jsonl(selected_path, selected)
    serialized_audit = materialize(root, model, 12288)
    report = {
        "before": original_mix,
        "after": dict(Counter(p["failure_type"] for p in selected)),
        "selected_tasks": len({p["task_id"] for p in selected}),
        "wrong_value_pair_id": replacement["pair_id"],
        "materialization_verdict": serialized_audit["verdict"],
    }
    (root / "quick_pair_selection_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if serialized_audit["verdict"] != "DPO_READY":
        raise ValueError("serialization gate failed after replacement")
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.model), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

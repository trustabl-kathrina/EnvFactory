"""Same frozen 80-state greedy next-action probe for both EOS ablations."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_pilot.prepare import read_jsonl
from repro_1p7b.graph_frontier.cycle3_pilot.probe import (
    OUT as FROZEN, TOOL, canonical, parse_action,
)
from repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.prepare import OUT

ROOT = Path(__file__).resolve().parents[3]


def collect(ratio: int, gpu: int = 1) -> dict:
    if ratio not in (2, 4) or gpu != 1:
        raise ValueError("only ratio 2/4 on GPU1; Rich24 uses GPU0")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frozen_manifest = json.loads((FROZEN / "manifest.json").read_text())
    states = read_jsonl(FROZEN / "states.jsonl")
    if (len(states) != 80 or frozen_manifest["states_sha256"] != sha256(FROZEN / "states.jsonl")
            or frozen_manifest["train_task_overlap"] != 0):
        raise RuntimeError("frozen next-action probe changed")
    schedule = json.loads((OUT / f"schedule_{ratio}.json").read_text())
    if (schedule["heldout_probe_manifest_sha256"] != sha256(FROZEN / "manifest.json")
            or schedule["heldout_probe_states_sha256"] != sha256(FROZEN / "states.jsonl")
            or schedule["train_task_probe_overlap"] != 0):
        raise RuntimeError("ratio/probe task-disjoint gate failed")
    model_path = ROOT / f"repro_1p7b/checkpoints/cycle3_execfd_eos_{ratio}to1_64"
    train = json.loads((OUT / f"train_{ratio}to1/result.json").read_text())
    if (train["status"] != "TRAINED" or train["ratio"] != ratio
            or sha256(model_path / "model.safetensors") != train["model_sha256"]):
        raise RuntimeError("ratio checkpoint invalid")
    destination = OUT / f"evaluation/eos_{ratio}to1/probe.jsonl"
    if destination.exists():
        raise RuntimeError("refusing to overwrite ratio probe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(gpu)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True).to(f"cuda:{gpu}").eval()
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("probe EOS token mismatch")
    records = []
    with torch.inference_mode():
        for index, row in enumerate(states):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=f"cuda:{gpu}")
            continuation = model.generate(ids, attention_mask=torch.ones_like(ids),
                                          max_new_tokens=256, do_sample=False,
                                          eos_token_id=151645, pad_token_id=151645)[0, ids.shape[1]:].tolist()
            text = tokenizer.decode(continuation, skip_special_tokens=False)
            action = parse_action(text, bool(continuation and continuation[-1] == 151645))
            prior_calls = TOOL.findall(tokenizer.decode(row["prompt_ids"], skip_special_tokens=False))
            prior_action = None
            if prior_calls:
                try:
                    prior_action = json.loads(prior_calls[-1])
                except json.JSONDecodeError:
                    pass
            repeated = bool(action["kind"] == "tool" and isinstance(prior_action, dict)
                            and action["name"] == prior_action.get("name")
                            and canonical(action["arguments"]) == canonical(prior_action.get("arguments")))
            if row["kind"] == "terminal_stop":
                category = ("stop_no_tool" if action["kind"] == "no_tool_termination"
                            else "extra_tool" if action["kind"] == "tool"
                            else "invalid_or_incomplete")
            else:
                gold = row["expected_action"]
                if action["kind"] == "tool":
                    category = ("correct_tool_and_arguments" if action["name"] == gold["name"]
                                and canonical(action["arguments"]) == canonical(gold["arguments"])
                                else "wrong_argument" if action["name"] == gold["name"]
                                else "wrong_tool")
                elif action["kind"] == "no_tool_termination":
                    category = "premature_stop"
                else:
                    category = "invalid_or_incomplete"
            records.append({"task_id": row["task_id"], "kind": row["kind"],
                            "category": category, "action": action,
                            "repeat_previous_exact": repeated,
                            "generated_token_count": len(continuation),
                            "eos_observed": bool(continuation and continuation[-1] == 151645)})
            if (index + 1) % 10 == 0:
                print(stable({"ratio": ratio, "completed": index + 1, "total": 80}), flush=True)
    destination.write_text("".join(stable(r) + "\n" for r in records))
    result = {"schema_version": "cycle3_eos_ratio_probe_v1", "ratio": ratio,
              "task_count": 80, "frozen_states_sha256": sha256(FROZEN / "states.jsonl"),
              "checkpoint_sha256": train["model_sha256"],
              "result_sha256": sha256(destination),
              "terminal_categories": dict(collections.Counter(r["category"] for r in records if r["kind"] == "terminal_stop")),
              "terminal_repeat_previous_exact": sum(r["repeat_previous_exact"] for r in records if r["kind"] == "terminal_stop"),
              "unfinished_categories": dict(collections.Counter(r["category"] for r in records if r["kind"] == "unfinished_fd"))}
    (destination.parent / "probe_summary.json").write_text(stable(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, choices=(2, 4), required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    print(stable(collect(args.ratio, args.gpu)), flush=True)


if __name__ == "__main__":
    main()

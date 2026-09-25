"""Materialize exact Qwen3 assistant-only CE examples from replay-verified states."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256
from repro_1p7b.graph_frontier.first_divergence_v1 import canonical
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import MODEL, checked_plan

MAX_LENGTH = 16384


def encode_one(tokenizer, sample: dict, rollout: dict) -> dict:
    if sample.get("independent_prefix_replay_valid") is not True:
        raise ValueError("sample lacks independent prefix replay")
    if sample["task_id"] != rollout["task_id"] or sample["frozen_overlap"] is not False:
        raise ValueError("task/Frozen provenance mismatch")
    action = sample["gold_action"]
    if not isinstance(action.get("arguments"), dict):
        raise ValueError("gold arguments not typed")
    target_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": action["name"], "arguments": action["arguments"]},
        }],
    }
    common = {"tools": rollout["tools"], "tokenize": True, "enable_thinking": False}
    prompt_ids = tokenizer.apply_chat_template(
        sample["conversation_prefix"], add_generation_prompt=True, **common
    )
    full_ids = tokenizer.apply_chat_template(
        sample["conversation_prefix"] + [target_message],
        add_generation_prompt=False, **common
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not exact prefix of assistant target")
    if len(full_ids) > MAX_LENGTH or len(full_ids) <= len(prompt_ids):
        raise ValueError("sample length exceeds 16k or target empty")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if any(label != -100 for label in labels[:len(prompt_ids)]):
        raise AssertionError("non-assistant prompt token included in loss")
    if any(label == -100 for label in labels[len(prompt_ids):]):
        raise AssertionError("assistant target token was masked")
    suffix = tokenizer.decode(full_ids[len(prompt_ids):], skip_special_tokens=False)
    if "<tool_call>" not in suffix or action["name"] not in suffix:
        raise ValueError("target did not serialize as tool call")
    return {
        "task_id": sample["task_id"],
        "state_identity": sample["state_identity"],
        "first_divergence_step": sample["first_divergence_step"],
        "divergence_type": sample["divergence_type"],
        "graph_depth": sample["graph_depth"],
        "gold_action_digest": canonical(action),
        "input_ids": full_ids,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(full_ids) - len(prompt_ids),
        "total_tokens": len(full_ids),
    }


def materialize(plan_path: Path, samples_path: Path, replay_summary: Path,
                rollouts: Path, output: Path) -> dict:
    from transformers import AutoTokenizer

    if output.exists():
        raise RuntimeError("refusing to overwrite CE materialization")
    plan, _, _, _ = checked_plan(plan_path)
    summary = json.loads(replay_summary.read_text())
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    if (
        summary.get("independent_replay_fail") != 0
        or summary.get("independent_replay_pass") != len(samples)
        or summary.get("plan_sha256") != sha256(plan_path)
        or any(s["task_id"] not in set(plan["task_ids"]) for s in samples)
        or len(samples) != len({s["state_identity"] for s in samples})
    ):
        raise RuntimeError("replay or source gate failed")
    student_model = Path(plan.get("student_model_path", str(MODEL)))
    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    examples = []
    for sample in samples:
        rollout = json.loads((rollouts / (sample["task_id"] + ".r0.json")).read_text())
        examples.append(encode_one(tokenizer, sample, rollout))
    output.mkdir(parents=True)
    with (output / "dataset.jsonl").open("w", encoding="utf-8") as out:
        for example in examples:
            out.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    counts = collections.Counter(x["divergence_type"] for x in examples)
    manifest = {
        "schema_version": "first_divergence_ce_v1",
        "plan_sha256": sha256(plan_path),
        "verified_samples_sha256": sha256(samples_path),
        "replay_summary_sha256": sha256(replay_summary),
        "student_model_path": str(student_model),
        "model_weights_sha256": sha256(student_model / "model.safetensors"),
        "task_count": len({x["task_id"] for x in examples}),
        "example_count": len(examples),
        "max_length": MAX_LENGTH,
        "target_tokens_total": sum(x["target_tokens"] for x in examples),
        "prompt_tokens_masked": sum(x["prompt_tokens"] for x in examples),
        "tool_observation_tokens_in_loss": 0,
        "divergence_types": dict(sorted(counts.items())),
        "length_max": max(x["total_tokens"] for x in examples),
        "length_over_8192": sum(x["total_tokens"] > 8192 for x in examples),
        "frozen_exact_id_overlap": 0,
        "dataset_sha256": sha256(output / "dataset.jsonl"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.plan, args.samples, args.replay_summary,
                                 args.rollouts, args.output),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

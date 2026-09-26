"""CPU-only token and masking probe for a replay-verified terminal state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_terminal
from repro_1p7b.graph_frontier.smoke_fd_terminal_v2_inference import classify
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask
from src.utils.utils import parse_structured_output

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b"
SOURCE = ROOT / "repro_1p7b/logs/first_divergence_terminal_v2/terminal_dataset_attempt4/dataset.jsonl"
OUT = Path(__file__).with_name("token_probe.json")
TASK_ID = "gf-rich-7372af308535cceb8e11"


def main() -> None:
    if OUT.exists():
        raise RuntimeError("refusing to overwrite terminal-stop probe")
    sample = next(
        json.loads(line) for line in SOURCE.read_text().splitlines()
        if line.strip() and json.loads(line)["task_id"] == TASK_ID
    )
    assert sample["replay_valid"] is True and sample["final_state_match"] is True
    assert sample["conversation_prefix"][-1]["role"] == "tool"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prefix = sample["conversation_prefix"]
    kwargs = {"tools": sample["tool_schema"], "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(
        prefix, tokenize=True, add_generation_prompt=True, **kwargs
    )
    old_encoded = encode_terminal(tokenizer, sample)
    validate_mask(old_encoded)
    empty = {"role": "assistant", "content": ""}
    empty_sample = dict(sample, assistant_target=empty)
    try:
        encode_terminal(tokenizer, empty_sample)
    except ValueError as exc:
        old_encoder_error = str(exc)
    else:
        raise AssertionError("formal terminal encoder unexpectedly accepted empty content")
    cases = {}
    for name, target in (
        ("A_existing", sample["assistant_target"]),
        ("B_empty", empty),
        ("C_minimal", {"role": "assistant", "content": "Done."}),
    ):
        full = tokenizer.apply_chat_template(
            prefix + [target], tokenize=True, add_generation_prompt=False, **kwargs
        )
        assert full[:len(prompt)] == prompt
        suffix = full[len(prompt):]
        cases[name] = {
            "rendered_suffix": tokenizer.decode(suffix, skip_special_tokens=False),
            "target_ids": suffix,
            "target_tokens": [tokenizer.convert_ids_to_tokens(i) for i in suffix],
            "target_count": len(suffix),
        }
    suffix = cases["B_empty"]["target_ids"]
    assert suffix == [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[0]]
    full = prompt + suffix
    all_suffix_labels = [-100] * len(prompt) + suffix
    direct_row = {
        "task_id": TASK_ID,
        "input_ids": full,
        "labels": all_suffix_labels,
        "prompt_tokens": len(prompt),
        "total_tokens": len(full),
    }
    validate_mask(direct_row)
    eos_only_labels = [-100] * len(prompt) + [suffix[0], -100]
    eos_only_row = {**direct_row, "labels": eos_only_labels}
    try:
        validate_mask(eos_only_row)
    except RuntimeError as exc:
        eos_only_validator_error = str(exc)
    else:
        raise AssertionError("existing trainer validator unexpectedly accepted partial suffix mask")
    logits = torch.zeros((1, len(tokenizer)), dtype=torch.float32, device="cpu")
    target = torch.tensor([suffix[0]], dtype=torch.long, device="cpu")
    one_token_ce = float(F.cross_entropy(logits, target).item())
    assert 0 < one_token_ce < 100
    assert classify({"message": {"content": "", "tool_calls": []}, "finish_reason": "stop"}) == ("final", None)
    parsed = parse_structured_output("")
    assert parsed.get("non_think") == "" and "tool_call" not in parsed
    token_table = []
    for index in range(len(prompt) - 5, len(full)):
        token_id = full[index]
        token_table.append({
            "position": index,
            "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
            "label_existing_all_suffix": all_suffix_labels[index],
            "label_proposed_eos_only": eos_only_labels[index],
            "loss_mask_eos_only": eos_only_labels[index] != -100,
        })
    result = {
        "schema_version": "terminal_stop_token_probe_v1",
        "task_id": TASK_ID,
        "source_path": str(SOURCE),
        "model_path": str(MODEL),
        "model_type": json.loads((MODEL / "config.json").read_text())["model_type"],
        "tokenizer_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "generation_eos_ids": json.loads((MODEL / "generation_config.json").read_text())["eos_token_id"],
        "prompt_token_count": len(prompt),
        "prompt_decoded_suffix": tokenizer.decode(prompt[-16:], skip_special_tokens=False),
        "history_roles": [message["role"] for message in prefix],
        "last_observation": prefix[-1].get("content"),
        "replay_valid": sample["replay_valid"],
        "final_state_match": sample["final_state_match"],
        "cases": cases,
        "old_terminal_encoder_rejects_empty": old_encoder_error,
        "existing_trainer_direct_two_token_mask_accepts": True,
        "existing_trainer_eos_only_mask_rejects": eos_only_validator_error,
        "direct_empty_active_target_tokens": len(suffix),
        "eos_only_active_target_tokens": 1,
        "attention_mask_tail": [1] * len(token_table),
        "token_table": token_table,
        "cpu_toy_logit_ce_loss": one_token_ce,
        "cpu_toy_logit_ce_note": "Analytic one-position CE sanity, not a model forward.",
        "empty_parser_kind": "final",
        "querygen_empty_non_think": parsed["non_think"],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "task_id": TASK_ID, "prompt_tokens": len(prompt),
        "B_suffix": cases["B_empty"],
        "old_encoder_error": old_encoder_error,
        "existing_trainer_direct_two_token_mask_accepts": True,
        "existing_trainer_eos_only_mask_rejects": eos_only_validator_error,
        "cpu_toy_logit_ce_loss": one_token_ce,
        "token_table": token_table,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


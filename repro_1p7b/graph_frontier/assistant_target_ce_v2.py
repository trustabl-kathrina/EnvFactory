"""Shared assistant-target CE encoding for tool corrections and terminal answers."""
from __future__ import annotations

from typing import Any, Mapping

MAX_LENGTH = 16384
TARGET_TYPES = frozenset({"first_divergence_tool", "terminal_final", "terminal_stop"})


def encode_assistant_target(
    tokenizer: Any, *,
    task_id: str,
    state_identity: str,
    target_type: str,
    conversation_prefix: list[dict[str, Any]],
    assistant_target: dict[str, Any],
    tool_schema: list[dict[str, Any]],
) -> dict[str, Any]:
    if target_type not in TARGET_TYPES:
        raise ValueError("unknown target type")
    if not conversation_prefix or not tool_schema or assistant_target.get("role") != "assistant":
        raise ValueError("incomplete assistant-target sample")
    has_calls = bool(assistant_target.get("tool_calls"))
    if target_type == "terminal_final":
        if has_calls or not isinstance(assistant_target.get("content"), str) or not assistant_target["content"].strip():
            raise ValueError("terminal target must be existing natural language")
    elif target_type == "terminal_stop":
        if has_calls or assistant_target.get("content") != "":
            raise ValueError("terminal-stop requires an empty assistant turn")
    elif not has_calls:
        raise ValueError("first-divergence target must be a structured tool call")
    kwargs = {"tools": tool_schema, "tokenize": True, "enable_thinking": False}
    prompt_ids = tokenizer.apply_chat_template(
        conversation_prefix, add_generation_prompt=True, **kwargs
    )
    full_ids = tokenizer.apply_chat_template(
        conversation_prefix + [assistant_target], add_generation_prompt=False, **kwargs
    )
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant target is not an exact prompt suffix")
    if not 0 < len(full_ids) - len(prompt_ids) or len(full_ids) > MAX_LENGTH:
        raise ValueError("empty target or context exceeds 16k")
    suffix = tokenizer.decode(full_ids[len(prompt_ids):], skip_special_tokens=False)
    if target_type == "terminal_final" and "<tool_call>" in suffix:
        raise ValueError("terminal target serialized as tool call")
    if target_type == "first_divergence_tool" and "<tool_call>" not in suffix:
        raise ValueError("tool target did not serialize as tool call")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    assistant_end_token_id = None
    if target_type == "terminal_stop":
        assistant_end_token_id = tokenizer.eos_token_id
        rendered = full_ids[len(prompt_ids):]
        if assistant_end_token_id != 151645 or not rendered or rendered[0] != assistant_end_token_id:
            raise ValueError("current Qwen3 assistant-end token mismatch")
        if tokenizer.decode(rendered[1:], skip_special_tokens=False).strip():
            raise ValueError("terminal-stop suffix contains non-whitespace content")
        labels[len(prompt_ids) + 1:] = [-100] * (len(rendered) - 1)
    return {
        "schema_version": "assistant_target_ce_v2",
        "task_id": task_id,
        "state_identity": state_identity,
        "target_type": target_type,
        "input_ids": full_ids,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": 1 if target_type == "terminal_stop" else len(full_ids) - len(prompt_ids),
        "assistant_end_token_id": assistant_end_token_id,
        "total_tokens": len(full_ids),
        "tool_observation_tokens_in_loss": 0,
        "historical_assistant_tokens_in_loss": 0,
    }


def encode_terminal(tokenizer: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    if sample.get("replay_valid") is not True or sample.get("final_state_match") is not True:
        raise ValueError("terminal replay gate failed")
    if sample.get("quality_status") != "structurally_valid_unverified_semantics":
        raise ValueError("terminal quality gate failed")
    return encode_assistant_target(
        tokenizer,
        task_id=sample["task_id"],
        state_identity=sample["gold_trajectory_id"] + ":terminal",
        target_type="terminal_final",
        conversation_prefix=sample["conversation_prefix"],
        assistant_target=sample["assistant_target"],
        tool_schema=sample["tool_schema"],
    )


def encode_terminal_stop(tokenizer: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    if sample.get("replay_valid") is not True or sample.get("final_state_match") is not True:
        raise ValueError("terminal-stop replay gate failed")
    prefix = sample.get("conversation_prefix")
    if not isinstance(prefix, list) or not prefix or prefix[-1].get("role") != "tool":
        raise ValueError("terminal-stop requires final gold tool observation")
    return encode_assistant_target(
        tokenizer,
        task_id=sample["task_id"], state_identity=sample["state_identity"],
        target_type="terminal_stop", conversation_prefix=prefix,
        assistant_target={"role": "assistant", "content": ""},
        tool_schema=sample["tool_schema"],
    )


def encode_first_divergence(tokenizer: Any, sample: Mapping[str, Any], rollout: Mapping[str, Any]) -> dict[str, Any]:
    if sample.get("independent_prefix_replay_valid") is not True or sample["task_id"] != rollout.get("task_id"):
        raise ValueError("first-divergence replay gate failed")
    action = sample["gold_action"]
    if not isinstance(action.get("arguments"), dict):
        raise ValueError("gold tool arguments not typed")
    return encode_assistant_target(
        tokenizer,
        task_id=sample["task_id"],
        state_identity=sample["state_identity"],
        target_type="first_divergence_tool",
        conversation_prefix=sample["conversation_prefix"],
        assistant_target={
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": action["name"], "arguments": action["arguments"]},
            }],
        },
        tool_schema=rollout["tools"],
    )

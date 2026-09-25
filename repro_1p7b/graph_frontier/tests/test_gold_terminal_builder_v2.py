"""CPU gates for replay-locked gold-terminal construction and shared CE masking."""
from __future__ import annotations

import json

import pytest

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import (
    encode_first_divergence, encode_terminal,
)
from repro_1p7b.graph_frontier.gold_terminal_builder import (
    build_one, terminal_messages, typed_trace_matches, validate_terminal_steps,
)


class TinyTemplate:
    def apply_chat_template(self, messages, *, tools, tokenize, enable_thinking, add_generation_prompt):
        assert tokenize is True and enable_thinking is False
        result = "TOOLS:" + json.dumps(tools, sort_keys=True) + "\n"
        for message in messages:
            result += message["role"] + ":"
            if message.get("tool_calls"):
                result += "<tool_call>" + json.dumps(message["tool_calls"], sort_keys=True) + "</tool_call>"
            else:
                result += message.get("content", "")
            result += "\n"
        if add_generation_prompt:
            result += "assistant:"
        else:
            # The terminal target is the last assistant message; its role prefix
            # must equal the generation prompt suffix.
            last = messages[-1]
            if last["role"] == "assistant":
                before = messages[:-1]
                result = "TOOLS:" + json.dumps(tools, sort_keys=True) + "\n"
                for message in before:
                    result += message["role"] + ":"
                    if message.get("tool_calls"):
                        result += "<tool_call>" + json.dumps(message["tool_calls"], sort_keys=True) + "</tool_call>"
                    else:
                        result += message.get("content", "")
                    result += "\n"
                result += "assistant:"
                if last.get("tool_calls"):
                    result += "<tool_call>" + json.dumps(last["tool_calls"], sort_keys=True) + "</tool_call>"
                else:
                    result += last.get("content", "")
                result += "<END>"
        return [ord(c) for c in result]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(chr(i) for i in ids)


@pytest.fixture
def case():
    sidecar = {"task_id": "task-1", "dependency_depth": 1, "environment_identifiers": ["TestServer"]}
    actions = [
        {"name": "TestServer-A", "arguments": {"x": 1}, "masked_arguments": []},
        {"name": "TestServer-B", "arguments": {"y": 2}, "masked_arguments": []},
    ]
    responses = ['{"v":2}', '{"ok":true}']
    steps = [{"role": "user", "content": "please do A then B"}]
    for action, response in zip(actions, responses):
        steps.extend([
            {"role": "tool_call", "content": [action]},
            {"role": "tool_response", "content": [response]},
        ])
    steps.append({"role": "assistant", "content": "Done."})
    row = {
        "task_id": "task-1",
        "prompt": json.dumps([{"role": "user", "content": "please do A then B"}]),
        "extra_info": {"graph_frontier": sidecar, "generation_seed": 7},
    }
    trace = {
        "task_id": "task-1",
        "events": [
            {"tool_name": a["name"], "tool_arguments": a["arguments"],
             "tool_response": json.loads(r), "execution_success": True}
            for a, r in zip(actions, responses)
        ],
    }
    reference = {
        "gold_trajectory_id": "gold-1", "actions": actions,
        "responses": responses, "final_text": "Done.",
    }
    tools = [
        {"type": "function", "function": {"name": a["name"], "parameters": {"type": "object"}}}
        for a in actions
    ]
    return row, sidecar, {"replay_ok": True, "static_pass": True, "final_state_match": True}, {"nodes": [{"steps": steps}]}, trace, reference, tools


def build(case):
    row, sidecar, ledger, raw, trace, reference, tools = case
    return build_one(row, sidecar, ledger, raw, trace, reference, tools, "fixture", {})


def test_final_tool_and_observation(case):
    _, _, _, raw, _, reference, _ = case
    call, response, final = validate_terminal_steps(raw["nodes"][0]["steps"], reference)
    assert call["name"] == "TestServer-B"
    assert json.loads(response) == {"ok": True}
    assert final == "Done."


def test_terminal_context_complete(case):
    sample, reason = build(case)
    assert reason == "ok"
    messages = sample["conversation_prefix"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant", "tool"]
    assert messages[-1]["content"] == '{"ok":true}'
    assert messages[1]["tool_calls"][0]["function"]["name"] == "TestServer-A"
    assert messages[3]["tool_calls"][0]["function"]["name"] == "TestServer-B"


def test_real_final_target_without_fake_stop(case):
    sample, _ = build(case)
    assert sample["assistant_target"] == {"role": "assistant", "content": "Done."}
    assert sample["target_type"] == "terminal_final"
    assert "STOP" not in json.dumps(sample)


def test_missing_final_skips(case):
    case[3]["nodes"][0]["steps"][-1]["content"] = ""
    assert build(case)[1] == "terminal_unreadable"


def test_missing_schema_skips(case):
    case[-1].clear()
    assert build(case)[1] == "schema_unrecoverable"


def test_replay_invalid_skips(case):
    case[2]["replay_ok"] = False
    assert build(case)[1] == "replay_invalid"


def test_final_state_mismatch_skips(case):
    case[2]["final_state_match"] = False
    assert build(case)[1] == "final_state_mismatch"


def test_observation_drift_skips(case):
    case[4]["events"][-1]["tool_response"] = {"ok": False}
    assert not typed_trace_matches(case[5], case[4])
    assert build(case)[1] == "raw_replay_drift"


def test_placeholder_and_tool_serialization_skips(case):
    case[3]["nodes"][0]["steps"][-1]["content"] = "TODO <tool_call>"
    case[5]["final_text"] = "TODO <tool_call>"
    assert build(case)[1].startswith("quality_")


def test_terminal_ce_masks_all_history_and_tool_observations(case):
    sample, _ = build(case)
    encoded = encode_terminal(TinyTemplate(), sample)
    p = encoded["prompt_tokens"]
    assert encoded["labels"][:p] == [-100] * p
    assert encoded["labels"][p:] == encoded["input_ids"][p:]
    assert encoded["tool_observation_tokens_in_loss"] == 0
    assert encoded["historical_assistant_tokens_in_loss"] == 0
    assert encoded["target_tokens"] > 0
    assert "TestServer-A" in TinyTemplate().decode(encoded["input_ids"][:p], skip_special_tokens=False)


def test_template_receives_schema_and_no_stop(case):
    sample, _ = build(case)
    encoded = encode_terminal(TinyTemplate(), sample)
    text = TinyTemplate().decode(encoded["input_ids"], skip_special_tokens=False)
    assert "TestServer-A" in text and "TestServer-B" in text
    assert text.endswith("assistant:Done.<END>")
    assert "<STOP>" not in text


def test_shared_encoder_supports_fd_and_terminal(case):
    sample, _ = build(case)
    tokenizer = TinyTemplate()
    terminal = encode_terminal(tokenizer, sample)
    fd = {
        "task_id": "task-1", "state_identity": "fd-1",
        "independent_prefix_replay_valid": True,
        "gold_action": {"name": "TestServer-A", "arguments": {"x": 1}},
        "conversation_prefix": [{"role": "user", "content": "please do A then B"}],
    }
    correction = encode_first_divergence(tokenizer, fd, {"task_id": "task-1", "tools": case[-1]})
    assert terminal["schema_version"] == correction["schema_version"]
    assert terminal["target_type"] == "terminal_final"
    assert correction["target_type"] == "first_divergence_tool"
    assert correction["labels"][:correction["prompt_tokens"]] == [-100] * correction["prompt_tokens"]


def test_tool_error_observation_is_not_terminal_label(case):
    case[4]["events"][-1]["tool_response"] = {"error": "failure"}
    case[5]["responses"][-1] = '{"error":"failure"}'
    case[3]["nodes"][0]["steps"][-2]["content"] = ['{"error":"failure"}']
    assert build(case)[1] == "quality_tool_error_observed"

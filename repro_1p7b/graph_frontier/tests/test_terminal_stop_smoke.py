"""Regression checks for the isolated EOS-only terminal-stop sample type."""
from __future__ import annotations

import copy

import pytest

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import (
    encode_assistant_target, encode_terminal_stop,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


class FakeTokenizer:
    eos_token_id = 151645
    end = 151645

    def apply_chat_template(self, messages, *, add_generation_prompt, **_):
        prompt = [10, 11, 12]
        if add_generation_prompt:
            return prompt
        if messages[-1].get("tool_calls"):
            return prompt + [700, self.end, 198]
        return prompt + [self.end, 198]

    def decode(self, ids, **_):
        return "".join("<tool_call>" if i == 700 else "\n" if i == 198
                       else "<|im_end|>" if i == 151645 else "x" for i in ids)


def sample():
    return {
        "task_id": "verified-1", "state_identity": "verified-1:stop",
        "replay_valid": True, "final_state_match": True,
        "conversation_prefix": [
            {"role": "user", "content": "do task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "tool_call_id": "c", "name": "server-tool", "content": "{}"},
        ],
        "tool_schema": [{"type": "function", "function": {"name": "server-tool"}}],
    }


def test_terminal_stop_eos_only_and_prompt_history_masked():
    row = encode_terminal_stop(FakeTokenizer(), sample())
    validate_mask(row)
    assert row["target_type"] == "terminal_stop"
    assert row["labels"] == [-100, -100, -100, 151645, -100]
    assert row["input_ids"] == [10, 11, 12, 151645, 198]
    assert row["target_tokens"] == 1


def test_terminal_stop_trailing_newline_is_not_supervised():
    row = encode_terminal_stop(FakeTokenizer(), sample())
    assert row["input_ids"][-1] == 198
    assert row["labels"][-1] == -100


def test_terminal_stop_rejects_missing_end_token():
    tok = FakeTokenizer()
    tok.end = 42
    with pytest.raises(ValueError, match="assistant-end"):
        encode_terminal_stop(tok, sample())


def test_terminal_stop_rejects_second_active_label():
    row = encode_terminal_stop(FakeTokenizer(), sample())
    row["labels"][-1] = 198
    with pytest.raises(RuntimeError, match="EOS-only"):
        validate_mask(row)


def test_terminal_stop_rejects_unverified_or_nonterminal_history():
    invalid = sample()
    invalid["replay_valid"] = False
    with pytest.raises(ValueError, match="replay gate"):
        encode_terminal_stop(FakeTokenizer(), invalid)
    invalid = sample()
    invalid["conversation_prefix"] = invalid["conversation_prefix"][:-1]
    with pytest.raises(ValueError, match="final gold tool observation"):
        encode_terminal_stop(FakeTokenizer(), invalid)


def test_first_divergence_mask_is_unchanged():
    target = {"role": "assistant", "content": "",
              "tool_calls": [{"id": "c", "type": "function",
                              "function": {"name": "server-tool", "arguments": {}}}]}
    row = encode_assistant_target(
        FakeTokenizer(), task_id="verified-1", state_identity="fd-1",
        target_type="first_divergence_tool",
        conversation_prefix=sample()["conversation_prefix"][:1],
        assistant_target=target, tool_schema=sample()["tool_schema"],
    )
    validate_mask(row)
    assert row["labels"] == [-100, -100, -100, 700, 151645, 198]
    assert row["target_tokens"] == 3

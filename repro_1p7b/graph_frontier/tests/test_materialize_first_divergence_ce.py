"""CPU tests for assistant-only CE materialization."""
from __future__ import annotations

import unittest

from repro_1p7b.graph_frontier.materialize_first_divergence_ce import encode_one


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, enable_thinking,
                            add_generation_prompt):
        prefix = "|".join(str(m.get("content", "")) for m in messages if m["role"] != "assistant")
        if messages[-1]["role"] == "assistant":
            action = messages[-1]["tool_calls"][0]["function"]
            value = prefix + "<assistant><tool_call>" + action["name"] + "</tool_call>"
        else:
            value = prefix + "<assistant>"
        return list(map(ord, value)) if tokenize else value

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(map(chr, tokens))


def sample() -> dict:
    return {
        "task_id": "t1", "state_identity": "state1",
        "independent_prefix_replay_valid": True, "frozen_overlap": False,
        "conversation_prefix": [
            {"role": "user", "content": "question"},
            {"role": "tool", "content": "secret-tool-observation"},
        ],
        "gold_action": {"name": "B", "arguments": {"id": 1}},
        "first_divergence_step": 2, "divergence_type": "wrong_tool", "graph_depth": 1,
    }


class MaterializationTests(unittest.TestCase):
    def test_only_assistant_target_tokens_have_loss(self):
        result = encode_one(FakeTokenizer(), sample(), {"task_id": "t1", "tools": []})
        prompt = result["prompt_tokens"]
        self.assertEqual(result["labels"][:prompt], [-100] * prompt)
        self.assertEqual(result["labels"][prompt:], result["input_ids"][prompt:])
        self.assertNotIn("secret-tool-observation",
                         FakeTokenizer().decode(result["labels"][prompt:]))

    def test_replay_certificate_required(self):
        bad = sample()
        bad["independent_prefix_replay_valid"] = False
        with self.assertRaises(ValueError):
            encode_one(FakeTokenizer(), bad, {"task_id": "t1", "tools": []})

    def test_frozen_overlap_rejected(self):
        bad = sample()
        bad["frozen_overlap"] = True
        with self.assertRaises(ValueError):
            encode_one(FakeTokenizer(), bad, {"task_id": "t1", "tools": []})


if __name__ == "__main__":
    unittest.main()

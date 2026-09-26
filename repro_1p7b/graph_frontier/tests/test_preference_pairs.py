from __future__ import annotations

import unittest
from unittest.mock import patch

from repro_1p7b.graph_frontier.build_preference_pairs import (
    classify_failure,
    eligible_actions,
    standard_candidates,
)
from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action


def _row():
    return {
        "task_id": "task-1",
        "reward_model": {
            "ground_truth": '[{"name":"S-A","arguments":{"name":"alice"},"masked_arguments":[]},'
            '{"name":"S-B","arguments":{"user_id":123},"masked_arguments":["user_id"]}]'
        },
        "extra_info": {
            "graph_frontier": {
                "parameters": [],
                "dependency_edges": [
                    {
                        "edge_id": "a-b",
                        "internal_parameter": True,
                        "required": True,
                        "value_semantics": "unknown",
                        "producer_tool": {"tool_name": "S-A"},
                        "consumer_tool": {"tool_name": "S-B"},
                        "producer_output_parameter": {"parameter_name": "user.id", "data_type": "integer"},
                        "consumer_input_parameter": {"parameter_name": "user_id", "data_type": "integer"},
                    }
                ],
            }
        },
    }


def _producer(success=True):
    return {
        "tool_name": "S-A",
        "tool_arguments": {"name": "alice"},
        "returned_fields": {"user": {"id": 123}},
        "execution_success": success,
        "state_before": {"x": 1},
        "state_after": {"x": 1},
    }


class PreferencePairTest(unittest.TestCase):
    def test_length_truncation_is_not_labeled_as_final_answer(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{
                        "finish_reason": "length",
                        "message": {"content": '<tool_call>{"name":"S-B"'},
                    }],
                    "usage": {"completion_tokens": 384},
                }

        with patch(
            "repro_1p7b.graph_frontier.collect_preference_rollouts.requests.post",
            return_value=Response(),
        ) as post:
            _, action, _, finish_reason = generate_action(
                "http://unused/v1", "model", [], [],
                seed=1, temperature=0.7, max_tokens=384,
            )
        self.assertEqual(finish_reason, "length")
        self.assertEqual(action["kind"], "invalid")
        self.assertEqual(action["reason"], "generation_length_truncated")
        request_body = post.call_args.kwargs["json"]
        self.assertEqual(request_body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("extra_body", request_body)

    def test_unique_executable_consumer_uses_typed_producer_value(self):
        actions, reasons = eligible_actions(_row(), [_producer()])
        self.assertFalse(reasons)
        self.assertEqual(list(actions), ["S-B"])
        self.assertEqual(actions["S-B"]["arguments"], {"user_id": 123})

    def test_producer_must_have_succeeded(self):
        actions, reasons = eligible_actions(_row(), [_producer(False)])
        self.assertFalse(actions)
        self.assertEqual(reasons["producer_not_successful"], 1)

    def test_failure_types_are_high_confidence(self):
        chosen = {"name": "S-B", "arguments": {"user_id": 123}, "provenance": [{"consumer_input_parameter": "user_id", "typed_value": 123}]}
        self.assertEqual(classify_failure({"kind": "final", "content": "done"}, chosen, [_producer()], {}), "premature_stop")
        self.assertEqual(classify_failure({"kind": "tool", "name": "S-C", "arguments": {}}, chosen, [_producer()], {}), "wrong_tool")
        self.assertEqual(classify_failure({"kind": "tool", "name": "S-B", "arguments": {"user_id": 999}}, chosen, [_producer()], {}), "wrong_value")
        self.assertIsNone(classify_failure({"kind": "tool", "name": "S-B", "arguments": {"user_id": 123}}, chosen, [_producer()], {}))
        self.assertIsNone(
            classify_failure(
                {"kind": "tool", "name": "S-C", "arguments": {}},
                chosen,
                [_producer()],
                {},
                {"S-C"},
            )
        )

    def test_useless_retry_requires_failed_identical_call_and_no_state_change(self):
        previous = _producer(False)
        chosen = {"name": "S-B", "arguments": {"user_id": 123}, "provenance": []}
        rejected = {"kind": "tool", "name": "S-A", "arguments": {"name": "alice"}}
        self.assertEqual(classify_failure(rejected, chosen, [previous], {"x": 1}), "useless_retry")

    def test_standard_pair_uses_strict_official_ordering(self):
        base = {
            "task_id": "task-1",
            "source_seed": 1,
            "initial_prompt": [{"role": "user", "content": "q"}],
            "tools": [],
        }
        low = base | {"rollout_index": 0, "semantic_success": False, "official_reward": 0.2, "conversation_messages": base["initial_prompt"] + [{"role": "assistant", "content": "bad"}]}
        high = base | {"rollout_index": 1, "semantic_success": True, "official_reward": 0.1, "conversation_messages": base["initial_prompt"] + [{"role": "assistant", "content": "good"}]}
        pairs = standard_candidates([low, high])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"][0]["content"], "good")
        self.assertEqual(pairs[0]["rejected"][0]["content"], "bad")


if __name__ == "__main__":
    unittest.main()


import unittest

from repro_1p7b.graph_frontier.guided_rl_reward import terminal_reward


def event(index, name, args, fields, success=True, before=None, after=None):
    item = {
        "step_index": index,
        "tool_name": name,
        "tool_arguments": args,
        "returned_fields": fields,
        "execution_success": success,
    }
    if before is not None:
        item["state_before"] = before
    if after is not None:
        item["state_after"] = after
    return item


SIDECAR = {
    "probe_eligibility": {"eligible": True, "reasons": []},
    "dependency_edges": [
        {
            "edge_id": "A->B:user_id",
            "internal_parameter": True,
            "producer_tool": {"tool_name": "A"},
            "producer_output_parameter": {"parameter_name": "user_id", "data_type": "integer"},
            "consumer_tool": {"tool_name": "B"},
            "consumer_input_parameter": {"parameter_name": "user_id", "data_type": "integer"},
            "value_semantics": "unknown",
        }
    ]
}
GT = [
    {"name": "A", "arguments": {}},
    {"name": "B", "arguments": {"user_id": 123}},
]


class GuidedRLRewardTest(unittest.TestCase):
    def score(self, mode, events, malformed=0):
        return terminal_reward(
            mode=mode,
            events=events,
            ground_truth=GT,
            final_state={},
            expected_state={},
            sidecar=SIDECAR,
            malformed_call_count=malformed,
            scheduler={"mode": "fix", "default_ratio": 1.0},
        )

    def test_standard_is_official_only(self):
        events = [event(0, "A", {}, {"user_id": 123}), event(1, "B", {"user_id": 123}, {})]
        result = self.score("official", events)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["graph_frontier"]["consumer_completion_fraction"], 1.0)

    def test_four_sanity_trajectories(self):
        correct = [event(0, "A", {}, {"user_id": 123}), event(1, "B", {"user_id": 123}, {})]
        missing_consumer = [event(0, "A", {}, {"user_id": 123})]
        wrong_value = [event(0, "A", {}, {"user_id": 123}), event(1, "B", {"user_id": 999}, {})]
        retry = [
            event(0, "A", {}, {}, False, {"x": 0}, {"x": 0}),
            event(1, "A", {}, {}, False, {"x": 0}, {"x": 0}),
        ]
        a = self.score("graph_frontier", correct)
        b = self.score("graph_frontier", missing_consumer)
        c = self.score("graph_frontier", wrong_value)
        d = self.score("graph_frontier", retry)
        self.assertGreater(a["score"], b["score"])
        self.assertGreater(a["score"], c["score"])
        self.assertGreater(a["score"], d["score"])
        self.assertEqual(b["graph_frontier"]["consumer_opportunities"], 1)
        self.assertEqual(b["graph_frontier"]["completed_consumer_opportunities"], 0)
        self.assertEqual(c["graph_frontier"]["propagation_fraction"], 0.0)
        self.assertEqual(c["graph_frontier"]["consumer_completion_fraction"], 1.0)
        self.assertEqual(d["graph_frontier"]["invalid_retry_count"], 1)

    def test_modified_retry_is_not_penalized(self):
        events = [
            event(0, "B", {"user_id": 1}, {}, False, {"x": 0}, {"x": 0}),
            event(1, "B", {"user_id": 2}, {}, True, {"x": 0}, {"x": 1}),
        ]
        result = self.score("graph_frontier", events)
        self.assertEqual(result["graph_frontier"]["invalid_retry_count"], 0)

    def test_external_or_unknown_edges_are_not_rewarded(self):
        events = [event(0, "A", {}, {"user_id": 123}), event(1, "B", {"user_id": 123}, {})]
        for marker in (False, None):
            sidecar = {
                "probe_eligibility": {"eligible": True, "reasons": []},
                "dependency_edges": [dict(SIDECAR["dependency_edges"][0])],
            }
            if marker is None:
                sidecar["dependency_edges"][0].pop("internal_parameter")
            else:
                sidecar["dependency_edges"][0]["internal_parameter"] = marker
            result = terminal_reward(
                mode="graph_frontier",
                events=events,
                ground_truth=GT,
                final_state={},
                expected_state={},
                sidecar=sidecar,
                scheduler={"mode": "fix", "default_ratio": 1.0},
            )
            self.assertEqual(result["graph_delta"], 0.0)
            self.assertEqual(result["graph_frontier"]["eligible_internal_edges"], 0)
            self.assertEqual(result["graph_frontier"]["skipped_non_internal_edges"], 1)

    def test_task_level_ineligible_sidecar_never_gets_graph_delta(self):
        events = [event(0, "A", {}, {"user_id": 123}), event(1, "B", {"user_id": 123}, {})]
        sidecar = dict(SIDECAR)
        sidecar["probe_eligibility"] = {
            "eligible": False,
            "reasons": ["dependency_semantics_not_selected_reference"],
        }
        result = terminal_reward(
            mode="graph_frontier",
            events=events,
            ground_truth=GT,
            final_state={},
            expected_state={},
            sidecar=sidecar,
            scheduler={"mode": "fix", "default_ratio": 1.0},
        )
        self.assertEqual(result["graph_delta"], 0.0)
        self.assertFalse(result["graph_frontier"]["task_graph_eligible"])
        self.assertEqual(result["graph_frontier"]["eligible_internal_edges"], 0)
        self.assertEqual(result["graph_frontier"]["skipped_task_ineligible_edges"], 1)
        self.assertEqual(
            result["graph_frontier"]["task_graph_skip_reasons"],
            ["dependency_semantics_not_selected_reference"],
        )

    def test_official_penalties(self):
        events = [
            event(0, "A", {}, {"user_id": 123}),
            event(1, "B", {"user_id": 123}, {}),
            event(2, "X", {}, {}),
        ]
        result = self.score("official", events, malformed=1)
        self.assertAlmostEqual(result["score"], 0.9)
        self.assertAlmostEqual(result["official_reward"]["length_penalty"], 0.05)
        self.assertAlmostEqual(result["official_reward"]["format_penalty"], 0.05)


if __name__ == "__main__":
    unittest.main()

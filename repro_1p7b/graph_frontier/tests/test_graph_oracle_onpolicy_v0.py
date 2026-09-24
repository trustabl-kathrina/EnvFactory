"""CPU contract tests for conservative graph-oracle supervision."""
import unittest
from repro_1p7b.graph_frontier.graph_oracle_onpolicy_v0 import (
    assistant_only_labels, frozen_overlap, get_local_supervision,
)

def fixture():
    names = ["A", "B", "C"]
    graph = {
        "dependency_semantics": "selected_reference",
        "structural_diagnosis_eligible": True,
        "cross_turn_dependency": False,
        "dependency_resolution": {"status": "resolved"},
        "gold_tool_sequence": [{"tool_name": n} for n in names],
        "required_tool_nodes": [{"tool_name": n} for n in names],
        "dependency_edges": [
            {"producer_tool": {"tool_name": "A"}, "consumer_tool": {"tool_name": "B"}},
            {"producer_tool": {"tool_name": "B"}, "consumer_tool": {"tool_name": "C"}},
        ],
        "expected_final_state": {"done": True},
    }
    refs = [{"name": n, "arguments": {}} for n in names]
    events = [{"tool_name": n, "tool_arguments": {}, "execution_success": True} for n in names]
    return graph, refs, events

class OracleTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.refs, self.events = fixture()

    def call(self, history=None, **kwargs):
        return get_local_supervision(sidecar=self.graph,
            history=self.events if history is None else history,
            reference_actions=self.refs, **kwargs)

    def test_terminal_ready_stop(self):
        self.assertEqual(self.call(current_environment_state={"done": True})["target_action"], "STOP")

    def test_unique_consumer(self):
        out = self.call(self.events[:1], certified_enabled_actions=["B"])
        self.assertEqual((out["tier"], out["target_action"]), ("A", "B"))

    def test_premature_stop_not_labeled_stop(self):
        out = self.call(self.events[:1], current_verifier_success=False,
                        certified_enabled_actions=["B"])
        self.assertEqual(out["target_action"], "B")
        self.assertIs(out["terminal_ready"], False)

    def test_producer_completed_consumer_missing(self):
        out = self.call(self.events[:1], certified_enabled_actions=["B"])
        self.assertEqual(out["completed_required_nodes"], ["A"])
        self.assertIn("B", out["remaining_required_nodes"])

    def test_duplicate_redundant_skipped(self):
        out = self.call([self.events[0], self.events[0]])
        self.assertEqual((out["tier"], out["target_type"]), ("C", "skip"))

    def test_multi_valid_not_hard_label(self):
        self.graph["dependency_edges"] = []
        out = self.call([], certified_enabled_actions=["A", "B"])
        self.assertEqual(out["tier"], "B")
        self.assertIsNone(out["target_action"])
        self.assertEqual(out["positive_action_set"], ["A", "B"])

    def test_unknown_action_space_skips(self):
        self.assertEqual(self.call([])["reason"], "selected_path_not_complete_action_oracle")

    def test_frozen_overlap(self):
        self.assertTrue(frozen_overlap("x", {"x"}, {"x"}))
        self.assertFalse(frozen_overlap("x", {"x"}, {"y"}))

    def test_replay_invalid_skips(self):
        self.assertEqual(self.call(replay_valid=False)["tier"], "C")

    def test_mask_excludes_tool_observation(self):
        self.assertEqual(assistant_only_labels(3, 2), [-100, -100, -100, 1, 1])

    def test_off_gold_wrong_arguments_skipped(self):
        bad = [{**self.events[0], "tool_arguments": {"wrong": 1}}]
        self.assertEqual(self.call(bad)["tier"], "C")

    def test_masked_reference_skipped(self):
        self.refs[0]["masked_arguments"] = ["id"]
        self.assertEqual(self.call(self.events[:1])["tier"], "C")

    def test_final_state_unknown_skipped(self):
        self.assertEqual(self.call()["tier"], "C")

if __name__ == "__main__":
    unittest.main()

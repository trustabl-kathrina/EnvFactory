"""CPU-only tests for evidence-constrained Rich-v3 differential classifications."""
import unittest

from repro_1p7b.graph_frontier.audit_rich_v3_trajectories import (
    first_pair_divergence,
    match_required,
    normalized_text,
    observed_failure_step,
    primary_failure,
)


def step(kind, name=None, arguments=None, response=None):
    action = {"kind": kind}
    if name:
        action.update({"name": name, "arguments": arguments or {}})
    elif kind == "final":
        action["content"] = "Done."
    value = {"parsed_action": action}
    if kind == "tool":
        value["typed_event"] = {
            "tool_name": name,
            "tool_arguments": arguments or {},
            "tool_response": response,
            "execution_success": True,
        }
    return value


class TrajectoryDifferentialTests(unittest.TestCase):
    def test_official_matching_is_order_insensitive_but_argument_sensitive(self):
        reference = {"actions": [
            {"name": "A", "arguments": {"x": 1}},
            {"name": "B", "arguments": {"y": 2}},
        ]}
        rollout = {"steps": [step("tool", "B", {"y": 2}),
                             step("tool", "A", {"x": 1})]}
        matched = match_required(rollout, reference)
        self.assertEqual(matched["missing_gold_indices"], [])
        self.assertEqual(matched["matched_gold_indices"], {0: 1, 1: 0})
        rollout["steps"][1] = step("tool", "A", {"x": 9})
        self.assertEqual(match_required(rollout, reference)["missing_gold_indices"], [0])

    def test_first_behavior_divergence_is_structured(self):
        base = {"steps": [step("tool", "A", {"x": 1}, {"id": 123}),
                          step("final")]}
        changed = {"steps": [step("tool", "A", {"x": 2}, {"id": 123}),
                             step("final")]}
        result = first_pair_divergence(base, changed)
        self.assertEqual((result["step"], result["field"]), (1, "structured_action"))
        changed["steps"][0] = step("tool", "A", {"x": 1}, {"id": 999})
        result = first_pair_divergence(base, changed)
        self.assertEqual((result["step"], result["field"]), (1, "tool_observation"))

    def test_wrong_argument_has_observed_step(self):
        reference = {"actions": [{"name": "A", "arguments": {"x": 1}}]}
        rollout = {"steps": [step("tool", "A", {"x": 9})],
                   "terminal_info": {"reward_parts": {
                       "official_reward": {"trace_score": 0, "state_score": 0},
                       "graph_frontier": {"edge_details": []}}}}
        matching = match_required(rollout, reference)
        family, failure, _ = primary_failure(rollout, reference, matching)
        self.assertEqual((family, failure), ("TOOL_POLICY", "WRONG_ARGUMENT"))
        self.assertEqual(observed_failure_step(rollout, reference, matching, failure), 1)

    def test_final_before_required_tool_is_policy_failure(self):
        reference = {"actions": [{"name": "A", "arguments": {"x": 1}}]}
        rollout = {"steps": [step("final")],
                   "terminal_info": {"reward_parts": {
                       "official_reward": {"trace_score": 0, "state_score": 1},
                       "graph_frontier": {"edge_details": []}}}}
        matching = match_required(rollout, reference)
        family, failure, _ = primary_failure(rollout, reference, matching)
        self.assertEqual((family, failure), ("TOOL_POLICY", "PREMATURE_STOP"))
        self.assertEqual(observed_failure_step(rollout, reference, matching, failure), 1)

    def test_terminal_target_normalization(self):
        self.assertEqual(normalized_text(" The  requested\noperations "),
                         normalized_text("the requested operations"))


if __name__ == "__main__":
    unittest.main()

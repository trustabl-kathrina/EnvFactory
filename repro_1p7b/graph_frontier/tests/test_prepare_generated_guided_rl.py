import unittest

from repro_1p7b.graph_frontier.prepare_generated_guided_rl import (
    materialize_sampling_schedule,
    structural,
)


def row(task_id, targeted):
    return {
        "prompt": [],
        "extra_info": {
            "task_id": task_id,
            "frontier_targeted": targeted,
        },
    }


class MaterializedSamplingScheduleTest(unittest.TestCase):
    def test_ineligible_internal_edge_stays_in_broad_pool(self):
        sidecar = {
            "probe_eligibility": {
                "eligible": False,
                "reasons": ["dependency_semantics_not_selected_reference"],
            },
            "dependency_depth": 2,
            "gold_tool_sequence": ["A", "B"],
            "dependency_edges": [
                {"internal_parameter": True, "consumer_tool": {"tool_name": "B"}}
            ],
        }
        metadata = structural(sidecar)
        self.assertFalse(metadata["graph_reward_eligible"])
        self.assertFalse(metadata["frontier_targeted"])
        self.assertEqual(metadata["sampling_weight"], 0.35)

    def test_explicitly_eligible_internal_edge_is_targeted(self):
        sidecar = {
            "probe_eligibility": {"eligible": True, "reasons": []},
            "dependency_depth": 2,
            "gold_tool_sequence": ["A", "B"],
            "dependency_edges": [
                {"internal_parameter": True, "consumer_tool": {"tool_name": "B"}}
            ],
        }
        metadata = structural(sidecar)
        self.assertTrue(metadata["graph_reward_eligible"])
        self.assertTrue(metadata["frontier_targeted"])
        self.assertEqual(metadata["sampling_weight"], 1.65)

    def test_exact_65_35_and_deterministic_order(self):
        rows = [row("t0", True), row("t1", True), row("b0", False)]
        first, audit = materialize_sampling_schedule(rows, total_samples=20, seed=7)
        second, second_audit = materialize_sampling_schedule(rows, total_samples=20, seed=7)
        self.assertEqual([x["extra_info"]["task_id"] for x in first],
                         [x["extra_info"]["task_id"] for x in second])
        self.assertEqual(audit, second_audit)
        self.assertEqual(audit["targeted_prompt_samples"], 13)
        self.assertEqual(audit["broad_prompt_samples"], 7)
        self.assertEqual(audit["targeted_fraction"], 0.65)
        self.assertEqual([x["extra_info"]["schedule_index"] for x in first], list(range(20)))

    def test_requires_both_exploration_buckets(self):
        with self.assertRaisesRegex(RuntimeError, "both frontier-targeted and broad"):
            materialize_sampling_schedule([row("t0", True)], total_samples=4)


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from repro_1p7b.graph_frontier.guided_rl_generate import (
    contamination_audit,
    enrich_official_rows,
    normalized_query,
)


class GuidedRLGenerateTest(unittest.TestCase):
    def test_normalized_query(self):
        self.assertEqual(normalized_query("  Create—Event!  "), "create—event!")

    def test_enrich_and_contamination_gate(self):
        initial = {"Calendar": {"events": {}}}
        row = {
            "prompt": json.dumps([{"role": "user", "content": "Create a new event."}]),
            "reward_model": {"ground_truth": "[]"},
            "extra_info": {
                "mcp_factory_kwargs": {"initial_config": json.dumps(initial)}
            },
        }
        sidecar = {
            "task_id": "generated-1",
            "seed": 7,
            "query": {"text": "Create a new event."},
            "initial_scenario": initial,
            "gold_tool_sequence": [{"tool_name": "Calendar-create_event"}],
            "dependency_edges": [],
        }
        enriched = enrich_official_rows([row], [sidecar])
        self.assertEqual(enriched[0]["source"], "EnvFactory-generated-guided-rl-v1")
        self.assertEqual(enriched[0]["extra_info"]["graph_frontier"]["seed"], 7)
        kept, report = contamination_audit(
            enriched,
            [{"task_id": "frozen-1", "query": "Different query", "gold_tools": []}],
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(report["frozen_300_exact_overlap"], 0)
        kept, report = contamination_audit(
            enriched,
            [{"task_id": "frozen-1", "query": "  create   A new event. ", "gold_tools": []}],
        )
        self.assertEqual(kept, [])
        self.assertEqual(report["removed_exact_count"], 1)


if __name__ == "__main__":
    unittest.main()

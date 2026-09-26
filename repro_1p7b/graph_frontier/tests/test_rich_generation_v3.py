from collections import Counter
from types import SimpleNamespace
import unittest

from repro_1p7b.graph_frontier.rich_generation_v3 import (
    bind_plan_arguments,
    profiler_unambiguous_path,
    select_balanced,
    selected_tool_sequence,
    trace_tool_execution_success,
    trace_follows_plan_dataflow,
    trace_leaks_plan_values,
    tool_sequence_from_steps,
    json_container,
)
from repro_1p7b.graph_frontier.replay_generated_guided_rl import (
    bind_runtime_dependency_arguments,
)
from repro_1p7b.graph_frontier.rich_generation_v3_validate import (
    compatible_sidecar,
    scalar_tokens,
    token_visible,
)


def _edge(edge_id, producer, consumer, environment="Demo"):
    return {
        "edge_id": edge_id,
        "producer_environment": environment,
        "consumer_environment": environment,
        "producer_tool": producer,
        "consumer_tool": consumer,
    }


class RichGenerationV3Tests(unittest.TestCase):
    def test_balanced_selection_enforces_shared_inventory_edge_cap(self):
        paths = [
            [_edge("a", "A", "B")],
            [_edge("a", "A", "B"), _edge("b", "B", "C")],
            [_edge("c", "D", "E")],
        ]
        counts = Counter()
        selected = select_balanced(paths, 3, seed=7, edge_counts=counts, max_edge_reuse=1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(max(counts.values()), 1)
        self.assertIn(set(counts), ({"a", "c"}, {"a", "b", "c"}))

    def test_profiler_unambiguous_path_rejects_flattened_collection_field(self):
        direct = [_edge("a", "A", "B") | {"producer_output_field": "id"}]
        flattened = [_edge("b", "A", "B") | {"producer_output_field": "items.id"}]
        self.assertTrue(profiler_unambiguous_path(direct))
        self.assertFalse(profiler_unambiguous_path(flattened))

    def test_selected_tool_sequence_uses_structured_calls_only(self):
        node = SimpleNamespace(steps=[
            {"role": "user", "content": "hello"},
            {"role": "tool_call", "content": [{"name": "Demo-A"}, {"name": "Demo-B"}]},
            {"role": "tool_response", "content": []},
        ])
        self.assertEqual(selected_tool_sequence(node), ["Demo-A", "Demo-B"])
        self.assertEqual(tool_sequence_from_steps(node.steps), ["Demo-A", "Demo-B"])

    def test_trace_execution_gate_requires_paired_failure_free_responses(self):
        success = [
            {"role": "tool_call", "content": [{"name": "Demo-A"}]},
            {"role": "tool_response", "content": ["{\"id\": 123}"]},
        ]
        failed = [
            {"role": "tool_call", "content": [{"name": "Demo-A"}]},
            {
                "role": "tool_response",
                "content": ["Demo-A failed: Error executing tool A: missing entity"],
            },
        ]
        missing = [{"role": "tool_call", "content": [{"name": "Demo-A"}]}]
        self.assertTrue(trace_tool_execution_success(success))
        self.assertFalse(trace_tool_execution_success(failed))
        self.assertFalse(trace_tool_execution_success(missing))

    def test_trace_dataflow_gate_accepts_member_and_rejects_unrelated_value(self):
        plan = {"edge_path": [{
            "producer_tool": "Demo-list",
            "producer_output_field": "items.id",
            "consumer_tool": "Demo-get",
            "consumer_argument": "item_id",
        }]}
        steps = [
            {"role": "tool_call", "content": [{"name": "Demo-list", "arguments": {}}]},
            {"role": "tool_response", "content": ['{"items":[{"id":"a"},{"id":"b"}]}']},
            {"role": "tool_call", "content": [{"name": "Demo-get", "arguments": {"item_id": "b"}}]},
            {"role": "tool_response", "content": ['{"id":"b"}']},
        ]
        self.assertTrue(trace_follows_plan_dataflow(steps, plan))
        steps[2]["content"][0]["arguments"]["item_id"] = "not-returned"
        self.assertFalse(trace_follows_plan_dataflow(steps, plan))

    def test_generation_rejects_visible_internal_flow_value(self):
        plan = {"edge_path": [{
            "producer_tool": "Demo-login",
            "producer_output_field": "user_id",
            "consumer_tool": "Demo-stats",
            "consumer_argument": "user_id",
        }]}
        steps = [
            {"role": "tool_call", "content": [{"name": "Demo-login", "arguments": {}}]},
            {"role": "tool_response", "content": ['{"user_id":"UID001"}']},
            {"role": "tool_call", "content": [{"name": "Demo-stats", "arguments": {"user_id": "UID001"}}]},
            {"role": "tool_response", "content": ['{"ok":true}']},
        ]
        self.assertTrue(trace_leaks_plan_values(steps, plan, "Show stats for UID001."))
        self.assertFalse(trace_leaks_plan_values(steps, plan, "Log in and show my stats."))
        self.assertFalse(trace_leaks_plan_values(steps, plan, "Show stats for XUID001Y."))

    def test_replay_rebinds_unique_nondeterministic_internal_value(self):
        sidecar = {"dependency_edges": [{
            "producer_tool": {"tool_name": "Demo-create"},
            "producer_output_parameter": {"parameter_name": "item.id"},
            "consumer_tool": {"tool_name": "Demo-get"},
            "consumer_input_parameter": {"parameter_name": "item_id"},
        }]}
        arguments, rebindings = bind_runtime_dependency_arguments(
            "Demo-get",
            {"item_id": "old-uuid", "keep": 1},
            sidecar,
            {"Demo-create": {"item": {"id": "new-uuid"}}},
        )
        self.assertEqual(arguments, {"item_id": "new-uuid", "keep": 1})
        self.assertEqual(len(rebindings), 1)

    def test_generation_rebinds_unique_value_but_rejects_ambiguous_mismatch(self):
        plan = {"edge_path": [{
            "producer_tool": "Demo-create",
            "producer_output_field": "item.id",
            "consumer_tool": "Demo-get",
            "consumer_argument": "item_id",
        }]}
        rebound = bind_plan_arguments(
            "Demo-get",
            {"item_id": "old-uuid"},
            plan,
            {"Demo-create": {"item": {"id": "new-uuid"}}},
        )
        self.assertEqual(rebound[0]["item_id"], "new-uuid")
        ambiguous = bind_plan_arguments(
            "Demo-get",
            {"item_id": "not-returned"},
            plan,
            {"Demo-create": {"item": [{"id": "a"}, {"id": "b"}]}},
        )
        self.assertIsNone(ambiguous)

    def test_internal_value_leakage_uses_token_boundaries(self):
        self.assertEqual(scalar_tokens([456]), ["456"])
        self.assertTrue(token_visible("456", "Please use order_id=456."))
        self.assertFalse(token_visible("456", "Please use order_id=14567."))

    def test_json_container_accepts_fenced_or_trailing_json(self):
        self.assertEqual(json_container("```json\n{\"x\": 1}\n```"), {"x": 1})
        self.assertEqual(json_container("schema:\n{\"x\": 1}\nextra"), {"x": 1})

    def test_legacy_rich_schema_is_normalized_for_profiler_compatibility(self):
        sidecar = compatible_sidecar({
            "schema_version": "envfactory_gold_sidecar_graph_frontier_rich_v1",
            "task_id": "x",
        })
        self.assertEqual(sidecar["schema_version"], "envfactory_gold_sidecar_v1")
        self.assertEqual(sidecar["rich_extension_schema_version"], "graph_frontier_rich_v1")


if __name__ == "__main__":
    unittest.main()

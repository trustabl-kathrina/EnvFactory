"""CPU-only regression checks for fixed-budget Cycle-3 pilot helpers."""
import unittest

from repro_1p7b.graph_frontier.cycle3_pilot.audit_rich24 import (
    bootstrap, mcnemar_exact,
)
from repro_1p7b.graph_frontier.cycle3_pilot.prepare import select_fd
from repro_1p7b.graph_frontier.cycle3_pilot.probe import parse_action


class Cycle3PilotTests(unittest.TestCase):
    def test_fd_sampling_is_deterministic_and_covers_buckets(self):
        rows = ([{"task_id": f"w{i}", "failure_type": "WRONG_ARGUMENT", "gold_step": 1}
                 for i in range(45)]
                + [{"task_id": f"d{i}", "failure_type": "WRONG_TOOL", "gold_step": 2}
                   for i in range(30)]
                + [{"task_id": f"o{i}", "failure_type": "WRONG_TOOL", "gold_step": 1}
                   for i in range(30)])
        selected = select_fd(rows)
        self.assertEqual(selected, select_fd(rows))
        self.assertEqual(len({r["task_id"] for r in selected}), 64)
        self.assertGreaterEqual(sum(r["failure_type"] == "WRONG_ARGUMENT" for r in selected), 32)
        self.assertGreaterEqual(sum(r["gold_step"] >= 2 for r in selected), 16)

    def test_tool_parser_handles_nested_arguments_and_truncation(self):
        text = '<tool_call>\n{"name":"A","arguments":{"x":{"id":123}}}\n</tool_call><|im_end|>'
        self.assertEqual(parse_action(text, True),
                         {"kind": "tool", "name": "A", "arguments": {"x": {"id": 123}}})
        self.assertEqual(parse_action("<tool_call>{", False), {"kind": "incomplete_tool"})
        self.assertEqual(parse_action("Done.<|im_end|>", True), {"kind": "no_tool_termination"})

    def test_paired_uncertainty_is_deterministic(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(4, 1), 0.375)
        self.assertEqual(bootstrap([0.0] * 24), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

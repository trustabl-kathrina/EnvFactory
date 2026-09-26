"""CPU-only exposure and source-lock regression checks for EOS ablations."""
import json
import random
import unittest

from repro_1p7b.graph_frontier.cycle3_pilot.prepare import OUT as DATA
from repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.prepare import (
    OUT, _bit_reverse, _extra,
)
from repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.train import load_plan


class RatioScheduleTests(unittest.TestCase):
    def test_nested_replacement_order(self):
        priority = sorted(range(64), key=_bit_reverse)
        self.assertEqual(len(set(priority)), 64)
        self.assertTrue(set(priority[:21]) < set(priority[:38]))
        self.assertEqual(sorted(priority[:4]), [0, 16, 32, 48])

    def test_stratified_extra_selection_no_duplicates(self):
        rows = ([{"task_id": f"w{i}", "failure_type": "WRONG_ARGUMENT", "gold_step": 1}
                 for i in range(60)]
                + [{"task_id": f"d{i}", "failure_type": "WRONG_TOOL", "gold_step": 2}
                   for i in range(30)]
                + [{"task_id": f"o{i}", "failure_type": "WRONG_TOOL", "gold_step": 1}
                   for i in range(30)])
        used = set()
        rng = random.Random(20260926)
        selected = _extra(rows, used, (11, 5, 5), rng) + _extra(rows, used, (8, 5, 4), rng)
        self.assertEqual(len(selected), 38)
        self.assertEqual(len({r["task_id"] for r in selected}), 38)
        self.assertEqual(sum(r["failure_type"] == "WRONG_ARGUMENT" for r in selected), 19)

    @unittest.skipUnless((OUT / "schedule_2.json").exists(), "remote frozen data unavailable")
    def test_frozen_plan_exposure_and_original_rank0(self):
        old = json.loads((DATA / "sampling_manifest.json").read_text())["formal_schedule"]
        for ratio, fd, stop in ((2, 85, 43), (4, 102, 26)):
            pairs, plan = load_plan(ratio)
            self.assertEqual(len(pairs), 64)
            self.assertEqual(plan["exposures"], {"FD": fd, "terminal_stop": stop})
            self.assertEqual([p[0]["task_id"] for p in pairs], [x["fd_task_id"] for x in old])


if __name__ == "__main__":
    unittest.main()

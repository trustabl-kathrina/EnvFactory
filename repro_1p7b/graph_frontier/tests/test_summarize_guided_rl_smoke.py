import json
import tempfile
import unittest
from pathlib import Path

from repro_1p7b.graph_frontier.summarize_guided_rl_smoke import summarize


class GuidedRLSmokeSummaryTest(unittest.TestCase):
    def test_graph_gate_and_gpu_peak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reward = {
                "score": 1.2,
                "reward_mode": "graph_frontier",
                "graph_frontier": {
                    "propagation_fraction": 1.0,
                    "consumer_completion_fraction": 1.0,
                    "invalid_retry_count": 0,
                },
            }
            other = dict(reward, score=0.0)
            root.joinpath("train.log").write_text(
                "ENVFACTORY_REWARD " + json.dumps(reward) + "\n"
                "ENVFACTORY_REWARD " + json.dumps(other) + "\n"
                "Training Progress: 100%\n"
                "step:1 - actor/grad_norm:1.250 - actor/kl_loss:0.010 - "
                "training/global_step:1.000 - timing_s/update_actor:2.000\n"
            )
            root.joinpath("gpu.csv").write_text(
                "2026-09-17T00:00:00+08:00\n"
                "0, 12000 MiB, 40960 MiB, 80 %\n"
                "1, 13000 MiB, 40960 MiB, 90 %\n"
            )
            result = summarize(root, "graph_frontier")
            self.assertTrue(result["gate"]["pass"])
            self.assertEqual(result["gpu"]["peak_memory_mib"]["1"], 13000)
            self.assertEqual(result["reward"]["count"], 2)


if __name__ == "__main__":
    unittest.main()

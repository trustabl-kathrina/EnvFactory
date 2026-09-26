import argparse
import json
import tempfile
import unittest
from pathlib import Path

from repro_1p7b.graph_frontier.rich_generation_v3 import write_jsonl
from repro_1p7b.graph_frontier.rich_generation_v3_resumable import bootstrap, run


def pair(directory: Path, task_id: str, seed: int) -> None:
    (directory / "raw").mkdir(parents=True, exist_ok=True)
    (directory / "sidecar").mkdir(parents=True, exist_ok=True)
    (directory / "raw" / f"{task_id}.json").write_text(json.dumps({"seed": seed}))
    (directory / "sidecar" / f"{task_id}.gold.json").write_text(
        json.dumps({"task_id": task_id, "generation_seed": seed})
    )


class RichGenerationV3ResumableTests(unittest.TestCase):
    def test_bootstrap_and_recover_interrupted_chunk_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.jsonl"
            graph = root / "graph.pkl"
            source = root / "source"
            output = root / "resumed"
            write_jsonl(plan, [
                {"plan_id": "first", "generation_seed": 11},
                {"plan_id": "second", "generation_seed": 22},
            ])
            graph.write_bytes(b"fixed graph")
            pair(source, "first", 11)
            (source / "status").write_text("SCRIPT_FAILED:143\n")
            common = dict(plan=plan, graph=graph, output_dir=output, chunk_size=2)
            bootstrap(argparse.Namespace(**common, source_run=source))
            self.assertEqual(len(list((output / "raw").glob("*.json"))), 1)

            interrupted = output / "chunks/chunk_0000/attempt_000"
            pair(interrupted, "second", 22)
            run(argparse.Namespace(
                **common, model_name="sglang,sglang1", pass_k=2,
                concurrency=4, attempts=4,
            ))
            report = json.loads((output / "generation_run.json").read_text())
            self.assertEqual(report["requested_plans"], 2)
            self.assertEqual(report["completed_chains"], 2)
            self.assertEqual(report["failed_chains"], 0)
            self.assertEqual(len(list((output / "raw").glob("*.json"))), 2)
            self.assertTrue((output / "chunks/chunk_0000/COMPLETE.json").exists())
            self.assertTrue((source / "raw/first.json").exists())


    def test_balanced_order_pauses_after_one_chunk_per_depth(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.jsonl"
            graph = root / "graph.pkl"
            source = root / "source"
            output = root / "resumed"
            rows = [{"plan_id": f"task{i}", "generation_seed": 100 + i,
                     "target_depth": depth, "bucket": f"depth{depth}"}
                    for i, depth in enumerate((3, 3, 2, 2, 1, 1))]
            write_jsonl(plan, rows)
            graph.write_bytes(b"fixed graph")
            for row in rows:
                pair(source, row["plan_id"], row["generation_seed"])
            (source / "status").write_text("PAUSED\n")
            common = dict(plan=plan, graph=graph, output_dir=output, chunk_size=1)
            bootstrap(argparse.Namespace(**common, source_run=source))
            run(argparse.Namespace(**common, model_name="sglang,sglang1", pass_k=2,
                                   concurrency=4, attempts=4, balanced_order=True,
                                   pause_after_completed_chunks=3))
            completed = sorted(int(path.parent.name[-4:])
                               for path in output.glob("chunks/chunk_*/COMPLETE.json"))
            self.assertEqual(completed, [0, 2, 4])
            self.assertEqual((output / "status").read_text().strip(), "PAUSED_FOR_PILOT")
            self.assertFalse((output / "generation_run.json").exists())


if __name__ == "__main__":
    unittest.main()

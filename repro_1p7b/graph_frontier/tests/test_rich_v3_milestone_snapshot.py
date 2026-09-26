import json
import tempfile
import unittest
from pathlib import Path

from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256, write_jsonl
from repro_1p7b.graph_frontier.rich_v3_milestone_snapshot import freeze


class RichV3MilestoneSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "raw").mkdir()
        (self.source / "sidecar").mkdir()
        self.plan = self.root / "plan.jsonl"
        self.plans = [
            {"plan_id": f"task{i}", "generation_seed": 100 + i, "bucket": "depth1", "target_depth": 1}
            for i in range(6)
        ]
        write_jsonl(self.plan, self.plans)
        (self.source / "resume_manifest.json").write_text(json.dumps({
            "plan_sha256": file_sha256(self.plan), "graph_sha256": "graph-hash", "chunk_size": 2
        }))
        (self.source / "status").write_text("PAUSED_FOR_PILOT\n")
        for index in range(2):
            chunk = self.source / "chunks" / f"chunk_{index:04d}"
            chunk.mkdir(parents=True)
            (chunk / "COMPLETE.json").write_text(json.dumps({
                "plan_ids": [f"task{index * 2}", f"task{index * 2 + 1}"]
            }))
        for index in (0, 1, 3, 4):
            (self.source / "raw" / f"raw{index}.json").write_text(json.dumps({"seed": 100 + index}))
            (self.source / "sidecar" / f"task{index}.gold.json").write_text(
                json.dumps({"task_id": f"task{index}", "generation_seed": 100 + index})
            )
        self.output = self.root / "pilot"

    def test_freezes_only_prefix_pairs_and_preserves_source(self):
        result = freeze(self.source, self.plan, self.output, prefix_plans=4, chunk_size=2)
        self.assertEqual(result["selected_plans"], 4)
        self.assertEqual(result["completed_pairs"], 3)
        self.assertEqual(len(list((self.output / "raw").glob("*.json"))), 3)
        self.assertFalse((self.output / "sidecar/task4.gold.json").exists())
        self.assertEqual(len(list((self.source / "raw").glob("*.json"))), 4)
        self.assertEqual(json.loads((self.output / "generation_run.json").read_text())["failed_chains"], 1)
        with self.assertRaises(FileExistsError):
            freeze(self.source, self.plan, self.output, prefix_plans=4, chunk_size=2)

    def test_completed_chunks_can_be_nonprefix_and_depth_balanced(self):
        (self.source / "chunks/chunk_0001/COMPLETE.json").unlink()
        chunk = self.source / "chunks/chunk_0002"
        chunk.mkdir()
        (chunk / "COMPLETE.json").write_text(json.dumps({
            "plan_ids": ["task4", "task5"]
        }))
        result = freeze(self.source, self.plan, self.output, chunk_size=2,
                        completed_chunks=True, min_completed_chunks=2)
        self.assertEqual(result["selected_plans"], 4)
        self.assertEqual(result["completed_pairs"], 3)
        snapshot = json.loads((self.output / "snapshot_manifest.json").read_text())
        self.assertEqual(snapshot["selected_chunk_indices"], [0, 2])
        self.assertEqual(snapshot["selection_mode"], "completed_chunks")
        self.assertFalse((self.output / "sidecar/task3.gold.json").exists())

    def test_refuses_live_generator_or_incomplete_chunk(self):
        (self.source / "status").write_text("GENERATING\n")
        with self.assertRaises(RuntimeError):
            freeze(self.source, self.plan, self.output, prefix_plans=4, chunk_size=2)
        (self.source / "status").write_text("PAUSED_FOR_PILOT\n")
        (self.source / "chunks/chunk_0001/COMPLETE.json").unlink()
        with self.assertRaises(RuntimeError):
            freeze(self.source, self.plan, self.output, prefix_plans=4, chunk_size=2)


if __name__ == "__main__":
    unittest.main()

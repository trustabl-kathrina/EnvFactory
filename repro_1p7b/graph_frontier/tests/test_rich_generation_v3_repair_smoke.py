import argparse
import json
import tempfile
import unittest
from pathlib import Path

from repro_1p7b.graph_frontier.rich_generation_v3_repair_smoke import assemble


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class RichGenerationV3RepairSmokeTests(unittest.TestCase):
    def test_assemble_copies_only_valid_tasks_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "old"
            new = root / "supplemental"
            output = root / "assembled"
            plan = root / "combined.jsonl"
            plans = [
                {
                    "plan_id": f"task-{depth}-{index}",
                    "target_depth": depth,
                    "generation_seed": depth * 1000 + index,
                }
                for depth in (1, 2, 3)
                for index in range(40)
            ]
            _jsonl(plan, plans)
            old_valid = plans[0]
            new_valid = plans[-1]
            _jsonl(old / "audit/final_ledger.jsonl", [
                {"task_id": item["plan_id"], "status": "FINAL_VALID" if item == old_valid else "INVALID"}
                for item in plans if item != new_valid
            ])
            _jsonl(new / "audit/final_ledger.jsonl", [
                {"task_id": new_valid["plan_id"], "status": "FINAL_VALID"}
            ])
            for run, item in ((old, old_valid), (new, new_valid)):
                (run / "audit_status").write_text("AUDIT_DONE\n")
                (run / "raw").mkdir()
                (run / "sidecar").mkdir()
                (run / "raw" / f"{item['plan_id']}.json").write_text(
                    json.dumps({"seed": item["generation_seed"]})
                )
                (run / "sidecar" / f"{item['plan_id']}.gold.json").write_text(
                    json.dumps({"task_id": item["plan_id"]})
                )
            assemble(argparse.Namespace(
                combined_plan=plan, original_run=old, supplemental_run=new, output_dir=output
            ))
            self.assertEqual((output / "status").read_text().strip(), "GENERATED")
            self.assertEqual(len(list((output / "raw").glob("*.json"))), 2)
            self.assertEqual(len(list((output / "sidecar").glob("*.gold.json"))), 2)
            manifest = json.loads((output / "assembly_manifest.json").read_text())
            self.assertEqual(manifest["retained_copied"], 2)
            self.assertEqual({row["task_id"] for row in manifest["lineage"]}, {
                old_valid["plan_id"], new_valid["plan_id"]
            })
            self.assertTrue((old / "raw" / f"{old_valid['plan_id']}.json").exists())
            self.assertTrue((new / "raw" / f"{new_valid['plan_id']}.json").exists())
            with self.assertRaises(FileExistsError):
                assemble(argparse.Namespace(
                    combined_plan=plan, original_run=old,
                    supplemental_run=new, output_dir=output
                ))


if __name__ == "__main__":
    unittest.main()

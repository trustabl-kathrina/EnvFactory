"""Freeze completed chunks of the resumable Rich-v3 formal plan as an isolated pilot.

This does not modify the 1219-plan source run. It is deliberately usable only
after the generator is paused and the selected 32-plan chunks are complete.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256, read_jsonl, write_jsonl
from repro_1p7b.graph_frontier.rich_generation_v3_resumable import paired_files, plan_maps


SCHEMA = "graph_frontier_rich_v3_milestone_snapshot_v1"


def freeze(source: Path, plan: Path, output: Path, prefix_plans: int = 512, chunk_size: int = 32, completed_chunks: bool = False, min_completed_chunks: int = 16) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite milestone snapshot: {output}")
    if not source.is_dir() or not plan.is_file():
        raise FileNotFoundError("source run or plan missing")
    status = (source / "status").read_text().strip()
    if status == "GENERATING":
        raise RuntimeError("pause the generator before freezing a milestone")
    manifest_path = source / "resume_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    plans, by_seed, by_id = plan_maps(plan)
    if manifest["plan_sha256"] != file_sha256(plan) or manifest["chunk_size"] != chunk_size:
        raise ValueError("source immutable plan/chunk hash mismatch")
    if completed_chunks:
        indices = sorted(int(path.parent.name.split("_")[-1])
                         for path in (source / "chunks").glob("chunk_*/COMPLETE.json"))
        if len(indices) < min_completed_chunks:
            raise RuntimeError(f"only {len(indices)} completed chunks; need {min_completed_chunks}")
    else:
        if prefix_plans <= 0 or prefix_plans > len(plans) or prefix_plans % chunk_size:
            raise ValueError("prefix_plans must be a positive complete-chunk prefix")
        indices = list(range(prefix_plans // chunk_size))
    selected = [row for index in indices for row in plans[index * chunk_size:(index + 1) * chunk_size]]
    if len({row["plan_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate selected plan ID")
    chunk_hashes = {}
    for index in indices:
        path = source / "chunks" / f"chunk_{index:04d}" / "COMPLETE.json"
        if not path.is_file():
            raise RuntimeError(f"chunk {index} is not complete")
        record = json.loads(path.read_text())
        expected = [row["plan_id"] for row in plans[index * chunk_size:(index + 1) * chunk_size]]
        if record["plan_ids"] != expected:
            raise ValueError(f"chunk {index} plan IDs changed")
        chunk_hashes[str(index)] = file_sha256(path)
    source_pairs = paired_files(source, by_seed, by_id, strict=True)
    selected_ids = {row["plan_id"] for row in selected}
    selected_pairs = {task: files for task, files in source_pairs.items() if task in selected_ids}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.building-", dir=output.parent))
    (temporary / "raw").mkdir()
    (temporary / "sidecar").mkdir()
    artifacts = {}
    for task_id in sorted(selected_pairs):
        raw, gold = selected_pairs[task_id]
        raw_target = temporary / "raw" / raw.name
        gold_target = temporary / "sidecar" / gold.name
        shutil.copy2(raw, raw_target)
        shutil.copy2(gold, gold_target)
        artifacts[task_id] = {
            "raw": raw_target.name,
            "raw_sha256": file_sha256(raw_target),
            "sidecar": gold_target.name,
            "sidecar_sha256": file_sha256(gold_target),
        }
    pilot_plan = temporary / "pilot_plan.jsonl"
    write_jsonl(pilot_plan, selected)
    summary = {
        "schema_version": SCHEMA,
        "selected_plans": len(selected),
        "completed_pairs": len(selected_pairs),
        "failed_or_missing_plans": len(selected) - len(selected_pairs),
        "bucket_plans": dict(Counter(row["bucket"] for row in selected)),
        "depth_plans": dict(Counter(str(row["target_depth"]) for row in selected)),
        "plan_sha256": file_sha256(pilot_plan),
        "source_plan_sha256": manifest["plan_sha256"],
        "source_graph_sha256": manifest["graph_sha256"],
    }
    (temporary / "pilot_plan.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    generation = {
        "schema_version": "graph_frontier_rich_generation_run_v1",
        "snapshot_schema_version": SCHEMA,
        "generator_model": "Qwen2.5-14B-Instruct",
        "requested_plans": len(selected),
        "completed_chains": len(selected_pairs),
        "failed_chains": len(selected) - len(selected_pairs),
        "raw_chain_files": len(selected_pairs),
        "sidecar_files": len(selected_pairs),
        "plan_sha256": file_sha256(pilot_plan),
        "graph_sha256": manifest["graph_sha256"],
        "frozen300_content_loaded": False,
    }
    (temporary / "generation_run.json").write_text(json.dumps(generation, indent=2, sort_keys=True) + "\n")
    snapshot = {
        "schema_version": SCHEMA,
        "source_run": str(source.resolve()),
        "source_status_at_freeze": status,
        "source_resume_manifest_sha256": file_sha256(manifest_path),
        "source_plan": str(plan.resolve()),
        "source_plan_sha256": manifest["plan_sha256"],
        "source_graph_sha256": manifest["graph_sha256"],
        "pilot_plan_sha256": file_sha256(pilot_plan),
        "prefix_plans": None if completed_chunks else prefix_plans,
        "selected_chunk_indices": indices,
        "selection_mode": "completed_chunks" if completed_chunks else "prefix",
        "completed_pairs": len(selected_pairs),
        "chunk_complete_sha256": chunk_hashes,
        "paired_artifacts": artifacts,
        "frozen300_content_loaded": False,
    }
    (temporary / "snapshot_manifest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    if len(paired_files(temporary, by_seed, by_id, strict=True)) != len(selected_pairs):
        raise RuntimeError("copied snapshot artifact pairing failed")
    os.replace(temporary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-plans", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--completed-chunks", action="store_true")
    parser.add_argument("--min-completed-chunks", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.plan, args.output, args.prefix_plans, args.chunk_size, args.completed_chunks, args.min_completed_chunks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

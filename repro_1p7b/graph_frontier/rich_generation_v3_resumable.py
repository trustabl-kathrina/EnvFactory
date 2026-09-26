"""Checkpointed, chunked driver around the unchanged rich-v3 generator.

Every completed chunk has a durable report. An interrupted chunk keeps its
successfully written raw/sidecar pairs; only its remaining plans are retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256, read_jsonl, write_jsonl


SCHEMA = "graph_frontier_rich_resumable_v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plan_maps(plan_path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    plans = read_jsonl(plan_path)
    by_seed = {int(item["generation_seed"]): item for item in plans}
    by_id = {str(item["plan_id"]): item for item in plans}
    if not plans or len(by_seed) != len(plans) or len(by_id) != len(plans):
        raise ValueError("plan empty or duplicate generation_seed/plan_id")
    return plans, by_seed, by_id


def paired_files(
    run_dir: Path,
    by_seed: dict[int, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    strict: bool,
) -> dict[str, tuple[Path, Path]]:
    raw_by_seed: dict[int, Path] = {}
    for path in (run_dir / "raw").glob("*.json"):
        seed = int(json.loads(path.read_text(encoding="utf-8"))["seed"])
        if seed not in by_seed or seed in raw_by_seed:
            raise ValueError(f"unknown or duplicate raw seed: {seed}")
        raw_by_seed[seed] = path
    pairs: dict[str, tuple[Path, Path]] = {}
    sidecar_count = 0
    for path in (run_dir / "sidecar").glob("*.gold.json"):
        sidecar_count += 1
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(sidecar["task_id"])
        if task_id not in by_id or task_id in pairs:
            raise ValueError(f"unknown or duplicate sidecar task: {task_id}")
        seed = int(sidecar['generation_seed'] if 'generation_seed' in sidecar else sidecar['seed'])
        if seed != int(by_id[task_id]["generation_seed"]):
            raise ValueError(f"sidecar seed does not match plan: {task_id}")
        raw = raw_by_seed.get(seed)
        if raw is not None:
            pairs[task_id] = (raw, path)
    if strict and (len(pairs) != len(raw_by_seed) or len(pairs) != sidecar_count):
        raise ValueError(f"unpaired artifacts in {run_dir}: raw={len(raw_by_seed)} sidecar={sidecar_count} pairs={len(pairs)}")
    return pairs


def copy_pairs(
    source: Path,
    target: Path,
    by_seed: dict[int, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    strict_source: bool,
) -> int:
    pairs = paired_files(source, by_seed, by_id, strict=strict_source)
    existing = paired_files(target, by_seed, by_id, strict=False)
    for task_id, (raw, sidecar) in pairs.items():
        if task_id in existing:
            continue
        for artifact, folder in ((raw, "raw"), (sidecar, "sidecar")):
            destination = target / folder / artifact.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if file_sha256(destination) != file_sha256(artifact):
                    raise ValueError(f"artifact filename collision: {destination}")
            else:
                shutil.copy2(artifact, destination)
    paired_files(target, by_seed, by_id, strict=True)
    return len(pairs)


def bootstrap(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite resume directory: {args.output_dir}")
    plans, by_seed, by_id = plan_maps(args.plan)
    source_pairs = paired_files(args.source_run, by_seed, by_id, strict=True)
    if not source_pairs:
        raise ValueError("source has no complete raw/sidecar pairs")
    args.output_dir.mkdir(parents=True)
    copy_pairs(args.source_run, args.output_dir, by_seed, by_id, strict_source=True)
    manifest = {
        "schema_version": SCHEMA,
        "source_run": str(args.source_run.resolve()),
        "source_status": (args.source_run / "status").read_text().strip(),
        "source_pairs": len(source_pairs),
        "source_artifacts": {
            task_id: {"raw_sha256": file_sha256(raw), "sidecar_sha256": file_sha256(sidecar)}
            for task_id, (raw, sidecar) in source_pairs.items()
        },
        "plan": str(args.plan.resolve()),
        "plan_sha256": file_sha256(args.plan),
        "graph": str(args.graph.resolve()),
        "graph_sha256": file_sha256(args.graph),
        "total_plans": len(plans),
        "chunk_size": args.chunk_size,
    }
    atomic_json(args.output_dir / "resume_manifest.json", manifest)
    (args.output_dir / "status").write_text("BOOTSTRAPPED\n")
    print(json.dumps({"source_pairs": len(source_pairs), "total_plans": len(plans), "output_dir": str(args.output_dir)}))


def validate_manifest(args: argparse.Namespace, plans: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = json.loads((args.output_dir / "resume_manifest.json").read_text())
    if (manifest["schema_version"] != SCHEMA or manifest["plan_sha256"] != file_sha256(args.plan)
            or manifest["graph_sha256"] != file_sha256(args.graph)
            or manifest["total_plans"] != len(plans) or manifest["chunk_size"] != args.chunk_size):
        raise ValueError("resume manifest does not match immutable plan/graph/chunk size")
    return manifest


def run(args: argparse.Namespace) -> None:
    plans, by_seed, by_id = plan_maps(args.plan)
    manifest = validate_manifest(args, plans)
    if (args.output_dir / "generation_run.json").exists():
        raise FileExistsError("completed generation report already exists")
    started = time.time()
    chunks_root = args.output_dir / "chunks"
    chunks_root.mkdir(exist_ok=True)
    total_chunks = (len(plans) + args.chunk_size - 1) // args.chunk_size
    if getattr(args, "balanced_order", False):
        groups = {1: [], 2: [], 3: []}
        for index in range(total_chunks):
            rows = plans[index * args.chunk_size:(index + 1) * args.chunk_size]
            counts = {depth: sum(int(row["target_depth"]) == depth for row in rows) for depth in groups}
            groups[max(counts, key=counts.get)].append(index)
        chunk_order = [index for triple in zip_longest(groups[3], groups[2], groups[1])
                       for index in triple if index is not None]
    else:
        chunk_order = list(range(total_chunks))
    completed_chunks = 0
    for chunk_index in chunk_order:
        if getattr(args, "pause_after_completed_chunks", 0) and completed_chunks >= args.pause_after_completed_chunks:
            (args.output_dir / "status").write_text("PAUSED_FOR_PILOT\n")
            print(json.dumps({"paused_after_completed_chunks": completed_chunks, "chunk_order": chunk_order}), flush=True)
            return
        offset = chunk_index * args.chunk_size
        chunk = plans[offset:offset + args.chunk_size]
        chunk_dir = chunks_root / f"chunk_{chunk_index:04d}"
        chunk_dir.mkdir(exist_ok=True)
        attempts = sorted(path for path in chunk_dir.glob("attempt_*" ) if path.is_dir())
        for attempt in attempts:
            copy_pairs(attempt, args.output_dir, by_seed, by_id, strict_source=False)
        successes = paired_files(args.output_dir, by_seed, by_id, strict=True)
        completion = chunk_dir / "COMPLETE.json"
        if completion.exists():
            record = json.loads(completion.read_text())
            if record["plan_ids"] != [item["plan_id"] for item in chunk]:
                raise ValueError(f"chunk completion changed: {chunk_index}")
            completed_chunks += 1
            continue
        completed_attempt = next((path for path in reversed(attempts) if (path / "generation_run.json").exists()), None)
        if completed_attempt is None:
            pending = [item for item in chunk if item["plan_id"] not in successes]
            if pending:
                attempt = chunk_dir / f"attempt_{len(attempts):03d}"
                attempt.mkdir()
                attempt_plan = attempt / "plan.jsonl"
                write_jsonl(attempt_plan, pending)
                command = [
                    sys.executable, "-m", "repro_1p7b.graph_frontier.rich_generation_v3",
                    "generate", "--graph", str(args.graph), "--plan", str(attempt_plan),
                    "--run-dir", str(attempt), "--model-name", args.model_name,
                    "--pass-k", str(args.pass_k), "--concurrency", str(args.concurrency),
                    "--attempts", str(args.attempts), "--max-failures", str(len(pending)),
                ]
                with (attempt / "generation.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"chunk {chunk_index} attempt {attempt.name} exited {result.returncode}")
                completed_attempt = attempt
                copy_pairs(attempt, args.output_dir, by_seed, by_id, strict_source=False)
                successes = paired_files(args.output_dir, by_seed, by_id, strict=True)
        if completed_attempt is not None:
            report = json.loads((completed_attempt / "generation_run.json").read_text())
            attempted = read_jsonl(completed_attempt / "plan.jsonl")
            if (report["requested_plans"] != len(attempted)
                    or report["plan_sha256"] != file_sha256(completed_attempt / "plan.jsonl")):
                raise ValueError(f"chunk {chunk_index} attempt report mismatch")
        successes = paired_files(args.output_dir, by_seed, by_id, strict=True)
        atomic_json(completion, {
            "schema_version": SCHEMA,
            "chunk_index": chunk_index,
            "plan_ids": [item["plan_id"] for item in chunk],
            "success_ids": [item["plan_id"] for item in chunk if item["plan_id"] in successes],
            "failed_ids": [item["plan_id"] for item in chunk if item["plan_id"] not in successes],
            "completed_attempt": str(completed_attempt) if completed_attempt else "preexisting_pairs_only",
        })
        print(json.dumps({"chunk": chunk_index, "total_chunks": (len(plans) + args.chunk_size - 1) // args.chunk_size,
                          "success": len(successes)}), flush=True)
        completed_chunks += 1
        if getattr(args, "pause_after_completed_chunks", 0) and completed_chunks >= args.pause_after_completed_chunks:
            (args.output_dir / "status").write_text("PAUSED_FOR_PILOT\n")
            print(json.dumps({"paused_after_completed_chunks": completed_chunks, "chunk_order": chunk_order}), flush=True)
            return
    successes = paired_files(args.output_dir, by_seed, by_id, strict=True)
    failed = [item["plan_id"] for item in plans if item["plan_id"] not in successes]
    atomic_json(args.output_dir / "generation_run.json", {
        "schema_version": "graph_frontier_rich_generation_run_v1",
        "resume_schema_version": SCHEMA,
        "generator_model": "Qwen2.5-14B-Instruct",
        "model_names": [item.strip() for item in args.model_name.split(",")],
        "requested_plans": len(plans),
        "completed_chains": len(successes),
        "failed_chains": len(failed),
        "failures": failed,
        "raw_chain_files": len(list((args.output_dir / "raw").glob("*.json"))),
        "sidecar_files": len(list((args.output_dir / "sidecar").glob("*.gold.json"))),
        "pass_k": args.pass_k,
        "concurrency": args.concurrency,
        "attempts_per_structure": args.attempts,
        "graph_sha256": manifest["graph_sha256"],
        "plan_sha256": manifest["plan_sha256"],
        "runtime_seconds_this_invocation": time.time() - started,
        "frozen300_content_loaded": False,
        "source_pairs": manifest["source_pairs"],
        "chunk_size": args.chunk_size,
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("bootstrap", "run"):
        item = sub.add_parser(name)
        item.add_argument("--plan", type=Path, required=True)
        item.add_argument("--graph", type=Path, required=True)
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--chunk-size", type=int, default=32)
        if name == "bootstrap":
            item.add_argument("--source-run", type=Path, required=True)
        else:
            item.add_argument("--model-name", default="sglang,sglang1")
            item.add_argument("--pass-k", type=int, default=2)
            item.add_argument("--concurrency", type=int, default=4)
            item.add_argument("--attempts", type=int, default=4)
            item.add_argument("--balanced-order", action="store_true")
            item.add_argument("--pause-after-completed-chunks", type=int, default=0)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    if arguments.command == "bootstrap":
        bootstrap(arguments)
    else:
        run(arguments)

"""Freeze replay-validated EnvFactory RL data and one shared pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

FROZEN_300_SHA256 = "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"
SOURCE = "EnvFactory-generated-guided-rl-v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def graph(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row["extra_info"]["graph_frontier"]


def group_seed(row: Mapping[str, Any]) -> str:
    seed = row.get("extra_info", {}).get("generation_seed")
    return str(seed if seed is not None else graph(row).get("seed", row["task_id"]))


def group_split(rows: list[dict[str, Any]], ratio: float, seed: int):
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_seed(row)].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = round(len(rows) * ratio)
    train_keys, count = [], 0
    for key in keys:
        if train_keys and count >= target:
            break
        train_keys.append(key)
        count += len(groups[key])
    if len(train_keys) == len(keys) and len(keys) > 1:
        train_keys.pop()
    train_set = set(train_keys)
    train = [row for row in rows if group_seed(row) in train_set]
    val = [row for row in rows if group_seed(row) not in train_set]
    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "rl_val"
    audit = {
        "method": "generation_seed_grouped_deterministic",
        "seed": seed,
        "target_train_ratio": ratio,
        "actual_train_ratio": len(train) / max(1, len(rows)),
        "train_groups": len(train_set),
        "val_groups": len(keys) - len(train_set),
        "group_overlap": [],
    }
    return train, val, audit


def depth_bucket(value: Any) -> str:
    return "3+" if isinstance(value, int) and value >= 3 else str(value)


def row_stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
    sidecar = graph(row)
    envs = "+".join(sorted(sidecar.get("environment_identifiers", []) or ["unknown"]))
    depth = depth_bucket(sidecar.get("dependency_depth", "unknown"))
    internal = sum(
        edge.get("internal_parameter") is True
        for edge in sidecar.get("dependency_edges", []) or []
    )
    return envs, depth, "2+" if internal >= 2 else str(internal)


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int):
    if len(rows) < size:
        raise RuntimeError(f"pilot requires {size} train tasks, only {len(rows)} available")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row_stratum(row)].append(row)
    rng = random.Random(seed)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])
    rng.shuffle(keys)
    selected, cursor = [], Counter()
    while len(selected) < size:
        progressed = False
        for key in keys:
            index = cursor[key]
            if index < len(buckets[key]):
                selected.append(buckets[key][index])
                cursor[key] += 1
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            raise RuntimeError("stratified sampler exhausted unexpectedly")
    audit = {
        "method": "deterministic_round_robin_env_depth_internal_edge",
        "seed": seed,
        "requested": size,
        "available_train": len(rows),
        "available_strata": len(buckets),
        "selected_strata": len({row_stratum(row) for row in selected}),
        "environment_distribution": dict(sorted(Counter(
            env for row in selected
            for env in graph(row).get("environment_identifiers", [])
        ).items())),
        "dependency_depth_distribution": dict(sorted(Counter(
            depth_bucket(graph(row).get("dependency_depth", "unknown")) for row in selected
        ).items())),
        "internal_edge_count_distribution": dict(sorted(Counter(
            sum(edge.get("internal_parameter") is True
                for edge in graph(row).get("dependency_edges", []) or [])
            for row in selected
        ).items())),
    }
    return selected, audit


def file_manifest(paths: Iterable[Path], base: Path):
    return [
        {"path": str(path.relative_to(base)), "sha256": file_sha256(path)}
        for path in sorted(paths)
    ]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    source, replay_dir, output = args.source_dir, args.replay_dir, args.output_dir
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen dataset: {output}")
    static_rows = load_jsonl(args.static_ledger)
    replayed = load_jsonl(replay_dir / "rl_data_v1_audit_replayed.jsonl")
    replay_pass = {
        str(row["task_id"]) for row in replayed if row.get("status") == "REPLAY_PASS"
    }
    if not replay_pass:
        raise RuntimeError("no REPLAY_PASS tasks; freeze gate failed")
    converted = load_json(source / "train.json") + load_json(source / "rl_val.json")
    by_id = {str(row["task_id"]): row for row in converted}
    missing = sorted(replay_pass - by_id.keys())
    if missing:
        raise RuntimeError(f"replay-pass tasks missing converted rows: {missing[:5]}")
    valid = [by_id[task_id] for task_id in sorted(replay_pass)]
    if any(row.get("frozen_300_overlap") is not False for row in valid):
        raise RuntimeError("frozen-300 contamination gate failed")
    train, val, split_audit = group_split(valid, args.train_ratio, args.seed)
    pilot, pilot_audit = stratified_sample(train, args.pilot_size, args.seed)
    smoke = pilot[:min(args.smoke_size, len(pilot))]

    output.mkdir(parents=True)
    raw_dir, gold_dir = output / "raw_valid", output / "gold_sidecar_valid"
    raw_dir.mkdir()
    gold_dir.mkdir()
    static_by_id = {str(row["task_id"]): row for row in static_rows}
    copied_raw: set[str] = set()
    for task_id in sorted(replay_pass):
        item = static_by_id[task_id]
        raw_source, gold_source = Path(item["raw_file"]), Path(item["sidecar_file"])
        if raw_source.name not in copied_raw:
            shutil.copy2(raw_source, raw_dir / raw_source.name)
            copied_raw.add(raw_source.name)
        shutil.copy2(gold_source, gold_dir / gold_source.name)

    write_json(output / "train.json", train)
    write_json(output / "val.json", val)
    write_jsonl(output / "audit.jsonl", replayed)
    contamination = load_json(source / "contamination_audit.json")
    contamination["FROZEN_300_EXACT_TRAIN_OVERLAP"] = 0
    write_json(output / "contamination_report.json", contamination)
    write_json(output / "generation_config.json", load_json(source / "generation_run.json"))
    replay_report = load_json(replay_dir / "executable_replay_report.json")
    static_report = load_json(args.static_report)
    quality = {
        "schema_version": "graph_frontier_rl_v1_frozen_quality_report",
        "generated": static_report.get("generated_sidecars"),
        "static_pass": static_report.get("static_pass"),
        "replay_pass": len(replay_pass),
        "dropped_or_quarantined": len(replayed) - len(replay_pass),
        "static_drop_reasons": static_report.get("drop_reasons", {}),
        "replay_failure_reasons": replay_report.get("failure_reasons", {}),
    }
    write_json(output / "quality_report.json", quality)

    pilot_ids = [str(row["task_id"]) for row in pilot]
    pilot_manifest = {
        "schema_version": "graph_frontier_rl_v1_shared_pilot_manifest",
        "task_ids_in_order": pilot_ids,
        "smoke_task_ids_in_order": [str(row["task_id"]) for row in smoke],
        "standard_graph_shared_order": True,
        "frontier_aware_sampling": False,
        "audit": pilot_audit,
        "task_order_sha256": stable_sha256(pilot_ids),
    }
    write_json(output / "pilot_128_manifest.json", pilot_manifest)
    write_json(output / "pilot_128.json", pilot)
    write_json(output / "smoke_8.json", smoke)
    write_json(output / "raw_manifest.json", file_manifest(raw_dir.glob("*.json"), output))
    write_json(output / "sidecar_manifest.json", file_manifest(gold_dir.glob("*.json"), output))

    hashes = {}
    for name in (
        "raw_manifest.json", "sidecar_manifest.json", "train.json", "val.json",
        "quality_report.json", "contamination_report.json", "pilot_128_manifest.json",
    ):
        hashes[name] = file_sha256(output / name)
    manifest = {
        "schema_version": "graph_frontier_rl_v1_frozen_manifest",
        "status": "RL_DATASET_V1_FROZEN",
        "source": SOURCE,
        "frozen_300_manifest_sha256": FROZEN_300_SHA256,
        "FROZEN_300_EXACT_TRAIN_OVERLAP": 0,
        "replay_valid_tasks": len(valid),
        "train_tasks": len(train),
        "val_tasks": len(val),
        "split_audit": split_audit,
        "shared_pilot_tasks": len(pilot),
        "hashes": hashes,
    }
    write_json(output / "manifest.json", manifest)
    (output / "status").write_text("RL_DATASET_V1_FROZEN\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--static-ledger", type=Path, required=True)
    result.add_argument("--static-report", type=Path, required=True)
    result.add_argument("--replay-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--train-ratio", type=float, default=0.90)
    result.add_argument("--seed", type=int, default=20260920)
    result.add_argument("--pilot-size", type=int, default=128)
    result.add_argument("--smoke-size", type=int, default=8)
    return result


if __name__ == "__main__":
    print(json.dumps(freeze(parser().parse_args()), indent=2, ensure_ascii=False))

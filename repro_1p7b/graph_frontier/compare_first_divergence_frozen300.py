"""Paired Frozen300 comparison with the unchanged Dynamic-v1 baseline."""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256

KEYS = (
    "task_success", "reference_path_complete_success", "final_state_success",
    "edge_complete", "gold_node_complete", "internal_param_complete",
)


def ci95(values: list[float]) -> list[float]:
    rng = random.Random(20260925)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(5000))
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def load_tasks(directory: Path) -> dict[str, dict]:
    tasks = {}
    for path in (directory / "tasks").glob("*.result.json"):
        row = json.loads(path.read_text())
        task_id = row["task_id"]
        if task_id in tasks:
            raise RuntimeError("duplicate task result")
        tasks[task_id] = row
    return tasks


def compare(manifest: Path, base_dir: Path, trained_dir: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite Frozen300 comparison")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    task_ids = {row["task_id"] for row in rows}
    if len(rows) != 300 or len(task_ids) != 300:
        raise RuntimeError("Frozen300 manifest count invalid")
    for shard in (0, 1):
        b_config = json.loads((base_dir / f"run_config.shard{shard}.json").read_text())
        t_config = json.loads((trained_dir / f"run_config.shard{shard}.json").read_text())
        if (
            b_config["manifest_sha256"] != sha256(manifest)
            or t_config["manifest_sha256"] != sha256(manifest)
            or b_config["inference"] != t_config["inference"]
            or b_config["shard_count"] != 2
            or t_config["shard_count"] != 2
            or b_config["shard_index"] != t_config["shard_index"]
        ):
            raise RuntimeError("Frozen300 protocol mismatch")
    base = load_tasks(base_dir)
    trained = load_tasks(trained_dir)
    if set(base) != task_ids or set(trained) != task_ids:
        raise RuntimeError("Frozen300 task coverage mismatch")
    if any(not x.get("valid_capability_probe") for x in base.values()):
        raise RuntimeError("baseline contains invalid system probe")
    if any(not x.get("valid_capability_probe") for x in trained.values()):
        raise RuntimeError("trained model contains invalid system probe")
    per_task = []
    totals = {"base": collections.Counter(), "trained": collections.Counter()}
    by_split = {"base": collections.defaultdict(collections.Counter),
                "trained": collections.defaultdict(collections.Counter)}
    for task_id in sorted(task_ids):
        b, t = base[task_id], trained[task_id]
        if b["split"] != t["split"]:
            raise RuntimeError("Frozen300 split mismatch")
        result = {"task_id": task_id, "split": b["split"]}
        for name, row in (("base", b), ("trained", t)):
            result[name] = {key: bool(row.get(key)) for key in KEYS}
            for key in KEYS:
                totals[name][key] += bool(row.get(key))
                by_split[name][row["split"]][key] += bool(row.get(key))
            totals[name]["tool_call_count"] += int(row.get("tool_call_count", 0))
            by_split[name][row["split"]]["tasks"] += 1
        per_task.append(result)
    paired = {}
    for key in KEYS:
        differences = [float(row["trained"][key]) - float(row["base"][key])
                       for row in per_task]
        paired[key] = {
            "delta_count": sum(differences),
            "delta_rate": sum(differences) / len(differences),
            "ci95_delta_rate": ci95(differences),
            "wins": sum(x > 0 for x in differences),
            "losses": sum(x < 0 for x in differences),
            "ties": sum(x == 0 for x in differences),
        }
    summary = {
        "schema_version": "first_divergence_frozen300_paired_v1",
        "manifest_sha256": sha256(manifest),
        "task_count": len(per_task), "valid_probe_count": 300,
        "baseline_model_path": json.loads((base_dir / "run_config.shard0.json").read_text())["model_path"],
        "trained_model_path": json.loads((trained_dir / "run_config.shard0.json").read_text())["model_path"],
        "protocol": json.loads((trained_dir / "run_config.shard0.json").read_text())["inference"],
        "base": dict(totals["base"]), "trained": dict(totals["trained"]),
        "by_split": {
            name: {split: dict(counter) for split, counter in sorted(by_split[name].items())}
            for name in ("base", "trained")
        },
        "paired": paired,
    }
    output.mkdir(parents=True)
    with (output / "per_task.jsonl").open("w", encoding="utf-8") as out:
        for row in per_task:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.manifest, args.base, args.trained, args.output),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

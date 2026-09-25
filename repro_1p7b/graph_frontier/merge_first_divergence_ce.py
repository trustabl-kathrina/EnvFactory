"""Gate and freeze cumulative replay-verified CE states from two policies."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256

MIN_NEW = 20
MIN_TOTAL = 48


def load_dataset(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "dataset.jsonl").read_text().splitlines()]
    if (
        manifest.get("dataset_sha256") != sha256(directory / "dataset.jsonl")
        or manifest.get("example_count") != len(rows)
        or manifest.get("frozen_exact_id_overlap") != 0
        or manifest.get("tool_observation_tokens_in_loss") != 0
    ):
        raise RuntimeError(f"CE manifest gate failed: {directory}")
    for row in rows:
        p = row["prompt_tokens"]
        if (
            len(row["input_ids"]) != len(row["labels"])
            or row["labels"][:p] != [-100] * p
            or row["labels"][p:] != row["input_ids"][p:]
            or len(row["input_ids"]) > 16384
        ):
            raise RuntimeError(f"assistant-only mask failed: {row['task_id']}")
    return manifest, rows


def merge(first: Path, second: Path, heldout_plan: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite cumulative CE dataset")
    a_manifest, a = load_dataset(first)
    b_manifest, b = load_dataset(second)
    heldout = json.loads(heldout_plan.read_text())
    ids_a = {x["task_id"] for x in a}
    ids_b = {x["task_id"] for x in b}
    states = [x["state_identity"] for x in a + b]
    if (
        len(b) < MIN_NEW or len(a) + len(b) < MIN_TOTAL
        or ids_a & ids_b
        or (ids_a | ids_b) & set(heldout["task_ids"])
        or len(states) != len(set(states))
        or b_manifest["model_weights_sha256"] == a_manifest["model_weights_sha256"]
    ):
        raise RuntimeError(
            f"DATA_NOT_READY new={len(b)} total={len(a)+len(b)} "
            f"task_overlap={len(ids_a & ids_b)} heldout_overlap={len((ids_a|ids_b)&set(heldout['task_ids']))}"
        )
    rows = a + b
    output.mkdir(parents=True)
    with (output / "dataset.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "schema_version": "first_divergence_cumulative_ce_v1",
        "source_datasets": [
            {"path": str(first), "manifest_sha256": sha256(first / "manifest.json"),
             "dataset_sha256": a_manifest["dataset_sha256"],
             "student_model_weights_sha256": a_manifest["model_weights_sha256"],
             "examples": len(a)},
            {"path": str(second), "manifest_sha256": sha256(second / "manifest.json"),
             "dataset_sha256": b_manifest["dataset_sha256"],
             "student_model_weights_sha256": b_manifest["model_weights_sha256"],
             "examples": len(b)},
        ],
        "heldout_plan_sha256": sha256(heldout_plan),
        "heldout_task_overlap": 0, "frozen_exact_id_overlap": 0,
        "example_count": len(rows), "task_count": len(ids_a | ids_b),
        "new_examples": len(b),
        "divergence_types": dict(sorted(collections.Counter(x["divergence_type"] for x in rows).items())),
        "max_length": max(x["total_tokens"] for x in rows),
        "target_tokens_total": sum(x["target_tokens"] for x in rows),
        "prompt_tokens_masked": sum(x["prompt_tokens"] for x in rows),
        "tool_observation_tokens_in_loss": 0,
        "dataset_sha256": sha256(output / "dataset.jsonl"),
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--heldout-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.first, args.second, args.heldout_plan, args.output),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

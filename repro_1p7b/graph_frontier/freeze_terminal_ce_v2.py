"""Freeze replay-gated terminal CE rows and the existing 50 FD rows for v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_terminal
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import MODEL

ROOT = Path(__file__).resolve().parents[2]
FD_SOURCE = ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_cumulative_ce_dataset"
HELDOUT = ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1/cycle2_heldout_plan.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_fd() -> tuple[list[dict], dict]:
    manifest = json.loads((FD_SOURCE / "manifest.json").read_text())
    rows = jsonl(FD_SOURCE / "dataset.jsonl")
    if (
        manifest.get("schema_version") != "first_divergence_cumulative_ce_v1"
        or manifest.get("dataset_sha256") != sha256(FD_SOURCE / "dataset.jsonl")
        or len(rows) != 50 or manifest.get("example_count") != 50
        or len({r["state_identity"] for r in rows}) != 50
        or manifest.get("heldout_task_overlap") != 0
        or manifest.get("frozen_exact_id_overlap") != 0
    ):
        raise RuntimeError("existing FD50 source lock failed")
    for row in rows:
        p = row["prompt_tokens"]
        if (len(row["input_ids"]) != len(row["labels"])
                or row["labels"][:p] != [-100] * p
                or row["labels"][p:] != row["input_ids"][p:]):
            raise RuntimeError("FD loss mask invalid")
    return rows, manifest


def freeze(terminal_source: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite v2 frozen data")
    terminal_manifest = json.loads((terminal_source / "manifest.json").read_text())
    terminal_rows = jsonl(terminal_source / "dataset.jsonl")
    fd_rows, fd_manifest = validate_fd()
    heldout_plan = json.loads(HELDOUT.read_text())
    heldout_ids = set(heldout_plan["task_ids"])
    if (
        terminal_manifest.get("schema_version") != "gold_terminal_dataset_v2"
        or terminal_manifest.get("verdict") != "DATA_READY"
        or terminal_manifest.get("dataset_sha256") != sha256(terminal_source / "dataset.jsonl")
        or terminal_manifest.get("heldout_plan_sha256") != sha256(HELDOUT)
        or terminal_manifest.get("heldout_task_overlap") != 0
        or terminal_manifest.get("frozen300_exact_id_overlap") != 0
        or terminal_manifest["counts"]["final_terminal_usable"] != len(terminal_rows)
        or len(terminal_rows) < 48
        or set(terminal_manifest["task_ids"]) != {r["task_id"] for r in terminal_rows}
        or set(terminal_manifest["task_ids"]) & heldout_ids
        or {r["task_id"] for r in fd_rows} & heldout_ids
    ):
        raise RuntimeError("terminal or heldout data gate failed")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    examples = [encode_terminal(tokenizer, row) for row in terminal_rows]
    if len({r["state_identity"] for r in examples}) != len(examples):
        raise RuntimeError("duplicate terminal state")
    if any(r["tool_observation_tokens_in_loss"] != 0 for r in examples):
        raise RuntimeError("tool observation loss leakage")
    output.mkdir(parents=True)
    data_path = output / "terminal_ce_dataset.jsonl"
    data_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in examples))
    lock = {
        "schema_version": "first_divergence_terminal_v2_frozen_data",
        "verdict": "FROZEN_READY",
        "fd_source_manifest_sha256": sha256(FD_SOURCE / "manifest.json"),
        "fd_source_dataset_sha256": fd_manifest["dataset_sha256"],
        "fd_task_ids": sorted({r["task_id"] for r in fd_rows}),
        "fd_examples": len(fd_rows),
        "terminal_source_manifest_sha256": sha256(terminal_source / "manifest.json"),
        "terminal_source_dataset_sha256": terminal_manifest["dataset_sha256"],
        "terminal_ce_dataset_sha256": sha256(data_path),
        "terminal_task_ids": sorted({r["task_id"] for r in examples}),
        "terminal_examples": len(examples),
        "heldout_plan_sha256": sha256(HELDOUT),
        "heldout_task_overlap": 0,
        "frozen300_exact_id_overlap": 0,
        "tokenizer_json_sha256": sha256(MODEL / "tokenizer.json"),
        "max_length": 16384,
        "tool_observation_tokens_in_loss": 0,
        "historical_assistant_tokens_in_loss": 0,
        "terminal_length_max": max(r["total_tokens"] for r in examples),
        "terminal_target_tokens_total": sum(r["target_tokens"] for r in examples),
    }
    (output / "manifest.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.terminal_source, args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Freeze one clean Official-RL train/heldout split for all OPD cycles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import MODEL, checked_source, sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import (
    FROZEN, RICH24, ROOT, SOURCE, OUT, overlap_index,
)

RUN = ROOT / "repro_1p7b/graph_frontier/recursive_opd_v1_run"
PROTOCOL = RUN / "protocol.json"
SEED = 20260926
CONFIG = {
    "temperature": 0.0, "top_p": 0.95, "max_tokens_per_turn": 1024,
    "max_turns": 12, "tool_call_limit": 11, "seed": SEED,
    "chat_template_enable_thinking": False,
}


def split_tasks(entries: list[dict]) -> tuple[list[str], list[str]]:
    ids = [str(e["audit_id"]) for e in entries]
    if len(ids) != 2158 or len(set(ids)) != len(ids):
        raise RuntimeError("clean task identity drift")
    order = sorted(ids, key=lambda task: (hashlib.sha256(f"{SEED}:{task}".encode()).hexdigest(), task))
    train = order[:1726]
    heldout = order[1726:]
    if len(heldout) != 432 or set(train) & set(heldout):
        raise RuntimeError("train/heldout split invalid")
    return train, heldout


def external_gate(entries: list[dict]) -> dict:
    ids = {e["audit_id"] for e in entries}
    queries = {e["query_hash"] for e in entries}
    result = {}
    for name, path, expected in (("Frozen300", FROZEN, 300), ("Rich24", RICH24, 24)):
        other = overlap_index(path)
        if not other["available"] or other["rows"] != expected:
            raise RuntimeError(f"{name} index missing or incomplete")
        collisions = {"task_ids": len(ids & other["task_ids"]),
                      "query_hashes": len(queries & other["query_hashes"])}
        if any(collisions.values()):
            raise RuntimeError(f"{name} overlap: {collisions}")
        result[name] = {"source_sha256": sha256(path), "rows": expected, **collisions}
    return result


def expected() -> dict:
    entries, _, _ = checked_source()
    train, heldout = split_tasks(entries)
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("original Dynamic-v1 checkpoint absent")
    return {
        "schema_version": "recursive_opd_v1_protocol", "seed": SEED,
        "source_sha256": sha256(SOURCE),
        "source_manifest_sha256": sha256(OUT / "opd_source_manifest.json"),
        "pi0_model": str(MODEL), "pi0_model_sha256": sha256(MODEL / "model.safetensors"),
        "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
        "config": CONFIG, "train_task_ids": train, "heldout_task_ids": heldout,
        "external_zero_overlap": external_gate(entries),
    }


def check(path: Path = PROTOCOL) -> dict:
    on_disk = json.loads(path.read_text())
    if on_disk != expected():
        raise RuntimeError("immutable Recursive OPD protocol drift")
    return on_disk


def create(path: Path = PROTOCOL) -> dict:
    if path.exists():
        raise RuntimeError("refusing to overwrite Recursive OPD protocol")
    value = expected()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable(value) + "\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "check"))
    args = parser.parse_args()
    value = create() if args.command == "create" else check()
    print(json.dumps({"train": len(value["train_task_ids"]),
                      "heldout": len(value["heldout_task_ids"]),
                      "source_sha256": value["source_sha256"],
                      "pi0_model_sha256": value["pi0_model_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

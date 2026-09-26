"""Immutable clean-source lock and deterministic one-rollout-per-task plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from repro_1p7b.graph_frontier.official_rl_audit_static import (
    OFFICIAL_SHA, OUT, ROOT, SOURCE, decode, query_from_row, query_hash,
)
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import MODEL

SOURCE_MANIFEST = OUT / "opd_source_manifest.json"
RUN = ROOT / "repro_1p7b/graph_frontier/cycle3_official_rl/run_dynamic_v1_t0"
CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_tokens_per_turn": 1024,
    "max_turns": 12,
    "tool_call_limit": 11,
    "seed": 20260926,
    "rollouts_per_task": 1,
    "served_model": "dynamic-v1",
    "chat_template_enable_thinking": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def checked_source() -> tuple[list[dict], list[dict], dict]:
    lock = json.loads(SOURCE_MANIFEST.read_text())
    if (lock.get("task_count") != 2158 or len(lock.get("rows", [])) != 2158
            or lock.get("source_sha256") != OFFICIAL_SHA
            or sha256(SOURCE) != OFFICIAL_SHA
            or len({r["source_row_index"] for r in lock["rows"]}) != 2158
            or len({r["audit_id"] for r in lock["rows"]}) != 2158):
        raise RuntimeError("authoritative Official-RL clean-source lock failed")
    raw = json.loads(SOURCE.read_text())
    if len(raw) != 3092:
        raise RuntimeError("Official-RL raw row count changed")
    for entry in lock["rows"]:
        index = entry["source_row_index"]
        row = raw[index]
        if entry["query_hash"] != query_hash(query_from_row(row)):
            raise RuntimeError(f"Official-RL query hash drift: {index}")
        servers = decode(row["extra_info"]["mcp_factory_kwargs"]["mcp_servers"])
        if entry["environment"] != ",".join(sorted(servers)):
            raise RuntimeError(f"Official-RL environment drift: {index}")
        replay = json.loads((OUT / "replay_tasks" / f"{index:04d}.json").read_text())
        if (replay.get("audit_id") != entry["audit_id"]
                or replay.get("replay_ok") is not True
                or replay.get("final_state_match") is not True):
            raise RuntimeError(f"strict gold replay lock failed: {index}")
    return lock["rows"], raw, lock


def create(path: Path = RUN / "rollout_manifest.json") -> dict:
    if path.exists():
        raise RuntimeError("refusing to overwrite Cycle-3 rollout plan")
    entries, _, _ = checked_source()
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("original Dynamic-v1 checkpoint missing")
    plan = {
        "schema_version": "cycle3_official_rl_rollout_plan_v1",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_sha256": OFFICIAL_SHA,
        "model_path": str(MODEL),
        "model_weights_sha256": sha256(MODEL / "model.safetensors"),
        "config": CONFIG,
        "task_count": len(entries),
        "tasks": [{
            "task_id": e["audit_id"],
            "source_row_index": e["source_row_index"],
            "environment": e["environment"],
            "query_hash": e["query_hash"],
            "gold_length": e["gold_length"],
        } for e in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return plan


def checked_plan(path: Path = RUN / "rollout_manifest.json") -> tuple[dict, list[dict]]:
    plan = json.loads(path.read_text())
    if (plan.get("schema_version") != "cycle3_official_rl_rollout_plan_v1"
            or plan.get("config") != CONFIG
            or plan.get("source_manifest_sha256") != sha256(SOURCE_MANIFEST)
            or plan.get("source_sha256") != sha256(SOURCE)
            or Path(plan.get("model_path", "")).resolve() != MODEL.resolve()
            or plan.get("model_weights_sha256") != sha256(MODEL / "model.safetensors")
            or plan.get("task_count") != 2158 or len(plan.get("tasks", [])) != 2158):
        raise RuntimeError("Cycle-3 plan/source/model/config drift")
    entries, raw, _ = checked_source()
    expected = [(e["audit_id"], e["source_row_index"]) for e in entries]
    actual = [(e["task_id"], e["source_row_index"]) for e in plan["tasks"]]
    if expected != actual:
        raise RuntimeError("Cycle-3 clean task order drift")
    return plan, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "check"))
    parser.add_argument("--path", type=Path, default=RUN / "rollout_manifest.json")
    args = parser.parse_args()
    plan = create(args.path) if args.command == "create" else checked_plan(args.path)[0]
    print(json.dumps({"task_count": plan["task_count"],
                      "model_sha256": plan["model_weights_sha256"],
                      "source_sha256": plan["source_sha256"],
                      "configuration": plan["config"]}, sort_keys=True))


if __name__ == "__main__":
    main()

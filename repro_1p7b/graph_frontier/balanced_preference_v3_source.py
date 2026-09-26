"""Audit and replay the next immutable Rich-v3 milestone for preference v3.

This module never edits the generator, the v2 pilot, or historical evaluation.
Each replay cache entry is keyed by task and graph edge and is safe to reuse
across larger, nested completed-chunk snapshots.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import inventory_record, replay_one
from repro_1p7b.graph_frontier.collect_preference_rollouts import EnvConfig, env_item, stable
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256


ROOT = Path(__file__).resolve().parents[2]
GF = ROOT / "repro_1p7b/graph_frontier"
V3 = GF / "preference_generation_v3"
OUT = GF / "balanced_preference_v3" / os.environ.get("BALANCED_V3_MILESTONE", "milestone_24")
SCHEMA = "balanced_preference_v3_source_v1"
HISTORICAL = (
    V3 / "balanced_effect_pilot_192_v1",
    V3 / "quick_effect_pilot_50_seed20260923",
)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(stable(value).encode()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values))
    temporary.replace(path)


def normalized_query(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    return " ".join(str(prompt[-1]["content"]).split()).casefold()


def exact_task_fingerprint(row: dict[str, Any]) -> str:
    graph = row["extra_info"]["graph_frontier"]
    return digest({"query": normalized_query(row),
                   "initial_scenario": graph.get("initial_scenario", "unknown"),
                   "gold_tool_sequence": graph.get("gold_tool_sequence", "unknown")})


def sealed_eval() -> tuple[set[str], set[str], set[str]]:
    task_ids: set[str] = set()
    query_hashes: set[str] = set()
    task_fingerprints: set[str] = set()
    for base in HISTORICAL:
        manifest = read(base / "quick_effect_manifest.json")
        ids = set(manifest["task_ids"]["heldout_eval"])
        task_ids.update(ids)
        for row in rows(base / "heldout_eval/source_tasks.jsonl"):
            if row["task_id"] not in ids:
                raise ValueError("historical heldout source identity changed")
            query_hashes.add(digest(normalized_query(row)))
            task_fingerprints.add(exact_task_fingerprint(row))
    return task_ids, query_hashes, task_fingerprints


def source_rows(snapshot: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = read(snapshot / "audit/validation_report.json")
    if (snapshot / "audit_status").read_text().strip() != "AUDIT_DONE":
        raise ValueError("Rich-v3 source audit not complete")
    # A bounded pilot snapshot may be formally audited yet remain below the
    # full 500-edge release gate.  Do not relabel it as a released dataset;
    # only the replay/leakage/overlap invariants are prerequisites here.
    if audit["mode"] not in {"smoke", "formal"} or audit["verdict"] not in {
            "SMOKE_READY", "GRAPH_FRONTIER_RICH_POOL_READY", "PARTIAL-STRUCTURAL-DATA-READY"}:
        raise ValueError("source structural audit failed")
    if audit["funnel"]["final_valid_tasks"] != audit["funnel"]["replay_pass"]:
        raise ValueError("source chosen replay rate below 100%")
    if audit["frozen300"]["exact_overlap"] != 0:
        raise ValueError("Frozen300 source overlap")
    path = snapshot / "valid/graph_frontier_rich_train_v1.jsonl"
    gold = snapshot / "valid/graph_frontier_rich_gold_v1.jsonl"
    if file_sha256(path) != audit["artifacts"]["valid_data_sha256"] or file_sha256(gold) != audit["artifacts"]["valid_gold_sha256"]:
        raise ValueError("source data/gold hashes changed")
    values = rows(path)
    if len(values) != audit["funnel"]["final_valid_tasks"] or len({r["task_id"] for r in values}) != len(values):
        raise ValueError("source task count/identity mismatch")
    eval_ids, eval_queries, eval_fingerprints = sealed_eval()
    retained = []
    exclusion = Counter()
    for row in values:
        if row.get("frozen_300_overlap") is not False:
            raise ValueError("source row has nonzero Frozen300 flag")
        if row["task_id"] in eval_ids:
            exclusion["historical_eval_task_id"] += 1
            continue
        if digest(normalized_query(row)) in eval_queries:
            exclusion["historical_eval_normalized_query"] += 1
            continue
        if exact_task_fingerprint(row) in eval_fingerprints:
            exclusion["historical_eval_exact_scenario_chain_query"] += 1
            continue
        retained.append(row)
    return retained, {"source_valid_tasks": len(values), "retained_tasks": len(retained),
                      "excluded": dict(exclusion), "source_data_sha256": file_sha256(path),
                      "source_gold_sha256": file_sha256(gold), "sealed_eval_task_ids": len(eval_ids),
                      "sealed_eval_query_hashes": len(eval_queries),
                      "sealed_eval_exact_fingerprints": len(eval_fingerprints),
                      "frozen300_exact_overlap": 0}


def historical_states() -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for base in HISTORICAL:
        for split in ("train", "preference_val"):
            path = base / split / "frontier_states.jsonl"
            if not path.exists():
                continue
            for state in rows(path):
                key = (state["task_id"], state["edge_id"])
                if key in seen:
                    continue
                seen.add(key)
                sample = base / split / "sampled_actions" / f"{state['state_id']}.json"
                item = copy.deepcopy(state)
                item["historical_sample_path"] = str(sample) if sample.exists() else None
                by_task[state["task_id"]].append(item)
    return by_task


def tools_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    from agent_system.environments.env_package.envfactory.official_envs import EnvFactoryBatchEnv

    os.environ.setdefault("ENVFACTORY_ROOT", str(ROOT))
    env = EnvFactoryBatchEnv(20260926, 1, 1, False, EnvConfig(8))
    try:
        env.reset([env_item(row)])
        return copy.deepcopy(env.states[0]["tools"])
    finally:
        env.close()


def frontier_identity(state: dict[str, Any]) -> str:
    identity = {"task_id": state["task_id"], "position": state["producer_position"],
                "gold_edge": state["edge_id"], "environment_state": state["state_signature"],
                "conversation_prefix": digest(state["prompt_messages"])}
    return "frontier-v3-" + digest(identity)[:24]


def stop_state(row: dict[str, Any], snapshot: Path, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    tid = row["task_id"]
    audit = read(snapshot / "replay/tasks" / f"{tid}.replay.json")
    if not (audit["replay_ok"] is True and audit["final_verifier_computed"] is True and
            audit["final_state_match"] is True):
        return None
    trace = read(snapshot / "replay/traces" / f"{tid}.rollout.json")
    if trace["task_id"] != tid or not trace["events"] or not all(e["execution_success"] is True for e in trace["events"]):
        raise ValueError(f"strict completed trace invalid: {tid}")
    prompt = row["prompt"]
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    messages = copy.deepcopy(prompt)
    for index, event in enumerate(trace["events"]):
        call_id = f"gold_stop_{index}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": event["tool_name"], "arguments": event["tool_arguments"]}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "name": event["tool_name"],
                         "content": stable(event["tool_response"])})
    last = trace["events"][-1]
    identity = {"task_id": tid, "execution_position": len(trace["events"]),
                "gold_node": last["tool_name"], "environment_state": digest(trace["final_environment_state"]),
                "conversation_prefix": digest(messages)}
    signature = digest({"prompt": messages, "state": trace["final_environment_state"], "tools": tools})
    return {"schema_version": SCHEMA, "task_id": tid, "state_type": "stop_required",
            "state_id": "stop-v3-" + digest(identity)[:24], "state_identity": identity,
            "state_signature": signature,
            "prompt_messages": messages, "tools": tools,
            "gold_tool_names": [e["tool_name"] for e in trace["events"]],
            "last_gold_action": {"name": last["tool_name"], "arguments": last["tool_arguments"]},
            "final_state_verifier_pass": True, "gold_replay_success": True,
            "dependency_depth": row["extra_info"]["graph_frontier"].get("dependency_depth", "unknown"),
            "environment_identifiers": trace["environment_identifiers"],
            "frozen_300_overlap": False}


def prepare(snapshot: Path, output: Path) -> dict[str, Any]:
    if (output / "source_manifest.json").exists():
        manifest = read(output / "source_manifest.json")
        if manifest["source_snapshot_manifest_sha256"] != file_sha256(snapshot / "snapshot_manifest.json"):
            raise ValueError("existing source preparation snapshot changed")
        return manifest
    source, source_audit = source_rows(snapshot)
    previous = historical_states()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "source_lock.json"
    lock = {"snapshot_manifest_sha256": file_sha256(snapshot / "snapshot_manifest.json"),
            "source_data_sha256": source_audit["source_data_sha256"]}
    if lock_path.exists() and read(lock_path) != lock:
        raise ValueError("interrupted source preparation identity changed")
    if not lock_path.exists():
        write(lock_path, lock)
    cache = GF / "balanced_preference_v3/replay_cache"
    cache.mkdir(parents=True, exist_ok=True)
    frontier: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    reused = 0
    for task_index, row in enumerate(source, 1):
        tid = row["task_id"]
        task_states = []
        for state in previous.get(tid, []):
            if state.get("chosen_executable") is True and state["chosen_execution"]["execution_success"] is True:
                task_states.append(state)
                reused += 1
        if not task_states:
            for edge_index, edge in enumerate(row["extra_info"]["graph_frontier"].get("dependency_edges", [])):
                record = inventory_record(row, edge, edge_index)
                if record["static_eligible"] is not True:
                    continue
                key = digest([tid, record["edge_id"], record["producer_position"]])[:24]
                path = cache / f"{key}.json"
                payload = read(path) if path.exists() else None
                retry_import_error = payload is not None and payload.get("status") == "dropped" and payload.get("reason", "").startswith("ModuleNotFoundError:No module named 'agent_system'")
                if retry_import_error:
                    write(cache / "import_error_archive" / path.name, payload)
                if payload is None or retry_import_error:
                    try:
                        seed = 20260926 + int(key[:8], 16) % 1000000
                        state = replay_one(row, record, output / "replay_runtime", seed)
                        payload = {"status": "valid", "task_id": tid, "edge_id": record["edge_id"], "state": state}
                    except Exception as exc:
                        payload = {"status": "dropped", "task_id": tid, "edge_id": record["edge_id"],
                                   "reason": f"{type(exc).__name__}:{str(exc)[:240]}"}
                    write(path, payload)
                if payload["task_id"] != tid or payload["edge_id"] != record["edge_id"]:
                    raise ValueError("replay cache identity mismatch")
                if payload["status"] == "valid":
                    task_states.append(payload["state"])
                else:
                    drops[payload["reason"]] += 1
        for state in task_states:
            item = copy.deepcopy(state)
            item["source_state_id"] = state["state_id"]
            item["state_id"] = frontier_identity(state)
            item["state_type"] = "downstream_continue" if state["producer_position"] >= 1 else "continue_required"
            item["source_snapshot"] = str(snapshot)
            frontier.append(item)
        replay_audit = read(snapshot / "replay/tasks" / f"{tid}.replay.json")
        if replay_audit["replay_ok"] is True and replay_audit["final_verifier_computed"] is True and replay_audit["final_state_match"] is True:
            tool_schema = task_states[0]["tools"] if task_states else tools_for(row)
            finished = stop_state(row, snapshot, tool_schema)
            if finished:
                stops.append(finished)
        print(f"V3_SOURCE {task_index}/{len(source)} frontier={len(frontier)} strict_stop={len(stops)}", flush=True)
    if len({s["state_id"] for s in frontier + stops}) != len(frontier) + len(stops):
        raise ValueError("duplicate v3 state identity")
    write_rows(output / "frontier_states.jsonl", sorted(frontier, key=lambda s: s["state_id"]))
    write_rows(output / "stop_states.jsonl", sorted(stops, key=lambda s: s["state_id"]))
    manifest = {"schema_version": SCHEMA, "source_snapshot": str(snapshot.resolve()),
                "source_snapshot_manifest_sha256": file_sha256(snapshot / "snapshot_manifest.json"),
                **source_audit, "frontier_states": len(frontier), "historical_frontier_reused": reused,
                "stop_states": len(stops), "frontier_types": dict(Counter(s["state_type"] for s in frontier)),
                "replay_drops": dict(drops), "state_hashes": {
                    "frontier": file_sha256(output / "frontier_states.jsonl"),
                    "stop": file_sha256(output / "stop_states.jsonl")}}
    write(output / "source_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.snapshot, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

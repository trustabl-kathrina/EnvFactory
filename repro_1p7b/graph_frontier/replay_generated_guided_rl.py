"""Deterministic typed replay gate for generated Graph-Frontier RL rows."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def task_rows(source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("train.json", "rl_val.json"):
        for row in json.loads((source / name).read_text()):
            result[str(row["task_id"])] = row
    return result


def raw_call(manager: Any, client_id: str, tool_name: str, arguments: Mapping[str, Any]) -> Any:
    client, _ = manager.get_client(client_id)
    server = client_id.split("-", 1)[0]
    short = tool_name.split("-", 1)[-1]

    async def execute() -> Any:
        if server in manager.stateless_clients:
            async with manager._stateless_lock:
                return await client.call_tool(short, dict(arguments))
        async with client:
            return await client.call_tool(short, dict(arguments))

    return asyncio.run_coroutine_threadsafe(execute(), manager._loop).result(timeout=60)


def saved_state(manager: Any, client_ids: list[str]) -> dict[str, Any]:
    values = manager.save_all_scenario(client_ids)
    if any(value is None for value in values.values()):
        raise RuntimeError("save_scenario_returned_none")
    return values


def replay_one(row: Mapping[str, Any], output: Path, manager: Any) -> dict[str, Any]:
    from repro_1p7b.graph_frontier.fastmcp_adapter import adapt_fastmcp_result
    from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace
    from repro_1p7b.graph_frontier.rollout_trace import TypedRolloutRecorder
    from repro_1p7b.graph_frontier.state_verifier import compare_final_states

    task_id = str(row["task_id"])
    sidecar = row["extra_info"]["graph_frontier"]
    factory = row["extra_info"]["mcp_factory_kwargs"]
    servers = decode(factory["mcp_servers"])
    initial = decode(factory["initial_config"])
    expected = decode(factory["final_config"])
    ground_truth = decode(row["reward_model"]["ground_truth"])
    if not isinstance(servers, list) or not isinstance(initial, Mapping):
        raise RuntimeError("invalid_environment_config")
    if not isinstance(expected, Mapping) or not isinstance(ground_truth, list):
        raise RuntimeError("invalid_reference_config")

    suffix = hashlib.sha256(task_id.encode()).hexdigest()[:16]
    client_ids = {server: f"{server}-replay-{suffix}" for server in servers}
    recorder = TypedRolloutRecorder(task_id, servers, initial, auto_timestamp=True)
    result: dict[str, Any] = {
        "task_id": task_id,
        "status": "REPLAY_FAILED",
        "replay_ok": False,
        "reset_ok": False,
        "tool_calls": len(ground_truth),
        "typed_success_calls": 0,
        "dependency_inspectable": False,
        "final_verifier_computed": False,
        "final_state_match": "unknown",
        "reasons": [],
    }
    try:
        for server in servers:
            if server not in initial:
                raise RuntimeError(f"initial_config_missing_server:{server}")
            manager.load_scenario(client_ids[server], initial[server], check=True)
        result["reset_ok"] = True
        for index, call in enumerate(ground_truth):
            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise RuntimeError(f"malformed_gold_call:{index}")
            server = name.split("-", 1)[0]
            if server not in client_ids:
                raise RuntimeError(f"gold_call_server_not_loaded:{server}")
            before = saved_state(manager, list(client_ids.values()))
            raw = raw_call(manager, client_ids[server], name, arguments)
            fields, success = adapt_fastmcp_result(raw)
            after = saved_state(manager, list(client_ids.values()))
            event = recorder.record_tool_event(
                index,
                name,
                dict(arguments),
                tool_response=fields,
                returned_fields=fields,
                execution_success=success,
                environment_id=client_ids[server],
                server_id=server,
            )
            event["state_before"] = before
            event["state_after"] = after
            if success is not True:
                raise RuntimeError(f"typed_tool_failure:{index}:{name}")
            result["typed_success_calls"] += 1

        final_state = saved_state(manager, list(client_ids.values()))
        verifier = compare_final_states(final_state, expected)
        result["final_verifier_computed"] = isinstance(verifier, bool)
        result["final_state_match"] = verifier
        recorder.finalize(
            "success",
            final_state,
            verifier,
            {"source": "deterministic_gold_replay"},
        )
        trace = recorder.as_dict()
        trace_path = output / "traces" / f"{task_id}.rollout.json"
        recorder.write(trace_path)
        profile = profile_sidecar_and_trace(sidecar, trace)
        dependency_checks = profile.get("dependency_edge_checks", []) or []
        result["dependency_inspectable"] = all(
            isinstance(item.get("success"), bool) for item in dependency_checks
        )
        result["profile_summary"] = {
            "path_adherence": profile.get("path_adherence", "unknown"),
            "dependency_edges": len(dependency_checks),
            "successful_edges": sum(item.get("success") is True for item in dependency_checks),
            "first_broken_edge": profile.get("first_broken_edge", "unknown"),
        }
        result["trace_path"] = str(trace_path)
        if not result["dependency_inspectable"]:
            raise RuntimeError("dependency_edges_not_inspectable")
        result["status"] = "REPLAY_PASS"
        result["replay_ok"] = True
    except Exception as exc:
        result["reasons"].append(f"{type(exc).__name__}:{exc}")
        try:
            recorder.finalize("failure", verifier_result="unknown")
            trace_path = output / "traces" / f"{task_id}.rollout.json"
            recorder.write(trace_path)
            result["trace_path"] = str(trace_path)
        except Exception:
            pass
    finally:
        for client_id in client_ids.values():
            try:
                manager.close_client(client_id)
            except Exception:
                pass
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["MCP_CONFIG_PATH"] = str(args.registry.resolve())
    from src.manager.mcp_client_manager import MCPManager

    rows = task_rows(args.source_dir)
    ledger = load_jsonl(args.audit_ledger)
    static_ids = [
        str(row["task_id"]) for row in ledger if row.get("status") == "STATIC_PASS"
    ]
    missing = sorted(set(static_ids) - set(rows))
    if missing:
        raise RuntimeError(f"static-pass tasks missing converted rows: {missing[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, task_id in enumerate(static_ids, 1):
        result_path = args.output_dir / "tasks" / f"{task_id}.replay.json"
        if result_path.exists() and args.resume:
            result = json.loads(result_path.read_text())
        else:
            result = replay_one(rows[task_id], args.output_dir, MCPManager)
            write_json(result_path, result)
        results.append(result)
        if index % 25 == 0 or index == len(static_ids):
            print(
                f"replay {index}/{len(static_ids)} "
                f"pass={sum(item['replay_ok'] for item in results)}",
                flush=True,
            )

    by_id = {row["task_id"]: row for row in results}
    replayed_ledger = []
    for source_row in ledger:
        row = dict(source_row)
        replay = by_id.get(str(row["task_id"]))
        if replay:
            row["executable_replay_ok"] = replay["replay_ok"]
            row["status"] = replay["status"]
            row["replay_reasons"] = replay["reasons"]
        replayed_ledger.append(row)
    write_jsonl(args.output_dir / "rl_data_v1_audit_replayed.jsonl", replayed_ledger)

    reason_counts: Counter[str] = Counter()
    for result in results:
        for reason in result["reasons"]:
            reason_counts[reason.split(":", 1)[0]] += 1
    summary = {
        "schema_version": "graph_frontier_rl_v1_executable_replay_report",
        "candidate_tasks": len(static_ids),
        "replay_pass": sum(row["replay_ok"] for row in results),
        "replay_failed": sum(not row["replay_ok"] for row in results),
        "reset_pass": sum(row["reset_ok"] for row in results),
        "typed_success_calls": sum(row["typed_success_calls"] for row in results),
        "dependency_inspectable": sum(row["dependency_inspectable"] for row in results),
        "final_verifier_computed": sum(row["final_verifier_computed"] for row in results),
        "final_state_match": sum(row["final_state_match"] is True for row in results),
        "failure_reasons": dict(reason_counts),
        "results": results,
    }
    write_json(args.output_dir / "executable_replay_report.json", summary)
    (args.output_dir / "status").write_text(
        ("REPLAY_COMPLETE" if results else "REPLAY_EMPTY") + "\n"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--audit-ledger", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--registry", type=Path, default=Path("configs/mcp_server.json")
    )
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))

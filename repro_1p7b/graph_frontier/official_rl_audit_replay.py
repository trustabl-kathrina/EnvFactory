"""CPU-only MCP schema and gold replay audit for official EnvFactory-RL.

Every row uses a fresh isolated MCP client ID; results checkpoint per row.
No model inference, training, or task generation.
"""
from __future__ import annotations

import argparse
import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq

from repro_1p7b.graph_frontier.official_rl_audit_static import (
    OUT, REGISTRY, SOURCE, decode, norm_query, query_from_row, stable,
)
from repro_1p7b.graph_frontier.fastmcp_adapter import adapt_fastmcp_result
from repro_1p7b.graph_frontier.state_verifier import compare_final_states


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def schema_audit(raw: list[dict], static: list[dict], manager: Any) -> list[dict]:
    results = []
    for item in static:
        index = item["row_index"]
        factory = raw[index]["extra_info"]["mcp_factory_kwargs"]
        servers = decode(factory["mcp_servers"])
        gold = decode(raw[index]["reward_model"]["ground_truth"])
        missing_tools, schema_errors = [], []
        for call_index, call in enumerate(gold):
            name = call.get("name")
            schema = manager.tools.get(name)
            if name is None or name.split("-", 1)[0] not in servers or schema is None:
                missing_tools.append({"call_index": call_index, "tool_name": name})
                continue
            try:
                jsonschema.validate(call.get("arguments"), schema["function"]["parameters"])
            except (jsonschema.ValidationError, jsonschema.SchemaError, TypeError, ValueError) as exc:
                schema_errors.append({"call_index": call_index, "tool_name": name,
                                      "error_type": type(exc).__name__,
                                      "message": str(exc).splitlines()[0][:180]})
        results.append({
            "row_index": index, "audit_id": item["audit_id"],
            "registered_servers": all(s in manager.server_to_path_mapping for s in servers),
            "missing_tool_count": len(missing_tools),
            "argument_schema_error_count": len(schema_errors),
            "tool_missing_json": stable(missing_tools),
            "schema_errors_json": stable(schema_errors),
            "schema_ok": not missing_tools and not schema_errors,
        })
    return results


def scalar_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        result = []
        for key, child in value.items():
            result.extend(scalar_paths(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(scalar_paths(child, f"{prefix}[{index}]"))
        return result
    if isinstance(value, str) and len(value) >= 4:
        return [(prefix, value)]
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 10:
        return [(prefix, value)]
    return []


def strict_observed_flows(gold: list[dict], observations: list[Any],
                          query: str, initial: dict) -> list[dict]:
    """Unique typed producer value absent from user/config/prior args; conservative proxy."""
    initial_values = {stable(v) for _, v in scalar_paths(initial)}
    query_text = norm_query(query)
    flows = []
    for consumer_index in range(1, min(len(gold), len(observations))):
        consumer = gold[consumer_index]
        for arg_path, arg_value in scalar_paths(consumer["arguments"]):
            encoded = stable(arg_value)
            if encoded in initial_values or norm_query(arg_value) in query_text:
                continue
            producers = []
            for producer_index in range(consumer_index):
                for output_path, output_value in scalar_paths(observations[producer_index]):
                    if type(output_value) is type(arg_value) and output_value == arg_value:
                        producers.append((producer_index, output_path))
            if len(producers) != 1:
                continue
            producer_index, output_path = producers[0]
            prior_arg_values = {
                stable(value) for earlier in gold[:consumer_index]
                for _, value in scalar_paths(earlier["arguments"])
            }
            if encoded in prior_arg_values:
                continue
            flows.append({
                "producer_call_index": producer_index,
                "producer_tool": gold[producer_index]["name"],
                "producer_output_path": output_path,
                "consumer_call_index": consumer_index,
                "consumer_tool": consumer["name"],
                "consumer_input_path": arg_path,
                "value_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            })
    return flows


def flow_depth(gold: list[dict], flows: list[dict]) -> int:
    depths = [0] * len(gold)
    for flow in flows:
        producer = flow["producer_call_index"]
        consumer = flow["consumer_call_index"]
        depths[consumer] = max(depths[consumer], depths[producer] + 1)
    return max(depths, default=0)


def raw_call(manager: Any, client_id: str, name: str, args: dict) -> Any:
    client, _ = manager.get_client(client_id)
    server = client_id.split("-", 1)[0]
    short_name = name.split("-", 1)[1]
    async def execute() -> Any:
        if server in manager.stateless_clients:
            async with manager._stateless_lock:
                return await client.call_tool(short_name, args)
        async with client:
            return await client.call_tool(short_name, args)
    return asyncio.run_coroutine_threadsafe(execute(), manager._loop).result(timeout=60)


def saved(manager: Any, client_ids: dict[str, str]) -> dict:
    result = manager.save_all_scenario(list(client_ids.values()))
    if any(value is None for value in result.values()):
        raise RuntimeError("save_scenario_failed")
    return result


def replay_one(index: int, raw_row: dict, static: dict, schema: dict,
               manager: Any) -> dict:
    started = time.monotonic()
    factory = raw_row["extra_info"]["mcp_factory_kwargs"]
    servers = decode(factory["mcp_servers"])
    initial = decode(factory["initial_config"])
    expected = decode(factory["final_config"])
    gold = decode(raw_row["reward_model"]["ground_truth"])
    suffix = f"officialaudit-{index}-{os.getpid()}"
    client_ids = {server: f"{server}-{suffix}" for server in servers}
    result = {
        "row_index": index, "audit_id": static["audit_id"],
        "environment": static["environment"], "gold_length": len(gold),
        "reset_ok": False, "tool_execution_success_calls": 0,
        "tool_execution_failure": False, "schema_ok": schema["schema_ok"],
        "final_state_match": "unknown", "replay_ok": False,
        "verified_observed_value_flow_count": 0, "verified_observed_dependency_depth": "unknown",
        "observed_flow_edges": [], "error_type": None, "error_detail": None,
    }
    if not schema["schema_ok"] or not static["static_gold_ok"] or not static["static_config_ok"]:
        result["error_type"] = "static_gate_failed"
        result["error_detail"] = (static["errors_json"] + " " +
                                  schema["tool_missing_json"] + " " +
                                  schema["schema_errors_json"])[:400]
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    try:
        for server in servers:
            manager.load_scenario(client_ids[server], initial[server], check=True)
        state = saved(manager, client_ids)
        if stable(state) != stable({server: initial[server] for server in servers}):
            raise RuntimeError("reset_state_mismatch")
        result["reset_ok"] = True
        observations = []
        for call_index, call in enumerate(gold):
            name, args = call["name"], call["arguments"]
            server = name.split("-", 1)[0]
            response = raw_call(manager, client_ids[server], name, args)
            fields, success = adapt_fastmcp_result(response)
            if success is not True:
                result["tool_execution_failure"] = True
                raise RuntimeError(f"typed_tool_failure:{call_index}:{name}")
            result["tool_execution_success_calls"] += 1
            observations.append(fields)
        final_state = saved(manager, client_ids)
        result["final_state_match"] = compare_final_states(
            final_state, {server: expected[server] for server in servers})
        flows = strict_observed_flows(gold, observations, query_from_row(raw_row), initial)
        result["observed_flow_edges"] = flows
        result["verified_observed_value_flow_count"] = len(flows)
        result["verified_observed_dependency_depth"] = flow_depth(gold, flows)
        if result["final_state_match"] is not True:
            raise RuntimeError("final_state_mismatch")
        result["replay_ok"] = True
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_detail"] = str(exc)[:400]
    finally:
        for client_id in client_ids.values():
            try:
                manager.close_client(client_id)
            except Exception:
                pass
    result["elapsed_seconds"] = time.monotonic() - started
    return result

def ordered_indices(static: list[dict], mode: str) -> tuple[list[int], int]:
    strata: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for row in static:
        bucket = min(row["gold_length"], 6)
        strata[(row["environment"], bucket)].append(row)
    selected, remaining = [], []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda item: hashlib.sha256(
            item["audit_id"].encode()).hexdigest())
        take = math.ceil(len(group) * .3)
        selected.append(group[:take])
        remaining.append(group[take:])
    def round_robin(groups: list[list[dict]]) -> list[int]:
        return [group[level]["row_index"]
                for level in range(max(map(len, groups), default=0))
                for group in groups if level < len(group)]
    first = round_robin(selected)
    rest = round_robin(remaining)
    return (first if mode == "stratified" else first + rest), len(first)


def write_replay_summary(static: list[dict], schema: list[dict],
                         ordered: list[int], target_count: int,
                         stratified_count: int) -> dict:
    tasks = OUT / "replay_tasks"
    results = []
    for index in ordered[:target_count]:
        path = tasks / f"{index:04d}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    if results:
        parquet_rows = []
        for row in results:
            encoded = dict(row)
            encoded["observed_flow_edges_json"] = stable(encoded.pop("observed_flow_edges"))
            encoded["final_state_match"] = str(encoded["final_state_match"])
            encoded["verified_observed_dependency_depth"] = str(encoded["verified_observed_dependency_depth"])
            parquet_rows.append(encoded)
        pq.write_table(pa.Table.from_pylist(parquet_rows), OUT / "replay_audit.parquet")
    errors = collections.Counter(row["error_type"] for row in results if row["error_type"])
    summary = {
        "schema_rows": len(schema),
        "registered_server_rows": sum(row["registered_servers"] for row in schema),
        "tool_exists_rows": sum(row["missing_tool_count"] == 0 for row in schema),
        "arguments_schema_valid_rows": sum(row["schema_ok"] for row in schema),
        "tool_missing_rows": sum(row["missing_tool_count"] > 0 for row in schema),
        "argument_schema_fail_rows": sum(row["argument_schema_error_count"] > 0 for row in schema),
        "stratified_target_count": stratified_count,
        "selected_target_count": target_count,
        "replay_recorded_rows": len(results),
        "reset_ok_rows": sum(row["reset_ok"] for row in results),
        "gold_tool_execution_complete_rows": sum(row["tool_execution_success_calls"] == row["gold_length"]
                                                  for row in results),
        "final_state_match_rows": sum(row["final_state_match"] is True for row in results),
        "replay_pass_rows": sum(row["replay_ok"] for row in results),
        "typed_success_calls": sum(row["tool_execution_success_calls"] for row in results),
        "observed_unique_value_flow_tasks": sum(row["verified_observed_value_flow_count"] > 0
                                                for row in results),
        "observed_unique_value_flow_edges": sum(row["verified_observed_value_flow_count"]
                                                for row in results),
        "error_types": dict(errors),
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in results),
        "gate": "FULL_REPLAY_COMPLETE" if len(results) == len(static) else
                "STRATIFIED_REPLAY_COMPLETE" if len(results) >= stratified_count else
                "REPLAY_PARTIAL",
    }
    write_json(OUT / "replay_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stratified", "full"), default="stratified")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional pilot cap on rows in stratified-first order.")
    parser.add_argument("--max-seconds", type=int, default=0,
                        help="Gracefully stop after this wall time; results are resumable.")
    args = parser.parse_args()
    if not (OUT / "schema_summary.json").exists():
        raise RuntimeError("run official_rl_audit_static before executable replay")
    raw = json.loads(SOURCE.read_text())
    static = pq.read_table(OUT / "row_audit.parquet").to_pylist()
    if len(raw) != len(static) or len(static) != 3092:
        raise RuntimeError("source/static row count mismatch")
    schema_path = OUT / "schema_resolution.parquet"
    os.environ["MCP_CONFIG_PATH"] = str(REGISTRY.resolve())
    from src.manager.mcp_client_manager import MCPManager
    started = time.monotonic()
    try:
        if schema_path.exists():
            schema = pq.read_table(schema_path).to_pylist()
        else:
            schema = schema_audit(raw, static, MCPManager)
            pq.write_table(pa.Table.from_pylist(schema), schema_path)
        if len(schema) != len(static):
            raise RuntimeError("schema resolution count mismatch")
        order, stratified_count = ordered_indices(static, args.mode)
        target_count = min(len(order), args.limit) if args.limit else len(order)
        by_schema = {row["row_index"]: row for row in schema}
        for position, index in enumerate(order[:target_count], 1):
            task_path = OUT / "replay_tasks" / f"{index:04d}.json"
            if task_path.exists():
                continue
            if args.max_seconds and time.monotonic() - started > args.max_seconds:
                print(f"graceful_time_limit {position-1}/{target_count}", flush=True)
                break
            result = replay_one(index, raw[index], static[index], by_schema[index], MCPManager)
            write_json(task_path, result)
            if position % 25 == 0 or position == target_count:
                print(f"replay {position}/{target_count} row={index} "
                      f"ok={result['replay_ok']} elapsed={time.monotonic()-started:.1f}s",
                      flush=True)
        summary = write_replay_summary(static, schema, order, target_count,
                                       stratified_count)
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        MCPManager.shutdown()


if __name__ == "__main__":
    main()


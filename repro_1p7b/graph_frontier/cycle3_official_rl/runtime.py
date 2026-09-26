"""Isolated Official-RL gold replay and one Dynamic-v1 on-policy rollout/task.

No training path is imported. Each task uses fresh FastMCP subprocess clients
for gold and student separately; all connections are explicitly closed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastmcp import Client

from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action
from repro_1p7b.graph_frontier.fastmcp_adapter import adapt_fastmcp_result
from repro_1p7b.graph_frontier.official_rl_audit_static import (
    OUT, ROOT, decode, query_from_row,
)
from repro_1p7b.graph_frontier.state_verifier import compare_final_states
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import (
    RUN, checked_plan, sha256, stable,
)


def typed_equal(left: Any, right: Any) -> bool:
    """JSON identity with unordered object keys and strict scalar types."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, path)


def immutable_json(path: Path, value: Any) -> str:
    encoded = stable(value)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    path = path / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text().strip() != encoded:
            raise RuntimeError("schema hash collision or drift")
        return digest
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(encoded + "\n")
    try:
        os.link(temp, path)
    except FileExistsError:
        if path.read_text().strip() != encoded:
            raise RuntimeError("concurrent schema drift")
    finally:
        temp.unlink(missing_ok=True)
    return digest


class IsolatedEpisode:
    def __init__(self, task_id: str, servers: list[str], initial: dict):
        self.task_id = task_id
        self.servers = servers
        self.initial = initial
        self.clients: dict[str, Client] = {}
        self.schemas: list[dict] = []

    async def open(self) -> None:
        try:
            for server in self.servers:
                path = ROOT / "envs/tools" / f"{server}.py"
                if not path.is_file():
                    raise FileNotFoundError(path)
                client = Client(str(path))
                await client._connect()
                self.clients[server] = client
                result = await client.call_tool("load_scenario", {"scenario": self.initial[server]})
                _, success = adapt_fastmcp_result(result)
                if success is not True:
                    raise RuntimeError(f"load_scenario_failed:{server}")
                for tool in await client.list_tools():
                    if tool.name in ("load_scenario", "save_scenario"):
                        continue
                    self.schemas.append({
                        "type": "function",
                        "function": {
                            "name": f"{server}-{tool.name}",
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    })
            actual = await self.save()
            if not typed_equal(actual, {s: self.initial[s] for s in self.servers}):
                raise RuntimeError("initial scenario reset mismatch")
        except Exception:
            await self.close()
            raise

    async def save(self) -> dict:
        result = {}
        for server, client in self.clients.items():
            raw = await client.call_tool("save_scenario", {})
            fields, success = adapt_fastmcp_result(raw)
            if success is not True or fields == "unknown":
                raise RuntimeError(f"save_scenario_failed:{server}")
            if isinstance(fields, dict) and set(fields) == {"result"}:
                fields = fields["result"]
            if isinstance(fields, str):
                try:
                    fields = json.loads(fields)
                except json.JSONDecodeError:
                    pass
            result[server] = fields
        return result

    async def call(self, name: str, arguments: dict) -> tuple[Any, bool, dict | None]:
        server = name.split("-", 1)[0]
        allowed = {t["function"]["name"] for t in self.schemas}
        if server not in self.clients or name not in allowed:
            return "unknown", False, {"type": "ValueError", "message": f"unexpected_tool:{name}"}
        try:
            raw = await self.clients[server].call_tool(name.split("-", 1)[1], arguments)
            fields, success = adapt_fastmcp_result(raw)
            if success is not True:
                return fields, False, {"type": "ToolError", "message": "typed MCP execution failure"}
            return fields, True, None
        except Exception as exc:
            return "unknown", False, {"type": type(exc).__name__, "message": str(exc)[:400]}

    async def close(self) -> None:
        for client in self.clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self.clients.clear()


async def gold_replay(entry: dict, row: dict) -> tuple[dict, list[dict]]:
    index, task_id = entry["source_row_index"], entry["task_id"]
    lock = json.loads((OUT / "replay_tasks" / f"{index:04d}.json").read_text())
    if (lock.get("audit_id") != task_id or lock.get("replay_ok") is not True
            or lock.get("final_state_match") is not True):
        raise RuntimeError("persisted strict gold replay lock failed")
    factory = row["extra_info"]["mcp_factory_kwargs"]
    servers = decode(factory["mcp_servers"])
    initial = decode(factory["initial_config"])
    expected = decode(factory["final_config"])
    gold = decode(row["reward_model"]["ground_truth"])
    if (not isinstance(gold, list) or len(gold) != entry["gold_length"]
            or any(not isinstance(a.get("arguments"), dict) for a in gold)):
        raise RuntimeError("ordered gold actions invalid")
    episode = IsolatedEpisode(task_id, servers, initial)
    await episode.open()
    try:
        initial_observed = await episode.save()
        observations, states_after = [], []
        for step, action in enumerate(gold):
            fields, success, error = await episode.call(action["name"], action["arguments"])
            if success is not True:
                raise RuntimeError(f"gold tool failed:{step}:{action['name']}:{error}")
            observations.append(fields)
            states_after.append(await episode.save())
        final_state = await episode.save()
        if compare_final_states(final_state, {s: expected[s] for s in servers}) is not True:
            raise RuntimeError("gold final-state verifier failed")
        return {
            "replay_valid": True, "final_state_match": True,
            "actions": [{"name": a["name"], "arguments": a["arguments"]} for a in gold],
            "observations": observations, "initial_state": initial_observed,
            "states_after": states_after, "final_state": final_state,
        }, episode.schemas
    finally:
        await episode.close()


async def student_rollout(entry: dict, row: dict, endpoint: str, config: dict,
                          schemas: list[dict]) -> dict:
    factory = row["extra_info"]["mcp_factory_kwargs"]
    servers = decode(factory["mcp_servers"])
    initial = decode(factory["initial_config"])
    episode = IsolatedEpisode(entry["task_id"], servers, initial)
    await episode.open()
    started = time.monotonic()
    try:
        if not typed_equal(episode.schemas, schemas):
            raise RuntimeError("gold/student tool schema drift")
        messages = [{"role": "user", "content": query_from_row(row)}]
        steps = []
        calls = 0
        for step_index in range(config["max_turns"]):
            before_state = await episode.save()
            seed = config["seed"] + entry["source_row_index"] * 1009 + step_index
            message, action, usage, finish_reason = generate_action(
                endpoint, config["served_model"], messages, schemas,
                seed=seed, temperature=config["temperature"],
                max_tokens=config["max_tokens_per_turn"],
            )
            prompt_count = len(messages)
            messages.append(message)
            event = None
            if action["kind"] == "tool":
                if calls >= config["tool_call_limit"]:
                    event = {"tool_name": action["name"], "tool_arguments": action["arguments"],
                             "tool_response": "unknown", "execution_success": False,
                             "exception": {"type": "ToolCallLimit", "message": "frozen budget exhausted"},
                             "state_before": before_state, "state_after": before_state}
                else:
                    fields, success, error = await episode.call(action["name"], action["arguments"])
                    after_state = await episode.save()
                    event = {"tool_name": action["name"], "tool_arguments": action["arguments"],
                             "tool_response": fields, "execution_success": success,
                             "exception": error, "state_before": before_state,
                             "state_after": after_state}
                    calls += 1
                    if success:
                        call_id = message["tool_calls"][0]["id"]
                        messages.append({
                            "role": "tool", "tool_call_id": call_id, "name": action["name"],
                            "content": stable(fields),
                        })
            steps.append({
                "step_index": step_index, "prompt_message_count": prompt_count,
                "state_before_action": before_state, "model_message": message,
                "parsed_action": action, "typed_event": event,
                "usage": usage, "finish_reason": finish_reason,
            })
            if action["kind"] != "tool" or event is None or event["execution_success"] is not True:
                break
        final_state = await episode.save()
        return {
            "status": "ROLLOUT_COMPLETE", "model_source": "original_dynamic_v1",
            "messages": messages, "steps": steps, "final_state": final_state,
            "turn_budget_exhausted": len(steps) == config["max_turns"]
                                     and steps[-1]["parsed_action"]["kind"] == "tool",
            "runtime_seconds": time.monotonic() - started,
        }
    finally:
        await episode.close()


async def process_task(entry: dict, row: dict, endpoint: str, config: dict,
                       run: Path, plan_sha: str) -> dict:
    started = time.monotonic()
    payload = {
        "schema_version": "cycle3_official_rl_episode_v1",
        "task_id": entry["task_id"], "source_row_index": entry["source_row_index"],
        "environment": entry["environment"], "query_hash": entry["query_hash"],
        "plan_sha256": plan_sha, "gold": None, "student": None,
        "gold_error": None, "student_error": None,
    }
    try:
        gold, schemas = await gold_replay(entry, row)
        payload["gold"] = gold
        payload["schema_sha256"] = immutable_json(run / "schemas", schemas)
    except Exception as exc:
        payload["gold_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        payload["runtime_seconds"] = time.monotonic() - started
        return payload
    try:
        payload["student"] = await student_rollout(entry, row, endpoint, config, schemas)
    except Exception as exc:
        payload["student_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
    payload["runtime_seconds"] = time.monotonic() - started
    return payload


def worker(plan_path: Path, endpoint: str, shard_index: int, shard_count: int,
           limit: int | None = None) -> dict:
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    plan, raw = checked_plan(plan_path)
    run = plan_path.parent
    plan_sha = sha256(plan_path)
    tasks = [(i, e) for i, e in enumerate(plan["tasks"]) if i % shard_count == shard_index]
    if limit is not None:
        tasks = tasks[:limit]
    counts = {"completed": 0, "skipped": 0, "gold_failed": 0, "student_failed": 0}
    started = time.monotonic()
    for position, entry in tasks:
        path = run / "episodes" / f"{entry['task_id']}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if (old.get("plan_sha256") != plan_sha
                    or old.get("task_id") != entry["task_id"]
                    or old.get("source_row_index") != entry["source_row_index"]):
                raise RuntimeError(f"persisted task identity drift: {entry['task_id']}")
            counts["skipped"] += 1
            continue
        payload = asyncio.run(process_task(
            entry, raw[entry["source_row_index"]], endpoint, plan["config"], run, plan_sha,
        ))
        atomic_json(path, payload)
        counts["completed"] += 1
        counts["gold_failed"] += payload["gold"] is None
        counts["student_failed"] += payload["student"] is None
        if counts["completed"] % 10 == 0 or payload["student"] is None:
            print(json.dumps({"shard": shard_index, "position": position,
                              "task_id": entry["task_id"], "counts": counts,
                              "elapsed_seconds": round(time.monotonic() - started, 1)},
                             sort_keys=True), flush=True)
    result = {"shard": shard_index, "shard_count": shard_count,
              "selected_tasks": len(tasks), "counts": counts,
              "elapsed_seconds": time.monotonic() - started}
    atomic_json(run / f"worker_{shard_index}_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=RUN / "rollout_manifest.json")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(worker(args.plan, args.endpoint, args.shard_index,
                            args.shard_count, args.limit), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

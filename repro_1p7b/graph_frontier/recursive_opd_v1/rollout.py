"""Full, resumable current-policy trajectories; analysis never truncates rollout.

Each task is isolated in a fresh MCP environment. Concurrent structured calls
are executed together, with every typed result returned to the model. A tool
error is observed by the model; it is not a collector stop condition.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action
from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode, atomic_json, immutable_json, typed_equal
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, decode, query_from_row
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import CONFIG, PROTOCOL, RUN, check

GOLD_RUN = RUN.parent / "cycle3_official_rl/run_dynamic_v1_t0"


def parse_calls(message: dict) -> tuple[list[dict], list[dict]]:
    calls, errors = [], []
    for index, raw in enumerate(message.get("tool_calls") or []):
        try:
            func = raw.get("function") or {}
            args = func.get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(func.get("name"), str) or not isinstance(args, dict):
                raise ValueError("non-typed tool call")
            calls.append({"position": index, "id": str(raw.get("id") or f"call_{index}"),
                          "name": func["name"], "arguments": args})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"position": index, "id": str(raw.get("id") or f"call_{index}"),
                           "type": type(exc).__name__, "message": str(exc)[:200]})
    return calls, errors


async def collect_one(entry: dict, row: dict, model_name: str, endpoint: str,
                      gold_episode: dict, schemas: list[dict]) -> dict:
    factory = row["extra_info"]["mcp_factory_kwargs"]
    env = IsolatedEpisode(entry["audit_id"], decode(factory["mcp_servers"]),
                          decode(factory["initial_config"]))
    await env.open()
    started = time.monotonic()
    try:
        if not typed_equal(env.schemas, schemas):
            raise RuntimeError("gold/student schema drift")
        messages = [{"role": "user", "content": query_from_row(row)}]
        steps, executed = [], 0
        termination = "MAX_TURNS"
        for index in range(CONFIG["max_turns"]):
            before = await env.save()
            seed = CONFIG["seed"] + entry["source_row_index"] * 1009 + index
            message, parsed, usage, finish_reason = generate_action(
                endpoint, model_name, messages, schemas, seed=seed,
                temperature=CONFIG["temperature"], max_tokens=CONFIG["max_tokens_per_turn"])
            prompt_count = len(messages)
            messages.append(message)
            calls, parse_errors = parse_calls(message)
            results = []
            if not (message.get("tool_calls") or []):
                termination = "MODEL_FINAL" if finish_reason != "length" else "GENERATION_LENGTH"
            else:
                async def execute(call: dict) -> dict:
                    if executed + call["position"] >= CONFIG["tool_call_limit"]:
                        return {"tool_name": call["name"], "tool_arguments": call["arguments"],
                                "tool_response": "unknown", "execution_success": False,
                                "exception": {"type": "ToolCallLimit", "message": "fixed call budget reached"}}
                    response, success, exception = await env.call(call["name"], call["arguments"])
                    return {"tool_name": call["name"], "tool_arguments": call["arguments"],
                            "tool_response": response, "execution_success": success,
                            "exception": exception}
                results = await asyncio.gather(*(execute(call) for call in calls))
                executed += len(calls)
                by_position = {call["position"]: result for call, result in zip(calls, results)}
                bad = {error["position"]: error for error in parse_errors}
                for position, raw in enumerate(message["tool_calls"]):
                    if position in by_position:
                        value = (by_position[position]["tool_response"] if by_position[position]["execution_success"]
                                 else {"error": by_position[position]["exception"]})
                        call_name = calls[next(i for i, c in enumerate(calls) if c["position"] == position)]["name"]
                    else:
                        value = {"error": bad[position]}
                        call_name = str((raw.get("function") or {}).get("name") or "unknown")
                    messages.append({"role": "tool", "tool_call_id": str(raw.get("id") or f"call_{position}"),
                                     "name": call_name, "content": stable(value)})
            after = await env.save()
            steps.append({"step_index": index, "prompt_message_count": prompt_count,
                          "state_before_action": before, "raw_assistant_output": message,
                          "model_message": message, "parsed_action": parsed,
                          "structured_calls": calls, "parse_errors": parse_errors,
                          "typed_events": results, "state_after_action": after,
                          "usage": usage, "finish_reason": finish_reason})
            if termination != "MAX_TURNS":
                break
        return {"status": "ROLLOUT_COMPLETE", "model_source": model_name,
                "messages": messages, "steps": steps, "final_state": await env.save(),
                "termination_reason": termination,
                "turn_budget_exhausted": termination == "MAX_TURNS",
                "runtime_seconds": time.monotonic() - started}
    finally:
        await env.close()


def worker(*, cycle: int, model: Path, endpoint: str, shard: int, shards: int,
           limit: int | None = None) -> dict:
    if cycle not in range(4) or shards != 2 or shard not in (0, 1):
        raise ValueError("expected cycles 0..3 and two shards")
    protocol = check()
    if cycle == 0 and model.resolve() != Path(protocol["pi0_model"]).resolve():
        raise RuntimeError("pi0 must be original Dynamic-v1")
    model_hash = sha256(model / "model.safetensors")
    if cycle == 0 and model_hash != protocol["pi0_model_sha256"]:
        raise RuntimeError("pi0 weight drift")
    run = RUN / f"cycle{cycle}"
    manifest = run / "rollout_manifest.json"
    if not manifest.exists():
        raise RuntimeError("cycle rollout manifest must be pre-created")
    lock = json.loads(manifest.read_text())
    if (lock.get("schema_version") != "recursive_opd_rollout_v1"
            or lock.get("cycle") != cycle or lock.get("model_sha256") != model_hash
            or lock.get("model_path") != str(model)
            or lock.get("protocol_sha256") != sha256(PROTOCOL)
            or lock.get("config") != CONFIG):
        raise RuntimeError("rollout manifest/model/protocol drift")
    raw = json.loads(SOURCE.read_text())
    by_id = {e["audit_id"]: e for e in json.loads((RUN.parent / "official_rl_audit/opd_source_manifest.json").read_text())["rows"]}
    task_ids = protocol["train_task_ids"] + protocol["heldout_task_ids"]
    selected = [(position, by_id[task]) for position, task in enumerate(task_ids) if position % shards == shard]
    if limit is not None:
        selected = selected[:limit]
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for _, entry in selected:
        task = entry["audit_id"]
        path = run / "rollouts" / f"{task}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if (old.get("task_id") != task or old.get("manifest_sha256") != sha256(manifest)
                    or old.get("model_sha256") != model_hash):
                raise RuntimeError(f"persisted rollout identity drift: {task}")
            counts["skipped"] += 1
            continue
        source = raw[entry["source_row_index"]]
        gold_source = GOLD_RUN / "episodes" / f"{task}.json"
        gold_episode = json.loads(gold_source.read_text())
        gold_source_kind = "historical"
        if not valid_gold(entry, source, gold_episode):
            cache = RUN / "gold_cache" / f"{task}.json"
            if not cache.exists():
                raise RuntimeError(f"authoritative gold replay invalid and no verified cache: {task}")
            restored = json.loads(cache.read_text())
            if restored.get("task_id") != task or not isinstance(restored.get("schemas"), list):
                raise RuntimeError(f"independent gold cache identity invalid: {task}")
            if gold_episode.get("schema_sha256") and not typed_equal(
                    restored["schemas"], schema_for(GOLD_RUN, gold_episode)):
                raise RuntimeError(f"independent gold cache/schema drift: {task}")
            gold_episode = {**gold_episode, "gold": restored["gold"]}
            gold_source, gold_source_kind = cache, "independent_replay_cache"
            if not valid_gold(entry, source, gold_episode):
                raise RuntimeError(f"independent cached gold replay invalid: {task}")
        schemas = (restored["schemas"] if gold_source_kind == "independent_replay_cache"
                   else schema_for(GOLD_RUN, gold_episode))
        schema_sha = (immutable_json(RUN / "schemas", schemas)
                      if gold_source_kind == "independent_replay_cache"
                      else gold_episode["schema_sha256"])
        payload = {"task_id": task, "source_row_index": entry["source_row_index"],
                   "pool": "train" if task in set(protocol["train_task_ids"]) else "heldout",
                   "cycle": cycle, "model_sha256": model_hash,
                   "manifest_sha256": sha256(manifest),
                   "gold_source_episode_sha256": sha256(gold_source),
                   "gold_source_kind": gold_source_kind,
                   "schema_sha256": schema_sha,
                   "gold": gold_episode["gold"], "student": None, "student_error": None}
        try:
            payload["student"] = asyncio.run(collect_one(entry, source, lock["served_model"], endpoint,
                                                           gold_episode, schemas))
        except Exception as exc:
            payload["student_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            counts["failed"] += 1
        atomic_json(path, payload)
        counts["completed"] += 1
        if counts["completed"] % 10 == 0 or payload["student_error"]:
            print(stable({"task_id": task, "shard": shard, "counts": counts}), flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(stable(worker(cycle=args.cycle, model=args.model, endpoint=args.endpoint,
                        shard=args.shard, shards=2, limit=args.limit)), flush=True)


if __name__ == "__main__":
    main()

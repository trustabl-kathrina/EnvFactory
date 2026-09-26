"""Collect Dynamic-v1 executable rollouts for preference construction.

Tool calls are consumed only from the OpenAI-compatible server's structured
``message.tool_calls`` field.  Raw assistant text is never regex-parsed back
into a call.  Every tool action is executed by the existing isolated
EnvFactory adapter, which records typed arguments, responses, success and
state snapshots.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import requests

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "repro_1p7b/data/graph_frontier_rl_v1_frozen_seed20260920"
EXPECTED_TRAIN_SHA256 = "1faabac69216efd36b4bb02822cec641fd623a6fefa5d581aa123fb441e66eeb"
EXPECTED_FROZEN300_SHA256 = "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class EnvConfig(dict):
    def __init__(self, max_steps: int) -> None:
        super().__init__(max_steps=max_steps, max_tool_calls=max_steps)
        self.max_steps = max_steps


def env_item(row: Mapping[str, Any]) -> dict[str, Any]:
    factory = row["extra_info"]["mcp_factory_kwargs"]
    sidecar = row["extra_info"]["graph_frontier"]
    item = {key: decoded(value) for key, value in factory.items()}
    item.update(
        source="EnvFactory-generated-guided-rl-v1",
        task_id=row["task_id"],
        query=decoded(row["prompt"])[-1]["content"],
        graph_frontier=sidecar,
        ground_truth=decoded(row["reward_model"]["ground_truth"]),
        frozen_300_overlap=False,
        reward_scheduler={"mode": "fix", "default_ratio": 0.5},
    )
    if item["task_id"] != sidecar["task_id"]:
        raise RuntimeError("task/sidecar mismatch")
    return item


def canonical_tool_call(call: Mapping[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    function = call.get("function") or {}
    name = function.get("name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("server returned a non-typed tool call")
    call_id = str(call.get("id") or f"call_{index}")
    message_call = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return message_call, {"kind": "tool", "name": name, "arguments": arguments}


def generate_action(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    response = requests.post(
        endpoint.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer placeholder"},
        json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "seed": seed,
            # This collector posts raw JSON rather than using the OpenAI SDK,
            # so SGLang's request field must be top-level. ``extra_body`` is
            # only an SDK convenience and would be silently ignored here.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    raw = choice["message"]
    finish_reason = str(choice.get("finish_reason") or "unknown")
    calls = raw.get("tool_calls") or []
    content = raw.get("content") or ""
    reasoning = raw.get("reasoning_content")
    if len(calls) == 1:
        message_call, action = canonical_tool_call(calls[0], 0)
        message = {"role": "assistant", "content": content, "tool_calls": [message_call]}
        if isinstance(reasoning, str) and reasoning:
            message["reasoning_content"] = reasoning
    elif len(calls) == 0:
        message = {"role": "assistant", "content": content}
        if isinstance(reasoning, str) and reasoning:
            message["reasoning_content"] = reasoning
        if finish_reason == "length":
            action = {
                "kind": "invalid",
                "raw": content,
                "reason": "generation_length_truncated",
            }
        else:
            action = {"kind": "final", "content": content, "raw": content}
    else:
        message = {"role": "assistant", "content": content, "tool_calls": calls}
        action = {"kind": "invalid", "raw": stable(raw), "reason": "parallel_tool_calls"}
    usage = payload.get("usage") or {}
    return message, action, usage, finish_reason


def collect_one(
    row: Mapping[str, Any],
    *,
    endpoint: str,
    model: str,
    rollout_index: int,
    seed: int,
    max_steps: int,
    temperature: float,
    max_tokens: int,
    output: Path,
) -> dict[str, Any]:
    from agent_system.environments.env_package.envfactory.official_envs import EnvFactoryBatchEnv

    task_id = str(row["task_id"])
    path = output / f"{task_id}.r{rollout_index}.json"
    if path.exists():
        return json.loads(path.read_text())
    item = env_item(row)
    env = EnvFactoryBatchEnv(seed, 1, 1, False, EnvConfig(max_steps))
    started = time.monotonic()
    try:
        env.reset([item])
        state = env.states[0]
        tools = copy.deepcopy(state["tools"])
        messages: list[dict[str, Any]] = [{"role": "user", "content": item["query"]}]
        steps: list[dict[str, Any]] = []
        final_info: dict[str, Any] = {}
        for step_index in range(max_steps):
            before_messages = copy.deepcopy(messages)
            before_state = env._save(state)
            message, action, usage, finish_reason = generate_action(
                endpoint,
                model,
                messages,
                tools,
                seed=seed + step_index,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            event_count = len(state["events"])
            _, _, dones, infos = env.step([action])
            event = copy.deepcopy(state["events"][-1]) if len(state["events"]) > event_count else None
            done = bool(dones[0])
            final_info = copy.deepcopy(infos[0])
            steps.append(
                {
                    "step_index": step_index,
                    "prompt_messages": before_messages,
                    "state_before_action": before_state,
                    "model_message": message,
                    "parsed_action": action,
                    "typed_event": event,
                    "done": done,
                    "usage": usage,
                    "finish_reason": finish_reason,
                }
            )
            messages.append(message)
            if event is not None and not done:
                call_id = message["tool_calls"][0]["id"]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": event["tool_name"],
                        "content": stable(event["tool_response"]),
                    }
                )
            if done:
                break
        result = {
            "schema_version": "envfactory_preference_rollout_v1",
            "task_id": task_id,
            "source_seed": row["extra_info"].get("generation_seed", "unknown"),
            "rollout_index": rollout_index,
            "sampling_seed": seed,
            "model": model,
            "model_source": "Dynamic-v1",
            "endpoint": endpoint,
            "tools": tools,
            "initial_prompt": [{"role": "user", "content": item["query"]}],
            "conversation_messages": messages,
            "steps": steps,
            "terminal_info": final_info,
            "official_reward": (final_info.get("reward_parts") or {}).get("official_reward", {}).get("score"),
            "semantic_success": bool(final_info.get("won", 0.0) >= 0.5),
            "frozen_300_overlap": False,
            "runtime_seconds": time.monotonic() - started,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="dynamic-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollouts-per-task", type=int, default=4)
    parser.add_argument("--rollout-start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--internal-only", action="store_true")
    parser.add_argument("--task-id", action="append", default=[])
    args = parser.parse_args()
    if not 0 <= args.rollout_start_index < args.rollouts_per_task:
        raise ValueError("rollout-start-index must be in [0, rollouts-per-task)")

    train_path = FROZEN / "train.json"
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    if sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("frozen train hash mismatch")
    if manifest.get("frozen_300_manifest_sha256") != EXPECTED_FROZEN300_SHA256:
        raise RuntimeError("Frozen300 manifest lock mismatch")
    if manifest.get("FROZEN_300_EXACT_TRAIN_OVERLAP") != 0:
        raise RuntimeError("Frozen300 contamination gate failed")
    rows = json.loads(train_path.read_text())
    if len(rows) != 314 or any(row.get("split") != "train" for row in rows):
        raise RuntimeError("unexpected frozen train split")
    if args.internal_only:
        rows = [row for row in rows if row["extra_info"]["graph_frontier"].get("dependency_edges")]
    if args.task_id:
        requested = set(args.task_id)
        known = {row["task_id"] for row in rows}
        missing = requested - known
        if missing:
            raise RuntimeError(f"requested task IDs are unavailable after filters: {sorted(missing)}")
        rows = [row for row in rows if row["task_id"] in requested]
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures = 0
    for task_index, row in enumerate(rows):
        for rollout_index in range(args.rollout_start_index, args.rollouts_per_task):
            sample_seed = args.seed + task_index * 1009 + rollout_index * 9176 + args.shard_index * 1000003
            try:
                collect_one(
                    row,
                    endpoint=args.endpoint,
                    model=args.model,
                    rollout_index=rollout_index,
                    seed=sample_seed,
                    max_steps=args.max_steps,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    output=args.output_dir,
                )
                completed += 1
            except Exception as exc:
                failures += 1
                print(f"ROLLOUT_FAILURE task={row['task_id']} r={rollout_index} error={type(exc).__name__}:{exc}", flush=True)
            print(
                f"PREFERENCE_ROLLOUT_PROGRESS shard={args.shard_index}/{args.shard_count} "
                f"tasks={task_index + 1}/{len(rows)} rollouts={completed} failures={failures}",
                flush=True,
            )
    summary = {
        "schema_version": "envfactory_preference_rollout_collection_v1",
        "selected_tasks": len(rows),
        "rollout_start_index": args.rollout_start_index,
        "rollout_end_index_exclusive": args.rollouts_per_task,
        "requested_rollouts": len(rows) * (args.rollouts_per_task - args.rollout_start_index),
        "completed_rollouts": completed,
        "failed_rollouts": failures,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_300_manifest_sha256": EXPECTED_FROZEN300_SHA256,
        "frozen_300_overlap": 0,
    }
    (args.output_dir / f"collection_summary.shard{args.shard_index}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


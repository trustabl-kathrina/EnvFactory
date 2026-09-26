"""Real-MCP CPU smoke for generated EnvFactory GRPO tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


class EnvConfig(dict):
    def __getattr__(self, name):
        return self[name]


def _stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from agent_system.environments.env_package.envfactory.official_envs import (
        EnvFactoryBatchEnv,
    )
    from repro_1p7b.graph_frontier.fastmcp_adapter import adapt_fastmcp_result

    rows = Dataset.from_parquet(str(args.parquet))
    candidates = []
    for row in rows:
        item = json.loads(row["env_kwargs"])
        if item.get("mcp_servers") == ["VehicleControl"] and item.get("ground_truth"):
            candidates.append(item)
    if not candidates:
        raise RuntimeError("no generated VehicleControl task available")
    item = max(candidates, key=lambda value: len(value["ground_truth"]))
    config = EnvConfig(max_steps=8, max_tool_calls=8)
    env = EnvFactoryBatchEnv(20260917, env_num=1, group_n=4, is_train=True, env_config=config)
    env.log_dir = args.output.parent / "real_env_rollouts"
    env.log_dir.mkdir(parents=True, exist_ok=True)

    observations, _ = env.reset([item] * 4)
    client_ids = [state["clients"]["VehicleControl"] for state in env.states]
    if len(set(client_ids)) != 4:
        raise AssertionError("G=4 did not receive four unique MCP client ids")
    initial_states = [env._save(state) for state in env.states]
    if len({_stable(value) for value in initial_states}) != 1:
        raise AssertionError("G=4 initial states differ")

    raw = env._call_typed(
        client_ids[0],
        "VehicleControl-set_climate_control",
        {"temperature": 19.0, "unit": "celsius", "fan_speed": 35, "mode": "cool"},
    )
    mutation_fields, mutation_success = adapt_fastmcp_result(raw)
    mutated_states = [env._save(state) for state in env.states]
    if _stable(mutated_states[0]) == _stable(initial_states[0]):
        raise AssertionError("rollout A mutation did not change its environment")
    if any(
        _stable(mutated_states[index]) != _stable(initial_states[index])
        for index in range(1, 4)
    ):
        raise AssertionError("rollout A mutation leaked into rollout B/C/D")

    # Reset all copies, then exercise the adapter's typed step boundary.
    env.reset([item] * 4)
    first = item["ground_truth"][0]
    action = {
        "kind": "tool",
        "name": first["name"],
        "arguments": first["arguments"],
        "raw": "typed-smoke",
    }
    _, rewards, dones, _ = env.step([action] * 4)
    if rewards.tolist() != [0.0] * 4 or dones.tolist() != [False] * 4:
        raise AssertionError("non-terminal rewards must be zero")
    events = [state["events"][0] for state in env.states]
    for event in events:
        if event.get("tool_arguments") != first["arguments"]:
            raise AssertionError("typed arguments changed")
        if not isinstance(event.get("execution_success"), bool):
            raise AssertionError("execution_success is not typed bool")
        if event.get("returned_fields") == "unknown":
            raise AssertionError("typed returned_fields unavailable")
        if not isinstance(event.get("state_before"), dict) or not isinstance(
            event.get("state_after"), dict
        ):
            raise AssertionError("state snapshots missing")

    _, terminal_rewards, terminal_dones, infos = env.step(
        [{"kind": "final", "content": "done", "raw": "done"}] * 4
    )
    if terminal_dones.tolist() != [True] * 4:
        raise AssertionError("final action did not terminate all rollouts")
    report = {
        "status": "PASS",
        "task_id": item["task_id"],
        "group_size": 4,
        "unique_client_ids": len(set(client_ids)),
        "mutation_success": mutation_success,
        "mutation_returned_fields": mutation_fields,
        "rollout_a_changed": _stable(mutated_states[0]) != _stable(initial_states[0]),
        "rollout_bcd_unchanged": all(
            _stable(mutated_states[index]) == _stable(initial_states[index])
            for index in range(1, 4)
        ),
        "typed_execution_statuses": [event["execution_success"] for event in events],
        "typed_returned_fields_available": [
            event["returned_fields"] != "unknown" for event in events
        ],
        "terminal_rewards": terminal_rewards.tolist(),
        "terminal_won": [info["won"] for info in infos],
        "observations": len(observations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()

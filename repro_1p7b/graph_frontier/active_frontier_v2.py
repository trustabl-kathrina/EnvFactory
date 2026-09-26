"""Active gold-frontier preference construction for EnvFactory.

This module deliberately separates four stages:

* ``inventory`` is a static, exhaustive audit of frozen TRAIN gold edges;
* ``replay`` executes the real gold prefix and validates the chosen action in
  a cloned EnvFactory state;
* ``sample`` asks Dynamic-v1 for K structured next actions at each state;
* ``finalize`` conservatively labels observed actions and builds task-grouped
  preference splits plus a retention-funnel audit.

No stage trains a model.  Frozen300 and external evaluations are only hash and
contamination locks; they are never used as data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from repro_1p7b.graph_frontier.build_preference_pairs import assistant_tool_message
from repro_1p7b.graph_frontier.collect_preference_rollouts import (
    EXPECTED_FROZEN300_SHA256,
    EXPECTED_TRAIN_SHA256,
    FROZEN,
    EnvConfig,
    env_item,
    generate_action,
    sha256,
    stable,
)
from repro_1p7b.graph_frontier.profiler import _path_values, _values_match

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "repro_1p7b/graph_frontier/preference_v2"
REPORT_ROOT = ROOT / "repro_1p7b/graph_frontier/reports"
UNKNOWN = "unknown"
FAILURE_ORDER = {"wrong_value": 0, "wrong_tool": 1, "premature_stop": 2, "useless_retry": 3}


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frozen_rows() -> list[dict[str, Any]]:
    train = FROZEN / "train.json"
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    if sha256(train) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("frozen TRAIN hash mismatch")
    if manifest.get("frozen_300_manifest_sha256") != EXPECTED_FROZEN300_SHA256:
        raise RuntimeError("Frozen300 manifest hash lock mismatch")
    if manifest.get("FROZEN_300_EXACT_TRAIN_OVERLAP") != 0:
        raise RuntimeError("Frozen300 contamination lock failed")
    rows = json.loads(train.read_text())
    if len(rows) != 314 or any(row.get("split") != "train" for row in rows):
        raise RuntimeError("unexpected frozen TRAIN split")
    return rows


def ground_truth(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = row["reward_model"]["ground_truth"]
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list):
        raise ValueError("ground truth is not a list")
    return value


def occurrences(actions: list[Mapping[str, Any]], name: str) -> list[int]:
    return [index for index, action in enumerate(actions) if action.get("name") == name]


def inventory_record(row: Mapping[str, Any], edge: Mapping[str, Any], edge_index: int) -> dict[str, Any]:
    actions = ground_truth(row)
    producer = edge.get("producer_tool", {}).get("tool_name")
    consumer = edge.get("consumer_tool", {}).get("tool_name")
    producer_positions = occurrences(actions, producer)
    consumer_positions = occurrences(actions, consumer)
    reasons: list[str] = []
    if edge.get("edge_type") != "parameter_flow":
        reasons.append("not_parameter_flow")
    if edge.get("internal_parameter") is not True:
        reasons.append("not_internal_parameter")
    if edge.get("required") is not True:
        reasons.append("not_required")
    if edge.get("optional") is True:
        reasons.append("optional_parameter")
    if len(producer_positions) != 1:
        reasons.append("producer_occurrence_not_unique")
    if len(consumer_positions) != 1:
        reasons.append("consumer_occurrence_not_unique")
    producer_position = producer_positions[0] if len(producer_positions) == 1 else None
    consumer_position = consumer_positions[0] if len(consumer_positions) == 1 else None
    if producer_position is not None and consumer_position is not None:
        if producer_position >= consumer_position:
            reasons.append("producer_not_before_consumer")
        elif producer_position + 1 != consumer_position:
            # The requested prompt ends at the producer observation.  A
            # non-adjacent reference would require inventing a reordering.
            reasons.append("consumer_not_immediate_after_producer")
    source = edge.get("producer_output_parameter", {}).get("parameter_name")
    target = edge.get("consumer_input_parameter", {}).get("parameter_name")
    if not isinstance(source, str) or not source:
        reasons.append("missing_producer_output_parameter")
    if not isinstance(target, str) or not target:
        reasons.append("missing_consumer_input_parameter")
    chosen = actions[consumer_position] if consumer_position is not None else None
    if not isinstance((chosen or {}).get("arguments"), dict):
        reasons.append("consumer_arguments_not_structured")
    return {
        "schema_version": "graph_frontier_edge_inventory_v2",
        "task_id": row["task_id"],
        "source_seed": row["extra_info"].get("generation_seed", UNKNOWN),
        "edge_index": edge_index,
        "edge_id": edge.get("edge_id", UNKNOWN),
        "producer_tool": producer,
        "producer_output_parameter": source,
        "consumer_tool": consumer,
        "consumer_input_parameter": target,
        "dependency_depth": edge.get("dependency_depth", row["extra_info"]["graph_frontier"].get("dependency_depth", UNKNOWN)),
        "environment_identifiers": row["extra_info"]["graph_frontier"].get("environment_identifiers", []),
        "internal_parameter": edge.get("internal_parameter", UNKNOWN),
        "user_provided": edge.get("consumer_input_parameter", {}).get("user_provided", UNKNOWN),
        "required": edge.get("required", UNKNOWN),
        "optional": edge.get("optional", UNKNOWN),
        "producer_position": producer_position if producer_position is not None else UNKNOWN,
        "consumer_position": consumer_position if consumer_position is not None else UNKNOWN,
        "static_eligible": not reasons,
        "static_drop_reasons": reasons,
        "edge": copy.deepcopy(edge),
    }


def run_inventory(output: Path) -> dict[str, Any]:
    rows = frozen_rows()
    records = []
    task_no_edges = 0
    for row in rows:
        edges = row["extra_info"]["graph_frontier"].get("dependency_edges", [])
        if not edges:
            task_no_edges += 1
        records.extend(inventory_record(row, edge, index) for index, edge in enumerate(edges))
    write_jsonl(output / "frontier_edge_inventory.jsonl", records)
    reasons = Counter(reason for record in records for reason in record["static_drop_reasons"])
    eligible = [record for record in records if record["static_eligible"]]
    report = {
        "schema_version": "graph_frontier_inventory_audit_v2",
        "source": "frozen EnvFactory generated-RL TRAIN split only",
        "train_tasks": len(rows),
        "tasks_without_dependency_edges": task_no_edges,
        "all_dependency_edges": len(records),
        "internal_required_edges": sum(r["internal_parameter"] is True and r["required"] is True for r in records),
        "static_eligible_edges": len(eligible),
        "static_drop_reason_distribution": dict(reasons),
        "depth_distribution": dict(Counter(str(r["dependency_depth"]) for r in eligible)),
        "environment_distribution": dict(Counter(env for r in eligible for env in r["environment_identifiers"])),
        "frozen_train_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_300_manifest_sha256": EXPECTED_FROZEN300_SHA256,
        "frozen_300_overlap": 0,
        "theoretical_max_unique_edge_states": len(eligible),
        "state_gate": 128,
        "state_gate_statically_reachable": len(eligible) >= 128,
    }
    (output / "inventory_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def action_message(action: Mapping[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": action["name"], "arguments": action["arguments"]}}],
    }


def set_argument(arguments: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = arguments
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise ValueError(f"cannot set nested argument {path}")
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise ValueError(f"cannot set nested argument {path}")
    current[parts[-1]] = value


def unique_matching_value(
    returned_fields: Any,
    source_path: str,
    chosen_arguments: Mapping[str, Any],
    target_path: str,
    edge: Mapping[str, Any],
) -> tuple[Any, list[Any], list[Any]]:
    source_values = _path_values(returned_fields, source_path)
    target_values = _path_values(chosen_arguments, target_path)
    if source_values == UNKNOWN or target_values == UNKNOWN or len(target_values) != 1:
        raise ValueError("source/target value unavailable or target non-unique")
    matches = [
        value for value in source_values
        if _values_match(
            [value],
            target_values,
            edge.get("producer_output_parameter", {}).get("data_type", UNKNOWN),
            edge.get("consumer_input_parameter", {}).get("data_type", UNKNOWN),
            edge.get("value_semantics", UNKNOWN),
        ) is True
    ]
    unique = {stable(value): value for value in matches}
    if len(unique) != 1:
        raise ValueError(f"matched producer value is not unique: {len(unique)}")
    return next(iter(unique.values())), source_values, target_values


def successful_event(state: Mapping[str, Any], before: int) -> Mapping[str, Any]:
    events = state.get("events", [])
    if len(events) <= before:
        raise RuntimeError("tool execution emitted no typed event")
    event = events[-1]
    if event.get("execution_success") is not True:
        raise RuntimeError(f"tool execution failed: {event.get('exception', UNKNOWN)}")
    return event


def legal_next_tools(row: Mapping[str, Any], completed_positions: set[int]) -> list[str]:
    actions = ground_truth(row)
    sidecar = row["extra_info"]["graph_frontier"]
    completed_names = {actions[index].get("name") for index in completed_positions}
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in sidecar.get("dependency_edges", []):
        if edge.get("required") is True and edge.get("internal_parameter") is True:
            incoming[edge["consumer_tool"]["tool_name"]].add(edge["producer_tool"]["tool_name"])
    legal = []
    for index, action in enumerate(actions):
        if index in completed_positions:
            continue
        name = action.get("name")
        if incoming.get(name, set()) <= completed_names:
            legal.append(name)
    return sorted(set(legal))


def replay_one(row: Mapping[str, Any], record: Mapping[str, Any], runtime_dir: Path, seed: int) -> dict[str, Any]:
    os.environ.setdefault("ENVFACTORY_ROOT", str(ROOT))
    from agent_system.environments.env_package.envfactory.official_envs import EnvFactoryBatchEnv

    actions = ground_truth(row)
    producer_position = int(record["producer_position"])
    consumer_position = int(record["consumer_position"])
    item = env_item(row)
    os.environ["ENVFACTORY_RL_ROLLOUT_DIR"] = str(runtime_dir)
    env = EnvFactoryBatchEnv(seed, 1, 1, False, EnvConfig(max(4, consumer_position + 2)))
    prompt = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else copy.deepcopy(row["prompt"])
    messages = copy.deepcopy(prompt)
    prefix_events = []
    try:
        env.reset([item])
        state = env.states[0]
        tools = copy.deepcopy(state["tools"])
        for position in range(producer_position + 1):
            gold = actions[position]
            action = {"kind": "tool", "name": gold["name"], "arguments": copy.deepcopy(gold["arguments"])}
            call_id = f"gold_prefix_{position}"
            before = len(state["events"])
            env.step([action])
            event = copy.deepcopy(successful_event(state, before))
            prefix_events.append(event)
            messages.append(action_message(action, call_id))
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": event["tool_name"],
                "content": stable(event["tool_response"]),
            })
        producer_event = prefix_events[-1]
        edge = record["edge"]
        chosen_gold = actions[consumer_position]
        chosen_arguments = copy.deepcopy(chosen_gold["arguments"])
        typed_value, source_values, target_values = unique_matching_value(
            producer_event.get("returned_fields", UNKNOWN),
            record["producer_output_parameter"],
            chosen_arguments,
            record["consumer_input_parameter"],
            edge,
        )
        set_argument(chosen_arguments, record["consumer_input_parameter"], typed_value)
        frontier_state = env._save(state)
    finally:
        env.close()

    clone_item = env_item(row)
    clone_item["initial_config"] = frontier_state
    clone = EnvFactoryBatchEnv(seed + 7919, 1, 1, False, EnvConfig(2))
    chosen_action = {"kind": "tool", "name": record["consumer_tool"], "arguments": chosen_arguments}
    try:
        clone.reset([clone_item])
        clone_state = clone.states[0]
        before = len(clone_state["events"])
        clone.step([chosen_action])
        chosen_event = copy.deepcopy(successful_event(clone_state, before))
    finally:
        clone.close()

    state_signature = hashlib.sha256(stable(frontier_state).encode()).hexdigest()
    identity = [row["task_id"], record["producer_tool"], producer_position, record["edge_id"], state_signature]
    state_id = "frontier-" + hashlib.sha256(stable(identity).encode()).hexdigest()[:24]
    completed_positions = set(range(producer_position + 1))
    return {
        "schema_version": "envfactory_active_frontier_state_v2",
        "state_id": state_id,
        "state_signature": state_signature,
        "task_id": row["task_id"],
        "source_seed": record["source_seed"],
        "edge_id": record["edge_id"],
        "producer_tool": record["producer_tool"],
        "producer_position": producer_position,
        "producer_output_parameter": record["producer_output_parameter"],
        "consumer_tool": record["consumer_tool"],
        "consumer_position": consumer_position,
        "consumer_input_parameter": record["consumer_input_parameter"],
        "dependency_depth": record["dependency_depth"],
        "environment_identifiers": record["environment_identifiers"],
        "prompt_messages": messages,
        "tools": tools,
        "state_before_chosen": frontier_state,
        "prefix_events": prefix_events,
        "legal_next_tools": legal_next_tools(row, completed_positions),
        "chosen_action": chosen_action,
        "chosen_message": action_message(chosen_action, "chosen_call"),
        "chosen_executable": True,
        "chosen_execution": {
            "tool_name": chosen_event["tool_name"],
            "tool_arguments": chosen_event["tool_arguments"],
            "returned_fields": chosen_event["returned_fields"],
            "execution_success": True,
        },
        "internal_value_source": {
            "producer_tool": record["producer_tool"],
            "producer_output_parameter": record["producer_output_parameter"],
            "consumer_tool": record["consumer_tool"],
            "consumer_input_parameter": record["consumer_input_parameter"],
            "typed_value": typed_value,
            "all_source_values": source_values,
            "gold_target_values": target_values,
        },
        "frozen_300_overlap": False,
    }


def run_replay(output: Path, limit: int | None) -> dict[str, Any]:
    rows = {row["task_id"]: row for row in frozen_rows()}
    inventory = read_jsonl(output / "frontier_edge_inventory.jsonl")
    eligible = [record for record in inventory if record.get("static_eligible")]
    if limit is not None:
        eligible = eligible[:limit]
    states = []
    drops: Counter[str] = Counter()
    runtime = output / "replay_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(eligible, 1):
        try:
            states.append(replay_one(rows[record["task_id"]], record, runtime, 20260920 + index * 17))
        except Exception as exc:
            key = f"{type(exc).__name__}:{str(exc)[:180]}"
            drops[key] += 1
            print(f"FRONTIER_REPLAY_DROP {record['task_id']} {record['edge_id']} {key}", flush=True)
        print(f"FRONTIER_REPLAY_PROGRESS {index}/{len(eligible)} valid={len(states)} dropped={sum(drops.values())}", flush=True)
    write_jsonl(output / "frontier_states.jsonl", states)
    report = {
        "schema_version": "envfactory_active_frontier_replay_audit_v2",
        "static_eligible_edges": len(eligible),
        "replay_valid_states": len(states),
        "replay_drop_reason_distribution": dict(drops),
        "chosen_executable_rate": 1.0 if states else 0.0,
        "unique_state_ids": len({state["state_id"] for state in states}),
        "unique_state_signatures": len({state["state_signature"] for state in states}),
        "frozen_300_overlap": 0,
    }
    (output / "replay_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_sample(args: argparse.Namespace) -> dict[str, Any]:
    states = read_jsonl(args.output_dir / "frontier_states.jsonl")
    states = [state for index, state in enumerate(states) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        states = states[:args.limit]
    action_dir = args.output_dir / "sampled_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    completed = failures = 0
    for state_index, state in enumerate(states):
        path = action_dir / f"{state['state_id']}.json"
        if path.exists():
            completed += 1
            continue
        samples = []
        for sample_index in range(args.samples_per_state):
            seed = args.seed + state_index * 1009 + sample_index * 9176 + args.shard_index * 1000003
            try:
                message, action, usage, finish_reason = generate_action(
                    args.endpoint,
                    args.model,
                    state["prompt_messages"],
                    state["tools"],
                    seed=seed,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                samples.append({
                    "sample_index": sample_index,
                    "sampling_seed": seed,
                    "model_message": message,
                    "parsed_action": action,
                    "usage": usage,
                    "finish_reason": finish_reason,
                })
            except Exception as exc:
                failures += 1
                samples.append({
                    "sample_index": sample_index,
                    "sampling_seed": seed,
                    "parsed_action": {"kind": "invalid", "reason": f"request_error:{type(exc).__name__}:{exc}"},
                    "model_message": {"role": "assistant", "content": ""},
                    "usage": {},
                    "finish_reason": "request_error",
                })
        payload = {
            "schema_version": "envfactory_active_frontier_samples_v2",
            "state_id": state["state_id"],
            "task_id": state["task_id"],
            "model": args.model,
            "samples": samples,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        completed += 1
        print(f"FRONTIER_SAMPLE_PROGRESS shard={args.shard_index}/{args.shard_count} states={completed}/{len(states)} request_failures={failures}", flush=True)
    summary = {
        "schema_version": "envfactory_active_frontier_sampling_summary_v2",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "states": len(states),
        "samples_per_state": args.samples_per_state,
        "request_failures": failures,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_tokens": args.max_tokens,
    }
    (args.output_dir / f"sampling_summary.shard{args.shard_index}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def classify_sample(sample: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    action = sample.get("parsed_action", {})
    if action.get("kind") == "invalid":
        return "invalid"
    if action.get("kind") == "final":
        return "premature_stop"
    if action.get("kind") != "tool" or not isinstance(action.get("arguments"), dict):
        return "invalid"
    for event in state.get("prefix_events", []):
        if event.get("tool_name") == action.get("name") and stable(event.get("tool_arguments")) == stable(action.get("arguments")):
            return "useless_retry"
    chosen = state["chosen_action"]
    if action.get("name") != chosen.get("name"):
        if action.get("name") in set(state.get("legal_next_tools", [])):
            return "legal_alternative"
        return "wrong_tool"
    flow = state["internal_value_source"]
    actual = _path_values(action["arguments"], flow["consumer_input_parameter"])
    expected = _path_values(chosen["arguments"], flow["consumer_input_parameter"])
    if actual == UNKNOWN or expected == UNKNOWN:
        return "invalid"
    kind = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(type(flow["typed_value"]).__name__, UNKNOWN)
    match = _values_match(expected, actual, kind, kind)
    if match is False:
        return "wrong_value"
    if match is not True:
        return "invalid"
    return "correct" if stable(action.get("arguments")) == stable(chosen.get("arguments")) else "ambiguous_same_tool"


def split_by_task(pairs: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = sorted({pair["task_id"] for pair in pairs}, key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).hexdigest())
    val_count = max(1, round(len(tasks) * 0.1)) if len(tasks) > 1 else 0
    val_tasks = set(tasks[:val_count])
    train = [pair for pair in pairs if pair["task_id"] not in val_tasks]
    val = [pair for pair in pairs if pair["task_id"] in val_tasks]
    if {pair["task_id"] for pair in train} & {pair["task_id"] for pair in val}:
        raise RuntimeError("task-group split leakage")
    return train, val


def stratified_samples(pairs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        buckets[pair["failure_type"]].append(pair)
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def render_samples(path: Path, pairs: list[dict[str, Any]]) -> None:
    lines = ["# Active gold-frontier preference samples", ""]
    for pair in pairs:
        lines += [
            f"## {pair['pair_id']}", "",
            f"- task: `{pair['task_id']}`",
            f"- state: `{pair['state_id']}`",
            f"- edge: `{pair['edge_id']}`",
            f"- failure: `{pair['failure_type']}`",
            f"- producer -> consumer: `{pair['producer_tool']}` -> `{pair['consumer_tool']}`", "",
            "Chosen:", "```json", json.dumps(pair["chosen_action"], ensure_ascii=False, indent=2), "```", "",
            "Rejected (real Dynamic-v1 output):", "```json", json.dumps(pair["rejected_action"], ensure_ascii=False, indent=2), "```", "",
            "Typed value provenance:", "```json", json.dumps(pair["internal_value_source"], ensure_ascii=False, indent=2), "```", "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_audit_md(report: Mapping[str, Any]) -> str:
    f = report["retention_funnel"]
    g = report["gates"]
    return f"""# Graph-targeted preference v2 data audit

## Verdict

**{report['verdict']}**

Active construction used only the frozen EnvFactory generated-RL TRAIN split.  It replayed real gold prefixes, captured the producer observation, and validated every chosen consumer call in a cloned real environment state.  Frozen300 and BFCL were not used as data.

## Retention funnel

| Stage | Count |
|---|---:|
| Frozen TRAIN tasks | {f['frozen_train_tasks']} |
| All gold dependency edges | {f['all_dependency_edges']} |
| Internal required edges | {f['internal_required_edges']} |
| Static high-confidence edges | {f['static_eligible_edges']} |
| Replay-valid / chosen-executable states | {f['replay_valid_states']} |
| States with at least one retained negative | {f['paired_states']} |
| Dynamic-v1 actions sampled | {f['sampled_actions']} |
| Valid preference pairs | {f['valid_pairs']} |

## Gates

| Gate | Observed | Required | Pass |
|---|---:|---:|---|
| Unique frontier states | {g['unique_frontier_states']['observed']} | {g['unique_frontier_states']['required']} | {g['unique_frontier_states']['pass']} |
| Preference pairs | {g['preference_pairs']['observed']} | {g['preference_pairs']['required']} | {g['preference_pairs']['pass']} |
| Chosen executable rate | {g['chosen_executable_rate']['observed']:.3f} | {g['chosen_executable_rate']['required']:.3f} | {g['chosen_executable_rate']['pass']} |
| Frozen300 overlap | {g['frozen_300_overlap']['observed']} | {g['frozen_300_overlap']['required']} | {g['frozen_300_overlap']['pass']} |

## Distributions

- Failure types: `{json.dumps(report['failure_type_distribution'], sort_keys=True)}`
- Depth: `{json.dumps(report['depth_distribution'], sort_keys=True)}`
- Environments: `{json.dumps(report['environment_distribution'], sort_keys=True)}`
- Sample classifications: `{json.dumps(report['sample_classification_distribution'], sort_keys=True)}`

## Hard bottleneck

The source TRAIN pool contains only **{f['internal_required_edges']} internal required edges** and only **{f['static_eligible_edges']} static high-confidence immediate frontiers**.  Therefore the protocol's 128-state/256-pair gate cannot be reached without generating new graph-bearing EnvFactory tasks or fabricating duplicate states; neither is permitted in this round.

## Protocol locks

- Frozen TRAIN SHA256: `{report['frozen_train_sha256']}`
- Frozen300 manifest SHA256: `{report['frozen_300_manifest_sha256']}`
- Frozen300 exact overlap: `0`
- Training started: `false`
"""


def run_finalize(output: Path, seed: int) -> dict[str, Any]:
    inventory = read_jsonl(output / "frontier_edge_inventory.jsonl")
    states = read_jsonl(output / "frontier_states.jsonl")
    pairs = []
    classifications: Counter[str] = Counter()
    sampled_actions = 0
    for state in states:
        sample_path = output / "sampled_actions" / f"{state['state_id']}.json"
        if not sample_path.exists():
            classifications["missing_state_samples"] += 1
            continue
        samples = json.loads(sample_path.read_text()).get("samples", [])
        sampled_actions += len(samples)
        candidates = []
        seen = set()
        for sample in samples:
            label = classify_sample(sample, state)
            classifications[label] += 1
            if label not in FAILURE_ORDER:
                continue
            signature = stable(sample.get("parsed_action"))
            if signature in seen:
                classifications["duplicate_rejected_action"] += 1
                continue
            seen.add(signature)
            candidates.append((FAILURE_ORDER[label], label, sample))
        candidates.sort(key=lambda item: (item[0], item[2]["sample_index"]))
        kept = []
        seen_labels = set()
        for item in candidates:
            if item[1] not in seen_labels:
                kept.append(item)
                seen_labels.add(item[1])
            if len(kept) == 2:
                break
        if len(kept) < 2:
            for item in candidates:
                if item not in kept:
                    kept.append(item)
                if len(kept) == 2:
                    break
        for _, label, sample in kept:
            pair_id = f"active-v2-{state['state_id']}-{sample['sample_index']}"
            pairs.append({
                "schema_version": "graph_targeted_active_preference_pair_v2",
                "pair_id": pair_id,
                "task_id": state["task_id"],
                "state_id": state["state_id"],
                "state_signature": state["state_signature"],
                "edge_id": state["edge_id"],
                "prompt": state["prompt_messages"],
                "tools": state["tools"],
                "chosen": [state["chosen_message"]],
                "rejected": [sample["model_message"]],
                "chosen_action": state["chosen_action"],
                "rejected_action": sample["parsed_action"],
                "failure_type": label,
                "producer_tool": state["producer_tool"],
                "consumer_tool": state["consumer_tool"],
                "internal_value_source": state["internal_value_source"],
                "dependency_depth": state["dependency_depth"],
                "environment_identifiers": state["environment_identifiers"],
                "chosen_executable": True,
                "rejected_is_observed_dynamic_v1_output": True,
                "sampling_seed": sample["sampling_seed"],
                "frozen_300_overlap": False,
            })
    pairs.sort(key=lambda pair: pair["pair_id"])
    train, val = split_by_task(pairs, seed)
    write_jsonl(output / "graph_targeted_pairs_train.jsonl", train)
    write_jsonl(output / "graph_targeted_pairs_val.jsonl", val)
    selected = stratified_samples(pairs.copy(), min(30, len(pairs)))
    render_samples(output / "graph_pair_sanity_30.md", selected)

    paired_states = {pair["state_id"] for pair in pairs}
    chosen_rate = sum(state.get("chosen_executable") is True for state in states) / len(states) if states else 0.0
    depth = Counter("3+" if isinstance(pair["dependency_depth"], int) and pair["dependency_depth"] >= 3 else str(pair["dependency_depth"]) for pair in pairs)
    envs = Counter(env for pair in pairs for env in pair["environment_identifiers"])
    failures = Counter(pair["failure_type"] for pair in pairs)
    gates = {
        "unique_frontier_states": {"observed": len(paired_states), "required": 128, "pass": len(paired_states) >= 128},
        "preference_pairs": {"observed": len(pairs), "required": 256, "pass": len(pairs) >= 256},
        "failure_diversity": {"observed": len(failures), "required": 2, "pass": len(failures) >= 2},
        "chosen_executable_rate": {"observed": chosen_rate, "required": 1.0, "pass": chosen_rate == 1.0},
        "frozen_300_overlap": {"observed": 0, "required": 0, "pass": True},
        "task_group_split_disjoint": {"observed": len({p['task_id'] for p in train} & {p['task_id'] for p in val}), "required": 0, "pass": True},
    }
    required_pass = all(gate["pass"] for gate in gates.values())
    verdict = "DATA-READY" if required_pass else ("PARTIAL-DATA-READY" if pairs else "DATA-NOT-READY")
    report = {
        "schema_version": "graph_targeted_preference_v2_data_audit",
        "verdict": verdict,
        "ready_for_dpo_smoke": required_pass,
        "training_started": False,
        "retention_funnel": {
            "frozen_train_tasks": 314,
            "all_dependency_edges": len(inventory),
            "internal_required_edges": sum(r.get("internal_parameter") is True and r.get("required") is True for r in inventory),
            "static_eligible_edges": sum(r.get("static_eligible") is True for r in inventory),
            "replay_valid_states": len(states),
            "paired_states": len(paired_states),
            "sampled_actions": sampled_actions,
            "valid_pairs": len(pairs),
        },
        "sample_classification_distribution": dict(classifications),
        "failure_type_distribution": dict(failures),
        "depth_distribution": dict(depth),
        "environment_distribution": dict(envs),
        "gates": gates,
        "splits": {
            "method": "task_id grouped deterministic 90/10",
            "seed": seed,
            "train_pairs": len(train),
            "val_pairs": len(val),
            "train_tasks": len({p["task_id"] for p in train}),
            "val_tasks": len({p["task_id"] for p in val}),
        },
        "frozen_train_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_300_manifest_sha256": EXPECTED_FROZEN300_SHA256,
        "frozen_300_overlap": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "preference_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "graph_targeted_preference_v2_data_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (REPORT_ROOT / "graph_targeted_preference_v2_data_audit.md").write_text(render_audit_md(report), encoding="utf-8")
    hashes = {name: sha256(output / name) for name in ("graph_targeted_pairs_train.jsonl", "graph_targeted_pairs_val.jsonl")}
    manifest = {
        "schema_version": "graph_targeted_preference_manifest_v2",
        "verdict": verdict,
        "split_method": "task_id grouped deterministic 90/10",
        "split_seed": seed,
        "hashes": hashes,
        "frozen_train_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_300_manifest_sha256": EXPECTED_FROZEN300_SHA256,
        "frozen_300_overlap": 0,
    }
    (output / "preference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    marker = output / "READY_FOR_DPO_SMOKE"
    if required_pass:
        marker.write_text("DATA-READY\n")
    elif marker.exists():
        marker.unlink()
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    replay = sub.add_parser("replay")
    replay.add_argument("--limit", type=int)
    sample = sub.add_parser("sample")
    sample.add_argument("--endpoint", required=True)
    sample.add_argument("--model", default="dynamic-v1")
    sample.add_argument("--samples-per-state", type=int, default=4)
    sample.add_argument("--temperature", type=float, default=0.7)
    sample.add_argument("--max-tokens", type=int, default=512)
    sample.add_argument("--seed", type=int, default=20260920)
    sample.add_argument("--shard-count", type=int, default=1)
    sample.add_argument("--shard-index", type=int, default=0)
    sample.add_argument("--limit", type=int)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--seed", type=int, default=20260920)
    return result


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "inventory":
        report = run_inventory(args.output_dir)
    elif args.command == "replay":
        report = run_replay(args.output_dir, args.limit)
    elif args.command == "sample":
        report = run_sample(args)
    else:
        report = run_finalize(args.output_dir, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

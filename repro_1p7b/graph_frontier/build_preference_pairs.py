"""Build audited Standard and Graph-targeted preference pairs from typed rollouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from repro_1p7b.graph_frontier.collect_preference_rollouts import (
    EXPECTED_FROZEN300_SHA256,
    EXPECTED_TRAIN_SHA256,
    FROZEN,
    EnvConfig,
    env_item,
    sha256,
    stable,
)
from repro_1p7b.graph_frontier.profiler import _path_values, _values_match

ROOT = Path(__file__).resolve().parents[2]
UNKNOWN = "unknown"
FAILURE_PRIORITY = {"premature_stop": 0, "wrong_tool": 1, "useless_retry": 2, "wrong_value": 3}


def task_rows() -> dict[str, dict[str, Any]]:
    path = FROZEN / "train.json"
    if sha256(path) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("frozen train hash mismatch")
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    if manifest.get("frozen_300_manifest_sha256") != EXPECTED_FROZEN300_SHA256:
        raise RuntimeError("Frozen300 manifest lock mismatch")
    if manifest.get("FROZEN_300_EXACT_TRAIN_OVERLAP") != 0:
        raise RuntimeError("Frozen300 contamination gate failed")
    rows = json.loads(path.read_text())
    if len(rows) != 314:
        raise RuntimeError(f"unexpected train count: {len(rows)}")
    return {row["task_id"]: row for row in rows}


def load_rollouts(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("collection_summary"):
            continue
        row = json.loads(path.read_text())
        if row.get("schema_version") == "envfactory_preference_rollout_v1":
            rows.append(row)
    return rows


def event_before(rollout: Mapping[str, Any], step_index: int) -> list[dict[str, Any]]:
    return [
        step["typed_event"]
        for step in rollout["steps"][:step_index]
        if isinstance(step.get("typed_event"), dict)
    ]


def parameter(sidecar: Mapping[str, Any], tool_name: str, role: str, name: str) -> Mapping[str, Any]:
    matches = [
        row for row in sidecar.get("parameters", [])
        if row.get("tool_name") == tool_name and row.get("role") == role and row.get("parameter_name") == name
    ]
    return matches[0] if len(matches) == 1 else {}


def ground_truth_action(row: Mapping[str, Any], tool_name: str) -> dict[str, Any] | None:
    ground = json.loads(row["reward_model"]["ground_truth"])
    matches = [item for item in ground if item.get("name") == tool_name]
    if len(matches) != 1 or not isinstance(matches[0].get("arguments"), dict):
        return None
    return {"kind": "tool", "name": tool_name, "arguments": copy.deepcopy(matches[0]["arguments"])}


def eligible_actions(
    row: Mapping[str, Any],
    prior_events: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    sidecar = row["extra_info"]["graph_frontier"]
    reasons: Counter[str] = Counter()
    if not sidecar.get("dependency_edges"):
        return {}, Counter(no_internal_edge=1)
    successful_consumers = {event["tool_name"] for event in prior_events if event.get("execution_success") is True}
    incoming: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in sidecar["dependency_edges"]:
        if edge.get("internal_parameter") is True and edge.get("required") is True:
            incoming[edge["consumer_tool"]["tool_name"]].append(edge)
    actions: dict[str, dict[str, Any]] = {}
    for consumer, edges in incoming.items():
        if consumer in successful_consumers:
            continue
        chosen = ground_truth_action(row, consumer)
        if chosen is None:
            reasons["missing_unique_ground_truth_action"] += 1
            continue
        provenance = []
        valid = True
        for edge in edges:
            producer = edge["producer_tool"]["tool_name"]
            producer_events = [
                event for event in prior_events
                if event.get("tool_name") == producer and event.get("execution_success") is True
            ]
            if not producer_events:
                valid = False
                reasons["producer_not_successful"] += 1
                break
            event = producer_events[-1]
            source_name = edge["producer_output_parameter"]["parameter_name"]
            target_name = edge["consumer_input_parameter"]["parameter_name"]
            source_values = _path_values(event.get("returned_fields", UNKNOWN), source_name)
            target_values = _path_values(chosen["arguments"], target_name)
            source_type = edge["producer_output_parameter"].get("data_type", UNKNOWN)
            target_type = edge["consumer_input_parameter"].get("data_type", UNKNOWN)
            match = _values_match(source_values, target_values, source_type, target_type, edge.get("value_semantics", UNKNOWN))
            if match is not True or len(source_values) != 1:
                valid = False
                reasons["internal_value_not_unique_or_mismatch"] += 1
                break
            chosen["arguments"][target_name] = source_values[0]
            provenance.append(
                {
                    "producer": producer,
                    "producer_output_parameter": source_name,
                    "consumer": consumer,
                    "consumer_input_parameter": target_name,
                    "typed_value": source_values[0],
                    "edge_id": edge.get("edge_id", UNKNOWN),
                }
            )
        if valid:
            chosen["provenance"] = provenance
            actions[consumer] = chosen
    return actions, reasons


def classify_failure(
    rejected: Mapping[str, Any],
    chosen: Mapping[str, Any],
    prior_events: list[dict[str, Any]],
    state_before: Any,
    other_legal_tools: set[str] | None = None,
) -> str | None:
    if rejected.get("kind") == "final":
        return "premature_stop"
    if rejected.get("kind") != "tool":
        return None
    if prior_events:
        previous = prior_events[-1]
        same_call = (
            previous.get("tool_name") == rejected.get("name")
            and stable(previous.get("tool_arguments")) == stable(rejected.get("arguments"))
        )
        no_change = stable(previous.get("state_before")) == stable(previous.get("state_after"))
        if previous.get("execution_success") is False and same_call and no_change:
            return "useless_retry"
    if rejected.get("name") != chosen.get("name"):
        # Another still-pending gold tool may be a valid reordering on an
        # independent branch.  Without a proof that it is illegal at this
        # state, do not manufacture a wrong-tool preference.
        if rejected.get("name") in (other_legal_tools or set()):
            return None
        return "wrong_tool"
    wrong = False
    for flow in chosen.get("provenance", []):
        name = flow["consumer_input_parameter"]
        actual = _path_values(rejected.get("arguments", {}), name)
        expected = _path_values(chosen.get("arguments", {}), name)
        if actual == UNKNOWN or expected == UNKNOWN:
            return None
        source_type = type(flow["typed_value"]).__name__
        kind = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(source_type, UNKNOWN)
        verdict = _values_match(expected, actual, kind, kind)
        if verdict is False:
            wrong = True
        elif verdict is not True:
            return None
    return "wrong_value" if wrong else None


def assistant_tool_message(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "chosen_call",
                "type": "function",
                "function": {"name": action["name"], "arguments": action["arguments"]},
            }
        ],
    }


def graph_candidates(
    rollouts: list[dict[str, Any]], rows: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pairs = []
    drops: Counter[str] = Counter()
    for rollout in rollouts:
        row = rows.get(rollout["task_id"])
        if row is None:
            drops["unknown_task"] += 1
            continue
        for step in rollout["steps"]:
            prior = event_before(rollout, int(step["step_index"]))
            actions, reasons = eligible_actions(row, prior)
            drops.update(reasons)
            if not actions:
                continue
            if len(actions) != 1:
                drops["ambiguous_multiple_executable_consumers"] += 1
                continue
            chosen = next(iter(actions.values()))
            completed = {event.get("tool_name") for event in prior if event.get("execution_success") is True}
            gold_tools = {
                item.get("name")
                for item in json.loads(row["reward_model"]["ground_truth"])
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            other_legal_tools = gold_tools - completed - {chosen["name"]}
            failure = classify_failure(
                step["parsed_action"], chosen, prior, step["state_before_action"], other_legal_tools
            )
            if failure is None:
                if step["parsed_action"].get("kind") == "invalid":
                    drops["invalid_model_action"] += 1
                elif step["parsed_action"].get("name") in other_legal_tools:
                    drops["possible_legal_reordering"] += 1
                else:
                    drops["next_action_not_target_failure"] += 1
                continue
            frontier_hash = hashlib.sha256(
                # The contamination cap is defined over task + normalized
                # environment state.  Prompt wording can vary across rollouts
                # without creating a genuinely new frontier state.
                stable([rollout["task_id"], step["state_before_action"]]).encode()
            ).hexdigest()
            consumer = chosen["name"]
            producer = chosen["provenance"][-1]["producer"]
            depth = row["extra_info"]["graph_frontier"].get("dependency_depth", UNKNOWN)
            pair = {
                "schema_version": "graph_targeted_preference_pair_v1",
                "pair_id": f"graph-{frontier_hash[:20]}-{rollout['rollout_index']}-{step['step_index']}",
                "task_id": rollout["task_id"],
                "source_seed": rollout["source_seed"],
                "prompt": step["prompt_messages"],
                "chosen": [assistant_tool_message(chosen)],
                "rejected": [step["model_message"]],
                "chosen_action": {"type": "tool_call", "tool_name": consumer, "arguments": chosen["arguments"]},
                "rejected_action": step["parsed_action"],
                "failure_type": failure,
                "producer": producer,
                "consumer": consumer,
                "internal_value_source": chosen["provenance"],
                "dependency_depth": depth,
                "environment": row["extra_info"]["graph_frontier"].get("environment_identifiers", []),
                "frontier_state_hash": frontier_hash,
                "state_before_action": step["state_before_action"],
                "chosen_executable": False,
                "rejected_is_observed_model_action": True,
                "tools": rollout["tools"],
                "source_rollout_index": rollout["rollout_index"],
            }
            pairs.append(pair)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["task_id"], pair["frontier_state_hash"])].append(pair)
    capped = []
    for candidates in grouped.values():
        candidates.sort(key=lambda p: (FAILURE_PRIORITY[p["failure_type"]], p["pair_id"]))
        unique = []
        seen_pair_signatures = set()
        for pair in candidates:
            signature = stable([pair["chosen"], pair["rejected"]])
            if signature in seen_pair_signatures:
                drops["exact_duplicate_pair"] += 1
                continue
            seen_pair_signatures.add(signature)
            unique.append(pair)
        kept = []
        seen_types = set()
        # Diversity first: at most one rejected per failure type.
        for pair in unique:
            if pair["failure_type"] in seen_types:
                continue
            kept.append(pair)
            seen_types.add(pair["failure_type"])
            if len(kept) == 2:
                break
        # The protocol permits 1-2 rejected actions per normalized state.
        # Fill the second slot only with a distinct observed model action.
        if len(kept) < 2:
            for pair in unique:
                if pair in kept:
                    continue
                kept.append(pair)
                if len(kept) == 2:
                    break
        capped.extend(kept)
        drops["duplicate_state_capped"] += len(unique) - len(kept)
    return sorted(capped, key=lambda p: p["pair_id"]), drops


def standard_candidates(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        if isinstance(rollout.get("official_reward"), (int, float)):
            grouped[rollout["task_id"]].append(rollout)
    pairs = []
    for task_id, items in sorted(grouped.items()):
        candidates = []
        for high in items:
            for low in items:
                success_gap = int(bool(high["semantic_success"])) - int(bool(low["semantic_success"]))
                reward_gap = float(high["official_reward"]) - float(low["official_reward"])
                strictly_ordered = success_gap > 0 or (success_gap == 0 and reward_gap > 1e-9)
                if not strictly_ordered:
                    continue
                prompt = high["initial_prompt"]
                if prompt != low["initial_prompt"]:
                    raise RuntimeError(f"standard prompt mismatch:{task_id}")
                candidates.append((success_gap, reward_gap, high, low))
        candidates.sort(key=lambda row: (-row[0], -row[1], int(row[2]["rollout_index"]), int(row[3]["rollout_index"])))
        kept_signatures = set()
        kept_for_task = 0
        for _, _, high, low in candidates:
            signature = stable([high["conversation_messages"], low["conversation_messages"]])
            if signature in kept_signatures:
                continue
            kept_signatures.add(signature)
            pairs.append({
                "schema_version": "standard_trajectory_preference_pair_v1",
                "pair_id": f"standard-{task_id}-{high['rollout_index']}-{low['rollout_index']}",
                "task_id": task_id,
                "source_seed": high["source_seed"],
                "prompt": prompt,
                "chosen": high["conversation_messages"][len(prompt):],
                "rejected": low["conversation_messages"][len(prompt):],
                "chosen_official": {"success": high["semantic_success"], "reward": high["official_reward"]},
                "rejected_official": {"success": low["semantic_success"], "reward": low["official_reward"]},
                "strict_official_ordering": True,
                "tools": high["tools"],
            })
            kept_for_task += 1
            if kept_for_task == 2:
                break
    return pairs


def validate_chosen(pairs: list[dict[str, Any]], rows: Mapping[str, Mapping[str, Any]], log_dir: Path) -> Counter[str]:
    from agent_system.environments.env_package.envfactory.official_envs import EnvFactoryBatchEnv

    os.environ["ENVFACTORY_RL_ROLLOUT_DIR"] = str(log_dir)
    drops: Counter[str] = Counter()
    for index, pair in enumerate(pairs, 1):
        row = rows[pair["task_id"]]
        item = env_item(row)
        item["initial_config"] = pair["state_before_action"]
        env = EnvFactoryBatchEnv(20260920 + index, 1, 1, False, EnvConfig(2))
        try:
            env.reset([item])
            state = env.states[0]
            action = {
                "kind": "tool",
                "name": pair["chosen_action"]["tool_name"],
                "arguments": pair["chosen_action"]["arguments"],
            }
            env.step([action])
            event = state["events"][-1] if state["events"] else None
            if event is None or event.get("execution_success") is not True:
                pair["chosen_validation_error"] = None if event is None else event.get("exception")
                drops["chosen_not_executable"] += 1
                continue
            pair["chosen_executable"] = True
            pair["chosen_execution"] = {
                "tool_name": event["tool_name"],
                "arguments": event["tool_arguments"],
                "returned_fields": event["returned_fields"],
                "execution_success": True,
            }
        except Exception as exc:
            pair["chosen_validation_error"] = f"{type(exc).__name__}:{exc}"
            drops["chosen_validation_exception"] += 1
        finally:
            env.close()
        if index % 20 == 0:
            print(f"CHOSEN_VALIDATION_PROGRESS {index}/{len(pairs)}", flush=True)
    pairs[:] = [pair for pair in pairs if pair["chosen_executable"] is True]
    return drops


def split_pairs(pairs: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_ids = sorted({pair["task_id"] for pair in pairs})
    ranked = sorted(task_ids, key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).hexdigest())
    val_count = max(1, round(0.1 * len(ranked))) if len(ranked) > 1 else 0
    val_ids = set(ranked[:val_count])
    train = [pair for pair in pairs if pair["task_id"] not in val_ids]
    val = [pair for pair in pairs if pair["task_id"] in val_ids]
    if {p["task_id"] for p in train} & {p["task_id"] for p in val}:
        raise RuntimeError("task-level split leakage")
    return train, val


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def render_samples(path: Path, pairs: list[dict[str, Any]], limit: int = 20) -> None:
    lines = ["# Graph-targeted preference pair sanity samples", ""]
    for pair in pairs[:limit]:
        flow = pair["internal_value_source"][-1]
        lines += [
            f"## {pair['pair_id']}", "",
            f"- task: `{pair['task_id']}`", f"- failure: `{pair['failure_type']}`",
            f"- producer output: `{json.dumps(flow, ensure_ascii=False)}`",
            f"- gold consumer: `{pair['consumer']}`", "",
            "Chosen:", "```json", json.dumps(pair["chosen_action"], ensure_ascii=False, indent=2), "```", "",
            "Rejected:", "```json", json.dumps(pair["rejected_action"], ensure_ascii=False, indent=2), "```", "",
            "Prompt frontier:", "```json", json.dumps(pair["prompt"], ensure_ascii=False, indent=2)[:12000], "```", "",
        ]
    path.write_text("\n".join(lines) + "\n")


def audit(graph: list[dict[str, Any]], standard: list[dict[str, Any]], drops: Counter[str], rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts = Counter((pair["task_id"], pair["frontier_state_hash"]) for pair in graph)
    signatures = Counter(stable([pair["chosen"], pair["rejected"]]) for pair in graph)
    return {
        "schema_version": "envfactory_preference_audit_v1",
        "dynamic_v1_rollouts": len(rollouts),
        "candidate_frontier_states": len(state_counts),
        "valid_graph_pairs": len(graph),
        "standard_pairs": len(standard),
        "dropped": sum(drops.values()),
        "drop_reason_distribution": dict(drops),
        "failure_type_distribution": dict(Counter(p["failure_type"] for p in graph)),
        "depth_distribution": dict(Counter("3+" if isinstance(p["dependency_depth"], int) and p["dependency_depth"] >= 3 else str(p["dependency_depth"]) for p in graph)),
        "environment_distribution": dict(Counter(env for p in graph for env in p["environment"])),
        "unique_task_ratio": len({p["task_id"] for p in graph}) / len(graph) if graph else 0.0,
        "unique_prompt_state_ratio": len(state_counts) / len(graph) if graph else 0.0,
        "unique_consumer_tool_ratio": len({p["consumer"] for p in graph}) / len(graph) if graph else 0.0,
        "duplicate_chosen_rejected_ratio": sum(count - 1 for count in signatures.values()) / len(graph) if graph else 0.0,
        "max_rejected_per_frontier_state": max(state_counts.values(), default=0),
        "chosen_executable_rate": sum(p["chosen_executable"] is True for p in graph) / len(graph) if graph else 0.0,
        "ambiguous_drop_count": drops.get("ambiguous_multiple_executable_consumers", 0),
        "duplicate_state_count": sum(count > 1 for count in state_counts.values()),
        "frozen_300_overlap": 0,
        "frozen_train_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_300_manifest_sha256": EXPECTED_FROZEN300_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--validate-chosen", action="store_true")
    args = parser.parse_args()
    rows = task_rows()
    rollouts = load_rollouts(args.rollout_dir)
    if not rollouts:
        raise RuntimeError("no typed Dynamic-v1 rollouts found")
    graph, drops = graph_candidates(rollouts, rows)
    if args.validate_chosen:
        drops.update(validate_chosen(graph, rows, args.output_dir / "chosen_validation_runtime"))
    standard = standard_candidates(rollouts)
    graph_train, graph_val = split_pairs(graph, args.seed)
    standard_train, standard_val = split_pairs(standard, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "graph_targeted_pairs_train.jsonl", graph_train)
    write_jsonl(args.output_dir / "graph_targeted_pairs_val.jsonl", graph_val)
    write_jsonl(args.output_dir / "standard_dpo_pairs_train.jsonl", standard_train)
    write_jsonl(args.output_dir / "standard_dpo_pairs_val.jsonl", standard_val)
    report = audit(graph, standard, drops, rollouts)
    report["splits"] = {
        "graph": {"train_pairs": len(graph_train), "val_pairs": len(graph_val), "train_tasks": len({p['task_id'] for p in graph_train}), "val_tasks": len({p['task_id'] for p in graph_val})},
        "standard": {"train_pairs": len(standard_train), "val_pairs": len(standard_val), "train_tasks": len({p['task_id'] for p in standard_train}), "val_tasks": len({p['task_id'] for p in standard_val})},
    }
    hashes = {}
    for name in ("graph_targeted_pairs_train.jsonl", "graph_targeted_pairs_val.jsonl", "standard_dpo_pairs_train.jsonl", "standard_dpo_pairs_val.jsonl"):
        hashes[name] = sha256(args.output_dir / name)
    manifest = {
        "schema_version": "envfactory_preference_manifest_v1",
        "source": "frozen EnvFactory generated-RL TRAIN split only",
        "initialization_model": "Dynamic-v1",
        "split_method": "task_id grouped deterministic 90/10",
        "split_seed": args.seed,
        "hashes": hashes,
        "frozen_train_sha256": EXPECTED_TRAIN_SHA256,
        "frozen_300_manifest_sha256": EXPECTED_FROZEN300_SHA256,
        "frozen_300_overlap": 0,
    }
    (args.output_dir / "preference_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "preference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Human-readable sanity review may span train and preference-val, while
    # the serialized split files remain task-disjoint.
    render_samples(args.output_dir / "graph_pair_sanity_20.md", graph_train + graph_val, 20)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


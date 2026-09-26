"""Terminal rewards for the controlled Standard-vs-Graph GRPO experiment.

The official component is a typed re-expression of RolandXMR/verl
``release/v0.6.1:EnvFactory/reward/tool_reward_fcn.py``.  It intentionally
keeps the official subset/permutation trace match, per-server state match,
extra-call penalty, malformed-call penalty, and tau scheduler.

Graph-Frontier only adds conservative evidence from generation-time gold
edges and the executed typed rollout.  It never reparses response prose.
"""

from __future__ import annotations

import json
import math

import numpy as np
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .adapters import UNKNOWN
from .profiler import _path_values, _values_match


def _json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_name(record: Mapping[str, Any], key: str) -> Any:
    value = record.get(key, UNKNOWN)
    if isinstance(value, Mapping):
        return value.get("tool_name", UNKNOWN)
    return value


def _parameter(record: Mapping[str, Any], key: str) -> tuple[Any, Any]:
    value = record.get(key, UNKNOWN)
    if not isinstance(value, Mapping):
        return UNKNOWN, UNKNOWN
    return value.get("parameter_name", UNKNOWN), value.get("data_type", UNKNOWN)


def _official_value_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(actual, expected, rel_tol=1e-6)
    return False


def official_call_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Match one call exactly as the official EnvFactory reward does."""

    if actual.get("name") != expected.get("name"):
        return False
    actual_args = actual.get("arguments", {})
    expected_args = expected.get("arguments", {})
    if not isinstance(actual_args, Mapping) or not isinstance(expected_args, Mapping):
        return False
    masked = set(expected.get("masked_arguments", []))
    return all(
        key in actual_args and _official_value_match(actual_args[key], value)
        for key, value in expected_args.items()
        if key not in masked
    )


def official_trace_score(
    actual_calls: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
) -> float:
    """Official subset-of-a-permutation trace success (binary)."""

    if not actual_calls:
        return 0.0
    if not ground_truth:
        return 1.0
    used: set[int] = set()
    for expected in ground_truth:
        match = next(
            (
                index
                for index, actual in enumerate(actual_calls)
                if index not in used and official_call_match(actual, expected)
            ),
            None,
        )
        if match is None:
            return 0.0
        used.add(match)
    return 1.0


def official_state_score(
    final_state: Mapping[str, Any],
    expected_state: Mapping[str, Any],
    ground_truth: Sequence[Mapping[str, Any]],
) -> float:
    """Official fraction of used MCP servers with an exact final state."""

    servers = {
        str(call.get("name", "")).split("-", 1)[0]
        for call in ground_truth
        if isinstance(call.get("name"), str) and "-" in call["name"]
    }
    if not servers:
        return 0.0
    return sum(final_state.get(server) == expected_state.get(server) for server in servers) / len(servers)


def reward_tau(
    *,
    mode: str = "fix",
    epoch: int = 1,
    total_epochs: int = 2,
    start_ratio: float = 0.9,
    end_ratio: float = 0.1,
    default_ratio: float = 0.5,
) -> float:
    if mode == "fix":
        return float(default_ratio)
    if mode == "linear":
        ratio = min(float(epoch) / max(int(total_epochs), 1), 1.0)
        return float(start_ratio + (end_ratio - start_ratio) * ratio)
    raise ValueError(f"unsupported reward scheduler mode: {mode}")


def official_reward(
    actual_calls: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    final_state: Mapping[str, Any],
    expected_state: Mapping[str, Any],
    *,
    malformed_call_count: int = 0,
    scheduler: Mapping[str, Any] | None = None,
    epoch: int = 1,
    total_epochs: int = 2,
) -> dict[str, float]:
    """Return ``R_official`` and its exact published decomposition."""

    cfg = dict(scheduler or {})
    tau = reward_tau(
        mode=cfg.get("mode", "fix"),
        epoch=epoch,
        total_epochs=total_epochs,
        start_ratio=cfg.get("start_ratio", 0.9),
        end_ratio=cfg.get("end_ratio", 0.1),
        default_ratio=cfg.get("default_ratio", 0.5),
    )
    trace = official_trace_score(actual_calls, ground_truth)
    state = official_state_score(final_state, expected_state, ground_truth)
    length_penalty = min(max(len(actual_calls) - len(ground_truth), 0) * 0.05, 0.5)
    format_penalty = min(max(int(malformed_call_count), 0) * 0.05, 0.5)
    score = max(0.0, tau * trace + (1.0 - tau) * state - length_penalty - format_penalty)
    return {
        "score": float(score),
        "trace_score": float(trace),
        "state_score": float(state),
        "length_penalty": float(length_penalty),
        "format_penalty": float(format_penalty),
        "penalty": float(length_penalty + format_penalty),
        "tau": float(tau),
    }


def _event_value(event: Mapping[str, Any], field: str, path: Any) -> Any:
    value = event.get(field, UNKNOWN)
    return _path_values(value, path)


def graph_frontier_components(
    events: Sequence[Mapping[str, Any]], sidecar: Mapping[str, Any]
) -> dict[str, Any]:
    """Score edge opportunities from typed execution evidence only."""

    eligibility = sidecar.get("probe_eligibility", {}) or {}
    task_graph_eligible = eligibility.get("eligible") is True
    skip_reasons = list(eligibility.get("reasons", []) or [])
    if not task_graph_eligible:
        if not skip_reasons:
            skip_reasons = ["probe_eligibility_not_explicitly_true"]
        return {
            "task_graph_eligible": False,
            "task_graph_skip_reasons": skip_reasons,
            "eligible_internal_edges": 0,
            "skipped_non_internal_edges": 0,
            "skipped_task_ineligible_edges": len(
                sidecar.get("dependency_edges", []) or []
            ),
            "consumer_opportunities": 0,
            "completed_consumer_opportunities": 0,
            "consumer_completion_fraction": 0.0,
            "propagation_checks": 0,
            "correct_propagations": 0,
            "propagation_fraction": 0.0,
            "invalid_retry_count": 0,
            "retry_opportunities": 0,
            "invalid_retry_fraction": 0.0,
            "edge_details": [],
        }

    opportunities = completed = propagation_checks = propagation_correct = 0
    eligible_edges = skipped_non_internal = 0
    edge_details = []
    for edge in sidecar.get("dependency_edges", []) or []:
        producer_name = _tool_name(edge, "producer_tool")
        consumer_name = _tool_name(edge, "consumer_tool")
        source_name, source_type = _parameter(edge, "producer_output_parameter")
        target_name, target_type = _parameter(edge, "consumer_input_parameter")
        producer_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.get("tool_name") == producer_name
                and event.get("execution_success") is True
                and _event_value(event, "returned_fields", source_name) != UNKNOWN
            ),
            None,
        )
        detail = {
            "edge_id": edge.get("edge_id", UNKNOWN),
            "eligible": edge.get("internal_parameter") is True,
            "opportunity": False,
            "consumer_executed": False,
            "propagation_match": UNKNOWN,
        }
        # Graph-Frontier v1 rewards only values that must flow from a tool
        # output.  External/user-provided or unknown edges are diagnostic-only;
        # rewarding them could encourage an unnecessary producer call.
        if edge.get("internal_parameter") is not True:
            skipped_non_internal += 1
            edge_details.append(detail)
            continue
        eligible_edges += 1
        if producer_index is None:
            edge_details.append(detail)
            continue
        opportunities += 1
        detail["opportunity"] = True
        consumer_index = next(
            (
                index
                for index in range(producer_index + 1, len(events))
                if events[index].get("tool_name") == consumer_name
            ),
            None,
        )
        if consumer_index is None:
            edge_details.append(detail)
            continue
        completed += 1
        detail["consumer_executed"] = True
        source_values = _event_value(events[producer_index], "returned_fields", source_name)
        target_values = _event_value(events[consumer_index], "tool_arguments", target_name)
        match = _values_match(
            source_values,
            target_values,
            source_type,
            target_type,
            edge.get("value_semantics", UNKNOWN),
        )
        detail["propagation_match"] = match
        if isinstance(match, bool):
            propagation_checks += 1
            propagation_correct += int(match)
        edge_details.append(detail)

    invalid_retries = 0
    retry_opportunities = 0
    for previous, current in zip(events, events[1:]):
        same_action = (
            previous.get("execution_success") is False
            and previous.get("tool_name") == current.get("tool_name")
            and _canonical(previous.get("tool_arguments", UNKNOWN))
            == _canonical(current.get("tool_arguments", UNKNOWN))
        )
        state_known = (
            previous.get("state_after", UNKNOWN) != UNKNOWN
            and current.get("state_before", UNKNOWN) != UNKNOWN
        )
        no_state_change = state_known and (
            _canonical(previous["state_after"]) == _canonical(current["state_before"])
        )
        retry_opportunities += int(
            previous.get("execution_success") is False and no_state_change
        )
        invalid_retries += int(same_action and no_state_change)

    completion = completed / opportunities if opportunities else 0.0
    propagation = propagation_correct / propagation_checks if propagation_checks else 0.0
    retry_fraction = invalid_retries / max(len(events), 1)
    return {
        "task_graph_eligible": True,
        "task_graph_skip_reasons": [],
        "eligible_internal_edges": eligible_edges,
        "skipped_non_internal_edges": skipped_non_internal,
        "skipped_task_ineligible_edges": 0,
        "consumer_opportunities": opportunities,
        "completed_consumer_opportunities": completed,
        "consumer_completion_fraction": float(completion),
        "propagation_checks": propagation_checks,
        "correct_propagations": propagation_correct,
        "propagation_fraction": float(propagation),
        "invalid_retry_count": invalid_retries,
        "retry_opportunities": retry_opportunities,
        "invalid_retry_fraction": float(retry_fraction),
        "edge_details": edge_details,
    }


def graph_v2_components(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Opportunity-normalized, bounded structural diagnostics for v2.

    Missing opportunities are neutral and explicitly marked unavailable.  The
    relative v1 weights are retained, but this score is only a tertiary
    within-class tie-break and is never added directly to official reward.
    """

    consumer_opportunities = int(graph.get("consumer_opportunities", 0) or 0)
    consumer_completed = int(graph.get("completed_consumer_opportunities", 0) or 0)
    prop_opportunities = int(graph.get("propagation_checks", 0) or 0)
    prop_correct = int(graph.get("correct_propagations", 0) or 0)
    retry_opportunities = int(graph.get("retry_opportunities", 0) or 0)
    invalid_retry = int(graph.get("invalid_retry_count", 0) or 0)
    consumer_score = consumer_completed / consumer_opportunities if consumer_opportunities else 0.0
    prop_score = prop_correct / prop_opportunities if prop_opportunities else 0.0
    retry_score = -(invalid_retry / retry_opportunities) if retry_opportunities else 0.0
    score = float(np.clip(0.30 * prop_score + 0.40 * consumer_score + 0.10 * retry_score, -1.0, 1.0))
    available = bool(consumer_opportunities or prop_opportunities or retry_opportunities)
    return {
        "eligible_consumer_opportunities": consumer_opportunities,
        "consumer_completed": consumer_completed,
        "consumer_score": float(consumer_score),
        "consumer_available": bool(consumer_opportunities),
        "eligible_prop_opportunities": prop_opportunities,
        "prop_correct": prop_correct,
        "prop_score": float(prop_score),
        "prop_available": bool(prop_opportunities),
        "retry_opportunities": retry_opportunities,
        "invalid_retry": invalid_retry,
        "retry_score": float(retry_score),
        "retry_available": bool(retry_opportunities),
        "graph_score": score,
        "graph_available": available,
        "bounded_range": [-1.0, 1.0],
    }


@dataclass(frozen=True)
class GraphRewardWeights:
    propagation: float = 0.30
    consumer_completion: float = 0.40
    invalid_retry: float = 0.10


def terminal_reward(
    *,
    mode: str,
    events: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    final_state: Mapping[str, Any],
    expected_state: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    malformed_call_count: int = 0,
    scheduler: Mapping[str, Any] | None = None,
    weights: GraphRewardWeights = GraphRewardWeights(),
) -> dict[str, Any]:
    """Compute the sole terminal reward; non-terminal steps must return zero."""

    if mode not in {"official", "graph_frontier"}:
        raise ValueError("mode must be 'official' or 'graph_frontier'")
    actual_calls = [
        {"name": event.get("tool_name"), "arguments": event.get("tool_arguments", {})}
        for event in events
    ]
    official = official_reward(
        actual_calls,
        ground_truth,
        final_state,
        expected_state,
        malformed_call_count=malformed_call_count,
        scheduler=scheduler,
    )
    graph = graph_frontier_components(events, sidecar)
    graph_v2 = graph_v2_components(graph)
    delta = (
        weights.propagation * graph["propagation_fraction"]
        + weights.consumer_completion * graph["consumer_completion_fraction"]
        - weights.invalid_retry * graph["invalid_retry_fraction"]
    )
    total = official["score"] if mode == "official" else official["score"] + delta
    return {
        "score": float(total),
        "reward_mode": mode,
        "official_reward": official,
        "graph_frontier": graph,
        "graph_v2": graph_v2,
        "graph_delta": float(delta),
        "weights": {
            "propagation": weights.propagation,
            "consumer_completion": weights.consumer_completion,
            "invalid_retry": weights.invalid_retry,
            "semantic": 0.0,
        },
        "semantic_not_duplicated": True,
    }

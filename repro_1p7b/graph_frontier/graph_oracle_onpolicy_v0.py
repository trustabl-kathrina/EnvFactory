"""Conservative action-level oracle for student-visited EnvFactory states.

Selected gold paths are not exhaustive action-space oracles. Unknown is skipped.
"""
from __future__ import annotations
import json
from typing import Any, Mapping, Sequence

UNKNOWN = "unknown"

def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def frozen_overlap(task_id: str, train_ids: set[str], frozen_ids: set[str]) -> bool:
    return not task_id or task_id not in train_ids or task_id in frozen_ids

def reference_matches(event: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    if event.get("tool_name") != reference.get("name") or event.get("execution_success") is not True:
        return False
    actual, expected = event.get("tool_arguments"), reference.get("arguments")
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    # Masked runtime values need a separate typed provenance certificate.
    if reference.get("masked_arguments"):
        return False
    return stable(actual) == stable(expected)

def get_local_supervision(
    *, sidecar: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
    reference_actions: Sequence[Mapping[str, Any]],
    current_environment_state: Any = UNKNOWN,
    current_verifier_success: Any = UNKNOWN,
    certified_enabled_actions: Sequence[str] | None = None,
    replay_valid: bool = True,
) -> dict[str, Any]:
    result = dict(tier="C", target_type="skip", target_action=None,
                  positive_action_set=[], terminal_ready=UNKNOWN,
                  completed_required_nodes=[], remaining_required_nodes=[],
                  enabled_required_nodes=[], reason="unknown", evidence={},
                  replay_valid=replay_valid, training_weight=0.0)
    if not replay_valid:
        result["reason"] = "replay_invalid"
        return result
    if current_verifier_success is True:
        # A current-state executable verifier is stronger than path adherence.
        result.update(tier="A", target_type="stop", target_action="STOP",
                      terminal_ready=True, reason="current_executable_verifier_success",
                      evidence={"source": "current_executable_verifier"}, training_weight=1.0)
        return result
    resolution = sidecar.get("dependency_resolution") or {}
    if (sidecar.get("dependency_semantics") != "selected_reference"
        or sidecar.get("structural_diagnosis_eligible") is not True
        or sidecar.get("cross_turn_dependency") is not False
        or resolution.get("status") != "resolved"):
        result["reason"] = "graph_unresolved"
        return result
    seq = [n.get("tool_name") for n in sidecar.get("gold_tool_sequence", [])]
    required = [n.get("tool_name") for n in sidecar.get("required_tool_nodes", [])]
    if not seq or len(seq) != len(set(seq)) or set(seq) != set(required):
        result["reason"] = "repeated_or_incomplete_reference"
        return result
    if len(history) > min(len(reference_actions), len(seq)):
        result["reason"] = "off_reference_path"
        return result
    for index, event in enumerate(history):
        if seq[index] != event.get("tool_name") or not reference_matches(event, reference_actions[index]):
            result["reason"] = "off_reference_path_or_unverified_arguments"
            return result
    completed, remaining = seq[:len(history)], seq[len(history):]
    result["completed_required_nodes"], result["remaining_required_nodes"] = completed, remaining
    edges = sidecar.get("dependency_edges") or []
    enabled = []
    for tool in remaining:
        incoming = [e for e in edges if (e.get("consumer_tool") or {}).get("tool_name") == tool]
        if all((e.get("producer_tool") or {}).get("tool_name") in completed for e in incoming):
            enabled.append(tool)
    result["enabled_required_nodes"] = enabled
    result["evidence"] = {"source": "selected_reference_prefix", "completed_count": len(completed)}
    expected = sidecar.get("expected_final_state", UNKNOWN)
    final_match = (expected != UNKNOWN and current_environment_state != UNKNOWN
                   and stable(current_environment_state) == stable(expected))
    if not remaining and (current_verifier_success is True or final_match):
        result.update(tier="A", target_type="stop", target_action="STOP", terminal_ready=True,
                      reason="verified_terminal_and_required_complete", training_weight=1.0)
        return result
    if not remaining:
        result["reason"] = "terminal_verifier_unknown"
        return result
    result["terminal_ready"] = False if current_verifier_success is False else UNKNOWN
    if certified_enabled_actions is None:
        result["reason"] = "selected_path_not_complete_action_oracle"
        return result
    certified = sorted(set(certified_enabled_actions) & set(enabled))
    if not certified:
        result["reason"] = "no_certified_progress_action"
    elif len(certified) == 1 and len(set(certified_enabled_actions)) == 1:
        result.update(tier="A", target_type="tool_call", target_action=certified[0],
                      reason="unique_exhaustively_certified_action", training_weight=1.0)
    else:
        result.update(tier="B", target_type="multi_action", positive_action_set=certified,
                      reason="multiple_or_uncertain_valid_actions")
    return result

def assistant_only_labels(prompt_token_count: int, assistant_token_count: int) -> list[int]:
    """-100 masks prompt and tool observations; only assistant decision may have loss."""
    if min(prompt_token_count, assistant_token_count) < 0:
        raise ValueError("negative token count")
    return [-100] * prompt_token_count + [1] * assistant_token_count

"""First-divergence correction from replayed gold and actual student trajectories.

Conservative v1: one gold call per assistant turn, no masked runtime arguments,
and exact typed response equality on every matching prefix step. Off-path
student states are never inspected for a correction target.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

UNKNOWN = "unknown"

def canonical(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def normalize_action(action: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = action.get("kind")
    if kind == "final":
        return {"kind": "final"}
    if kind != "tool":
        return None
    name, arguments = action.get("name"), action.get("arguments")
    if not isinstance(name, str):
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {"kind": "tool", "name": name, "arguments": arguments}

def reference_from_raw(row: Mapping[str, Any], audit: Mapping[str, Any],
                       raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    task_id = row.get("task_id")
    status = audit.get("status")
    if audit.get("task_id") != task_id or status not in {"REPLAY_PASS", "FINAL_VALID"}:
        return None, "gold_replay_invalid"
    if status == "FINAL_VALID" and (audit.get("replay_ok") is not True
                                    or audit.get("static_pass") is not True
                                    or audit.get("leakage_hits")):
        return None, "gold_replay_invalid"
    if row.get("frozen_300_overlap") is not False or audit.get("frozen_overlap", False) is not False:
        return None, "frozen_overlap"
    try:
        index = int(audit["node_index"]) if "node_index" in audit else int(str(task_id).rsplit("-t", 1)[1])
        node = raw["nodes"][index]
        steps = node["steps"]
        ground_truth = row["reward_model"]["ground_truth"]
        actions = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None, "reference_unreadable"
    calls, responses = [], []
    final_text = None
    for step in steps:
        role, content = step.get("role"), step.get("content")
        if role == "tool_call":
            if not isinstance(content, list) or len(content) != 1:
                return None, "multi_call_or_malformed_gold_turn"
            calls.extend(content)
        elif role == "tool_response":
            if not isinstance(content, list) or len(content) != 1:
                return None, "multi_response_or_malformed_gold_turn"
            responses.extend(content)
        elif role == "assistant" and isinstance(content, str):
            final_text = content
    if not calls or len(calls) != len(responses) or len(calls) != len(actions):
        return None, "reference_length_mismatch"
    if any(call.get("masked_arguments") or ref.get("masked_arguments") for call, ref in zip(calls, actions)):
        return None, "masked_gold_arguments"
    for call, ref in zip(calls, actions):
        if (call.get("name") != ref.get("name")
            or canonical(call.get("arguments")) != canonical(ref.get("arguments"))
            or not isinstance(ref.get("arguments"), dict)):
            return None, "ground_truth_raw_mismatch"
    digest = hashlib.sha256(canonical([task_id, calls]).encode()).hexdigest()[:20]
    return {"task_id": task_id, "gold_trajectory_id": "gold-" + digest,
            "actions": actions, "responses": responses,
            "final_text": final_text, "raw_file": audit.get("raw_file"),
            "audit_status": "REPLAY_PASS"}, "ok"

def _target(action: Mapping[str, Any]) -> str:
    return "<tool_call>\n" + canonical({"name": action["name"], "arguments": action["arguments"]}) + "\n</tool_call>"

def align_first_divergence(row: Mapping[str, Any], rollout: Mapping[str, Any],
                           reference: Mapping[str, Any], *,
                           student_checkpoint: str, policy_version: str) -> dict[str, Any]:
    """Return exactly one correction, a perfect result, or a conservative skip."""
    task_id = row["task_id"]
    base = {"task_id": task_id, "trajectory_id": f"{task_id}.r{rollout.get('rollout_index', 'unknown')}",
            "gold_trajectory_id": reference["gold_trajectory_id"],
            "gold_step_count": len(reference["actions"]),
            "env_seed": row.get("extra_info", {}).get("generation_seed", UNKNOWN),
            "student_checkpoint": student_checkpoint, "student_policy_version": policy_version,
            "gold_replay_valid": reference["audit_status"] == "REPLAY_PASS",
            "frozen_overlap": False, "alignment_valid": False,
            "prefix_replay_valid": False, "sample": None, "status": "skip",
            "reason": "unknown"}
    if rollout.get("task_id") != task_id or row.get("frozen_300_overlap") is not False:
        return {**base, "reason": "task_or_frozen_mismatch"}
    steps = rollout.get("steps")
    if not isinstance(steps, list) or not steps:
        return {**base, "reason": "no_student_steps"}
    gold = reference["actions"]
    for index, step in enumerate(steps):
        action = normalize_action(step.get("parsed_action") or {})
        if action is None:
            return {**base, "reason": "invalid_alignment",
                    "first_divergence_step": index + 1}
        if index < len(gold):
            expected = gold[index]
            if action["kind"] == "final":
                failure = "premature_stop"
            elif action["name"] != expected["name"]:
                failure = "wrong_tool"
            elif canonical(action["arguments"]) != canonical(expected["arguments"]):
                failure = "wrong_arguments"
            else:
                failure = None
            if failure:
                sample = {
                    "task_id": task_id, "env_seed": base["env_seed"],
                    "trajectory_id": base["trajectory_id"],
                    "student_checkpoint": student_checkpoint,
                    "student_policy_version": policy_version,
                    "gold_trajectory_id": reference["gold_trajectory_id"],
                    "gold_step_count": len(gold),
                    "first_divergence_step": index + 1, "prefix_length": index,
                    "conversation_prefix": step.get("prompt_messages"),
                    "current_observation": step.get("state_before_action"),
                    "student_action": step.get("parsed_action"),
                    "student_action_canonical": action,
                    "gold_action": expected,
                    "gold_action_canonical": {"kind": "tool", "name": expected["name"],
                                              "arguments": expected["arguments"]},
                    "target_text": _target(expected),
                    "divergence_type": failure, "gold_replay_valid": True,
                    "prefix_replay_valid": True, "alignment_valid": True,
                    "frontier_bucket": row.get("extra_info", {}).get("graph_frontier", {}).get("target_bucket", UNKNOWN),
                    "graph_depth": row.get("extra_info", {}).get("graph_frontier", {}).get("dependency_depth", UNKNOWN),
                    "failure_type": failure, "training_weight": 1.0, "frozen_overlap": False,
                    "source_raw_file": reference.get("raw_file"),
                }
                return {**base, "status": "first_divergence", "reason": failure,
                        "first_divergence_step": index + 1, "prefix_length": index,
                        "alignment_valid": True, "prefix_replay_valid": True, "sample": sample}
            event = step.get("typed_event")
            if not isinstance(event, dict) or event.get("execution_success") is not True:
                return {**base, "reason": "student_prefix_execution_invalid"}
            if (event.get("tool_name") != expected["name"]
                or canonical(event.get("tool_arguments")) != canonical(expected["arguments"])
                or canonical(event.get("tool_response")) != canonical(reference["responses"][index])):
                return {**base, "reason": "student_prefix_observation_mismatch"}
            if index + 1 < len(steps):
                next_state = steps[index + 1].get("state_before_action")
                if (event.get("state_after") is not None and next_state is not None
                    and canonical(event["state_after"]) != canonical(next_state)):
                    return {**base, "reason": "student_prefix_state_discontinuity"}
            continue
        if action["kind"] == "tool":
            return {**base, "status": "skip", "reason": "extra_tool_stop_target_unverified",
                    "first_divergence_step": index + 1, "prefix_length": index,
                    "alignment_valid": True, "prefix_replay_valid": True}
        if action["kind"] == "final":
            return {**base, "status": "perfect", "reason": "gold_tool_path_and_final",
                    "alignment_valid": True, "prefix_replay_valid": True,
                    "prefix_length": len(gold)}
    return {**base, "status": "skip", "reason": "incomplete_student_rollout",
            "alignment_valid": True, "prefix_replay_valid": True}

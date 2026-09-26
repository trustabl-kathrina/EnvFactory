"""Strict first-divergence alignment and conservative wrong-argument taxonomy.

Pure-Python logic, deliberately independent of the model/runtime libraries.
"""
from __future__ import annotations

import json
import re
from typing import Any


def typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def scalar_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [(path, scalar) for key, child in value.items()
                for path, scalar in scalar_paths(child, f"{prefix}.{key}" if prefix else str(key))]
    if isinstance(value, list):
        return [(path, scalar) for i, child in enumerate(value)
                for path, scalar in scalar_paths(child, f"{prefix}[{i}]")]
    return [(prefix, value)]


def _query_contains(query: str, value: Any) -> bool:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return False
    text = str(value).strip().casefold()
    if len(text) < 3:
        return False
    return re.search(r"(?<!\w)" + re.escape(text) + r"(?!\w)", query.casefold()) is not None


def _observation_sources(observations: list[Any], value: Any) -> list[dict]:
    return [{"step": step + 1, "field": path}
            for step, observation in enumerate(observations)
            for path, candidate in scalar_paths(observation)
            if typed_equal(candidate, value)]


def classify_wrong_arguments(gold: dict, actual: dict, schema: dict | None,
                             observations: list[Any], query: str) -> dict:
    expected, got = gold["arguments"], actual["arguments"]
    if not isinstance(expected, dict) or not isinstance(got, dict):
        raise ValueError("WRONG_ARGUMENT requires structured argument objects")
    required = set(((schema or {}).get("function") or {}).get("parameters", {}).get("required") or [])
    missing = sorted(k for k in expected if k not in got)
    extra = sorted(k for k in got if k not in expected)
    differing = sorted(k for k in expected.keys() & got.keys()
                       if not typed_equal(expected[k], got[k]))
    wrong = missing + extra + differing
    correct = sorted(k for k in expected.keys() & got.keys()
                     if typed_equal(expected[k], got[k]))
    if not wrong:
        raise ValueError("argument classifier called on equal arguments")
    evidence = []
    for key in differing:
        gold_value, actual_value = expected[key], got[key]
        gold_sources = _observation_sources(observations, gold_value)
        student_sources = _observation_sources(observations, actual_value)
        evidence.append({
            "argument": key,
            "gold_source": gold_sources,
            "student_source": student_sources,
            "gold_appears_in_query": _query_contains(query, gold_value),
            "same_type": type(gold_value) is type(actual_value),
        })
    flags = set()
    if any(k in required for k in missing):
        flags.add("MISSING_REQUIRED_ARG")
    if extra:
        flags.add("EXTRA_ARG")
    if any(not item["same_type"] for item in evidence):
        flags.add("TYPE_OR_FORMAT_ERROR")
    if len(expected) >= 2 and correct and wrong:
        flags.add("PARTIAL_MULTI_ARG_ERROR")
    if any(item["gold_source"] and not item["gold_appears_in_query"] for item in evidence):
        flags.add("UPSTREAM_PROVENANCE_ERROR")
    if any(item["gold_appears_in_query"] for item in evidence):
        flags.add("USER_QUERY_EXTRACTION_ERROR")
    if any(item["student_source"] and item["gold_source"]
           and max(s["step"] for s in item["student_source"])
               < max(s["step"] for s in item["gold_source"])
           for item in evidence):
        flags.add("STALE_VALUE")
    if any(re.search(r"(^|_)(id|uuid|sku|symbol|email|name)($|_)", item["argument"], re.I)
           and item["same_type"] for item in evidence):
        flags.add("WRONG_ENTITY_OR_ID")
    if differing and all(isinstance(expected[key], (str, int, float, bool))
                         for key in differing):
        flags.add("WRONG_LITERAL_VALUE")
    priority = (
        "UPSTREAM_PROVENANCE_ERROR", "STALE_VALUE", "USER_QUERY_EXTRACTION_ERROR",
        "MISSING_REQUIRED_ARG", "TYPE_OR_FORMAT_ERROR", "EXTRA_ARG",
        "WRONG_ENTITY_OR_ID", "PARTIAL_MULTI_ARG_ERROR", "WRONG_LITERAL_VALUE",
    )
    primary = next((flag for flag in priority if flag in flags), "UNKNOWN_ARGUMENT_ERROR")
    return {
        "primary_subtype": primary, "secondary_subtypes": sorted(flags - {primary}),
        "gold_arg_count": len(expected), "wrong_arg_slots": len(wrong),
        "correct_arg_slots": len(correct), "missing_keys": missing,
        "extra_keys": extra, "differing_keys": differing,
        "first_differing_arg": wrong[0],
        "previous_observation_count": len(observations),
        "upstream_derived_arg_exists": "UPSTREAM_PROVENANCE_ERROR" in flags,
        "evidence": evidence,
    }


def align_episode(episode: dict) -> dict:
    """Return one primary failure and at most one gold-next-action correction."""
    gold = episode.get("gold")
    student = episode.get("student")
    if not gold or gold.get("replay_valid") is not True or gold.get("final_state_match") is not True:
        return {"category": "ENV_FAILURE", "failure_type": "ENVIRONMENT_FAILURE",
                "reason": "independent gold replay invalid"}
    if not student or not student.get("steps"):
        return {"category": "ENV_FAILURE", "failure_type": "ENVIRONMENT_FAILURE",
                "reason": "student rollout unavailable"}
    actions, observations, gold_states = (
        gold["actions"], gold["observations"], gold["states_after"])
    steps = student["steps"]
    if not (len(actions) == len(observations) == len(gold_states)):
        return {"category": "ALIGNMENT_UNCERTAIN", "failure_type": "ALIGNMENT_UNCERTAIN",
                "reason": "gold trace length mismatch"}
    if not typed_equal(steps[0]["state_before_action"], gold["initial_state"]):
        return {"category": "ALIGNMENT_UNCERTAIN", "failure_type": "ALIGNMENT_UNCERTAIN",
                "reason": "student/gold initial state mismatch"}
    for index, step in enumerate(steps):
        expected_state = gold["initial_state"] if index == 0 else gold_states[min(index - 1, len(gold_states) - 1)]
        if not typed_equal(step["state_before_action"], expected_state):
            return {"category": "ALIGNMENT_UNCERTAIN", "failure_type": "ALIGNMENT_UNCERTAIN",
                    "reason": "gold-exact prefix state mismatch", "gold_step": index + 1}
        action = step.get("parsed_action") or {}
        kind = action.get("kind")
        if index >= len(actions):
            if kind == "tool":
                return {"category": "TERMINAL_ONLY",
                        "failure_type": "EXTRA_TOOL_AFTER_COMPLETION",
                        "gold_step": len(actions) + 1,
                        "matched_gold_calls": len(actions)}
            if kind == "final":
                return {"category": "NO_DIVERGENCE",
                        "failure_type": "NO_TOOL_DIVERGENCE",
                        "gold_step": len(actions) + 1,
                        "matched_gold_calls": len(actions)}
            return {"category": "TERMINAL_ONLY",
                    "failure_type": "PARSER_OR_FORMAT_ERROR",
                    "gold_step": len(actions) + 1,
                    "matched_gold_calls": len(actions)}
        expected = actions[index]
        failure = None
        if kind == "final":
            failure = "PREMATURE_STOP"
        elif kind != "tool" or not isinstance(action.get("arguments"), dict):
            failure = "PARSER_OR_FORMAT_ERROR"
        elif action["name"] != expected["name"]:
            failure = "WRONG_TOOL"
        elif not typed_equal(action["arguments"], expected["arguments"]):
            failure = "WRONG_ARGUMENT"
        if failure is not None:
            if (index > 0 and kind == "tool"
                    and typed_equal(action, steps[index - 1].get("parsed_action"))):
                failure = "LOOP_OR_REPEAT"
            prompt_count = step["prompt_message_count"]
            messages = student["messages"]
            if not isinstance(prompt_count, int) or prompt_count < 1 or prompt_count > len(messages):
                return {"category": "ALIGNMENT_UNCERTAIN",
                        "failure_type": "ALIGNMENT_UNCERTAIN",
                        "reason": "student prompt history cannot be reconstructed"}
            return {
                "category": "FD_CANDIDATE", "failure_type": failure,
                "gold_step": index + 1, "student_step": step["step_index"] + 1,
                "matched_gold_calls": index,
                "gold_next_action": expected,
                "student_action": action,
                "student_state": {
                    "prompt_messages": messages[:prompt_count],
                    "environment_state": step["state_before_action"],
                    "previous_tool_observations": observations[:index],
                },
                "alignment_evidence": {
                    "independent_gold_replay": True,
                    "gold_target_executed": True,
                    "initial_state_exact": True,
                    "matched_prefix_calls": index,
                    "matched_prefix_observations": index,
                    "matched_prefix_states": index,
                    "off_path_states_supervised": 0,
                },
            }
        event = step.get("typed_event")
        if not isinstance(event, dict) or event.get("execution_success") is not True:
            return {"category": "ALIGNMENT_UNCERTAIN",
                    "failure_type": "TOOL_EXECUTION_ERROR",
                    "reason": "matched gold action did not execute", "gold_step": index + 1}
        if (event.get("tool_name") != expected["name"]
                or not typed_equal(event.get("tool_arguments"), expected["arguments"])
                or not typed_equal(event.get("tool_response"), observations[index])
                or not typed_equal(event.get("state_after"), gold_states[index])):
            return {"category": "ALIGNMENT_UNCERTAIN",
                    "failure_type": "ALIGNMENT_UNCERTAIN",
                    "reason": "matched action observation/state mismatch",
                    "gold_step": index + 1}
    return {"category": "ALIGNMENT_UNCERTAIN",
            "failure_type": "ALIGNMENT_UNCERTAIN",
            "reason": "student trajectory ended before gold path/terminal action",
            "matched_gold_calls": min(len(steps), len(actions))}

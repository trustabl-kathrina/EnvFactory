"""CPU-only strict alignment and wrong-argument taxonomy regression tests."""
from __future__ import annotations

from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import (
    align_episode, classify_wrong_arguments, typed_equal,
)


def episode(actions=None, observations=None, student_actions=None):
    actions = actions or [
        {"name": "S-A", "arguments": {"name": "alice"}},
        {"name": "S-B", "arguments": {"user_id": 123}},
    ]
    observations = observations or [{"user_id": 123}, {"order_id": 456}]
    student_actions = student_actions or [
        {"kind": "tool", "name": "S-A", "arguments": {"name": "alice"}},
        {"kind": "tool", "name": "S-B", "arguments": {"user_id": 999}},
    ]
    states = [{"S": {"step": i + 1}} for i in range(max(len(actions), len(student_actions)))]
    messages = [{"role": "user", "content": "Find alice then inspect user 123"}]
    steps = []
    for i, action in enumerate(student_actions):
        before = {"S": {"step": i}} if i == 0 else states[i - 1]
        count = len(messages)
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [] if action["kind"] != "tool" else [{"id": f"c{i}"}]})
        event = None
        if action["kind"] == "tool":
            event = {"tool_name": action["name"], "tool_arguments": action["arguments"],
                     "tool_response": observations[i] if i == 0 else {},
                     "execution_success": True, "state_after": states[i]}
            messages.append({"role": "tool", "content": "{}"})
        steps.append({"step_index": i, "prompt_message_count": count,
                      "state_before_action": before,
                      "parsed_action": action, "typed_event": event})
    return {
        "gold": {"replay_valid": True, "final_state_match": True,
                 "actions": actions, "observations": observations,
                 "initial_state": {"S": {"step": 0}}, "states_after": states[:len(actions)]},
        "student": {"steps": steps, "messages": messages},
    }


def test_type_aware_argument_comparison():
    assert typed_equal({"a": [1, 2]}, {"a": [1, 2]})
    assert not typed_equal({"a": 1}, {"a": "1"})
    assert not typed_equal({"a": True}, {"a": 1})
    assert not typed_equal([1, 2], [2, 1])


def test_first_wrong_argument_after_exact_gold_prefix():
    result = align_episode(episode())
    assert result["category"] == "FD_CANDIDATE"
    assert result["failure_type"] == "WRONG_ARGUMENT"
    assert result["gold_step"] == 2
    assert result["matched_gold_calls"] == 1
    assert result["student_state"]["previous_tool_observations"] == [{"user_id": 123}]
    assert result["gold_next_action"]["arguments"]["user_id"] == 123


def test_premature_stop_has_gold_tool_target():
    e = episode(student_actions=[
        {"kind": "tool", "name": "S-A", "arguments": {"name": "alice"}},
        {"kind": "final", "content": ""},
    ])
    result = align_episode(e)
    assert result["failure_type"] == "PREMATURE_STOP"
    assert result["gold_next_action"]["name"] == "S-B"


def test_extra_tool_after_gold_path_is_not_fd():
    e = episode(student_actions=[
        {"kind": "tool", "name": "S-A", "arguments": {"name": "alice"}},
        {"kind": "tool", "name": "S-B", "arguments": {"user_id": 123}},
        {"kind": "tool", "name": "S-C", "arguments": {}},
    ])
    e["student"]["steps"][1]["typed_event"]["tool_response"] = {"order_id": 456}
    e["student"]["steps"][2]["state_before_action"] = {"S": {"step": 2}}
    result = align_episode(e)
    assert result["category"] == "TERMINAL_ONLY"
    assert result["failure_type"] == "EXTRA_TOOL_AFTER_COMPLETION"
    assert "gold_next_action" not in result


def test_observation_mismatch_blocks_off_path_target():
    e = episode()
    e["student"]["steps"][0]["typed_event"]["tool_response"] = {"user_id": 555}
    result = align_episode(e)
    assert result["category"] == "ALIGNMENT_UNCERTAIN"
    assert "gold_next_action" not in result


def test_gold_replay_failure_is_environment_failure():
    e = episode()
    e["gold"]["final_state_match"] = False
    assert align_episode(e)["category"] == "ENV_FAILURE"


def test_provenance_uses_actual_observation():
    gold = {"name": "S-B", "arguments": {"user_id": 123, "limit": 10}}
    actual = {"name": "S-B", "arguments": {"user_id": 999, "limit": 10}}
    audit = classify_wrong_arguments(gold, actual, None, [{"user_id": 123}], "inspect user")
    assert audit["primary_subtype"] == "UPSTREAM_PROVENANCE_ERROR"
    assert "PARTIAL_MULTI_ARG_ERROR" in audit["secondary_subtypes"]
    assert audit["wrong_arg_slots"] == 1 and audit["correct_arg_slots"] == 1


def test_missing_required_and_type_are_separate():
    schema = {"function": {"parameters": {"required": ["user_id"]}}}
    gold = {"name": "S-B", "arguments": {"user_id": 123}}
    assert classify_wrong_arguments(gold, {"arguments": {}}, schema, [], "")[
        "primary_subtype"] == "MISSING_REQUIRED_ARG"
    assert classify_wrong_arguments(gold, {"arguments": {"user_id": "123"}}, schema, [], "")[
        "primary_subtype"] == "TYPE_OR_FORMAT_ERROR"

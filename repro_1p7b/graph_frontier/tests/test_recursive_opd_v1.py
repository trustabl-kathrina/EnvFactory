"""CPU tests for immutable split, complete-call parsing and first-error semantics."""
from __future__ import annotations

import asyncio

from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify, exact_parallel_progress
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import parse_calls


def fixture(final: bool = False) -> tuple[dict, dict, dict]:
    gold = {"actions": [{"name": "S-A", "arguments": {"x": 1}},
                        {"name": "S-B", "arguments": {"x": 2}}],
            "observations": [{"user_id": 123}, {"order_id": 456}],
            "initial_state": {"S": {"v": 0}},
            "states_after": [{"S": {"v": 1}}, {"S": {"v": 2}}],
            "final_state": {"S": {"v": 2}}}
    row = {"extra_info": {"mcp_factory_kwargs": {
        "mcp_servers": ["S"], "final_config": {"S": {"v": 2}}}}}
    first = {"step_index": 0, "prompt_message_count": 1,
             "state_before_action": gold["initial_state"],
             "state_after_action": gold["states_after"][0],
             "model_message": {"role": "assistant", "tool_calls": [{"id": "a", "function": {
                 "name": "S-A", "arguments": {"x": 1}}}]},
             "structured_calls": [{"position": 0, "id": "a", "name": "S-A", "arguments": {"x": 1}}],
             "parse_errors": [],
             "typed_events": [{"execution_success": True, "tool_response": {"user_id": 123}}]}
    second = {"step_index": 1, "prompt_message_count": 3,
              "state_before_action": gold["states_after"][0],
              "state_after_action": gold["states_after"][0],
              "model_message": {"role": "assistant", "content": "done"},
              "structured_calls": [], "parse_errors": [], "typed_events": []}
    student = {"steps": [first, second], "final_state": gold["states_after"][0],
               "termination_reason": "MODEL_FINAL",
               "messages": [{"role": "user", "content": "q"}, first["model_message"],
                            {"role": "tool", "content": "obs"}, second["model_message"]]}
    return {"gold": gold, "student": student}, {"audit_id": "task"}, row


def test_parse_structured_single_and_multi():
    calls, errors = parse_calls({"tool_calls": [
        {"id": "1", "function": {"name": "S-A", "arguments": '{"x":1}'}},
        {"id": "2", "function": {"name": "S-B", "arguments": {"x": 2}}}]})
    assert len(calls) == 2 and not errors
    assert calls[0]["arguments"] == {"x": 1}


def test_parallel_progress_requires_response_and_state():
    episode, _, _ = fixture()
    gold = episode["gold"]
    step = {"structured_calls": [
        {"name": "S-A", "arguments": {"x": 1}},
        {"name": "S-B", "arguments": {"x": 2}}],
        "parse_errors": [],
        "typed_events": [{"execution_success": True, "tool_response": gold["observations"][0]},
                         {"execution_success": True, "tool_response": gold["observations"][1]}],
        "state_after_action": gold["final_state"]}
    assert exact_parallel_progress(step, gold, 0) == 2
    step["typed_events"][1]["tool_response"] = {"order_id": 999}
    assert exact_parallel_progress(step, gold, 0) == 0


def test_premature_stop_is_advance_not_terminal():
    episode, entry, row = fixture()
    verdict = asyncio.run(classify(episode, entry, row, []))
    assert verdict["status"] == "FIRST_ERROR"
    assert verdict["error_regime"] == "ADVANCE"
    assert verdict["frontier_depth"] == 1
    assert verdict["gold_next_action"] == episode["gold"]["actions"][1]


def test_unchanged_final_state_does_not_fake_read_only_success():
    episode, entry, row = fixture()
    episode["student"]["final_state"] = episode["gold"]["final_state"]
    verdict = asyncio.run(classify(episode, entry, row, []))
    assert verdict["status"] == "FIRST_ERROR"
    assert verdict["error_regime"] == "ADVANCE"



def test_failed_first_tool_is_start():
    episode, entry, row = fixture()
    step = episode["student"]["steps"][0]
    step["typed_events"][0]["execution_success"] = False
    verdict = asyncio.run(classify(episode, entry, row, []))
    assert verdict["error_regime"] == "START" and verdict["frontier_depth"] == 0


def test_over_continue_is_terminate():
    episode, entry, row = fixture()
    gold = episode["gold"]
    second = episode["student"]["steps"][1]
    second.update(state_before_action=gold["final_state"],
                  model_message={"role": "assistant", "tool_calls": [{"id": "c",
                                 "function": {"name": "S-A", "arguments": {"x": 1}}}]},
                  structured_calls=[{"name": "S-A", "arguments": {"x": 1}}])
    episode["student"]["steps"].insert(1, {
        "step_index": 1, "prompt_message_count": 3,
        "state_before_action": gold["states_after"][0],
        "state_after_action": gold["final_state"],
        "model_message": {"role": "assistant", "tool_calls": [{"id": "b", "function": {
            "name": "S-B", "arguments": {"x": 2}}}]},
        "structured_calls": [{"name": "S-B", "arguments": {"x": 2}}],
        "parse_errors": [], "typed_events": [{"execution_success": True,
                                                  "tool_response": gold["observations"][1]}]})
    second["step_index"] = 2
    second["prompt_message_count"] = 5
    episode["student"]["messages"] += [{"role": "tool", "content": "obs2"}]
    episode["student"]["termination_reason"] = "MAX_TURNS"
    verdict = asyncio.run(classify(episode, entry, row, []))
    assert verdict["error_regime"] == "TERMINATE"
    assert verdict["frontier_depth"] == len(gold["actions"])

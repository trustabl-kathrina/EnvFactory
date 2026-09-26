"""CPU-only deterministic tests for Cycle-3 multi-call semantic screening."""
import json

from repro_1p7b.graph_frontier.cycle3_official_rl.semantic_audit import (
    classify_static, summarize_depth,
)


SCHEMA = [
    {"function": {"name": "S-A", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"function": {"name": "S-B", "parameters": {"type": "object", "properties": {
        "id": {"type": "string"}}, "required": ["id"]}}},
]


def fixture(calls, *, gold=None, observations=None, states=None, reason="parallel_tool_calls"):
    gold = gold or [{"name": "S-A", "arguments": {}},
                    {"name": "S-B", "arguments": {"id": "ID_1234"}}]
    observations = observations or [{"result": "ok"}, {"result": "done"}]
    states = states or [{}, {}]
    fd = {"task_id": "t", "failure_type": "PARSER_OR_FORMAT_ERROR",
          "student_step": 1, "gold_step": 1, "environment": "S",
          "query_hash": "q", "gold_next_action": gold[0],
          "student_action": {"kind": "invalid", "reason": reason}}
    episode = {"gold": {"actions": gold, "observations": observations,
                        "initial_state": {}, "states_after": states},
               "student": {"steps": [{"model_message": {"tool_calls": [
                   {"function": {"name": name, "arguments": args}} for name, args in calls]}}]}}
    return fd, episode


def test_exact_readonly_gold_prefix_needs_replay():
    fd, episode = fixture([("S-A", "{}"), ("S-B", '{"id":"ID_1234"}')])
    result = classify_static(fd, episode, SCHEMA, "ID_1234")
    assert result["replay_candidate"] is True
    assert result["disposition"] == "UNCERTAIN_EXCLUDE"
    assert result["exact_gold_matches"] == {"0": 1, "1": 2}


def test_future_observation_dependency_is_true_failure():
    fd, episode = fixture([("S-A", "{}"), ("S-B", '{"id":"ID_1234"}')],
                          observations=[{"id": "ID_1234"}, {"result": "done"}])
    result = classify_static(fd, episode, SCHEMA, "unrelated query")
    assert result["new_failure_type"] == "DEPENDENCY_VIOLATION"
    assert result["semantic_validity"] == "TRUE_FAILURE"
    assert result["dependency_evidence"]


def test_partial_schema_error_is_reclassified():
    fd, episode = fixture([("S-A", "{}"), ("S-B", "{}")])
    result = classify_static(fd, episode, SCHEMA, "unrelated query")
    assert result["semantic_validity"] == "PARTIALLY_VALID"
    assert result["new_failure_type"] == "TOOL_SCHEMA_INVALID_IN_MULTI_CALL"
    assert result["disposition"] == "RECLASSIFY_AND_KEEP"


def test_wrong_argument_requires_local_gold_tool():
    gold = [{"name": "S-B", "arguments": {"id": "ID_1234"}},
            {"name": "S-A", "arguments": {}}]
    fd, episode = fixture([("S-B", "{}"), ("S-A", "{}")], gold=gold)
    result = classify_static(fd, episode, SCHEMA, "unrelated query")
    assert result["new_failure_type"] == "WRONG_ARGUMENT_IN_MULTI_CALL"


def test_malformed_json_is_true_format_failure():
    fd, episode = fixture([("S-A", "{"), ("S-B", '{"id":"ID_1234"}')])
    result = classify_static(fd, episode, SCHEMA, "unrelated query")
    assert result["case_subtype"] == "MALFORMED_JSON"
    assert result["disposition"] == "KEEP_AS_FD"


def test_single_structured_call_marked_invalid_is_uncertain():
    fd, episode = fixture([("S-A", "{}")])
    result = classify_static(fd, episode, SCHEMA, "unrelated query")
    assert result["case_subtype"] == "TOOL_CALL_PARSE_FAILURE"
    assert result["disposition"] == "UNCERTAIN_EXCLUDE"


def test_no_call_after_length_stop_is_true_format_failure():
    fd, episode = fixture([], reason="generation_length_truncated")
    result = classify_static(fd, episode, SCHEMA, "unrelated query")
    assert result["case_subtype"] == "GENERATION_LENGTH_TRUNCATED"
    assert result["disposition"] == "KEEP_AS_FD"


def test_depth_summary():
    got = summarize_depth([{"gold_step": step} for step in [1, 1, 2, 4]])
    assert got["step1"] == 2 and got["depth_ge2"] == 2
    assert got["mean"] == 2 and got["p75"] == 2.5

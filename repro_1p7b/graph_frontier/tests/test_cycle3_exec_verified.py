"""No model calls; executable decision-core and pointer regression tests."""
from repro_1p7b.graph_frontier.cycle3_official_rl.exec_verified import resolve_evidence
from repro_1p7b.graph_frontier.cycle3_official_rl.exec_verified_miner import resume_after_parallel
from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import align_episode


def action(name, **arguments):
    return {"name": name, "arguments": arguments}


def schema(name, required=()):
    return {"function": {"name": name, "parameters": {
        "type": "object", "properties": {key: {"type": "string"} for key in required},
        "required": list(required), "additionalProperties": False}}}


def decide(calls, *, actions=None, state=None, progress_states=None,
           responses=None, results=None, suffix=None, dependency=None):
    actions = actions or [action("A"), action("B")]
    return resolve_evidence(
        calls=calls, actions=actions, start=0,
        schemas=[schema("A"), schema("B", ("id",)), schema("X")],
        observations=responses or [{"id": "123"}, {"ok": True}], query="test",
        student_state=state if state is not None else {"done": True},
        gold_progress_states=progress_states or [{"done": False}, {"done": True}],
        student_results=results or [({"id": "123"}, True, None), ({"ok": True}, True, None)],
        gold_progress_responses=responses or [{"id": "123"}, {"ok": True}],
        suffix_verified=suffix if suffix is not None else {2: True},
        pre_state={"done": False}, dependency=dependency if dependency is not None else [])


def test_a_parallel_gold_progress():
    result = decide([action("A"), action("B", id="123")],
                    actions=[action("A"), action("B", id="123")])
    assert result["status"] == "VALID_PARALLEL"
    assert result["gold_progress_consumed"] == 2


def test_b_future_dependency_violation():
    result = decide([action("A"), action("B", id="123")],
                    actions=[action("A"), action("B", id="123")],
                    dependency=[{"producer": "A", "field": "id"}])
    assert result["status"] == "DEPENDENCY_VIOLATION"


def test_c_wrong_argument():
    result = decide([action("A"), action("B", id="wrong")],
                    actions=[action("A"), action("B", id="123")],
                    state={"corrupt": True}, suffix={1: False})
    assert result["status"] == "RECLASSIFY_WRONG_ARGUMENT"


def test_d_wrong_tool():
    result = decide([action("A"), action("X")],
                    actions=[action("A"), action("B", id="123")],
                    state={"corrupt": True}, suffix={1: False})
    assert result["status"] == "RECLASSIFY_WRONG_TOOL"


def test_e_harmless_read_extra():
    result = decide([action("A"), action("X")],
                    actions=[action("A"), action("B", id="123")],
                    state={"done": False}, progress_states=[{"done": False}],
                    results=[({"id": "123"}, True, None), ({"read": True}, True, None)],
                    suffix={1: True})
    assert result["status"] == "VALID_PARALLEL"
    assert result["gold_progress_consumed"] == 1
    assert result["redundant_read_calls"] == 1


def test_f_unverified_alternative_excluded():
    result = decide([action("A"), action("B", id="123")],
                    actions=[action("A"), action("B", id="123")],
                    state={"other": True}, suffix={1: False, 2: False})
    assert result["status"] == "UNCERTAIN"


def _episode():
    a, b, c = action("A"), action("B"), action("C")
    return {
        "gold": {"replay_valid": True, "final_state_match": True,
                 "initial_state": {"n": 0}, "actions": [a, b, c],
                 "observations": [{"a": 1}, {"b": 2}, {"c": 3}],
                 "states_after": [{"n": 1}, {"n": 2}, {"n": 3}]},
        "student": {"messages": [{"role": "user", "content": "q"}],
                    "steps": [{"step_index": 0, "prompt_message_count": 1,
                               "state_before_action": {"n": 0},
                               "parsed_action": {"kind": "tool", **a},
                               "typed_event": {"execution_success": True, "tool_name": "A",
                                               "tool_arguments": {}, "tool_response": {"a": 1},
                                               "state_after": {"n": 1}}}]}}


def test_g_exact_single_fast_path_unchanged():
    result = align_episode(_episode())
    assert result["failure_type"] == "ALIGNMENT_UNCERTAIN"
    assert result["matched_gold_calls"] == 1


def test_realign_pointer_exposes_later_wrong_argument():
    episode = _episode()
    episode["student"]["steps"] = [
        {"step_index": 0, "state_before_action": {"n": 0},
         "parsed_action": {"kind": "invalid"}},
        {"step_index": 1, "state_before_action": {"n": 2},
         "parsed_action": {"kind": "tool", "name": "C", "arguments": {"wrong": 1}}},
    ]
    result = resume_after_parallel(episode, 0, 2)
    assert result["status"] == "DEEPER_FD"
    assert result["gold_step"] == 3
    assert result["failure_type"] == "WRONG_ARGUMENT"

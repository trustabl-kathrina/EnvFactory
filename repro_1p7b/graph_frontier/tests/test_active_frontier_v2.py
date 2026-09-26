from repro_1p7b.graph_frontier.active_frontier_v2 import (
    classify_sample,
    inventory_record,
    split_by_task,
)


def _row():
    edge = {
        "edge_id": "A->B:id->id:parameter_flow",
        "edge_type": "parameter_flow",
        "internal_parameter": True,
        "required": True,
        "optional": False,
        "dependency_depth": 1,
        "producer_tool": {"tool_name": "A"},
        "producer_output_parameter": {"parameter_name": "id", "data_type": "string"},
        "consumer_tool": {"tool_name": "B"},
        "consumer_input_parameter": {"parameter_name": "id", "data_type": "string", "user_provided": False},
    }
    return {
        "task_id": "t1",
        "reward_model": {"ground_truth": '[{"name":"A","arguments":{}},{"name":"B","arguments":{"id":"123"}}]'},
        "extra_info": {
            "generation_seed": 7,
            "graph_frontier": {"dependency_depth": 1, "environment_identifiers": ["Mock"], "dependency_edges": [edge]},
        },
    }, edge


def test_inventory_accepts_unique_immediate_internal_edge():
    row, edge = _row()
    result = inventory_record(row, edge, 0)
    assert result["static_eligible"] is True
    assert result["producer_position"] == 0
    assert result["consumer_position"] == 1


def test_inventory_rejects_user_provided_edge():
    row, edge = _row()
    edge = dict(edge, internal_parameter=False)
    result = inventory_record(row, edge, 0)
    assert result["static_eligible"] is False
    assert "not_internal_parameter" in result["static_drop_reasons"]


def _state():
    return {
        "chosen_action": {"kind": "tool", "name": "B", "arguments": {"id": "123"}},
        "internal_value_source": {"consumer_input_parameter": "id", "typed_value": "123"},
        "legal_next_tools": ["B", "D"],
        "prefix_events": [{"tool_name": "A", "tool_arguments": {"q": 1}}],
    }


def test_classification_is_conservative_and_typed():
    state = _state()
    assert classify_sample({"parsed_action": {"kind": "final", "content": "done"}}, state) == "premature_stop"
    assert classify_sample({"parsed_action": {"kind": "tool", "name": "D", "arguments": {}}}, state) == "legal_alternative"
    assert classify_sample({"parsed_action": {"kind": "tool", "name": "C", "arguments": {}}}, state) == "wrong_tool"
    assert classify_sample({"parsed_action": {"kind": "tool", "name": "B", "arguments": {"id": "999"}}}, state) == "wrong_value"
    assert classify_sample({"parsed_action": {"kind": "tool", "name": "B", "arguments": {"id": "123"}}}, state) == "correct"
    assert classify_sample({"parsed_action": {"kind": "tool", "name": "A", "arguments": {"q": 1}}}, state) == "useless_retry"


def test_split_is_task_grouped():
    pairs = [{"task_id": f"t{i}", "pair_id": f"p{i}-{j}"} for i in range(10) for j in range(2)]
    train, val = split_by_task(pairs, 20260920)
    assert not ({p["task_id"] for p in train} & {p["task_id"] for p in val})
    assert len({p["task_id"] for p in val}) == 1

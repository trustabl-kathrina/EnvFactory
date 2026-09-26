import argparse
import json

import pytest

from repro_1p7b.graph_frontier import dynamic_v2 as dynamic


def metric(rate, attempts=40):
    successes = round(rate * attempts)
    return dynamic.beta_metric(successes, attempts)


def capability():
    buckets = {}
    for index, bucket in enumerate(dynamic.BUCKETS[1:]):
        buckets[bucket] = {
            "task_support": 40,
            "internal_reach": metric(0.60 + index * 0.05),
            "conditional_propagation": metric(0.75 - index * 0.20),
            "producer_success": metric(0.80),
            "consumer_after_producer_success": metric(0.70 - index * 0.20),
            "retry_after_error_tasks": metric(0.10 + index * 0.10),
            "semantic_success": metric(0.30 + index * 0.10),
        }
    return {
        "overall": {
            "task_support": 120,
            "internal_reach": metric(0.65),
            "conditional_propagation": metric(0.55),
            "producer_success": metric(0.80),
            "consumer_after_producer_success": metric(0.50),
            "retry_after_error_tasks": metric(0.20, 120),
            "semantic_success": metric(0.40, 120),
        },
        "buckets": buckets,
    }


def sample(output, instruction="do it", history=None):
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "system": "tools",
        "history": history or [],
    }


def call(name, value=1):
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": {"x": value}})
        + "</tool_call>"
    )


def test_beta_metric_keeps_raw_counts_and_smooths():
    row = dynamic.beta_metric(3, 8)
    assert row["numerator"] == 3
    assert row["denominator"] == 8
    assert row["raw_rate"] == pytest.approx(3 / 8)
    assert row["smoothed_rate"] == pytest.approx(4 / 10)


def test_priority_formula_is_normalized_and_floored():
    rows = dynamic.priority_rows(capability(), 0.05)
    assert sum(row["normalized_probability"] for row in rows.values()) == pytest.approx(1)
    assert all(row["normalized_probability"] >= 0.05 for row in rows.values())
    assert rows["depth3plus_internal"]["F_consumer"] > rows["depth1_internal"]["F_consumer"]
    expected = (
        0.30 * rows["depth2_internal"]["F_prop"]
        + 0.35 * rows["depth2_internal"]["F_consumer"]
        + 0.20 * rows["depth2_internal"]["F_retry"]
        + 0.15 * rows["depth2_internal"]["F_semantic"]
    )
    assert rows["depth2_internal"]["raw_priority"] == pytest.approx(expected)


def test_consumer_completion_requires_successful_producer_then_later_consumer():
    row = {
        "manifest": {
            "dependency_edges": [{
                "producer_tool": {"tool_name": "A"},
                "consumer_tool": {"tool_name": "B"},
            }]
        },
        "rollout": {
            "events": [
                {"step_index": 0, "tool_name": "A", "execution_success": True},
                {"step_index": 1, "tool_name": "B", "execution_success": True},
            ]
        },
    }
    counts = dynamic._edge_consumer_counts([row])
    assert counts == {
        "gold_edges": 1,
        "producer_successes": 1,
        "consumers_after_successful_producer": 1,
    }


def test_quality_audit_excludes_retry_after_error():
    first = call("A", 1)
    retry = sample(
        call("A", 1),
        instruction='<tool_response>{"error":"boom"}</tool_response>',
        history=[["start", first]],
    )
    audit = dynamic.quality_audit_v2(retry)
    assert audit["label"] in {"retry_present", "loop_present"}
    assert not audit["eligible_v2"]


def test_quality_audit_keeps_clean_dependency_trace():
    clean = sample(
        call("B", 2),
        instruction='<tool_response>{"success":true,"id":123}</tool_response>',
        history=[["start", call("A", 1)]],
    )
    audit = dynamic.quality_audit_v2(clean)
    assert audit["label"] == "clean_success"
    assert audit["eligible_v2"]


def test_quality_audit_downranks_nonconsecutive_redundancy():
    redundant = sample(
        call("A", 1),
        history=[
            ["first", call("A", 1)],
            ["second", call("B", 2)],
        ],
    )
    audit = dynamic.quality_audit_v2(redundant)
    assert audit["label"] == "redundant_present"
    assert audit["eligible_v2"]


def test_missing_boundary_blocks_without_fallback(tmp_path):
    stage2 = tmp_path / "stage2.json"
    config = tmp_path / "config.yaml"
    stage2.write_text("[]")
    config.write_text("model: test\n")
    report = tmp_path / "audit.json"
    args = argparse.Namespace(
        boundary=tmp_path / "checkpoint-207",
        stage2=stage2,
        config=config,
        report=report,
    )
    with pytest.raises(SystemExit) as raised:
        dynamic.continuation_audit(args)
    assert raised.value.code == 20
    result = json.loads(report.read_text())
    assert result["status"] == "BLOCKED_STAGE1_STATE_MISSING"
    assert "checkpoint-414" in result["forbidden_fallbacks"]

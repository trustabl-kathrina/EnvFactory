"""CPU-only invariants for the larger balanced preference selection."""
from __future__ import annotations

from collections import Counter

from repro_1p7b.graph_frontier.balanced_preference_v3_pairs import (
    TYPES, _balanced_train, _failure, _split,
)


def _pair(kind: str, index: int, repeat: int = 0) -> dict:
    return {"state_type": kind, "state_id": f"{kind}-{index}",
            "task_id": f"task-{index}", "pair_id": f"{kind}-{index}-{repeat}",
            "failure_type": "wrong_tool" if repeat == 0 else "premature_stop"}


def test_balanced_selection_enforces_state_cap_and_type_mix() -> None:
    pool = [_pair(TYPES[0], i, r) for i in range(150) for r in range(2)]
    pool += [_pair(TYPES[1], i + 200, r) for i in range(120) for r in range(2)]
    pool += [_pair(TYPES[2], i + 400, r) for i in range(110) for r in range(2)]
    selected = _balanced_train(pool)
    count = Counter(p["state_type"] for p in selected)
    assert 280 <= len(selected) <= 500
    assert len({p["state_id"] for p in selected}) >= 180
    assert max(Counter(p["state_id"] for p in selected).values()) <= 2
    assert count[TYPES[1]] / len(selected) >= .25
    assert count[TYPES[2]] / len(selected) >= .20


def test_split_is_grouped_by_task() -> None:
    pool = [_pair(t, i) for t in TYPES for i in range(100)]
    train, val, _, _ = _split(pool)
    assert {p["task_id"] for p in train}.isdisjoint({p["task_id"] for p in val})
    assert len({p["state_id"] for p in val if p["state_type"] == TYPES[1]}) >= 10


def test_stop_failure_must_be_observed_tool_call() -> None:
    state = {"state_type": "stop_required", "last_gold_action": {
        "name": "A", "arguments": {"id": 1}}}
    assert _failure(state, {"parsed_action": {"kind": "final", "content": "done"}}) is None
    assert _failure(state, {"parsed_action": {"kind": "tool", "name": "A",
                                              "arguments": {"id": 1}}}) == "useless_retry"
    assert _failure(state, {"parsed_action": {"kind": "tool", "name": "B",
                                              "arguments": {"id": 2}}}) == "extra_tool_after_completion"

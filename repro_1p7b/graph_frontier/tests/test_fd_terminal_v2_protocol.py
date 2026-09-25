"""CPU protocol tests for deterministic, masked FD/terminal v2 exposure."""
from __future__ import annotations

import copy

import pytest

from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import (
    build_schedule, validate_mask,
)


def rows(prefix: str, count: int, base_length: int):
    return [{
        "task_id": f"{prefix}{i}", "state_identity": f"{prefix}{i}",
        "input_ids": list(range(base_length + i)),
        "labels": [-100] * 2 + list(range(2, base_length + i)),
        "prompt_tokens": 2, "total_tokens": base_length + i,
    } for i in range(count)]


@pytest.mark.parametrize(
    "ratio,steps,expected_fd,expected_terminal",
    [(1, 16, 16, 16), (2, 16, 11, 21), (1, 64, 64, 64), (2, 64, 43, 85)],
)
def test_exact_balanced_exposure(ratio, steps, expected_fd, expected_terminal):
    fd, terminal = rows("fd", 50, 30), rows("terminal", 75, 100)
    selected, counts = build_schedule(fd, terminal, ratio=ratio, steps=steps)
    again, again_counts = build_schedule(fd, terminal, ratio=ratio, steps=steps)
    assert counts == again_counts == {
        "first_divergence_tool": expected_fd, "terminal_final": expected_terminal
    }
    assert [r["state_identity"] for r in selected] == [r["state_identity"] for r in again]
    assert len(selected) == 2 * steps
    if steps == 16:
        assert max(r["total_tokens"] for r in selected) == max(r["total_tokens"] for r in terminal)


def test_mask_gate_rejects_prompt_or_observation_loss():
    row = rows("fd", 1, 10)[0]
    validate_mask(row)
    bad = copy.deepcopy(row)
    bad["labels"][0] = 0
    with pytest.raises(RuntimeError):
        validate_mask(bad)


def test_mask_gate_rejects_missing_target():
    row = rows("terminal", 1, 10)[0]
    row["labels"] = [-100] * len(row["labels"])
    with pytest.raises(RuntimeError):
        validate_mask(row)


def test_protocol_rejects_unapproved_step_or_ratio():
    fd, terminal = rows("fd", 50, 30), rows("terminal", 75, 100)
    with pytest.raises(ValueError):
        build_schedule(fd, terminal, ratio=3, steps=64)
    with pytest.raises(ValueError):
        build_schedule(fd, terminal, ratio=1, steps=200)

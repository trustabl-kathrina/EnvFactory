"""CPU tests for first-divergence alignment and conservative gold admission."""
import copy
import unittest

from repro_1p7b.graph_frontier.first_divergence_v1 import (
    align_first_divergence, canonical, reference_from_raw,
)
from repro_1p7b.graph_frontier.graph_oracle_onpolicy_v0 import assistant_only_labels

TASK = "gf-rl-v1-s1-t0"

def fixture():
    calls = [
        {"name": "A", "arguments": {"id": 1}, "masked_arguments": []},
        {"name": "B", "arguments": {"id": 37, "name": "A"}, "masked_arguments": []},
    ]
    raw = {"nodes": [{"steps": [
        {"role": "user", "content": "do task"},
        {"role": "tool_call", "content": [calls[0]]},
        {"role": "tool_response", "content": ['{"ok":true}']},
        {"role": "tool_call", "content": [calls[1]]},
        {"role": "tool_response", "content": ['{"done":true}']},
        {"role": "assistant", "content": "Done."},
    ]}]}
    row = {"task_id": TASK, "frozen_300_overlap": False,
           "extra_info": {"generation_seed": 1, "graph_frontier": {"dependency_depth": 1}},
           "reward_model": {"ground_truth": calls}}
    audit = {"task_id": TASK, "status": "REPLAY_PASS", "frozen_overlap": False,
             "raw_file": "fixture.json"}
    reference, reason = reference_from_raw(row, audit, raw)
    assert reason == "ok"
    return row, audit, raw, reference

def student_step(kind, name=None, arguments=None, response=None, before=None, after=None):
    action = {"kind": kind}
    if kind == "tool":
        action.update(name=name, arguments=arguments)
    event = None
    if kind == "tool":
        event = {"tool_name": name, "tool_arguments": arguments,
                 "tool_response": response, "execution_success": True,
                 "state_after": after}
    return {"parsed_action": action, "typed_event": event, "prompt_messages": [{"role": "user", "content": "do task"}],
            "state_before_action": before}

def align(steps, modify=None):
    row, audit, raw, reference = fixture()
    if modify:
        modify(row, audit, raw, reference)
    return align_first_divergence(
        row, {"task_id": TASK, "rollout_index": 0, "steps": steps}, reference,
        student_checkpoint="dynamic-v1", policy_version="v1")

A = student_step("tool", "A", {"id": 1}, {"ok": True}, {"start": True}, {"a": True})
B = student_step("tool", "B", {"name": "A", "id": 37}, {"done": True}, {"a": True}, {"done": True})
STOP = student_step("final", before={"done": True})

class FirstDivergenceTests(unittest.TestCase):
    def test_wrong_tool_step_two(self):
        x = align([A, student_step("tool", "X", {}, {}, {"a": True})])
        self.assertEqual((x["status"], x["reason"], x["first_divergence_step"]),
                         ("first_divergence", "wrong_tool", 2))
        self.assertEqual(x["sample"]["gold_action"]["name"], "B")

    def test_premature_stop(self):
        x = align([A, student_step("final", before={"a": True})])
        self.assertEqual(x["reason"], "premature_stop")
        self.assertEqual(x["sample"]["target_text"].count("B"), 1)

    def test_perfect(self):
        x = align([A, B, STOP])
        self.assertEqual(x["status"], "perfect")
        self.assertIsNone(x["sample"])

    def test_argument_key_order(self):
        x = align([A, B, STOP])
        self.assertEqual(x["status"], "perfect")

    def test_wrong_arguments(self):
        x = align([A, student_step("tool", "B", {"id": 999, "name": "A"}, {}, {"a": True})])
        self.assertEqual(x["reason"], "wrong_arguments")

    def test_first_only_discard_later(self):
        x = align([A, student_step("tool", "X", {}, {}, {"a": True}),
                   student_step("tool", "Y", {}, {}, {"bad": True})])
        self.assertEqual(x["first_divergence_step"], 2)
        self.assertEqual(x["sample"]["prefix_length"], 1)

    def test_replay_invalid(self):
        row, audit, raw, _ = fixture()
        audit["status"] = "REPLAY_FAILED"
        ref, why = reference_from_raw(row, audit, raw)
        self.assertIsNone(ref)
        self.assertEqual(why, "gold_replay_invalid")

    def test_frozen_overlap(self):
        row, audit, raw, _ = fixture()
        row["frozen_300_overlap"] = True
        self.assertEqual(reference_from_raw(row, audit, raw)[1], "frozen_overlap")

    def test_loss_mask(self):
        self.assertEqual(assistant_only_labels(3, 2), [-100, -100, -100, 1, 1])

    def test_prefix_response_mismatch(self):
        bad = copy.deepcopy(A)
        bad["typed_event"]["tool_response"] = {"ok": False}
        self.assertEqual(align([bad, B])["reason"], "student_prefix_observation_mismatch")

    def test_prefix_state_discontinuity(self):
        bad = copy.deepcopy(B)
        bad["state_before_action"] = {"impossible": True}
        self.assertEqual(align([A, bad])["reason"], "student_prefix_state_discontinuity")

    def test_multi_call_gold_rejected(self):
        row, audit, raw, _ = fixture()
        raw["nodes"][0]["steps"][1]["content"].append(
            {"name": "C", "arguments": {}, "masked_arguments": []})
        self.assertEqual(reference_from_raw(row, audit, raw)[1], "multi_call_or_malformed_gold_turn")

    def test_masked_gold_rejected(self):
        row, audit, raw, _ = fixture()
        row["reward_model"]["ground_truth"][1]["masked_arguments"] = ["id"]
        self.assertEqual(reference_from_raw(row, audit, raw)[1], "masked_gold_arguments")

    def test_invalid_student_alignment(self):
        self.assertEqual(align([{"parsed_action": {"kind": "invalid"}}])["reason"], "invalid_alignment")

    def test_canonical_json(self):
        self.assertEqual(canonical({"b": 2, "a": 1}), canonical({"a": 1, "b": 2}))

if __name__ == "__main__":
    unittest.main()

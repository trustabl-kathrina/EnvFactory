"""Independently replay gold-matching student prefixes in isolated EnvFactory envs."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from repro_1p7b.graph_frontier.collect_preference_rollouts import EnvConfig, env_item
from repro_1p7b.graph_frontier.first_divergence_v1 import canonical
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import (
    checked_plan, reference_for,
)
from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256


def verify_one(sample: dict, rollout: dict, row: dict, reference: dict) -> tuple[bool, str]:
    from agent_system.environments.env_package.envfactory.official_envs import EnvFactoryBatchEnv

    prefix = sample["prefix_length"]
    if (
        sample["task_id"] != rollout.get("task_id")
        or sample["frozen_overlap"] is not False
        or sample["gold_action"] != reference["actions"][prefix]
        or sample["conversation_prefix"] != rollout["steps"][prefix]["prompt_messages"]
        or sample["current_observation"] != rollout["steps"][prefix]["state_before_action"]
    ):
        return False, "provenance_or_sample_mismatch"
    if prefix >= len(reference["actions"]):
        return False, "prefix_beyond_gold"
    env = EnvFactoryBatchEnv(
        int(rollout["sampling_seed"]), 1, 1, False, EnvConfig(8)
    )
    try:
        env.reset([env_item(row)])
        state = env.states[0]
        if canonical(env._save(state)) != canonical(rollout["steps"][0]["state_before_action"]):
            return False, "initial_state_mismatch"
        for index in range(prefix):
            expected = reference["actions"][index]
            observed = rollout["steps"][index]["parsed_action"]
            if observed.get("kind") != "tool" or observed.get("name") != expected["name"]:
                return False, "prefix_action_mismatch"
            if canonical(observed.get("arguments")) != canonical(expected["arguments"]):
                return False, "prefix_argument_mismatch"
            _, _, dones, _ = env.step([{
                "kind": "tool",
                "name": expected["name"],
                "arguments": expected["arguments"],
            }])
            event = state["events"][-1]
            logged = rollout["steps"][index]["typed_event"]
            if (
                event.get("execution_success") is not True
                or canonical(event.get("tool_response")) != canonical(logged.get("tool_response"))
                or canonical(event.get("tool_response")) != canonical(reference["responses"][index])
            ):
                return False, "prefix_response_mismatch"
            if dones[0]:
                return False, "prefix_terminated_early"
            if canonical(env._save(state)) != canonical(
                rollout["steps"][index + 1]["state_before_action"]
            ):
                return False, "prefix_state_mismatch"
        if canonical(env._save(state)) != canonical(sample["current_observation"]):
            return False, "divergence_state_mismatch"
        return True, "replay_pass"
    finally:
        env.close()


def audit(plan_path: Path, samples_path: Path, rollouts: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite prefix audit")
    plan, rows, ledger, raw_by_seed = checked_plan(plan_path)
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    allowed = set(plan["task_ids"])
    if len(samples) != len({s["state_identity"] for s in samples}):
        raise RuntimeError("duplicate sample identity")
    if any(s["task_id"] not in allowed for s in samples):
        raise RuntimeError("sample outside frozen plan")
    output.mkdir(parents=True)
    reasons = collections.Counter()
    ledger_rows = []
    verified = []
    for sample in samples:
        task_id = sample["task_id"]
        reference, why = reference_for(rows[task_id], ledger[task_id], raw_by_seed)
        if reference is None:
            passed, reason = False, "gold_" + why
        else:
            rollout = json.loads((rollouts / (task_id + ".r0.json")).read_text())
            try:
                passed, reason = verify_one(sample, rollout, rows[task_id], reference)
            except Exception as exc:
                passed, reason = False, "replay_exception:" + type(exc).__name__ + ":" + str(exc)[:200]
        reasons[reason] += 1
        ledger_rows.append({
            "task_id": task_id, "state_identity": sample["state_identity"],
            "prefix_length": sample["prefix_length"],
            "independent_prefix_replay_valid": passed, "reason": reason,
        })
        if passed:
            verified.append({**sample, "independent_prefix_replay_valid": True})
        print(json.dumps(ledger_rows[-1], ensure_ascii=False, sort_keys=True), flush=True)
    with (output / "ledger.jsonl").open("w", encoding="utf-8") as out:
        for row in ledger_rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "verified_samples.jsonl").open("w", encoding="utf-8") as out:
        for row in verified:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "first_divergence_prefix_replay_audit_v1",
        "plan_sha256": sha256(plan_path),
        "sample_source_sha256": sha256(samples_path),
        "samples": len(samples),
        "independent_replay_pass": len(verified),
        "independent_replay_fail": len(samples) - len(verified),
        "reasons": dict(sorted(reasons.items())),
        "frozen_exact_id_overlap": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.plan, args.samples, args.rollouts, args.output),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

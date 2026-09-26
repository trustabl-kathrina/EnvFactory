"""Task-disjoint next-action and serial full-task evaluation for Balanced DPO v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import classify_sample
from repro_1p7b.graph_frontier.balanced_preference_v2_pipeline import GF, OLD, OUT, SNAP, frozen
from repro_1p7b.graph_frontier.collect_preference_rollouts import collect_one, generate_action, stable
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256
from repro_1p7b.graph_frontier.state_verifier import compare_final_states

RESULT = OUT / "evaluation"
POSTMORTEM = GF / "reports/graph_targeted_dpo_local_global_postmortem.json"
OLD_STOP = GF / "reports/graph_targeted_dpo_stop_required_eval.json"
SEED = 20260925


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def lock() -> dict[str, Any]:
    source = frozen()
    preference = read(OUT / "manifest.json")
    training = read(OUT.parent.parent / "checkpoints/balanced_graph_dpo_v2_pilot/run_manifest.json")
    if preference["verdict"] != "DATA_READY" or preference["source_data_sha256"] != source["source_data_sha256"]:
        raise ValueError("preference dataset source changed")
    if training["preference_manifest_sha256"] != file_sha256(OUT / "manifest.json"):
        raise ValueError("training preference manifest hash changed")
    if preference["frozen300_train_overlap"] != 0:
        raise ValueError("Frozen300 overlap")
    train_ids = {r["task_id"] for r in rows(OUT / "pairs_train.jsonl")}
    val_ids = {r["task_id"] for r in rows(OUT / "pairs_val.jsonl")}
    heldout_ids = set(source["task_ids"]["heldout_eval"])
    if train_ids & val_ids or train_ids & heldout_ids or val_ids & heldout_ids:
        raise ValueError("task split leakage")
    return source


def frontier_states(source: dict[str, Any]) -> list[dict[str, Any]]:
    states = rows(OLD / "heldout_eval/frontier_states.jsonl")
    if len(states) != 29 or any(s["task_id"] not in source["task_ids"]["heldout_eval"] for s in states):
        raise ValueError("independent 29-state frontier changed")
    return states


def stop_inputs(source: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = read(OLD_STOP)
    paired = {r["task_id"]: r for r in baseline["paired"]}
    strict = set(read(POSTMORTEM)["stop_required_eval"]["strict_task_ids"])
    if len(strict) != 10 or not strict <= set(source["task_ids"]["heldout_eval"]):
        raise ValueError("independent 10 strict-stop tasks changed")
    result = []
    for index, tid in enumerate(source["task_ids"]["heldout_eval"]):
        if tid not in strict:
            continue
        audit = read(SNAP / "replay/tasks" / f"{tid}.replay.json")
        if audit["final_state_match"] is not True or audit["replay_ok"] is not True:
            raise ValueError("heldout completed state invalid")
        trace = read(SNAP / "replay/traces" / f"{tid}.rollout.json")
        baseline_rollout = read(OLD / "heldout_eval/base" / f"{tid}.r0.json")
        messages = list(baseline_rollout["initial_prompt"])
        for j, event in enumerate(trace["events"]):
            call_id = f"gold_stop_{j}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": event["tool_name"], "arguments": event["tool_arguments"]}}]})
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "name": event["tool_name"], "content": stable(event["tool_response"])})
        signature = hashlib.sha256(stable(messages).encode()).hexdigest()
        if signature != paired[tid]["prompt_signature_sha256"]:
            raise ValueError(f"strict-stop prompt changed: {tid}")
        result.append({"task_id": tid, "messages": messages, "tools": baseline_rollout["tools"],
                       "sampling_seed": SEED + index * 1009, "prompt_signature_sha256": signature})
    return result


def isolated(endpoint: str, model: str) -> dict[str, Any]:
    source = lock()
    states = frontier_states(source)
    frontier_dir = RESULT / "frontier"
    found = []
    for index, state in enumerate(states):
        path = frontier_dir / f"{state['state_id']}.json"
        seed = SEED + index * 1009
        if path.exists():
            item = read(path)
        else:
            message, action, usage, finish = generate_action(endpoint, model, state["prompt_messages"],
                                                              state["tools"], seed=seed, temperature=0.0,
                                                              max_tokens=512)
            category = classify_sample({"model_message": message, "parsed_action": action}, state)
            item = {"task_id": state["task_id"], "state_id": state["state_id"], "sampling_seed": seed,
                    "classification": category, "parsed_action": action, "model_message": message,
                    "usage": usage, "finish_reason": finish,
                    "downstream": state["producer_position"] >= 1}
            write(path, item)
        if item["state_id"] != state["state_id"] or item["sampling_seed"] != seed:
            raise ValueError("frontier cache identity changed")
        found.append(item)
        print(f"FRONTIER {len(found)}/{len(states)}", flush=True)
    strict = []
    for state in stop_inputs(source):
        path = RESULT / "strict_stop" / f"{state['task_id']}.json"
        if path.exists():
            item = read(path)
        else:
            message, action, usage, finish = generate_action(endpoint, model, state["messages"], state["tools"],
                                                              seed=state["sampling_seed"], temperature=0.0,
                                                              max_tokens=512)
            item = {"task_id": state["task_id"], "sampling_seed": state["sampling_seed"],
                    "prompt_signature_sha256": state["prompt_signature_sha256"],
                    "parsed_action": action, "model_message": message,
                    "usage": usage, "finish_reason": finish,
                    "correct_stop": action["kind"] == "final" and bool(str(action.get("content") or "").strip())}
            write(path, item)
        if item["prompt_signature_sha256"] != state["prompt_signature_sha256"]:
            raise ValueError("stop cache identity changed")
        strict.append(item)
        print(f"STRICT_STOP {len(strict)}/10", flush=True)
    down = [x for x in found if x["downstream"]]
    summary = {
        "schema_version": "balanced_preference_v2_isolated_eval_v1",
        "frontier_states": len(found), "frontier_exact": sum(x["classification"] == "correct" for x in found),
        "frontier_flow": sum(x["classification"] in ("correct", "ambiguous_same_tool") for x in found),
        "frontier_classes": dict(Counter(x["classification"] for x in found)),
        "downstream_states": len(down), "downstream_exact": sum(x["classification"] == "correct" for x in down),
        "downstream_flow": sum(x["classification"] in ("correct", "ambiguous_same_tool") for x in down),
        "downstream_classes": dict(Counter(x["classification"] for x in down)),
        "strict_stop_tasks": len(strict), "strict_stop_correct": sum(x["correct_stop"] for x in strict),
        "stop_action_kinds": dict(Counter(x["parsed_action"]["kind"] for x in strict)),
        "request_errors": 0,
    }
    write(RESULT / "isolated_summary.json", summary)
    return summary


def full(endpoint: str, model: str) -> dict[str, Any]:
    source = lock()
    tasks = rows(OLD / "heldout_eval/source_tasks.jsonl")
    if [x["task_id"] for x in tasks] != source["task_ids"]["heldout_eval"]:
        raise ValueError("heldout21 source changed")
    output = RESULT / "full_heldout21"
    output.mkdir(parents=True, exist_ok=True)
    found = []
    for index, row in enumerate(tasks):
        result = collect_one(row, endpoint=endpoint, model=model, rollout_index=0,
                             seed=SEED + index * 1009, max_steps=8, temperature=0.0,
                             max_tokens=384, output=output)
        if result["task_id"] != row["task_id"] or result["sampling_seed"] != SEED + index * 1009:
            raise ValueError("heldout rollout cache identity changed")
        found.append(result)
        print(f"FULL {len(found)}/{len(tasks)} success={sum(x['semantic_success'] for x in found)}", flush=True)
    summary = {"schema_version": "balanced_preference_v2_full_eval_v1", "tasks": len(found),
               "success": sum(x["semantic_success"] for x in found),
               "official_reward": sum(x["official_reward"] for x in found) / len(found)}
    write(RESULT / "full_summary.json", summary)
    return summary


def report() -> dict[str, Any]:
    source = lock()
    isolated_result = read(RESULT / "isolated_summary.json")
    full_result = read(RESULT / "full_summary.json")
    baseline = read(POSTMORTEM)
    old_tasks = {x["task_id"]: x for x in baseline["tasks"]}
    rollouts = [read(RESULT / "full_heldout21" / f"{tid}.r0.json")
                for tid in source["task_ids"]["heldout_eval"]]
    metrics = Counter()
    parts = Counter()
    task_cases = []
    for rollout in rollouts:
        tid = rollout["task_id"]
        steps = rollout["steps"]
        tool_steps = [(i, s) for i, s in enumerate(steps) if s["parsed_action"]["kind"] == "tool"]
        calls = [s["parsed_action"] for _, s in tool_steps]
        metrics["final_answer"] += any(s["parsed_action"]["kind"] == "final" for s in steps)
        metrics["8_step_still_tool_calling"] += len(steps) == 8 and steps[-1]["parsed_action"]["kind"] == "tool"
        metrics["adjacent_repeated_call_tasks"] += any(a["name"] == b["name"] and
                                                      stable(a["arguments"]) == stable(b["arguments"])
                                                      for a, b in zip(calls, calls[1:]))
        metrics["total_tool_calls"] += len(calls)
        reward = rollout["terminal_info"]["reward_parts"]["official_reward"]
        for key in ("length_penalty", "trace_score", "state_score", "format_penalty"):
            parts[key] += reward[key]
        first = old_tasks[tid]["base"]["first_edge"]
        producer = first["producer_name"]
        consumer = first["consumer_name"]
        producer_attempts = [(i, s) for i, s in tool_steps if s["parsed_action"]["name"] == producer]
        producer_hits = [(i, s) for i, s in tool_steps if s["parsed_action"]["name"] == producer
                         and s["typed_event"] and s["typed_event"]["execution_success"] is True]
        producer_index = producer_hits[0][0] if producer_hits else None
        consumer_hits = [(i, s) for i, s in tool_steps if producer_index is not None and i > producer_index
                         and s["parsed_action"]["name"] == consumer]
        consumer_success = [(i, s) for i, s in consumer_hits if s["typed_event"] and
                            s["typed_event"]["execution_success"] is True]
        metrics["producer_reach"] += bool(producer_attempts)
        metrics["producer_success"] += producer_index is not None
        metrics["consumer_opportunity"] += producer_index is not None
        metrics["consumer_success"] += bool(consumer_success)
        parameter_correct = False
        if consumer_hits and producer_hits:
            sidecars = list((SNAP / "sidecar").glob(f"{tid}-*.gold.json"))
            if len(sidecars) != 1:
                raise ValueError("heldout sidecar identity ambiguous")
            edges = read(sidecars[0])["dependency_edges"]
            edge = next((e for e in edges if e["producer_tool"]["tool_name"] == producer
                         and e["consumer_tool"]["tool_name"] == consumer and e["internal_parameter"]), None)
            if edge:
                output_key = edge["producer_output_parameter"]["parameter_name"]
                input_key = edge["consumer_input_parameter"]["parameter_name"]
                observed = producer_hits[0][1]["typed_event"].get("returned_fields", {})
                arguments = consumer_hits[0][1]["parsed_action"]["arguments"]
                parameter_correct = output_key in observed and input_key in arguments and observed[output_key] == arguments[input_key]
        metrics["parameter_correct"] += parameter_correct
        downstream = False
        if consumer_success and first["downstream_exists"]:
            sidecars = list((SNAP / "sidecar").glob(f"{tid}-*.gold.json"))
            sequence = read(sidecars[0])["gold_tool_sequence"]
            sequence = [v["tool_name"] if isinstance(v, dict) else v for v in sequence]
            try:
                next_tool = sequence[sequence.index(consumer) + 1]
            except (ValueError, IndexError):
                next_tool = None
            if next_tool:
                downstream = any(i > consumer_success[0][0] and s["parsed_action"]["name"] == next_tool
                                 and s["typed_event"] and s["typed_event"]["execution_success"] is True
                                 for i, s in tool_steps)
        metrics["downstream_eligible"] += bool(consumer_success and first["downstream_exists"])
        metrics["post_consumer_continuation"] += downstream
        sidecars = list((SNAP / "sidecar").glob(f"{tid}-*.gold.json"))
        expected = read(sidecars[0])["expected_final_state"]
        actual = next((s["typed_event"]["state_after"] for _, s in reversed(tool_steps)
                       if s["typed_event"] and "state_after" in s["typed_event"]), "unknown")
        raw_sidecar_exact = compare_final_states(actual, expected)
        # Match the prior postmortem's terminal-state measure: the official
        # verifier state_score=1.0, not raw state JSON identity.
        state_exact = reward["state_score"] == 1.0
        metrics["final_state_match"] += state_exact is True
        task_cases.append({"task_id": tid, "producer_reach": bool(producer_attempts),
                           "producer_success": producer_index is not None,
                           "consumer_success": bool(consumer_success), "parameter_correct": parameter_correct,
                           "downstream_continuation": downstream, "final_state_match": state_exact,
                           "raw_sidecar_state_exact": raw_sidecar_exact,
                           "final_answer": any(s["parsed_action"]["kind"] == "final" for s in steps),
                           "official_reward": rollout["official_reward"], "success": rollout["semantic_success"]})
    report = {
        "schema_version": "balanced_dpo_v2_pilot_report_v1",
        "data_audit": read(OUT / "audit.json"), "serialization_audit": read(OUT / "serialization_audit.json"),
        "smoke": read(OUT / "smoke_verdict.json"),
        "training": read(OUT.parent.parent / "checkpoints/balanced_graph_dpo_v2_pilot/run_manifest.json"),
        "isolated": isolated_result, "full": full_result,
        "full_metrics": {**dict(metrics), **{k + "_mean": v / len(rollouts) for k, v in parts.items()}},
        "task_cases": task_cases,
        "historical": {"base": baseline["aggregate"]["base"], "old_dpo": baseline["aggregate"]["dpo"]},
        "limitations": ["21 heldout tasks and one deterministic trajectory per task",
                        "only one real stop-negative validation task; stop validation accuracy is high variance",
                        "re-split was selected after observing available stop negatives but before training; heldout remained sealed"],
    }
    gates = {
        "consumer_local_preserved": isolated_result["frontier_exact"] > 7 and isolated_result["frontier_flow"] > 14,
        "stop_shortcut_repaired": isolated_result["strict_stop_correct"] >= 5,
        "downstream_full_improved": metrics["post_consumer_continuation"] > 0,
        "overexecution_reduced": (metrics["8_step_still_tool_calling"] < 11
                                  and metrics["adjacent_repeated_call_tasks"] < 10
                                  and parts["length_penalty"] / len(rollouts) < 0.135714),
        "official_not_regressed": full_result["official_reward"] >= 0.242857 - 0.02,
    }
    report["mechanism_gates"] = gates
    report["verdict"] = "GO" if all(gates.values()) else (
        "PARTIAL-GO" if gates["consumer_local_preserved"] and gates["stop_shortcut_repaired"] else "NO-GO")
    data = report["data_audit"]
    data_lines = ["# Balanced Preference v2 data audit", "",
                  f"Verdict: {data['verdict']}; unique train states: {data['train_unique_states']}; pairs: {data['train_pairs']}.",
                  f"A/B/C: {data['train_selected_counts']}; chosen tool/final ratios: {data['chosen_tool_call_rate']:.4f}/{data['chosen_final_answer_rate']:.4f}.",
                  f"Chosen valid: {data['chosen_valid_rate']:.4f}; TRAIN-validation task overlap: {data['split_task_overlap']}; Frozen300 overlap: {data['frozen300_train_overlap']}.",
                  f"Validation A/B/C: {read(OUT / 'manifest.json')['val_type_counts']}; promoted whole TRAIN task: {data['validation_promoted_task_ids']}.",
                  f"State pair cap: {data['state_pair_cap']}; observed failure types: {data['failure_types']}.",
                  "", "Caveat: only eight train stop states produced real extra-tool behavior at K=8; one such task was moved wholly into preference-validation before training.",
                  "The independent 21-task/29-frontier/10-strict-stop heldout tasks were not used in preference construction."]
    (GF / "reports/balanced_preference_v2_data_audit.md").write_text("\n".join(data_lines) + "\n")
    write(GF / "reports/balanced_dpo_v2_pilot.json", report)
    lines = ["# Balanced Conditional Transition Preference v2", "", f"Verdict: {report['verdict']} for a bounded mechanism pilot, not authorization for full training.",
             f"Mechanism gates: {gates}.",
             f"Train: {report['data_audit']['train_pairs']} pairs / {report['data_audit']['train_unique_states']} states; A/B/C = {report['data_audit']['train_selected_counts']}.",
             f"Frozen300 overlap: {report['data_audit']['frozen300_train_overlap']}; task split overlap: {report['data_audit']['split_task_overlap']}.",
             f"Frontier exact: {isolated_result['frontier_exact']}/29; flow: {isolated_result['frontier_flow']}/29; strict stop: {isolated_result['strict_stop_correct']}/10.",
             f"Downstream exact: {isolated_result['downstream_exact']}/{isolated_result['downstream_states']}.",
             f"Official reward: {full_result['official_reward']:.6f}; success: {full_result['success']}/21.",
             f"Full metrics: {report['full_metrics']}", "", "Limitations:"] + [f"- {x}" for x in report["limitations"]]
    (GF / "reports/balanced_dpo_v2_pilot.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("isolated", "full", "report"))
    p.add_argument("--endpoint")
    p.add_argument("--model", default="balanced-v2")
    args = p.parse_args()
    if args.command != "report" and not args.endpoint:
        p.error("model endpoint required")
    result = isolated(args.endpoint, args.model) if args.command == "isolated" else (
        full(args.endpoint, args.model) if args.command == "full" else report())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()

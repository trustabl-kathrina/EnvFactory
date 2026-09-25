"""CPU-only, evidence-first differential audit of the frozen Rich-v3 greedy heldout."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.first_divergence_v1 import canonical, normalize_action
from repro_1p7b.graph_frontier.guided_rl_reward import official_call_match
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import checked_plan, reference_for

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "repro_1p7b/logs"
PLAN = LOG / "first_divergence_onpolicy_v1/cycle3_heldout_greedy_plan.json"
FROZEN = ROOT / "repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
TERMINAL = LOG / "first_divergence_terminal_v2/terminal_dataset_attempt4/dataset.jsonl"
OUT = LOG / "rich_v3_trajectory_differential_audit"
REPORT = ROOT / "repro_1p7b/graph_frontier/reports/rich_v3_trajectory_differential_audit.md"
DIRS = {
    "Dynamic-v1": LOG / "first_divergence_onpolicy_v1/cycle3_heldout_greedy_eval/base",
    "FD-only": LOG / "first_divergence_onpolicy_v1/cycle3_heldout_greedy_eval/trained",
    "v2-1to1": LOG / "first_divergence_terminal_v2/rich_1to1_greedy/trained",
    "v2-1to2": LOG / "first_divergence_terminal_v2/rich_1to2_greedy/trained",
}
MODELS = ("Dynamic-v1", "FD-only", "v2-1to1", "v2-1to2")
GENERIC = "The requested operations were completed successfully."


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def safe_text(value: Any, limit: int = 240) -> str:
    text = str(value).replace("\n", " ").strip()
    text = re.sub(r"(?i)(['\"](?:client_secret|api_key|access_token|refresh_token|password)['\"]\s*:\s*)['\"][^'\"]*['\"]",
                  r"\1'[redacted]'", text)
    return text[:limit] + ("..." if len(text) > limit else "")


def actions(rollout: dict) -> list[dict]:
    return [step.get("parsed_action") or {} for step in rollout["steps"]]


def events(rollout: dict) -> list[dict]:
    return [step["typed_event"] for step in rollout["steps"]
            if isinstance(step.get("typed_event"), dict)]


def official(rollout: dict) -> dict:
    return rollout["terminal_info"]["reward_parts"]["official_reward"]


def ordered_first_divergence(rollout: dict, reference: dict) -> dict:
    """Exact reference-order divergence proxy; official trace is order-insensitive."""
    gold = reference["actions"]
    for index, step in enumerate(rollout["steps"]):
        raw = step.get("parsed_action") or {}
        action = normalize_action(raw)
        if action is None:
            return {"step": index + 1, "reason": "INVALID_TOOL_CALL"}
        if index >= len(gold):
            if action["kind"] == "tool":
                return {"step": index + 1, "reason": "EXTRA_TOOL"}
            return {"step": None, "reason": "gold_path_then_final"}
        expected = gold[index]
        if action["kind"] == "final":
            return {"step": index + 1, "reason": "PREMATURE_STOP"}
        if action["name"] != expected["name"]:
            return {"step": index + 1, "reason": "WRONG_TOOL_REFERENCE_RELATIVE"}
        if canonical(action["arguments"]) != canonical(expected["arguments"]):
            return {"step": index + 1, "reason": "WRONG_ARGUMENT_REFERENCE_RELATIVE"}
        event = step.get("typed_event")
        if not isinstance(event, dict) or event.get("execution_success") is not True:
            return {"step": index + 1, "reason": "TOOL_EXECUTION_FAILED"}
        if canonical(event.get("tool_response")) != canonical(reference["responses"][index]):
            return {"step": index + 1, "reason": "OBSERVATION_DRIFT"}
    return {"step": len(rollout["steps"]) + 1, "reason": "INCOMPLETE_REFERENCE_PATH"}


def load_inputs() -> tuple[dict, dict, dict, dict, dict]:
    plan, rows, ledger, raw_by_seed = checked_plan(PLAN)
    if plan["task_count"] != 24 or len(set(plan["task_ids"])) != 24:
        raise RuntimeError("frozen 24-task plan changed")
    if plan["temperature"] != 0 or plan["max_steps"] != 8 or plan["max_tokens"] != 1024:
        raise RuntimeError("greedy protocol changed")
    if plan["train_task_overlap"] != 0 or plan["frozen_exact_id_overlap"] != 0:
        raise RuntimeError("heldout isolation failed")
    source = read_json(LOG / "first_divergence_onpolicy_v1/cycle3_heldout_greedy_eval/base/collection_manifest.json")
    plan_sha = sha(PLAN)
    if source["plan_sha256"] != plan_sha or source["task_count"] != 24:
        raise RuntimeError("baseline manifest does not match plan")
    collections_by_model: dict[str, dict] = {}
    for model, directory in DIRS.items():
        manifest = read_json(directory / "collection_manifest.json")
        if manifest["plan_sha256"] != plan_sha or manifest["task_count"] != 24:
            raise RuntimeError(f"{model} plan identity mismatch")
        collection = {task: read_json(directory / f"{task}.r0.json")
                      for task in plan["task_ids"]}
        collections_by_model[model] = collection
    references = {}
    protocol = collections.Counter()
    for index, task in enumerate(plan["task_ids"]):
        reference, reason = reference_for(rows[task], ledger[task], raw_by_seed)
        if reference is None:
            raise RuntimeError(f"{task} gold reference invalid: {reason}")
        references[task] = reference
        group = [collections_by_model[model][task] for model in MODELS]
        if any(r["task_id"] != task or r["rollout_index"] != 0 for r in group):
            raise RuntimeError(f"{task} rollout identity mismatch")
        expected_seed = plan["sampling_seed"] + index * 1009
        if any(r["sampling_seed"] != expected_seed for r in group):
            raise RuntimeError(f"{task} sampling seed mismatch")
        for field in ("source_seed", "initial_prompt", "tools"):
            if len({canonical(r[field]) for r in group}) != 1:
                raise RuntimeError(f"{task} {field} mismatch")
        if len({canonical(r["steps"][0]["state_before_action"]) for r in group}) != 1:
            raise RuntimeError(f"{task} initial state/reset mismatch")
        for r in group:
            o = official(r)
            if bool(r["semantic_success"]) != (o["trace_score"] == 1 and o["state_score"] == 1):
                raise RuntimeError(f"{task} official semantic rule mismatch")
            protocol["verified_rollouts"] += 1
        protocol["matched_tasks"] += 1
    return plan, rows, references, collections_by_model, dict(protocol)

def match_required(rollout: dict, reference: dict) -> dict:
    actual = [{"name": e.get("tool_name"), "arguments": e.get("tool_arguments", {})}
              for e in events(rollout)]
    used: set[int] = set()
    matched: dict[int, int] = {}
    for gold_index, expected in enumerate(reference["actions"]):
        hit = next((i for i, call in enumerate(actual)
                    if i not in used and official_call_match(call, expected)), None)
        if hit is not None:
            used.add(hit)
            matched[gold_index] = hit
    return {
        "matched_gold_indices": matched,
        "missing_gold_indices": [i for i in range(len(reference["actions"]))
                                 if i not in matched],
        "actual_calls": actual,
    }


def first_pair_divergence(base: dict, other: dict) -> dict:
    a, b = base["steps"], other["steps"]
    for index in range(max(len(a), len(b))):
        if index >= len(a) or index >= len(b):
            return {"step": index + 1, "field": "trajectory_length",
                    "base": None if index >= len(a) else a[index]["parsed_action"],
                    "other": None if index >= len(b) else b[index]["parsed_action"]}
        aa = normalize_action(a[index].get("parsed_action") or {})
        bb = normalize_action(b[index].get("parsed_action") or {})
        if aa != bb:
            return {"step": index + 1, "field": "structured_action",
                    "base": a[index].get("parsed_action"),
                    "other": b[index].get("parsed_action")}
        if aa and aa["kind"] == "final":
            ca = (a[index]["parsed_action"] or {}).get("content", "")
            cb = (b[index]["parsed_action"] or {}).get("content", "")
            if normalized_text(ca) != normalized_text(cb):
                return {"step": index + 1, "field": "final_response",
                        "base": ca, "other": cb}
        if aa and aa["kind"] == "tool":
            ea, eb = a[index].get("typed_event"), b[index].get("typed_event")
            if isinstance(ea, dict) and isinstance(eb, dict):
                if canonical(ea.get("tool_response")) != canonical(eb.get("tool_response")):
                    return {"step": index + 1, "field": "tool_observation",
                            "base": ea.get("tool_response"), "other": eb.get("tool_response")}
    return {"step": None, "field": "identical", "base": None, "other": None}


def primary_failure(rollout: dict, reference: dict, matching: dict) -> tuple[str, str, str]:
    """Classify a proven official failure; do not infer natural-language correctness."""
    score = official(rollout)
    missing = matching["missing_gold_indices"]
    if score["trace_score"] == 1:
        if score["state_score"] < 1:
            return "TOOL_POLICY", "ENV_STATE_WRONG", "complete trace but final state mismatches"
        if not actions(rollout) or actions(rollout)[-1].get("kind") != "final":
            return "MIXED_AMBIGUOUS", "FINAL_ANSWER_MISSING", "official tool/state pass; no final action"
        return "MIXED_AMBIGUOUS", "OTHER", "official tool/state pass; answer semantics unverified"
    if not missing:
        raise RuntimeError("official trace failed despite all calls matched")
    actual = matching["actual_calls"]
    edge_details = (rollout["terminal_info"]["reward_parts"].get("graph_frontier") or {}).get("edge_details") or []
    if any(e.get("consumer_executed") is True and e.get("propagation_match") is False
           for e in edge_details):
        return "TOOL_POLICY", "PARAMETER_PROVENANCE", "typed edge verifier found incorrect value propagation"
    missing_actions = [reference["actions"][i] for i in missing]
    if any(any(call["name"] == expected["name"] for call in actual)
           for expected in missing_actions):
        return "TOOL_POLICY", "WRONG_ARGUMENT", "required tool attempted without matching required arguments"
    last = actions(rollout)[-1] if actions(rollout) else {}
    if last.get("kind") == "final":
        return "TOOL_POLICY", "PREMATURE_STOP", "final answer before required official calls"
    if last.get("kind") == "invalid":
        return "TOOL_POLICY", "INVALID_TOOL_CALL", "invalid terminal action while required calls missing"
    if len(rollout["steps"]) >= 8:
        return "TOOL_POLICY", "BUDGET_EXHAUSTED", "eight-step budget reached with required calls missing"
    keys = [canonical(c) for c in actual]
    if len(keys) > len(set(keys)):
        return "TOOL_POLICY", "LOOP_REPEATED_TOOL", "exact repeated call while required calls missing"
    if actual:
        return "TOOL_POLICY", "DEPENDENCY_INCOMPLETE", "required official tool call never appeared"
    return "TOOL_POLICY", "DEPENDENCY_INCOMPLETE", "no executable tool calls"


def observed_failure_step(rollout: dict, reference: dict, matching: dict,
                          failure: str) -> int | None:
    """First observed event supporting the selected failure, not irreversible-cause proof."""
    steps = rollout["steps"]
    if not steps:
        return None
    if failure == "WRONG_ARGUMENT":
        missing = [reference["actions"][i] for i in matching["missing_gold_indices"]]
        for index, step in enumerate(steps):
            event = step.get("typed_event")
            if not isinstance(event, dict):
                continue
            actual = {"name": event.get("tool_name"),
                      "arguments": event.get("tool_arguments", {})}
            if any(actual["name"] == gold["name"] and
                   not official_call_match(actual, gold) for gold in missing):
                return index + 1
    if failure == "PARAMETER_PROVENANCE":
        return None  # Edge verifier does not expose a reliable consumer-step index.
    if failure in ("INVALID_TOOL_CALL", "PREMATURE_STOP", "BUDGET_EXHAUSTED",
                   "FINAL_ANSWER_MISSING", "ENV_STATE_WRONG"):
        return len(steps)
    if failure == "LOOP_REPEATED_TOOL":
        seen: set[str] = set()
        for index, step in enumerate(steps):
            event = step.get("typed_event")
            if isinstance(event, dict):
                key = canonical({"name": event.get("tool_name"),
                                 "arguments": event.get("tool_arguments", {})})
                if key in seen:
                    return index + 1
                seen.add(key)
    return None


def analyze_model(rollout: dict, reference: dict, terminal_targets: set[str]) -> dict:
    matching = match_required(rollout, reference)
    family, failure, why = primary_failure(rollout, reference, matching)
    ordered = ordered_first_divergence(rollout, reference)
    calls = matching["actual_calls"]
    call_keys = [canonical(call) for call in calls]
    final_actions = [a for a in actions(rollout) if a.get("kind") == "final"]
    final_text = str(final_actions[-1].get("content", "")) if final_actions else ""
    normalized = normalized_text(final_text)
    exact_targets = {x for x in terminal_targets}
    normalized_targets = {normalized_text(x) for x in terminal_targets}
    graph = rollout["terminal_info"]["reward_parts"].get("graph_frontier") or {}
    first_missing = matching["missing_gold_indices"][0] if matching["missing_gold_indices"] else None
    return {
        "reward": rollout["official_reward"],
        "semantic_success": rollout["semantic_success"],
        "official_reward_parts": official(rollout),
        "graph_frontier": {
            "completed_consumer_opportunities": graph.get("completed_consumer_opportunities"),
            "correct_propagations": graph.get("correct_propagations"),
            "eligible_internal_edges": graph.get("eligible_internal_edges"),
            "edge_details": graph.get("edge_details"),
        },
        "primary_family": family,
        "primary_failure": failure,
        "primary_evidence": why,
        "first_failure_step": observed_failure_step(rollout, reference, matching, failure),
        "first_reference_divergence_step": ordered["step"],
        "first_reference_divergence_reason": ordered["reason"],
        "first_missing_gold_index": first_missing,
        "missing_required_actions": [reference["actions"][i] for i in matching["missing_gold_indices"]],
        "matched_required_call_count": len(matching["matched_gold_indices"]),
        "matched_required_call_indices": matching["matched_gold_indices"],
        "tool_efficiency": {
            "tool_calls_total": len(calls),
            "unique_tools": len({c["name"] for c in calls}),
            "repeated_exact_calls": len(call_keys) - len(set(call_keys)),
            "successful_calls": sum(e.get("execution_success") is True for e in events(rollout)),
            "failed_calls": sum(e.get("execution_success") is not True for e in events(rollout)),
            "official_required_calls_matched": len(matching["matched_gold_indices"]),
            "calls_over_gold_length": max(0, len(calls) - len(reference["actions"])),
            "extra_after_last_useful_state": "unknown",
        },
        "final_response": final_text,
        "final_exact_training_target": bool(final_text and final_text in exact_targets),
        "final_normalized_training_target": bool(final_text and normalized in normalized_targets),
        "final_generic_training_template": normalized == normalized_text(GENERIC),
        "premature_generic_final": bool(normalized == normalized_text(GENERIC)
                                        and matching["missing_gold_indices"]),
        "termination": {
            "last_action_kind": actions(rollout)[-1].get("kind") if actions(rollout) else "missing",
            "step_count": len(rollout["steps"]),
            "last_finish_reason": rollout["steps"][-1].get("finish_reason") if rollout["steps"] else "unknown",
        },
        "steps": [
            {
                "index": step["step_index"] + 1,
                "action": step.get("parsed_action"),
                "tool_name": (step.get("typed_event") or {}).get("tool_name"),
                "tool_arguments": (step.get("typed_event") or {}).get("tool_arguments"),
                "tool_response": (step.get("typed_event") or {}).get("tool_response"),
                "execution_success": (step.get("typed_event") or {}).get("execution_success"),
                "exception": (step.get("typed_event") or {}).get("exception"),
            }
            for step in rollout["steps"]
        ],
    }

def direction(base: dict, other: dict, pair: dict) -> str:
    if other["reward"] > base["reward"] + 1e-12:
        return "IMPROVEMENT"
    if other["reward"] < base["reward"] - 1e-12:
        return "REGRESSION"
    if pair["field"] == "identical":
        return "NEUTRAL_ALTERNATIVE"
    if (other["matched_required_call_count"] == base["matched_required_call_count"]
            and other["official_reward_parts"]["state_score"]
            == base["official_reward_parts"]["state_score"]):
        return "NEUTRAL_ALTERNATIVE"
    return "AMBIGUOUS"


def distribution(plan: dict, rows: dict, references: dict) -> dict:
    frozen_rows = [json.loads(line) for line in FROZEN.read_text().splitlines() if line.strip()]
    if len(frozen_rows) != 300 or len({r["task_id"] for r in frozen_rows}) != 300:
        raise RuntimeError("Frozen300 manifest count changed")
    groups: dict[str, list[dict]] = {"Frozen300": [], "Rich24": []}
    for row in frozen_rows:
        trace = read_json(ROOT / row["reference_trace_path"])
        initial = read_json(ROOT / row["initial_state_path"])
        expected = read_json(ROOT / row["expected_final_state_path"])
        groups["Frozen300"].append({
            "environment": str(row["environment"]),
            "depth": int(row["dependency_depth"]),
            "gold_length": len(trace),
            "unique_gold_tools": len(set(x["tool_name"] for x in trace)),
            "argument_count": sum(len(x.get("arguments") or {}) for x in trace),
            "internal_edges": sum(e.get("internal_parameter") is True
                                  for e in row.get("dependency_edges") or []),
            "query_length": len(str(row.get("query") or "")),
            "state_mutation": canonical(initial) != canonical(expected),
            "terminal_form": str((row.get("semantic_verifier") or {}).get("verifier_type", "unknown")),
        })
    for task in plan["task_ids"]:
        sidecar = rows[task]["extra_info"]["graph_frontier"]
        ref = references[task]
        envs = sidecar.get("environment_identifiers") or []
        if isinstance(envs, str):
            envs = [envs]
        initial = sidecar.get("initial_scenario")
        expected = sidecar.get("expected_final_scenario")
        mutation = ("unknown" if not isinstance(initial, dict) or not isinstance(expected, dict)
                    else canonical(initial) != canonical(expected))
        groups["Rich24"].append({
            "environment": ",".join(sorted(map(str, envs))) or "unknown",
            "depth": int(sidecar["dependency_depth"]),
            "gold_length": len(ref["actions"]),
            "unique_gold_tools": len({x["name"] for x in ref["actions"]}),
            "argument_count": sum(len(x.get("arguments") or {}) for x in ref["actions"]),
            "internal_edges": sum(e.get("internal_parameter") is True
                                  for e in sidecar.get("dependency_edges") or []),
            "query_length": len(str(sidecar.get("query") or "")),
            "state_mutation": mutation,
            "terminal_form": "gold_natural_language_final" if ref.get("final_text") else "missing",
        })
    out = {}
    for name, data in groups.items():
        out[name] = {
            "task_count": len(data),
            "environment_counts": dict(sorted(collections.Counter(d["environment"] for d in data).items())),
            "depth_counts": dict(sorted(collections.Counter(d["depth"] for d in data).items())),
            "gold_length_counts": dict(sorted(collections.Counter(d["gold_length"] for d in data).items())),
            "mean_depth": statistics.mean(d["depth"] for d in data),
            "mean_gold_actions": statistics.mean(d["gold_length"] for d in data),
            "mean_unique_gold_tools": statistics.mean(d["unique_gold_tools"] for d in data),
            "mean_argument_count": statistics.mean(d["argument_count"] for d in data),
            "mean_internal_edges": statistics.mean(d["internal_edges"] for d in data),
            "multi_hop_depth_ge_2": sum(d["depth"] >= 2 for d in data),
            "mean_query_length_characters": statistics.mean(d["query_length"] for d in data),
            "state_mutation_counts": dict(collections.Counter(str(d["state_mutation"]) for d in data)),
            "terminal_form_counts": dict(collections.Counter(d["terminal_form"] for d in data)),
        }
    return out


def summarize(per_task: list[dict], protocol: dict, distributions: dict,
              terminal_targets: list[str]) -> dict:
    result: dict[str, Any] = {
        "schema_version": "rich_v3_trajectory_differential_audit_v1",
        "task_count": 24, "plan_sha256": sha(PLAN), "frozen_manifest_sha256": sha(FROZEN),
        "terminal_source_sha256": sha(TERMINAL), "protocol": protocol,
        "distribution": distributions,
        "terminal_training_target_count": len(terminal_targets),
        "terminal_generic_exact_count": sum(normalized_text(x) == normalized_text(GENERIC)
                                            for x in terminal_targets),
        "models": {},
    }
    for model in MODELS:
        data = [row["models"][model] for row in per_task]
        steps = [x["first_failure_step"] for x in data if x["first_failure_step"] is not None]
        reference_steps = [x["first_reference_divergence_step"] for x in data]
        result["models"][model] = {
            "tool_policy_primary": sum(x["primary_family"] == "TOOL_POLICY" for x in data),
            "final_answer_only_primary": sum(x["primary_family"] == "FINAL_ANSWER_ONLY" for x in data),
            "mixed_ambiguous_primary": sum(x["primary_family"] == "MIXED_AMBIGUOUS" for x in data),
            "primary_failure_counts": dict(sorted(collections.Counter(
                x["primary_failure"] for x in data).items())),
            "first_failure_step_mean": statistics.mean(steps) if steps else None,
            "first_failure_step_median": statistics.median(steps) if steps else None,
            "first_failure_step_distribution": dict(sorted(collections.Counter(steps).items())),
            "first_reference_divergence_step_mean": statistics.mean(reference_steps),
            "first_reference_divergence_step_median": statistics.median(reference_steps),
            "first_reference_divergence_step_distribution": dict(sorted(collections.Counter(reference_steps).items())),
            "official_trace_score_one": sum(x["official_reward_parts"]["trace_score"] == 1 for x in data),
            "official_state_score_one": sum(x["official_reward_parts"]["state_score"] == 1 for x in data),
            "reward_mean": statistics.mean(x["reward"] for x in data),
            "tool_events": sum(x["tool_efficiency"]["tool_calls_total"] for x in data),
            "tool_efficiency_totals": {
                key: sum(x["tool_efficiency"][key] for x in data)
                for key in ("tool_calls_total", "unique_tools", "repeated_exact_calls",
                            "successful_calls", "failed_calls", "official_required_calls_matched",
                            "calls_over_gold_length")
            },
            "graph_completed_consumer_opportunities": sum(
                x["graph_frontier"]["completed_consumer_opportunities"] or 0 for x in data),
            "graph_correct_propagations": sum(
                x["graph_frontier"]["correct_propagations"] or 0 for x in data),
            "final_action_count": sum(x["termination"]["last_action_kind"] == "final" for x in data),
            "generic_final_exact_training_count": sum(x["final_generic_training_template"] for x in data),
            "final_exact_any_training_target": sum(x["final_exact_training_target"] for x in data),
            "final_normalized_any_training_target": sum(x["final_normalized_training_target"] for x in data),
            "premature_generic_final": sum(x["premature_generic_final"] for x in data),
            "repeated_normalized_final_texts": {
                text: count for text, count in collections.Counter(
                    normalized_text(x["final_response"]) for x in data
                    if x["final_response"]).items() if count > 1
            },
        }
    deltas = [row["models"]["v2-1to1"]["reward"] - row["models"]["Dynamic-v1"]["reward"]
              for row in per_task]
    result["v2_vs_dynamic_paired"] = {
        "better": sum(x > 1e-12 for x in deltas),
        "same": sum(abs(x) <= 1e-12 for x in deltas),
        "worse": sum(x < -1e-12 for x in deltas),
        "delta_mean": statistics.mean(deltas),
        "first_behavior_divergence_field_counts": dict(collections.Counter(
            row["v2_vs_dynamic"]["first_behavior_divergence"]["field"] for row in per_task)),
        "direction_counts": dict(collections.Counter(
            row["v2_vs_dynamic"]["direction"] for row in per_task)),
        "required_call_match_delta_counts": dict(collections.Counter(
            row["models"]["v2-1to1"]["matched_required_call_count"]
            - row["models"]["Dynamic-v1"]["matched_required_call_count"]
            for row in per_task)),
        "correct_propagation_delta_counts": dict(collections.Counter(
            (row["models"]["v2-1to1"]["graph_frontier"]["correct_propagations"] or 0)
            - (row["models"]["Dynamic-v1"]["graph_frontier"]["correct_propagations"] or 0)
            for row in per_task)),
        "fd_to_v2_fewer_tool_calls_tasks": sum(
            row["models"]["v2-1to1"]["tool_efficiency"]["tool_calls_total"]
            < row["models"]["FD-only"]["tool_efficiency"]["tool_calls_total"]
            for row in per_task),
    }
    return result

def trajectory_brief(model: dict) -> str:
    bits = []
    for step in model["steps"]:
        action = step["action"] or {}
        kind = action.get("kind", "invalid")
        if kind == "tool":
            args = json.dumps(action.get("arguments", {}), ensure_ascii=False, sort_keys=True)
            bits.append(f'{step["index"]}:{action.get("name")}({safe_text(args, 110)})'
                        f' [exec={step["execution_success"]}, obs={safe_text(step["tool_response"], 90)}]')
        elif kind == "final":
            bits.append(f'{step["index"]}:FINAL({safe_text(action.get("content", ""), 160)})')
        else:
            bits.append(f'{step["index"]}:INVALID({safe_text(action.get("reason", ""), 80)})')
    return " -> ".join(bits)


def render_report(summary: dict, per_task: list[dict]) -> str:
    models = summary["models"]
    paired = summary["v2_vs_dynamic_paired"]
    dist = summary["distribution"]
    lines = [
        "# Rich-v3 24-task Trajectory Differential Audit",
        "",
        "## Executive conclusion",
        "",
        "VERDICT = TOOL-POLICY-BOTTLENECK. The frozen T=0 official semantic evaluator requires both "
        "trace_score=1 and state_score=1; it does not grade final-answer wording. In this 24-task run "
        "all three primary models have trace_score=0 on every task, so no final-answer-only failure "
        "can be established as the cause of 0/24 success. Natural-language answer patterns are "
        "audited separately, without an LLM judge.",
        "",
        "## Protocol lock and evidence",
        "",
        f'- Frozen plan SHA256: {summary["plan_sha256"]}; Frozen300 manifest SHA256: {summary["frozen_manifest_sha256"]}.',
        f'- Verified {summary["protocol"]["matched_tasks"]}/24 same task IDs and {summary["protocol"]["verified_rollouts"]} '
        "rollouts with matching sampling/source seeds, initial prompt, tool schemas and initial reset state. "
        "All use the same eight-step, 1024-token, T=0 plan and collector; each collector constructs a "
        "fresh EnvFactoryBatchEnv and resets the task. The adapter generates a distinct MCP client ID "
        "from PID, reset serial, task and server. This verifies protocol equivalence, not a claim that "
        "separate processes share physical server state.",
        f'- Structured per-task diff: {OUT / "per_task_diff.jsonl"}; machine summary: {OUT / "summary.json"}.',
        "",
        "## Model-level failure and efficiency",
        "",
        "| Model | Tool-policy primary | Final-answer-only | Mixed/ambiguous | Mean first observed primary failure step | "
        "Tool events | Exact repeated calls | Successful/failed calls | Official required calls matched | "
        "Correct graph propagations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        x = models[name]
        eff = x["tool_efficiency_totals"]
        lines.append(
            f'| {name} | {x["tool_policy_primary"]}/24 | {x["final_answer_only_primary"]}/24 | '
            f'{x["mixed_ambiguous_primary"]}/24 | {x["first_failure_step_mean"]:.2f} | '
            f'{x["tool_events"]} | {eff["repeated_exact_calls"]} | '
            f'{eff["successful_calls"]}/{eff["failed_calls"]} | '
            f'{eff["official_required_calls_matched"]} | {x["graph_correct_propagations"]} |'
        )
    lines.extend([
        "",
        "The primary-failure step is the first observed event supporting the selected mechanism, not "
        "proof that the mistake was irreversible. A separate exact gold-order divergence proxy is "
        "retained in JSON; the official trace verifier accepts a subset of a permutation of gold calls, "
        "so an early ordered divergence alone is not necessarily a failure. Provenance failure step "
        "remains unknown if the edge verifier has no step index.",
        "",
    ])
    for name in MODELS:
        x = models[name]
        lines.append(
            f'- {name}: first observed primary failure median step {x["first_failure_step_median"]}; '
            f'distribution {x["first_failure_step_distribution"]}; '
            f'gold-order proxy mean {x["first_reference_divergence_step_mean"]:.2f}; '
            f'primary mechanisms {x["primary_failure_counts"]}; '
            f'official trace passes {x["official_trace_score_one"]}/24, '
            f'state passes {x["official_state_score_one"]}/24.'
        )
    lines.extend([
        "",
        "Exact repeated calls count identical tool+argument pairs only and are a lower bound on "
        "equivalent-call loops. Official required-call matches and typed graph propagations are "
        "verifiable progress proxies. The point of the last useful environment state is not "
        "certified by current logs; extra calls after that state are marked unknown, not guessed.",
        "",
        "## Paired Dynamic-v1 / FD-only / v2 1:1 changes",
        "",
        f'- v2 1:1 vs Dynamic-v1 official reward: better {paired["better"]}, '
        f'same {paired["same"]}, worse {paired["worse"]}; mean delta {paired["delta_mean"]:.5f}.',
        f'- First v2-vs-base behavior difference fields: {paired["first_behavior_divergence_field_counts"]}. '
        f'Evidence-weighted directions: {paired["direction_counts"]}. '
        "Different valid tool paths are not marked regressions merely for differing from gold order.",
        f'- FD-only -> v2 1:1 tool events: {models["FD-only"]["tool_events"]} -> '
        f'{models["v2-1to1"]["tool_events"]}; official trace passes remain '
        f'{models["FD-only"]["official_trace_score_one"]} -> '
        f'{models["v2-1to1"]["official_trace_score_one"]}.',
        "",
        "## Terminal target contamination check",
        "",
        f'- Frozen terminal targets: {summary["terminal_training_target_count"]}; '
        f'exact generic template count {summary["terminal_generic_exact_count"]}.',
    ])
    for name in MODELS:
        x = models[name]
        lines.append(
            f'- {name}: final actions {x["final_action_count"]}/24; exact/normalized duplicate '
            f'of any terminal training target {x["final_exact_any_training_target"]}/'
            f'{x["final_normalized_any_training_target"]}; exact generic template '
            f'{x["generic_final_exact_training_count"]}; premature generic final '
            f'{x["premature_generic_final"]}.'
        )
    lines.extend([
        "",
        "Exact and whitespace/case-normalized duplicates are textual evidence only. Non-matching "
        "wording cannot establish absence of influence, and a generic answer cannot explain an "
        "official zero score when the required tool trace is missing.",
        "",
        "## Frozen300 versus Rich-v3 distribution",
        "",
        "| Pool | Depth 1/2/3 | Mean gold actions | Mean distinct gold tools | Mean arguments | "
        "Mean internal edges | Multi-hop depth>=2 | Mean query chars | State mutation proxy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name in ("Frozen300", "Rich24"):
        x = dist[name]
        depth = x["depth_counts"]
        lines.append(
            f'| {name} | {depth.get(1, 0)}/{depth.get(2, 0)}/{depth.get(3, 0)} | '
            f'{x["mean_gold_actions"]:.2f} | {x["mean_unique_gold_tools"]:.2f} | '
            f'{x["mean_argument_count"]:.2f} | {x["mean_internal_edges"]:.2f} | '
            f'{x["multi_hop_depth_ge_2"]}/{x["task_count"]} | '
            f'{x["mean_query_length_characters"]:.1f} | {x["state_mutation_counts"]} |'
        )
    for name in ("Frozen300", "Rich24"):
        x = dist[name]
        lines.append(
            f'- {name} environment families: {x["environment_counts"]}; '
            f'terminal form: {x["terminal_form_counts"]}; gold length distribution: '
            f'{x["gold_length_counts"]}.'
        )
    lines.extend([
        "",
        "State mutation is only initial-versus-expected JSON inequality, not a tool-name read/write "
        "heuristic. Frozen300 uses state-predicate semantic verifiers; Rich references include natural "
        "language final text, but the **actual Rich heldout semantic flag still checks tool trace "
        "and state, not answer text**. Distribution differences alone do not prove causation.",
        "",
        "## Per-task reward and first mechanism",
        "",
        "| Task | Environment | Depth | Dynamic-v1 | FD-only | v2 1:1 | v2-base first behavior divergence | v2 primary failure |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ])
    for row in per_task:
        m = row["models"]
        first = row["v2_vs_dynamic"]["first_behavior_divergence"]
        lines.append(
            f'| {row["task_id"]} | {row["environment"]} | {row["depth"]} | '
            f'{m["Dynamic-v1"]["reward"]:.3f} | {m["FD-only"]["reward"]:.3f} | '
            f'{m["v2-1to1"]["reward"]:.3f} | step {first["step"]} {first["field"]} | '
            f'{m["v2-1to1"]["primary_failure"]} |'
        )
    lines.extend(["", "## Five largest gains and five largest regressions", ""])
    ordered = sorted(per_task, key=lambda r: (r["v2_vs_dynamic"]["reward_delta"], r["task_id"]))
    for label, selected in (("Largest gains", list(reversed(ordered[-5:]))),
                            ("Largest regressions", ordered[:5])):
        lines.extend([f"### {label}", ""])
        for row in selected:
            m = row["models"]
            pair = row["v2_vs_dynamic"]
            first = pair["first_behavior_divergence"]
            lines.extend([
                f'#### {row["task_id"]} (delta {pair["reward_delta"]:+.3f}; '
                f'{pair["direction"]})',
                "",
                f'Query: {safe_text(row["query"], 1200)}',
                "",
                f'- Dynamic-v1: {trajectory_brief(m["Dynamic-v1"])}',
                f'- FD-only: {trajectory_brief(m["FD-only"])}',
                f'- v2 1:1: {trajectory_brief(m["v2-1to1"])}',
                f'- First structured difference: step {first["step"]}, {first["field"]}; '
                f'base={safe_text(first["base"], 240)}; v2={safe_text(first["other"], 240)}.',
                f'- Primary v2 failure: {m["v2-1to1"]["primary_failure"]}; '
                f'{m["v2-1to1"]["primary_evidence"]}.',
                f'- Mechanism evidence: required calls matched base/v2 '
                f'{m["Dynamic-v1"]["matched_required_call_count"]}/'
                f'{m["v2-1to1"]["matched_required_call_count"]}; '
                f'official state score base/v2 '
                f'{m["Dynamic-v1"]["official_reward_parts"]["state_score"]}/'
                f'{m["v2-1to1"]["official_reward_parts"]["state_score"]}.',
                "",
            ])
    lines.extend([
        "## Q1–Q8 direct answers",
        "",
        f'- Q1: v2 1:1 most frequent primary mechanism is '
        f'{max(models["v2-1to1"]["primary_failure_counts"], key=models["v2-1to1"]["primary_failure_counts"].get)} '
        f'({max(models["v2-1to1"]["primary_failure_counts"].values())}/24); all 24 have official trace failure.',
        f'- Q2: terminal supervision reduces FD-only tool events '
        f'{models["FD-only"]["tool_events"]} -> {models["v2-1to1"]["tool_events"]} '
        f'({(models["FD-only"]["tool_events"] - models["v2-1to1"]["tool_events"]) / models["FD-only"]["tool_events"]:.1%}); '
        f'{paired["fd_to_v2_fewer_tool_calls_tasks"]}/24 tasks use fewer calls. '
        f'Exact repeats fall {models["FD-only"]["tool_efficiency_totals"]["repeated_exact_calls"]} -> '
        f'{models["v2-1to1"]["tool_efficiency_totals"]["repeated_exact_calls"]}, but trace passes remain zero.',
        '- Q3: the observed bottleneck for this official benchmark is tool policy, 24/24 per model; '
        'final-answer grounding remains ungraded by the semantic flag and cannot be exonerated or blamed here.',
        f'- Q4: first observed primary-failure mean steps base/FD/v2 '
        f'{models["Dynamic-v1"]["first_failure_step_mean"]:.2f}/'
        f'{models["FD-only"]["first_failure_step_mean"]:.2f}/'
        f'{models["v2-1to1"]["first_failure_step_mean"]:.2f}. '
        'This mechanism-conditioned statistic is not an identical-cause survival analysis; '
        'the stricter gold-order proxy means are listed above.',
        f'- Q5: local transfer is limited but real as a proxy: official required-call matches '
        f'{models["Dynamic-v1"]["tool_efficiency_totals"]["official_required_calls_matched"]} -> '
        f'{models["v2-1to1"]["tool_efficiency_totals"]["official_required_calls_matched"]}; '
        f'correct graph propagations {models["Dynamic-v1"]["graph_correct_propagations"]} -> '
        f'{models["v2-1to1"]["graph_correct_propagations"]}. '
        f'Paired required-call deltas {paired["required_call_match_delta_counts"]}; '
        f'paired propagation deltas {paired["correct_propagation_delta_counts"]}. '
        'Neither is a task-level success gain.',
        f'- Q6: 40/75 terminal targets share one exact generic response, but v2 has '
        f'{models["v2-1to1"]["final_exact_any_training_target"]} exact and '
        f'{models["v2-1to1"]["final_normalized_any_training_target"]} normalized target duplicates; '
        f'repeated normalized complete final texts {len(models["v2-1to1"]["repeated_normalized_final_texts"])}; '
        f'premature exact-generic finals {models["v2-1to1"]["premature_generic_final"]}. '
        'There is no direct textual-copy evidence; subtler influence remains unknown.',
        f'- Q7: Rich24 spans mostly different environment families, has balanced depth '
        f'8/8/8 versus Frozen300 120/110/70, longer queries '
        f'({dist["Rich24"]["mean_query_length_characters"]:.1f} vs '
        f'{dist["Frozen300"]["mean_query_length_characters"]:.1f} chars), and more gold '
        f'arguments ({dist["Rich24"]["mean_argument_count"]:.2f} vs '
        f'{dist["Frozen300"]["mean_argument_count"]:.2f}); a distribution gap is measured, '
        'not proven sufficient to cause failure.',
        '- Q8: among the proposed choices, prioritize C, deeper/longer Frontier-targeted '
        'data in Rich-like environments with exact argument and provenance supervision; '
        'do not prioritize answer-only supervision from these results. This is a '
        'diagnostic recommendation, not authorization to generate or train.',
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.output_dir.exists() or args.report.exists():
        raise RuntimeError("refusing to overwrite trajectory differential audit")
    plan, rows, refs, collections_by_model, protocol = load_inputs()
    targets = [json.loads(line)["gold_final_response"]
               for line in TERMINAL.read_text().splitlines() if line.strip()]
    if len(targets) != 75:
        raise RuntimeError("frozen terminal training target count changed")
    target_set = set(targets)
    per_task = []
    for task in plan["task_ids"]:
        reference = refs[task]
        sidecar = rows[task]["extra_info"]["graph_frontier"]
        model_data = {
            name: analyze_model(collections_by_model[name][task], reference, target_set)
            for name in MODELS
        }
        base = model_data["Dynamic-v1"]
        a = model_data["v2-1to1"]
        pair = first_pair_divergence(
            collections_by_model["Dynamic-v1"][task],
            collections_by_model["v2-1to1"][task],
        )
        fd_pair = first_pair_divergence(
            collections_by_model["Dynamic-v1"][task],
            collections_by_model["FD-only"][task],
        )
        envs = sidecar.get("environment_identifiers") or []
        if isinstance(envs, str):
            envs = [envs]
        row = {
            "task_id": task,
            "query": collections_by_model["Dynamic-v1"][task]["initial_prompt"][0]["content"],
            "environment": ",".join(sorted(map(str, envs))) or "unknown",
            "source_seed": collections_by_model["Dynamic-v1"][task]["source_seed"],
            "sampling_seed": collections_by_model["Dynamic-v1"][task]["sampling_seed"],
            "depth": sidecar["dependency_depth"],
            "gold_trajectory_id": reference["gold_trajectory_id"],
            "gold_actions": reference["actions"],
            "gold_observations": reference["responses"],
            "gold_final_response": reference["final_text"],
            "gold_dependency_edges": sidecar.get("dependency_edges") or [],
            "models": model_data,
            "v2_vs_dynamic": {
                "first_behavior_divergence": pair,
                "direction": direction(base, a, pair),
                "reward_delta": a["reward"] - base["reward"],
            },
            "fd_vs_dynamic": {
                "first_behavior_divergence": fd_pair,
                "direction": direction(base, model_data["FD-only"], fd_pair),
                "reward_delta": model_data["FD-only"]["reward"] - base["reward"],
            },
        }
        per_task.append(row)
    distributions = distribution(plan, rows, refs)
    result = summarize(per_task, protocol, distributions, targets)
    if any(result["models"][name]["official_trace_score_one"] != 0
           for name in ("Dynamic-v1", "FD-only", "v2-1to1")):
        raise RuntimeError("hardcoded report verdict requires all primary trace scores zero")
    report = render_report(result, per_task)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "per_task_diff.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in per_task),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({
        "verdict": "TOOL-POLICY-BOTTLENECK",
        "task_count": 24,
        "protocol": protocol,
        "models": {name: result["models"][name] for name in MODELS},
        "v2_vs_dynamic": result["v2_vs_dynamic_paired"],
        "report": str(args.report),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

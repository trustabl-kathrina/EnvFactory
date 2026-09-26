#!/usr/bin/env python3
"""Dynamic Graph-Frontier SFT v2: diagnosis, sampling, and continuation gates.

Only the frozen Stage-1 diagnosis split may influence allocation.  The held-out
split and the final Dynamic-v1 evaluation are intentionally outside this path.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from repro_1p7b.data_analysis import parameter_aware_data as feature_lib
from repro_1p7b.graph_frontier.confirm_analyze import (
    events,
    successful,
    task_rows_for_model,
)
from repro_1p7b.graph_frontier.dynamic_v1 import (
    BUCKETS,
    EXPLORATION_FLOOR,
    OVERLAP_CAP_RATE,
    SEED,
    STAGE1_COUNT,
    STAGE2_COUNT,
    UNIQUE_RATIO_FLOOR,
    bucket_name,
    constrain_allocation,
    content_id,
    feature_summary,
    largest_remainder,
    load_json_array,
    load_jsonl,
    load_source_and_features,
    sha256_path,
    tool_calls,
    write_json,
)

BETA_ALPHA = 1.0
BETA_BETA = 1.0
PRIORITY_WEIGHTS = {
    "propagation": 0.30,
    "consumer_completion": 0.35,
    "retry_after_error": 0.20,
    "semantic_frontier": 0.15,
}
FAILURE_STATUS = {"error", "failed", "failure", "exception", "invalid"}
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def beta_metric(successes: int, attempts: int) -> dict[str, Any]:
    successes = int(successes)
    attempts = int(attempts)
    if attempts < 0 or successes < 0 or successes > attempts:
        raise ValueError(f"invalid metric counts: {successes}/{attempts}")
    return {
        "numerator": successes,
        "denominator": attempts,
        "raw_rate": successes / attempts if attempts else None,
        "smoothed_rate": (
            (successes + BETA_ALPHA) / (attempts + BETA_ALPHA + BETA_BETA)
        ),
        "prior": "Beta(1,1)",
    }


def _bucket_for_depth(depth: int) -> str:
    if depth <= 0:
        return "shallow_general"
    if depth == 1:
        return "depth1_internal"
    if depth == 2:
        return "depth2_internal"
    return "depth3plus_internal"


def _edge_consumer_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    gold_edges = 0
    producer_successes = 0
    consumers_after_success = 0
    for row in rows:
        rollout_events = events(row["rollout"])
        for edge in row["manifest"].get("dependency_edges", []):
            gold_edges += 1
            producer = edge["producer_tool"]["tool_name"]
            consumer = edge["consumer_tool"]["tool_name"]
            producer_events = [
                event
                for event in rollout_events
                if event.get("tool_name") == producer and successful(event)
            ]
            if not producer_events:
                continue
            producer_successes += 1
            first_success = min(
                int(event.get("step_index", -1)) for event in producer_events
            )
            if any(
                event.get("tool_name") == consumer
                and int(event.get("step_index", -1)) > first_success
                for event in rollout_events
            ):
                consumers_after_success += 1
    return {
        "gold_edges": gold_edges,
        "producer_successes": producer_successes,
        "consumers_after_successful_producer": consumers_after_success,
    }


def summarize_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    supported = [
        row for row in rows
        if row["semantic"].get("semantic_verifier_supported") is True
    ]
    semantic_success = sum(
        row["semantic"].get("semantic_success") is True for row in supported
    )
    gold = sum(
        int(row["funnel"].get("gold_internal_opportunities", 0)) for row in rows
    )
    reached = sum(int(row["funnel"].get("consumer_reached", 0)) for row in rows)
    correct = sum(int(row["funnel"].get("correct_propagated", 0)) for row in rows)
    retry_tasks = sum(
        int(
            row["calls"]["repeated_pattern_counts"].get(
                "retry_after_tool_error", 0
            )
            > 0
        )
        for row in rows
    )
    edge = _edge_consumer_counts(rows)
    return {
        "task_support": len(rows),
        "semantic_success": beta_metric(semantic_success, len(supported)),
        "internal_reach": beta_metric(reached, gold),
        "conditional_propagation": beta_metric(correct, reached),
        "producer_success": beta_metric(edge["producer_successes"], edge["gold_edges"]),
        "consumer_after_producer_success": beta_metric(
            edge["consumers_after_successful_producer"],
            edge["producer_successes"],
        ),
        "retry_after_error_tasks": beta_metric(retry_tasks, len(rows)),
        "retry_event_count": sum(
            int(
                row["calls"]["repeated_pattern_counts"].get(
                    "retry_after_tool_error", 0
                )
            )
            for row in rows
        ),
    }


def build_capability_map(args: argparse.Namespace) -> None:
    manifest = load_jsonl(args.manifest)
    diagnosis = [row for row in manifest if row.get("split") == "diagnosis"]
    heldout = [row for row in manifest if row.get("split") == "heldout"]
    if len(diagnosis) != 240:
        raise RuntimeError(f"diagnosis manifest count must be 240, got {len(diagnosis)}")
    if len(heldout) != 60:
        raise RuntimeError(f"heldout manifest count must be 60, got {len(heldout)}")
    rows, runtime = task_rows_for_model(args.result_root, diagnosis, args.label)
    if runtime["valid"] < args.minimum_valid:
        raise RuntimeError(f"valid diagnosis gate failed: {runtime}")

    grouped = {
        bucket: [
            row
            for row in rows.values()
            if _bucket_for_depth(int(row["dependency_depth"])) == bucket
        ]
        for bucket in BUCKETS
        if bucket != "shallow_general"
    }
    overall = summarize_bucket(list(rows.values()))
    capability = {
        "schema_version": "graph_frontier_stage1_capability_v2",
        "status": "PASS",
        "source_model_label": args.label,
        "consumed_split": "diagnosis",
        "diagnosis_tasks_consumed": len(diagnosis),
        "heldout_tasks_consumed": 0,
        "final_v1_evaluation_consumed": 0,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_path(args.manifest),
        "runtime": runtime,
        "overall": overall,
        "buckets": {
            bucket: summarize_bucket(grouped[bucket])
            for bucket in grouped
        },
        "definitions": {
            "retry_after_error": (
                "task has at least one retry_after_tool_error event; "
                "denominator is tasks in bucket"
            ),
            "consumer_completion": (
                "consumer executed after an actually successful producer; "
                "denominator is gold edges whose producer succeeded"
            ),
            "smoothing": "all rates use Beta(1,1) posterior means",
        },
    }
    write_json(args.output, capability)
    print(json.dumps(capability, indent=2))


def priority_rows(
    capability: Mapping[str, Any], floor: float = EXPLORATION_FLOOR
) -> dict[str, Any]:
    if not 0.0 <= floor < 1.0 / len(BUCKETS):
        raise ValueError("invalid exploration floor")
    rows: dict[str, Any] = {}
    global_row = capability["overall"]
    for bucket in BUCKETS:
        if bucket == "shallow_general":
            row = global_row
            reach = 0.0
            conditional = 0.0
            producer = 0.0
            consumer = 1.0
        else:
            row = capability["buckets"][bucket]
            reach = float(row["internal_reach"]["smoothed_rate"])
            conditional = float(row["conditional_propagation"]["smoothed_rate"])
            producer = float(row["producer_success"]["smoothed_rate"])
            consumer = float(
                row["consumer_after_producer_success"]["smoothed_rate"]
            )
        semantic = float(row["semantic_success"]["smoothed_rate"])
        retry = float(row["retry_after_error_tasks"]["smoothed_rate"])
        f_prop = reach * (1.0 - conditional)
        f_consumer = producer * (1.0 - consumer)
        f_retry = retry
        f_semantic = 4.0 * semantic * (1.0 - semantic)
        raw = (
            PRIORITY_WEIGHTS["propagation"] * f_prop
            + PRIORITY_WEIGHTS["consumer_completion"] * f_consumer
            + PRIORITY_WEIGHTS["retry_after_error"] * f_retry
            + PRIORITY_WEIGHTS["semantic_frontier"] * f_semantic
        )
        rows[bucket] = {
            "task_support": int(row["task_support"]),
            "F_prop": f_prop,
            "F_consumer": f_consumer,
            "F_retry": f_retry,
            "F_semantic": f_semantic,
            "raw_priority": raw,
            "source_rates": {
                "reach": reach,
                "conditional_propagation": conditional,
                "producer_success": producer,
                "consumer_after_producer_success": consumer,
                "retry_after_error": retry,
                "semantic_success": semantic,
            },
        }
    total = sum(row["raw_priority"] for row in rows.values())
    if total <= 0:
        raise RuntimeError("frontier priority has zero mass")
    residual = 1.0 - floor * len(BUCKETS)
    for row in rows.values():
        row["normalized_probability"] = (
            floor + residual * row["raw_priority"] / total
        )
    return rows


def _response_failed(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("success") is False:
            return True
        if any(key in value for key in ("error", "exception", "traceback")):
            return True
        status = str(value.get("status", "")).strip().lower()
        if status in FAILURE_STATUS:
            return True
        return any(_response_failed(item) for item in value.values())
    if isinstance(value, list):
        return any(_response_failed(item) for item in value)
    return False


def _input_has_failed_response(text: str) -> bool:
    for fragment in TOOL_RESPONSE_RE.findall(text or ""):
        try:
            value = json.loads(fragment)
        except (json.JSONDecodeError, TypeError):
            continue
        if _response_failed(value):
            return True
    return False


def quality_audit_v2(sample: Mapping[str, Any]) -> dict[str, Any]:
    required = {"instruction", "input", "output", "system", "history"}
    malformed_example = (
        not isinstance(sample, Mapping)
        or not required.issubset(sample)
        or not isinstance(sample.get("history"), list)
    )
    if malformed_example:
        return {
            "label": "malformed_or_failed",
            "eligible_v2": False,
            "malformed_example": True,
            "malformed_tool_calls": 0,
            "retry_after_error": False,
            "redundant_exact_calls": 0,
            "loop_present": False,
            "tool_sequence_signature": "invalid",
        }

    parsed, malformed_calls = tool_calls(sample)
    sequence: list[tuple[str, str]] = []
    retry_after_error = False
    previous_name: str | None = None
    previous_signature: str | None = None
    exact_seen: Counter[str] = Counter()
    immediate_repeat = False
    response_failure_count = 0
    for prompt, output in feature_lib.pair_sequence(dict(sample)):
        failed_response = _input_has_failed_response(prompt or "")
        response_failure_count += int(failed_response)
        fragments = feature_lib.TOOL_CALL_RE.findall(output or "")
        malformed_calls += abs(
            (output or "").count("<tool_call>")
            - (output or "").count("</tool_call>")
        )
        for fragment in fragments:
            try:
                call = json.loads(fragment)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                continue
            signature = json.dumps(
                call, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            name = call["name"]
            if failed_response and previous_name == name:
                retry_after_error = True
            immediate_repeat = immediate_repeat or signature == previous_signature
            exact_seen[signature] += 1
            sequence.append((name, signature))
            previous_name = name
            previous_signature = signature

    repeated_exact = sum(max(0, count - 1) for count in exact_seen.values())
    names = [name for name, _ in sequence]
    signatures = [signature for _, signature in sequence]
    repeated_block = any(
        signatures[index - 4:index - 2] == signatures[index - 2:index]
        for index in range(4, len(signatures) + 1)
    )
    loop_present = immediate_repeat or repeated_block
    malformed = malformed_calls > 0
    if malformed:
        label = "malformed_or_failed"
    elif loop_present:
        label = "loop_present"
    elif retry_after_error:
        label = "retry_present"
    elif repeated_exact:
        label = "redundant_present"
    else:
        label = "clean_success"
    return {
        "label": label,
        "eligible_v2": not (malformed or loop_present or retry_after_error),
        "malformed_example": False,
        "malformed_tool_calls": malformed_calls,
        "retry_after_error": retry_after_error,
        "failed_tool_response_turns": response_failure_count,
        "redundant_exact_calls": repeated_exact,
        "loop_present": loop_present,
        "tool_sequence_signature": (
            " -> ".join(names) if names else "no_tool_call"
        ),
    }


def _select_bucket(
    rows: list[tuple[int, str, dict[str, Any]]],
    count: int,
    stage1_hashes: set[str],
    rng: random.Random,
) -> list[tuple[int, str, dict[str, Any]]]:
    groups: dict[tuple[bool, bool], list[tuple[int, str, dict[str, Any]]]] = {}
    for is_overlap in (False, True):
        for redundant in (False, True):
            group = [
                row for row in rows
                if (row[1] in stage1_hashes) == is_overlap
                and (row[2]["label"] == "redundant_present") == redundant
            ]
            rng.shuffle(group)
            groups[(is_overlap, redundant)] = group
    ordered = (
        groups[(False, False)]
        + groups[(False, True)]
        + groups[(True, False)]
        + groups[(True, True)]
    )
    if len(ordered) < count:
        raise RuntimeError(f"bucket capacity {len(ordered)} below allocation {count}")
    return ordered[:count]


def build_stage2(args: argparse.Namespace) -> None:
    source, features = load_source_and_features(args.source, args.features)
    stage1 = load_json_array(args.stage1)
    capability = json.loads(args.capability.read_text())
    if (
        capability.get("consumed_split") != "diagnosis"
        or capability.get("heldout_tasks_consumed") != 0
        or capability.get("final_v1_evaluation_consumed") != 0
    ):
        raise RuntimeError("capability input is not diagnosis-only")

    stage1_hashes = {content_id(sample) for sample in stage1}
    candidates: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    labels = Counter()
    rejection = Counter()
    seen: set[str] = set()
    for pos, (sample, feature) in enumerate(zip(source, features)):
        digest = content_id(sample)
        if digest in seen:
            rejection["duplicate_source_content"] += 1
            continue
        seen.add(digest)
        audit = quality_audit_v2(sample)
        labels[audit["label"]] += 1
        if not audit["eligible_v2"]:
            rejection[audit["label"]] += 1
            continue
        bucket = bucket_name(feature)
        if (
            bucket != "shallow_general"
            and int(feature.get("dependent_calls", 0)) <= 0
        ):
            rejection["unresolved_internal_binding_proxy"] += 1
            continue
        candidates[bucket].append((pos, digest, audit))

    frontier = priority_rows(capability, args.exploration_floor)
    probabilities = {
        bucket: frontier[bucket]["normalized_probability"] for bucket in BUCKETS
    }
    requested = largest_remainder(probabilities, args.count)
    total_capacity = {bucket: len(candidates[bucket]) for bucket in BUCKETS}
    unseen_capacity = {
        bucket: sum(
            digest not in stage1_hashes
            for _, digest, _ in candidates[bucket]
        )
        for bucket in BUCKETS
    }
    max_overlap = math.floor(args.count * args.overlap_cap)
    floor_count = math.floor(args.count * args.exploration_floor)
    allocation = constrain_allocation(
        requested,
        total_capacity,
        unseen_capacity,
        args.count,
        max_overlap,
        floor_count,
        frontier,
    )

    rng = random.Random(args.seed)
    selected_rows: list[tuple[int, str, dict[str, Any], str]] = []
    for bucket in BUCKETS:
        chosen = _select_bucket(
            candidates[bucket], allocation[bucket], stage1_hashes, rng
        )
        selected_rows.extend((*row, bucket) for row in chosen)
    rng.shuffle(selected_rows)
    selected = [source[pos] for pos, _, _, _ in selected_rows]
    selected_features = [features[pos] for pos, _, _, _ in selected_rows]
    selected_hashes = [digest for _, digest, _, _ in selected_rows]
    selected_labels = Counter(audit["label"] for _, _, audit, _ in selected_rows)
    actual_buckets = Counter(bucket for _, _, _, bucket in selected_rows)
    actual_overlap = len(set(selected_hashes) & stage1_hashes)
    signatures = Counter(
        audit["tool_sequence_signature"] for _, _, audit, _ in selected_rows
    )
    total_unique = len(stage1_hashes | set(selected_hashes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    )
    plan = {
        "schema_version": "graph_frontier_dynamic_allocation_v2",
        "status": "PASS",
        "seed": args.seed,
        "formula": {
            "F_prop": "reach * (1 - conditional_propagation)",
            "F_consumer": (
                "producer_success_rate * "
                "(1 - consumer_after_producer_success_rate)"
            ),
            "F_retry": (
                "Beta(1,1) smoothed tasks_with_retry_after_error / "
                "tasks_in_bucket"
            ),
            "F_semantic": "4 * p_semantic * (1 - p_semantic)",
            "priority": (
                "0.30*F_prop + 0.35*F_consumer + "
                "0.20*F_retry + 0.15*F_semantic"
            ),
            "exploration_floor": args.exploration_floor,
        },
        "capability_map_path": str(args.capability),
        "capability_map_sha256": sha256_path(args.capability),
        "capability_map": capability,
        "bucket_priority": frontier,
        "requested_allocation": requested,
        "target_allocation": allocation,
        "quality_eligible_capacity": total_capacity,
        "unseen_quality_eligible_capacity": unseen_capacity,
        "overlap_cap_rate": args.overlap_cap,
        "overlap_cap_count": max_overlap,
        "actual_overlap": actual_overlap,
        "quality_policy": {
            "clean_success": "first choice",
            "redundant_present": "eligible but selected after clean rows",
            "retry_present": "excluded to avoid teaching retry-after-error",
            "loop_present": "excluded",
            "malformed_or_failed": "excluded",
        },
        "source_quality_labels": dict(labels),
        "rejection_counts": dict(rejection),
    }
    stats = {
        "schema_version": "graph_frontier_dynamic_stage2_v2",
        "status": "PASS",
        "sample_count": len(selected),
        "unique_content_count": len(set(selected_hashes)),
        "cross_stage_overlap": actual_overlap,
        "cross_stage_overlap_rate": actual_overlap / len(selected),
        "overall_unique_content": total_unique,
        "overall_unique_ratio": total_unique / (len(stage1) + len(selected)),
        "output_sha256": sha256_path(args.output),
        "bucket_counts": {
            bucket: actual_buckets[bucket] for bucket in BUCKETS
        },
        "selected_quality_labels": dict(selected_labels),
        "retry_present_selected": selected_labels["retry_present"],
        "loop_present_selected": selected_labels["loop_present"],
        "malformed_or_failed_selected": selected_labels["malformed_or_failed"],
        "top_tool_sequence_signatures": dict(signatures.most_common(20)),
        "max_tool_sequence_signature_rate": (
            max(signatures.values(), default=0) / len(selected)
        ),
        **feature_summary(selected_features),
    }
    write_json(args.plan, plan)
    write_json(args.stats, stats)
    print(json.dumps({"plan": plan, "stats": stats}, indent=2))


def validate_datasets(args: argparse.Namespace) -> None:
    stage1 = load_json_array(args.stage1)
    stage2 = load_json_array(args.stage2)
    plan = json.loads(args.plan.read_text())
    stats = json.loads(args.stats.read_text())
    h1 = [content_id(sample) for sample in stage1]
    h2 = [content_id(sample) for sample in stage2]
    audits = [quality_audit_v2(sample) for sample in stage2]
    labels = Counter(audit["label"] for audit in audits)
    overlap = len(set(h1) & set(h2))
    unique_ratio = len(set(h1) | set(h2)) / (len(stage1) + len(stage2))
    checks = {
        "stage1_exact_count": len(stage1) == STAGE1_COUNT,
        "stage2_exact_count": len(stage2) == STAGE2_COUNT,
        "stage1_unique": len(set(h1)) == len(stage1),
        "stage2_unique": len(set(h2)) == len(stage2),
        "retry_after_error_excluded": labels["retry_present"] == 0,
        "loop_excluded": labels["loop_present"] == 0,
        "malformed_excluded": labels["malformed_or_failed"] == 0,
        "overlap_cap": overlap <= math.floor(STAGE2_COUNT * OVERLAP_CAP_RATE),
        "overall_unique_ratio": unique_ratio >= UNIQUE_RATIO_FLOOR,
        "allocation_count": sum(plan["target_allocation"].values()) == STAGE2_COUNT,
        "allocation_matches_stats": (
            plan["target_allocation"] == stats["bucket_counts"]
        ),
        "five_percent_floor": all(
            value >= math.floor(STAGE2_COUNT * EXPLORATION_FLOOR)
            for value in plan["target_allocation"].values()
        ),
        "diagnosis_only": (
            plan["capability_map"]["consumed_split"] == "diagnosis"
            and plan["capability_map"]["heldout_tasks_consumed"] == 0
            and plan["capability_map"]["final_v1_evaluation_consumed"] == 0
        ),
        "source_hash_matches": (
            stats["output_sha256"] == sha256_path(args.stage2)
        ),
        "no_signature_domination": (
            stats["max_tool_sequence_signature_rate"] <= 0.25
        ),
    }
    report = {
        "schema_version": "graph_frontier_dynamic_validation_v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "stage2_quality_labels": dict(labels),
        "cross_stage_overlap": overlap,
        "cross_stage_overlap_rate": overlap / len(stage2),
        "overall_unique_ratio": unique_ratio,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def continuation_audit(args: argparse.Namespace) -> None:
    required_root = [
        args.boundary / "trainer_state.json",
        args.boundary / "scheduler.pt",
        args.boundary / "model.safetensors",
    ]
    optimizer = sorted(args.boundary.glob("global_step207/*optim_states.pt"))
    model_states = sorted(args.boundary.glob("global_step207/*model_states.pt"))
    rng_states = sorted(args.boundary.glob("rng_state_*.pth"))
    checks = {
        "boundary_directory_exists": args.boundary.is_dir(),
        "trainer_state_exists": required_root[0].is_file(),
        "scheduler_exists": required_root[1].is_file(),
        "model_weights_exist": required_root[2].is_file(),
        "optimizer_rank_count_is_2": len(optimizer) == 2,
        "model_state_rank_count_is_2": len(model_states) == 2,
        "rng_rank_count_is_2": len(rng_states) == 2,
        "stage2_dataset_exists": args.stage2.is_file(),
        "stage2_config_exists": args.config.is_file(),
    }
    global_step = None
    if checks["trainer_state_exists"]:
        try:
            global_step = int(
                json.loads(required_root[0].read_text()).get("global_step")
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            global_step = None
    checks["global_step_is_207"] = global_step == 207
    status = (
        "PASS"
        if all(checks.values())
        else "BLOCKED_STAGE1_STATE_MISSING"
    )
    report = {
        "schema_version": "graph_frontier_dynamic_v2_continuation_audit",
        "status": status,
        "boundary": str(args.boundary),
        "expected_global_step": 207,
        "observed_global_step": global_step,
        "required_resume_mode": "native_deepspeed_full_state",
        "forbidden_fallbacks": [
            "dynamic_v1_final",
            "checkpoint-390",
            "checkpoint-414",
            "weights_only_resume",
            "fresh_optimizer",
        ],
        "checks": checks,
        "found_optimizer_states": [str(path) for path in optimizer],
        "found_model_states": [str(path) for path in model_states],
        "found_rng_states": [str(path) for path in rng_states],
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(20)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    command = sub.add_parser("capability")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--result-root", type=Path, required=True)
    command.add_argument("--label", default="stage1")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--minimum-valid", type=int, default=228)
    command.set_defaults(func=build_capability_map)

    command = sub.add_parser("stage2")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--stage1", type=Path, required=True)
    command.add_argument("--capability", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--stats", type=Path, required=True)
    command.add_argument("--count", type=int, default=STAGE2_COUNT)
    command.add_argument("--seed", type=int, default=SEED)
    command.add_argument("--overlap-cap", type=float, default=OVERLAP_CAP_RATE)
    command.add_argument(
        "--exploration-floor", type=float, default=EXPLORATION_FLOOR
    )
    command.set_defaults(func=build_stage2)

    command = sub.add_parser("validate")
    command.add_argument("--stage1", type=Path, required=True)
    command.add_argument("--stage2", type=Path, required=True)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--stats", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.set_defaults(func=validate_datasets)

    command = sub.add_parser("continuation-audit")
    command.add_argument("--boundary", type=Path, required=True)
    command.add_argument("--stage2", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.set_defaults(func=continuation_audit)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

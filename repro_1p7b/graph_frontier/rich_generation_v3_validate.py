"""Static, executable, flow, leakage, and contamination gates for rich v3.

This module never participates in task generation.  Frozen300 is opened only
by ``finalize`` for a one-way, post-hoc normalized-query hash comparison; no
Frozen300 text, graph, trajectory, or state is copied into an output artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from repro_1p7b.graph_frontier.rich_generation_v3 import (
    EXPECTED_FROZEN300_SHA256,
    canonical,
    edge_signature,
    file_sha256,
    read_jsonl,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FROZEN = ROOT / (
    "repro_1p7b/results/graph_frontier/confirm_300/frozen/"
    "confirm_300_seed_20260914.jsonl"
)


def decode(value: Any) -> Any:
    for _ in range(3):
        if not isinstance(value, str):
            return value
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def normalized_query(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def query_hash(value: Any) -> str:
    return hashlib.sha256(normalized_query(value).encode("utf-8")).hexdigest()


def sidecar_query(sidecar: Mapping[str, Any]) -> str:
    query = sidecar.get("query") or {}
    return str(query.get("text", "")) if isinstance(query, Mapping) else ""


def compatible_sidecar(value: Mapping[str, Any]) -> dict[str, Any]:
    sidecar = dict(value)
    if sidecar.get("schema_version") == "envfactory_gold_sidecar_graph_frontier_rich_v1":
        sidecar["schema_version"] = "envfactory_gold_sidecar_v1"
        sidecar["rich_extension_schema_version"] = "graph_frontier_rich_v1"
    return sidecar


def internal_edges(sidecar: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(edge)
        for edge in sidecar.get("dependency_edges", []) or []
        if isinstance(edge, Mapping)
        and edge.get("required") is True
        and edge.get("internal_parameter") is True
    ]


def normalized_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    producer = edge.get("producer_tool") or {}
    consumer = edge.get("consumer_tool") or {}
    source = edge.get("producer_output_parameter") or {}
    target = edge.get("consumer_input_parameter") or {}
    producer_name = str(producer.get("tool_name", "unknown"))
    consumer_name = str(consumer.get("tool_name", "unknown"))
    return {
        "producer_environment": producer_name.split("-", 1)[0],
        "producer_tool": producer_name,
        "producer_output_field": source.get("parameter_name", "unknown"),
        "consumer_environment": consumer_name.split("-", 1)[0],
        "consumer_tool": consumer_name,
        "consumer_argument": target.get("parameter_name", "unknown"),
    }


def direct_tool_hint(query: str, sidecar: Mapping[str, Any]) -> list[str]:
    lowered = query.lower()
    hits = []
    for item in sidecar.get("gold_tool_sequence", []) or []:
        name = str(item.get("tool_name", "")) if isinstance(item, Mapping) else ""
        short = name.split("-", 1)[-1]
        # Exact implementation identifiers are hard hints.  Natural-language
        # paraphrases (for example "search users") are intentionally allowed.
        if name and name.lower() in lowered:
            hits.append(name)
        elif short and ("_" in short or "-" in short) and short.lower() in lowered:
            hits.append(short)
    return sorted(set(hits))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from repro_1p7b.graph_frontier.audit_generated_guided_rl import (
        metadata_index,
        structural_checks,
    )
    from repro_1p7b.graph_frontier.guided_rl_generate import enrich_official_rows
    from src.utils.data_process import convert_to_rl_data, load_tool_chains

    plan = read_jsonl(args.plan)
    plans = {str(row["plan_id"]): row for row in plan}
    sidecars = [
        compatible_sidecar(json.loads(path.read_text()))
        for path in sorted((args.run_dir / "sidecar").glob("*.gold.json"))
    ]
    sidecars_by_id = {str(row["task_id"]): row for row in sidecars}
    converted_dir = args.run_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    official_path = converted_dir / "official_all.json"
    chains = load_tool_chains([str(args.run_dir / "raw")])
    convert_to_rl_data(chains, str(official_path), shuffle=False, seed=args.seed)
    official = json.loads(official_path.read_text())
    enriched = enrich_official_rows(official, sidecars)
    rows = {str(row["task_id"]): row for row in enriched}

    registry_payload = json.loads(args.registry.read_text())
    registry = registry_payload.get("mcpServers", {})
    metadata = metadata_index(args.metadata_dir)
    ledger = []
    for task_id, target in plans.items():
        reasons: list[str] = []
        sidecar = sidecars_by_id.get(task_id)
        row = rows.get(task_id)
        graph_ok = False
        tool_ok = False
        if sidecar is None or row is None:
            reasons.append("query_generation_failure")
        else:
            graph_ok, tool_ok, structural_reasons = structural_checks(
                sidecar, registry, metadata, ROOT
            )
            reasons.extend(structural_reasons)
            edges = internal_edges(sidecar)
            planned = {edge_signature(edge) for edge in target["edge_path"]}
            resolved = {edge_signature(normalized_edge(edge)) for edge in edges}
            if sidecar.get("structure_selected_before_query_generation") is not True:
                reasons.append("structure_not_selected_before_query")
            if sidecar.get("dependency_resolution", {}).get("status") != "resolved":
                reasons.append("ambiguous_graph")
            if sidecar.get("dependency_depth") != target["target_depth"]:
                reasons.append("depth_mismatch")
            if len(edges) != target["target_depth"]:
                reasons.append("internal_edge_count_mismatch")
            if resolved != planned:
                reasons.append("target_edge_mismatch")
            ground_truth = decode((row.get("reward_model") or {}).get("ground_truth", []))
            selected = [
                str(call.get("name")) for call in ground_truth
                if isinstance(call, Mapping) and isinstance(call.get("name"), str)
            ] if isinstance(ground_truth, list) else []
            expected = [str(item.get("tool_name")) for item in sidecar.get("gold_tool_sequence", []) or []]
            if selected != expected:
                reasons.append("gold_trajectory_path_mismatch")
            hints = direct_tool_hint(sidecar_query(sidecar), sidecar)
            if hints:
                reasons.append("direct_tool_identifier_in_query")
        status = "STATIC_PASS" if not reasons and graph_ok and tool_ok else "STATIC_FAILED"
        ledger.append({
            "task_id": task_id,
            "bucket": target["bucket"],
            "target_depth": target["target_depth"],
            "status": status,
            "static_pass": status == "STATIC_PASS",
            "graph_ok": graph_ok,
            "tool_schema_and_mapping_ok": tool_ok,
            "reasons": sorted(set(reasons)),
        })

    write_json(converted_dir / "train.json", list(rows.values()))
    write_json(converted_dir / "rl_val.json", [])
    ledger_path = args.run_dir / "audit" / "static_ledger.jsonl"
    write_jsonl(ledger_path, ledger)
    summary = {
        "schema_version": "graph_frontier_rich_static_audit_v1",
        "sampled_structures": len(plan),
        "query_generated": len(sidecars),
        "converted_rows": len(rows),
        "static_pass": sum(row["status"] == "STATIC_PASS" for row in ledger),
        "mcp_pass": sum(row["tool_schema_and_mapping_ok"] for row in ledger),
        "by_bucket": {
            bucket: {
                "planned": sum(row["bucket"] == bucket for row in ledger),
                "generated": sum(row["bucket"] == bucket and row["task_id"] in sidecars_by_id for row in ledger),
                "static_pass": sum(row["bucket"] == bucket and row["status"] == "STATIC_PASS" for row in ledger),
            }
            for bucket in sorted({row["bucket"] for row in ledger})
        },
        "drop_reasons": dict(Counter(reason for row in ledger for reason in row["reasons"])),
        "ledger": str(ledger_path),
        "converted_dir": str(converted_dir),
        "frozen300_content_loaded": False,
    }
    write_json(args.run_dir / "audit" / "static_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def visible_user_text(row: Mapping[str, Any], sidecar: Mapping[str, Any]) -> str:
    values = [sidecar_query(sidecar)]
    prompt = decode(row.get("prompt", []))
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, Mapping) and message.get("role") == "user":
                values.append(str(message.get("content", "")))
    return "\n".join(values)


def scalar_tokens(values: Any) -> list[str]:
    if not isinstance(values, list) or len(values) != 1:
        return []
    value = values[0]
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


def token_visible(token: str, text: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9_-]+", token):
        return re.search(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])", text, re.I) is not None
    return token.casefold() in text.casefold()


def frozen_query_hashes(path: Path) -> set[str]:
    if file_sha256(path) != EXPECTED_FROZEN300_SHA256:
        raise RuntimeError("Frozen300 manifest SHA256 mismatch")
    hashes = set()
    for row in read_jsonl(path):
        query = row.get("query")
        if query is None:
            prompt = decode(row.get("prompt", []))
            if isinstance(prompt, list):
                users = [message.get("content", "") for message in prompt if isinstance(message, Mapping) and message.get("role") == "user"]
                query = users[-1] if users else ""
        hashes.add(query_hash(query))
    return hashes


def profile_for_task(task_id: str, sidecar: Mapping[str, Any], replay_dir: Path) -> dict[str, Any]:
    from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace

    trace = replay_dir / "traces" / f"{task_id}.rollout.json"
    return profile_sidecar_and_trace(sidecar, trace)


def write_formal_reports(report: Mapping[str, Any], args: argparse.Namespace) -> None:
    capacity = json.loads(args.capacity_report.read_text())
    plan_summary = json.loads(args.plan_summary.read_text())
    generation = json.loads((args.run_dir / "generation_run.json").read_text())
    plans = read_jsonl(args.plan)
    structural = report["structural_result"]
    forecast_states = int(structural["internal_required_edges"] * 9 / 33)
    forecast_pairs = forecast_states * 2
    manifest = {
        "schema_version": "graph_frontier_rich_train_v1_manifest",
        "verdict": report["verdict"],
        "generator_model": generation.get("generator_model"),
        "generation_config": {
            key: generation.get(key)
            for key in ("model_names", "pass_k", "concurrency", "attempts_per_structure", "runtime_seconds")
        },
        "generation_seeds": {
            "count": len(plans),
            "min": min((row["generation_seed"] for row in plans), default=None),
            "max": max((row["generation_seed"] for row in plans), default=None),
        },
        "toolgraph_sha256": capacity.get("graph_sha256"),
        "plan_sha256": file_sha256(args.plan),
        "task_count": structural["valid_tasks"],
        "total_gold_edges": structural["total_gold_edges"],
        "internal_required_edges": structural["internal_required_edges"],
        "depth_distribution": {
            "depth1": structural["depth1_edges"],
            "depth2": structural["depth2_edges"],
            "depth3plus": structural["depth3plus_edges"],
        },
        "environment_distribution": structural["environment_edge_incidence"],
        "unique_internal_edge_signatures": structural["unique_internal_edge_signatures"],
        "drop_reasons": report["drop_reasons"],
        "frozen300_exact_overlap": report["frozen300"]["exact_overlap"],
        "artifacts": report["artifacts"],
    }
    manifest_path = args.run_dir / "valid" / "manifest.json"
    write_json(manifest_path, manifest)
    combined = {
        "schema_version": "graph_frontier_rich_generation_audit_v1",
        "verdict": report["verdict"],
        "toolgraph_capacity": capacity,
        "plan": plan_summary,
        "generation": generation,
        "validation": dict(report),
        "preference_forecast": {
            "basis": "observed Active-v2 conversion: 33 internal edges -> 9 unique paired states -> 18 pairs",
            "expected_unique_paired_frontier_states": forecast_states,
            "expected_preference_pairs": forecast_pairs,
            "forecast_only_not_observed": True,
        },
        "manifest": {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path)},
        "protocol": {
            "no_dpo_training": True,
            "no_dynamic_v1_sampling": True,
            "no_full_rl": True,
            "frozen300_used_only_for_post_hoc_one_way_query_hash_comparison": True,
        },
    }
    write_json(args.audit_json, combined)
    environments = sorted(
        capacity.get("environment_capacity", []),
        key=lambda row: (-row.get("internal_compatible_edges", 0), row.get("environment", "")),
    )
    capacity_rows = "\n".join(
        f"| {row['environment']} | {row.get('relevant_nodes', 'unknown')} | {row.get('tool_nodes', 'unknown')} | "
        f"{row.get('parameter_nodes', 'unknown')} | {row.get('incident_graph_edges', 'unknown')} | "
        f"{row.get('internal_compatible_edges', 'unknown')} | {row.get('max_dependency_depth', 'unknown')} |"
        for row in environments
    )
    funnel = report["funnel"]
    drops = report["drop_reasons"]
    drop_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(drops.items())) or "- none: 0"
    md = f"""# Graph-Frontier-Rich generation audit

## Final verdict

`{report['verdict']}`

## A. ToolGraph capacity

The graph contains {capacity.get('tool_nodes')} tools, {capacity.get('parameter_nodes')} parameters,
{capacity.get('networkx_edges')} NetworkX edges, and {capacity.get('internal_compatible_edge_signatures')}
unique internal-compatible edge signatures. It can support 500 diverse edge instances without reuse:
`{capacity.get('can_support_500_edge_instances_without_signature_reuse')}`.

| environment | nodes | tools | parameters | graph edges | internal-compatible edges | max depth |
|---|---:|---:|---:|---:|---:|---:|
{capacity_rows}

## B. Generation funnel

- sampled structures: {funnel['sampled_structures']}
- query generated: {funnel['query_generated']}
- static pass: {funnel['static_pass']}
- MCP pass: {funnel['mcp_pass']}
- replay pass: {funnel['replay_pass']}
- no-leak pass: {funnel['no_leak_pass']}
- final valid tasks: {funnel['final_valid_tasks']}

## C. Structural result

- valid tasks: {structural['valid_tasks']}
- total gold edges: {structural['total_gold_edges']}
- internal required edges: {structural['internal_required_edges']}
- depth1 edges: {structural['depth1_edges']}
- depth2 edges: {structural['depth2_edges']}
- depth3+ edges: {structural['depth3plus_edges']}
- unique internal edge signatures: {structural['unique_internal_edge_signatures']}
- unique producer tools: {structural['unique_producer_tools']}
- unique consumer tools: {structural['unique_consumer_tools']}
- environments: {structural['unique_environments']}

## D. Drop reasons

{drop_lines}

## E. Frozen300 isolation

- exact overlap in retained pool: {report['frozen300']['exact_overlap']}
- method: post-hoc one-way normalized-query hash comparison only
- Frozen query text, graph, trajectory, and state were not persisted or used for generation.

## F. Preference forecast

Using the prior observed conversion only as a forecast (33 edges -> 9 states -> 18 pairs):

- expected unique paired frontier states: about {forecast_states}
- expected preference pairs: about {forecast_pairs}

These are estimates, not actual Dynamic-v1 sampling results.

## Protocol stop

No Dynamic-v1 sampling, DPO, SFT, GRPO, PPO, full RL, Frozen300 evaluation, or BFCL was started.
"""
    args.audit_md.parent.mkdir(parents=True, exist_ok=True)
    args.audit_md.write_text(md)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    static = read_jsonl(args.run_dir / "audit" / "static_ledger.jsonl")
    replay_report = json.loads((args.replay_dir / "executable_replay_report.json").read_text())
    replay_by_id = {str(row["task_id"]): row for row in replay_report.get("results", [])}
    rows = {
        str(row["task_id"]): row
        for row in json.loads((args.run_dir / "converted" / "train.json").read_text())
    }
    sidecars = {
        str(row["task_id"]): row
        for row in (
            compatible_sidecar(json.loads(path.read_text()))
            for path in sorted((args.run_dir / "sidecar").glob("*.gold.json"))
        )
    }
    final_ledger = []
    valid_rows = []
    valid_sidecars = []
    for base in static:
        record = dict(base)
        task_id = str(record["task_id"])
        reasons = list(record.get("reasons", []))
        replay = replay_by_id.get(task_id)
        sidecar = sidecars.get(task_id)
        row = rows.get(task_id)
        flow_checks = []
        leakage_hits = []
        if record.get("status") == "STATIC_PASS":
            if not replay or replay.get("replay_ok") is not True:
                reasons.append("replay_failure")
            elif replay.get("final_verifier_computed") is not True:
                reasons.append("final_verifier_not_computed")
            else:
                profile = profile_for_task(task_id, sidecar, args.replay_dir)
                internal_ids = {edge["edge_id"] for edge in internal_edges(sidecar)}
                flow_checks = [
                    check for check in profile.get("dependency_edge_checks", [])
                    if check.get("edge_id") in internal_ids
                ]
                if len(flow_checks) != len(internal_ids):
                    reasons.append("dependency_edge_not_inspectable")
                if any(check.get("success") is not True or check.get("value_match") is not True for check in flow_checks):
                    reasons.append("internal_parameter_flow_failure")
                visible = visible_user_text(row, sidecar)
                for check in flow_checks:
                    for token in scalar_tokens(check.get("source_values")):
                        if token_visible(token, visible):
                            leakage_hits.append({"edge_id": check.get("edge_id"), "value_sha256": hashlib.sha256(token.encode()).hexdigest()})
                if leakage_hits:
                    reasons.append("internal_value_leakage")
        is_valid = record.get("status") == "STATIC_PASS" and not reasons
        record.update({
            "status": "FINAL_VALID" if is_valid else "DROPPED",
            "replay_ok": bool(replay and replay.get("replay_ok") is True),
            "final_verifier_computed": bool(replay and replay.get("final_verifier_computed") is True),
            "final_state_match": replay.get("final_state_match", "unknown") if replay else "unknown",
            "internal_flow_edges_checked": len(flow_checks),
            "internal_flow_edges_passed": sum(check.get("success") is True and check.get("value_match") is True for check in flow_checks),
            "leakage_hits": leakage_hits,
            "reasons": sorted(set(reasons)),
        })
        final_ledger.append(record)
        if is_valid:
            valid_rows.append(row)
            valid_sidecars.append(sidecar)

    frozen_hashes = frozen_query_hashes(args.frozen_manifest)
    overlap_ids = [sidecar["task_id"] for sidecar in valid_sidecars if query_hash(sidecar_query(sidecar)) in frozen_hashes]
    if overlap_ids:
        invalid = set(overlap_ids)
        valid_rows = [row for row in valid_rows if row["task_id"] not in invalid]
        valid_sidecars = [row for row in valid_sidecars if row["task_id"] not in invalid]
        for record in final_ledger:
            if record["task_id"] in invalid:
                record["status"] = "DROPPED"
                record["reasons"] = sorted(set(record["reasons"] + ["frozen300_exact_overlap"]))

    all_edges = [(sidecar, edge) for sidecar in valid_sidecars for edge in internal_edges(sidecar)]
    bucket_edges = Counter()
    edge_signatures = set()
    producers = set()
    consumers = set()
    environments = Counter()
    for sidecar, edge in all_edges:
        bucket_edges[str(sidecar.get("target_bucket", "unknown"))] += 1
        normalized = normalized_edge(edge)
        edge_signatures.add(edge_signature(normalized, edge.get("dependency_depth") if isinstance(edge.get("dependency_depth"), int) else None))
        producers.add(normalized["producer_tool"])
        consumers.add(normalized["consumer_tool"])
        environments[normalized["producer_environment"]] += 1
        if normalized["consumer_environment"] != normalized["producer_environment"]:
            environments[normalized["consumer_environment"]] += 1

    valid_dir = args.run_dir / "valid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    valid_data = valid_dir / "graph_frontier_rich_train_v1.jsonl"
    valid_gold = valid_dir / "graph_frontier_rich_gold_v1.jsonl"
    write_jsonl(valid_data, valid_rows)
    write_jsonl(valid_gold, valid_sidecars)
    write_jsonl(args.run_dir / "audit" / "final_ledger.jsonl", final_ledger)

    final_tasks = len(valid_rows)
    total_internal = len(all_edges)
    depth2plus = bucket_edges["depth2"] + bucket_edges["depth3plus"]
    leakage_count = sum("internal_value_leakage" in row["reasons"] for row in final_ledger)
    replay_retained = sum(row["status"] == "FINAL_VALID" and row["replay_ok"] for row in final_ledger)
    environment_incidence_total = sum(environments.values())
    max_environment_share = (
        max(environments.values(), default=0) / environment_incidence_total
        if environment_incidence_total else 1.0
    )
    if args.mode == "smoke":
        hard_gates = {
            "ten_valid_depth1": sum(row.get("bucket") == "depth1" and row["status"] == "FINAL_VALID" for row in final_ledger) >= 10,
            "ten_valid_depth2": sum(row.get("bucket") == "depth2" and row["status"] == "FINAL_VALID" for row in final_ledger) >= 10,
            "ten_valid_depth3plus": sum(row.get("bucket") == "depth3plus" and row["status"] == "FINAL_VALID" for row in final_ledger) >= 10,
            "retained_replay_rate_100pct": replay_retained == final_tasks,
            "internal_value_leakage_zero": not any(row["status"] == "FINAL_VALID" and "internal_value_leakage" in row["reasons"] for row in final_ledger),
            "frozen300_exact_overlap_zero": True,
            "multiple_environments": len(environments) >= 2,
            "max_environment_share_le_40pct": max_environment_share <= 0.40,
        }
    else:
        hard_gates = {
            "internal_required_edges_ge_500": total_internal >= 500,
            "depth2plus_edges_ge_200": depth2plus >= 200,
            "consumer_schema_resolvable_100pct": all(row.get("tool_schema_and_mapping_ok") is True for row in final_ledger if row["status"] == "FINAL_VALID"),
            "retained_replay_rate_100pct": replay_retained == final_tasks,
            "internal_value_leakage_zero": not any(row["status"] == "FINAL_VALID" and "internal_value_leakage" in row["reasons"] for row in final_ledger),
            "frozen300_exact_overlap_zero": True,
            "multiple_environments": len(environments) >= 2,
            "max_environment_share_le_40pct": max_environment_share <= 0.40,
        }
    verdict = (
        "SMOKE_READY" if args.mode == "smoke" and all(hard_gates.values())
        else "SMOKE_NOT_READY" if args.mode == "smoke"
        else "GRAPH_FRONTIER_RICH_POOL_READY" if all(hard_gates.values())
        else "PARTIAL-STRUCTURAL-DATA-READY" if final_tasks and total_internal
        else "STRUCTURAL-DATA-NOT-READY"
    )
    drop_reasons = Counter(reason for row in final_ledger if row["status"] != "FINAL_VALID" for reason in row["reasons"])
    report = {
        "schema_version": "graph_frontier_rich_validation_v1",
        "mode": args.mode,
        "verdict": verdict,
        "funnel": {
            "sampled_structures": len(final_ledger),
            "query_generated": len(sidecars),
            "static_pass": sum(row.get("static_pass") is True for row in final_ledger),
            "mcp_pass": sum(row.get("tool_schema_and_mapping_ok") is True for row in final_ledger),
            "replay_pass": sum(row.get("replay_ok") is True for row in final_ledger),
            "no_leak_pass": sum(row.get("replay_ok") is True and "internal_value_leakage" not in row["reasons"] for row in final_ledger),
            "final_valid_tasks": final_tasks,
        },
        "structural_result": {
            "valid_tasks": final_tasks,
            "total_gold_edges": sum(len(sidecar.get("dependency_edges", []) or []) for sidecar in valid_sidecars),
            "internal_required_edges": total_internal,
            "depth1_edges": bucket_edges["depth1"],
            "depth2_edges": bucket_edges["depth2"],
            "depth3plus_edges": bucket_edges["depth3plus"],
            "depth2plus_edges": depth2plus,
            "unique_internal_edge_signatures": len(edge_signatures),
            "unique_producer_tools": len(producers),
            "unique_consumer_tools": len(consumers),
            "unique_environments": len(environments),
            "environment_edge_incidence": dict(environments),
            "max_environment_edge_incidence_share": max_environment_share,
            "final_state_match": sum(row.get("final_state_match") is True for row in final_ledger if row["status"] == "FINAL_VALID"),
        },
        "drop_reasons": dict(drop_reasons),
        "hard_gates": hard_gates,
        "frozen300": {
            "manifest_sha256": EXPECTED_FROZEN300_SHA256,
            "comparison": "post_hoc_one_way_normalized_query_hash_only",
            "candidate_overlap_detected_and_dropped": len(overlap_ids),
            "exact_overlap": 0,
            "frozen_content_persisted": False,
        },
        "artifacts": {
            "valid_data": str(valid_data),
            "valid_data_sha256": file_sha256(valid_data),
            "valid_gold": str(valid_gold),
            "valid_gold_sha256": file_sha256(valid_gold),
        },
    }
    write_json(args.run_dir / "audit" / "validation_report.json", report)
    if args.mode == "formal":
        write_formal_reports(report, args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--plan", type=Path, required=True)
    prep.add_argument("--registry", type=Path, default=ROOT / "configs/mcp_server.json")
    prep.add_argument("--metadata-dir", type=Path, default=ROOT / "envs/metadata")
    prep.add_argument("--seed", type=int, default=20260922)
    prep.set_defaults(func=prepare)
    final = sub.add_parser("finalize")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--replay-dir", type=Path, required=True)
    final.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN)
    final.add_argument("--mode", choices=("smoke", "formal"), required=True)
    final.add_argument("--capacity-report", type=Path, default=ROOT / "repro_1p7b/graph_frontier/preference_generation_v3/audit/toolgraph_capacity.json")
    final.add_argument("--plan-summary", type=Path, default=ROOT / "repro_1p7b/graph_frontier/preference_generation_v3/manifests/formal_plan_v1_scaled_deep.summary.json")
    final.add_argument("--plan", type=Path, default=ROOT / "repro_1p7b/graph_frontier/preference_generation_v3/manifests/formal_plan_v1_scaled_deep.jsonl")
    final.add_argument("--audit-json", type=Path, default=ROOT / "repro_1p7b/graph_frontier/reports/graph_frontier_rich_generation_audit.json")
    final.add_argument("--audit-md", type=Path, default=ROOT / "repro_1p7b/graph_frontier/reports/graph_frontier_rich_generation_audit.md")
    final.set_defaults(func=finalize)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)

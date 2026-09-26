"""Full static and contamination audit for generated Graph-Frontier RL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

UNKNOWN = "unknown"
KEY_ERROR = re.compile(r"KeyError\(['\"]([^'\"]+)['\"]\)")
SEED_IN_NAME = re.compile(r"-(\d+)-(?:sglang|sglang1)\.json$")
TURN_IN_TASK = re.compile(r"-t(\d+)(?:-|$)")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_query(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("text")
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.casefold().strip())


def known(value: Any) -> bool:
    return value not in (None, "", UNKNOWN)


def load_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text()), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_frozen(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metadata_index(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(metadata_dir.glob("*_metadata.json")):
        payload, error = load_json(path)
        if error or not isinstance(payload, Mapping):
            continue
        server = payload.get("class_name") or path.name.removesuffix("_metadata.json")
        tools = {}
        for tool in payload.get("tools", []) or []:
            if isinstance(tool, Mapping) and isinstance(tool.get("name"), str):
                tools[tool["name"]] = tool
        result[str(server)] = {"path": str(path), "tools": tools}
    return result


def schema_has_path(schema: Any, dotted: Any) -> bool:
    if not isinstance(schema, Mapping) or not isinstance(dotted, str) or not dotted:
        return False
    current = schema
    for part in dotted.split("."):
        properties = current.get("properties", {}) if isinstance(current, Mapping) else {}
        if part not in properties:
            if isinstance(current, Mapping) and current.get("type") == "array":
                current = current.get("items", {})
                properties = current.get("properties", {}) if isinstance(current, Mapping) else {}
            if part not in properties:
                return False
        current = properties[part]
        if isinstance(current, Mapping) and current.get("type") == "array":
            current = current.get("items", current)
    return True


def tool_parts(tool_name: Any) -> tuple[str, str]:
    if not isinstance(tool_name, str) or "-" not in tool_name:
        return "", ""
    return tuple(tool_name.split("-", 1))  # type: ignore[return-value]


def sequence(sidecar: Mapping[str, Any]) -> list[str]:
    return [
        item["tool_name"]
        for item in sidecar.get("gold_tool_sequence", []) or []
        if isinstance(item, Mapping) and isinstance(item.get("tool_name"), str)
    ]


def structural_checks(
    sidecar: Mapping[str, Any],
    registry: Mapping[str, Any],
    metadata: Mapping[str, Any],
    repo_root: Path,
) -> tuple[bool, bool, list[str]]:
    reasons: list[str] = []
    names = sequence(sidecar)
    required = {
        item.get("tool_name")
        for item in sidecar.get("required_tool_nodes", []) or []
        if isinstance(item, Mapping)
    }
    edges = sidecar.get("dependency_edges")
    depth = sidecar.get("dependency_depth")
    if not names:
        reasons.append("gold_nodes_empty")
    if not isinstance(edges, list):
        reasons.append("gold_edges_not_list")
        edges = []
    if not isinstance(depth, int) or depth < 0:
        reasons.append("dependency_depth_invalid")
    nodes = set(names) | required
    for index, edge in enumerate(edges):
        prefix = f"edge_{index}"
        if not isinstance(edge, Mapping):
            reasons.append(prefix + "_not_object")
            continue
        producer = (edge.get("producer_tool") or {}).get("tool_name")
        consumer = (edge.get("consumer_tool") or {}).get("tool_name")
        source = (edge.get("producer_output_parameter") or {}).get("parameter_name")
        target = (edge.get("consumer_input_parameter") or {}).get("parameter_name")
        if producer not in nodes:
            reasons.append(prefix + "_producer_dangling")
        if consumer not in nodes:
            reasons.append(prefix + "_consumer_dangling")
        if producer == consumer:
            reasons.append(prefix + "_self_loop")
        if not isinstance(source, str) or not source:
            reasons.append(prefix + "_producer_field_invalid")
        if not isinstance(target, str) or not target:
            reasons.append(prefix + "_consumer_field_invalid")
        if edge.get("required") not in (True, False, UNKNOWN):
            reasons.append(prefix + "_required_invalid")
        if edge.get("optional") not in (True, False, UNKNOWN):
            reasons.append(prefix + "_optional_invalid")

    graph_ok = not any(
        marker in reason
        for reason in reasons
        for marker in ("gold_", "dependency_", "dangling", "self_loop", "field_invalid")
    )
    tool_ok = True
    tool_specs: dict[str, Mapping[str, Any]] = {}
    for name in sorted(nodes):
        server, short = tool_parts(name)
        entry = registry.get(server)
        meta = metadata.get(server)
        if not entry:
            reasons.append(f"server_unregistered:{server}")
            tool_ok = False
        else:
            tool_path = repo_root / str(entry.get("tool_path", ""))
            if not tool_path.is_file():
                reasons.append(f"server_implementation_missing:{server}")
                tool_ok = False
        if not meta:
            reasons.append(f"server_metadata_missing:{server}")
            tool_ok = False
        elif short not in meta["tools"]:
            reasons.append(f"tool_schema_missing:{name}")
            tool_ok = False
        else:
            tool_specs[name] = meta["tools"][short]

    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            continue
        producer = (edge.get("producer_tool") or {}).get("tool_name")
        consumer = (edge.get("consumer_tool") or {}).get("tool_name")
        source = (edge.get("producer_output_parameter") or {}).get("parameter_name")
        target = (edge.get("consumer_input_parameter") or {}).get("parameter_name")
        if producer in tool_specs and not schema_has_path(tool_specs[producer].get("output_schema", {}), source):
            reasons.append(f"edge_{index}_producer_schema_field_missing")
            tool_ok = False
        if consumer in tool_specs and not schema_has_path(tool_specs[consumer].get("input_schema", {}), target):
            reasons.append(f"edge_{index}_consumer_schema_field_missing")
            tool_ok = False
    return graph_ok, tool_ok, sorted(set(reasons))


def raw_chain_conversion_compatible(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    for node in payload.get("nodes", []) or []:
        if not isinstance(node, Mapping):
            continue
        for step in node.get("steps", []) or []:
            if not isinstance(step, Mapping) or step.get("role") != "tool_call":
                continue
            content = step.get("content")
            if not isinstance(content, list) or not content:
                return False
            for call in content:
                if not isinstance(call, Mapping):
                    return False
                if not isinstance(call.get("name"), str):
                    return False
                if not isinstance(call.get("arguments"), Mapping):
                    return False
    return True


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_dir.resolve()
    raw_dir, sidecar_dir, log_dir = source / "raw", source / "gold", source / "querygen_logs"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    registry_payload, registry_error = load_json(args.registry)
    if registry_error:
        raise RuntimeError(registry_error)
    registry = registry_payload.get("mcpServers", {})
    metadata = metadata_index(args.metadata_dir)

    raw_by_seed: dict[int, tuple[Path, Any, str | None]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        payload, error = load_json(path)
        seed = payload.get("seed") if isinstance(payload, Mapping) else None
        if isinstance(seed, int):
            raw_by_seed[seed] = (path, payload, error)

    conversion_invalid_seeds = {
        seed
        for seed, (_, payload, error) in raw_by_seed.items()
        if error is not None or not raw_chain_conversion_compatible(payload)
    }

    sidecars: list[tuple[Path, Any, str | None]] = []
    for path in sorted(sidecar_dir.glob("*.gold.json")):
        payload, error = load_json(path)
        sidecars.append((path, payload, error))

    frozen = load_frozen(args.frozen_manifest)
    frozen_queries = {normalized_query(row.get("query")) for row in frozen if normalized_query(row.get("query"))}
    frozen_ids = {str(row.get("task_id")) for row in frozen if known(row.get("task_id"))}
    frozen_sequences = {
        sha256_value(row.get("gold_tools", []))
        for row in frozen
        if isinstance(row.get("gold_tools"), list)
    }
    frozen_initial_hashes = {
        str(row.get("initial_state_sha256"))
        for row in frozen
        if known(row.get("initial_state_sha256"))
    }

    ledger = []
    contamination_rows = []
    warning_tasks: dict[str, set[str]] = defaultdict(set)
    warnings: Counter[str] = Counter()
    log_agents: Counter[str] = Counter()

    for log_path in sorted(log_dir.glob("*.json")):
        payload, _ = load_json(log_path)
        match = SEED_IN_NAME.search(log_path.name)
        seed = int(match.group(1)) if match else None
        task_ids = {
            str(item[1].get("task_id"))
            for item in sidecars
            if isinstance(item[1], Mapping) and item[1].get("seed") == seed
        }
        if isinstance(payload, Mapping):
            for rows in payload.values():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    log_agents[str(row.get("agent", "unknown"))] += 1
                    text = str(row.get("user", "")) + "\n" + str(row.get("assistant", ""))
                    for server in KEY_ERROR.findall(text):
                        warnings[server] += 1
                        warning_tasks[server].update(task_ids)

    static_pass_ids = []
    drop_reasons: Counter[str] = Counter()
    depths: Counter[str] = Counter()
    environments: Counter[str] = Counter()
    chain_lengths: Counter[str] = Counter()
    internal_edges_distribution: Counter[str] = Counter()

    for sidecar_path, sidecar, sidecar_error in sidecars:
        task_id = sidecar.get("task_id") if isinstance(sidecar, Mapping) else sidecar_path.stem
        seed = sidecar.get("seed") if isinstance(sidecar, Mapping) else None
        turn_match = TURN_IN_TASK.search(str(task_id))
        turn = int(turn_match.group(1)) if turn_match else 0
        raw_entry = raw_by_seed.get(seed)
        raw_path = raw_entry[0] if raw_entry else None
        raw = raw_entry[1] if raw_entry else None
        reasons: list[str] = []

        sidecar_ok = sidecar_error is None and isinstance(sidecar, Mapping)
        raw_ok = raw_entry is not None and raw_entry[2] is None and isinstance(raw, Mapping)
        node = None
        if raw_ok:
            nodes = raw.get("nodes")
            if isinstance(nodes, list) and turn < len(nodes) and isinstance(nodes[turn], Mapping):
                node = nodes[turn]
            else:
                reasons.append("raw_turn_missing")
                raw_ok = False
        if not sidecar_ok:
            reasons.append("sidecar_json_invalid")
        if not raw_ok:
            reasons.append("raw_json_or_mapping_invalid")

        graph_ok = tool_ok = mapping_ok = False
        if sidecar_ok:
            if not isinstance(seed, int):
                reasons.append("seed_missing")
            if raw_ok and raw.get("seed") != seed:
                reasons.append("seed_mismatch")
            if seed in conversion_invalid_seeds:
                reasons.append("raw_chain_conversion_incompatible")
            if not known(sidecar.get("task_id")):
                reasons.append("task_id_missing")
            query = sidecar.get("query", {})
            query_text = query.get("text") if isinstance(query, Mapping) else query
            if not known(query_text):
                reasons.append("query_missing")
            initial = node.get("initial_scenario") if node else sidecar.get("initial_scenario")
            final = node.get("final_scenario") if node else sidecar.get("expected_final_state")
            if not isinstance(initial, Mapping):
                reasons.append("initial_config_missing")
            if not isinstance(final, Mapping):
                reasons.append("final_config_missing")
            if node is not None:
                if node.get("decision") is not True:
                    reasons.append("decision_not_true")
                if not isinstance(node.get("steps"), list) or not node.get("steps"):
                    reasons.append("trajectory_missing")
                if normalized_query(node.get("query")) != normalized_query(query_text):
                    reasons.append("query_raw_sidecar_mismatch")
                gold_names = sequence(sidecar)
                actual_names = []
                for step in node.get("steps", []) or []:
                    if not isinstance(step, Mapping):
                        reasons.append("reference_step_not_object")
                        continue
                    if step.get("role") != "tool_call":
                        continue
                    content = step.get("content")
                    if not isinstance(content, list) or not content:
                        reasons.append("reference_tool_call_content_not_list")
                        continue
                    for call in content:
                        if not isinstance(call, Mapping):
                            reasons.append("reference_tool_call_not_object")
                            continue
                        name = call.get("name")
                        if not isinstance(name, str):
                            reasons.append("reference_tool_name_invalid")
                            continue
                        if not isinstance(call.get("arguments"), Mapping):
                            reasons.append("reference_tool_arguments_invalid")
                            continue
                        if name not in gold_names:
                            candidates = [item for item in gold_names if item.endswith("-" + name)]
                            if len(candidates) == 1:
                                name = candidates[0]
                        actual_names.append(name)
                missing_reference_tools = sorted(set(gold_names) - set(actual_names))
                if missing_reference_tools:
                    reasons.append(
                        "reference_trajectory_missing_gold_tools:"
                        + ",".join(missing_reference_tools)
                    )
            graph_ok, tool_ok, structural_reasons = structural_checks(
                sidecar, registry, metadata, args.repo_root
            )
            reasons.extend(structural_reasons)
            mapping_ok = not any(
                reason.startswith(("server_unregistered:", "server_implementation_missing:", "server_metadata_missing:"))
                for reason in structural_reasons
            )

            names = sequence(sidecar)
            qnorm = normalized_query(query_text)
            query_overlap = bool(qnorm and qnorm in frozen_queries)
            task_overlap = str(task_id) in frozen_ids
            seq_overlap = sha256_value(names) in frozen_sequences if names else False
            initial_hash = sha256_value(initial) if isinstance(initial, Mapping) else ""
            state_overlap = initial_hash in frozen_initial_hashes
            contamination_rows.append(
                {
                    "task_id": task_id,
                    "normalized_query_sha256": hashlib.sha256(qnorm.encode()).hexdigest() if qnorm else UNKNOWN,
                    "query_exact_overlap": query_overlap,
                    "task_id_overlap": task_overlap,
                    "tool_sequence_overlap_audit_only": seq_overlap,
                    "initial_state_overlap": state_overlap,
                    "drop_for_exact_overlap": query_overlap or task_overlap or state_overlap,
                }
            )
            depths[str(sidecar.get("dependency_depth", UNKNOWN))] += 1
            chain_lengths[str(len(names))] += 1
            internal_count = sum(
                edge.get("internal_parameter") is True
                for edge in sidecar.get("dependency_edges", []) or []
                if isinstance(edge, Mapping)
            )
            internal_edges_distribution[str(internal_count)] += 1
            environments.update(sidecar.get("environment_identifiers", []) or [])

        reasons = sorted(set(reasons))
        frozen_overlap = any(
            row["task_id"] == task_id and row["drop_for_exact_overlap"]
            for row in contamination_rows
        )
        static_pass = (
            raw_ok and sidecar_ok and graph_ok and mapping_ok and tool_ok
            and not reasons and not frozen_overlap
        )
        status = "STATIC_PASS" if static_pass else "QUARANTINED"
        if static_pass:
            static_pass_ids.append(str(task_id))
        else:
            for reason in reasons or ["exact_contamination"]:
                drop_reasons[reason] += 1
        ledger.append(
            {
                "task_id": task_id,
                "seed": seed,
                "raw_file": str(raw_path) if raw_path else UNKNOWN,
                "sidecar_file": str(sidecar_path),
                "raw_ok": raw_ok,
                "sidecar_ok": sidecar_ok,
                "graph_ok": graph_ok,
                "server_mapping_ok": mapping_ok,
                "tool_schema_ok": tool_ok,
                "executable_replay_ok": None,
                "frozen_overlap": frozen_overlap,
                "status": status,
                "reasons": reasons,
            }
        )

    with (output / "rl_data_v1_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    generation_text = (source / "generation.log").read_text(errors="replace")
    generation_key_counts = Counter(KEY_ERROR.findall(generation_text))
    warning_pattern_counts = {
        pattern: len(re.findall(pattern, generation_text, flags=re.IGNORECASE))
        for pattern in (
            r"missing server mapping", r"registry miss", r"unresolved MCP",
            r"unknown server", r"fallback mapping", r"lookup exception",
        )
    }
    warning_report = []
    registry_casefold = {name.casefold(): name for name in registry}
    warning_servers = sorted(
        set(warnings) | set(generation_key_counts),
        key=lambda name: (-generation_key_counts.get(name, warnings[name]), name),
    )
    for server in warning_servers:
        count = generation_key_counts.get(server, warnings[server])
        entry = registry.get(server)
        alias = registry_casefold.get(server.casefold())
        metadata_ok = server in metadata
        implementation_ok = bool(entry) and (args.repo_root / str(entry.get("tool_path", ""))).is_file()
        if entry and metadata_ok and implementation_ok:
            status = "fixed_mapping"
            resolution = "server exists; generation wrapper omitted MCP_CONFIG_PATH; wrapper fixed, regeneration required"
        elif alias and alias != server:
            status = "valid_alias"
            resolution = f"normalize alias {server} -> {alias}, then revalidate"
        else:
            status = "invalid_task"
            resolution = "no complete registry + implementation + metadata triple; quarantine"
        warning_report.append(
            {
                "server_name": server,
                "affected_tasks": sorted(warning_tasks[server]),
                "count": count,
                "generation_log_occurrences": generation_key_counts[server],
                "querygen_log_occurrences": warnings[server],
                "registry_entry": bool(entry),
                "implementation": implementation_ok,
                "metadata_schema": metadata_ok,
                "resolution": resolution,
                "status": status,
            }
        )
    write_json(
        output / "mapping_warning_report.json",
        {
            "schema_version": "graph_frontier_rl_v1_mapping_warning_report",
            "generation_log": str(source / "generation.log"),
            "warning_servers": warning_report,
            "warning_server_count": len(warning_report),
            "warning_pattern_counts": warning_pattern_counts,
            "known_focus": {
                name: next((row for row in warning_report if row["server_name"] == name), None)
                for name in ("HugeiconsServer", "FatSecretPlatform")
            },
        },
    )

    exact_overlap = sum(row["drop_for_exact_overlap"] for row in contamination_rows)
    write_json(
        output / "contamination_report.json",
        {
            "schema_version": "graph_frontier_rl_v1_contamination_report",
            "frozen_manifest": str(args.frozen_manifest.resolve()),
            "frozen_manifest_sha256": file_sha256(args.frozen_manifest),
            "frozen_count": len(frozen),
            "generated_checked": len(contamination_rows),
            "exact_query_overlap": sum(row["query_exact_overlap"] for row in contamination_rows),
            "exact_task_id_overlap": sum(row["task_id_overlap"] for row in contamination_rows),
            "exact_initial_state_overlap": sum(row["initial_state_overlap"] for row in contamination_rows),
            "tool_sequence_overlap_audit_only": sum(row["tool_sequence_overlap_audit_only"] for row in contamination_rows),
            "FROZEN_300_EXACT_TRAIN_OVERLAP": exact_overlap,
            "rows": contamination_rows,
        },
    )

    quality = {
        "schema_version": "graph_frontier_rl_v1_quality_report",
        "source_dir": str(source),
        "paths": {
            "raw": str(raw_dir),
            "gold_sidecar": str(sidecar_dir),
            "generation_log": str(source / "generation.log"),
            "generation_manifest": str(source / "generation_run.json"),
            "mcp_registry": str(args.registry.resolve()),
        },
        "generated_raw_files": len(list(raw_dir.glob("*.json"))),
        "generated_sidecars": len(sidecars),
        "raw_seed_mappings": len(raw_by_seed),
        "static_pass": len(static_pass_ids),
        "replay_pass": 0,
        "dropped": 0,
        "quarantined": len(ledger) - len(static_pass_ids),
        "drop_reasons": dict(drop_reasons.most_common()),
        "dependency_depth_distribution": dict(sorted(depths.items())),
        "internal_edge_count_distribution": dict(sorted(internal_edges_distribution.items())),
        "tool_chain_length_distribution": dict(sorted(chain_lengths.items())),
        "environment_distribution": dict(environments.most_common()),
        "querygen_agent_counts": dict(log_agents),
        "query_generator_completed_tasks": sum(
            row.get("agent") == "QueryGenerator"
            for path in log_dir.glob("*.json")
            for payload in [load_json(path)[0]]
            if isinstance(payload, Mapping)
            for rows in payload.values()
            if isinstance(rows, list)
            for row in rows
            if isinstance(row, Mapping)
        ),
        "dataset_gate": "STATIC_PASS" if static_pass_ids else "PIPELINE_ISSUE",
        "diagnosis": (
            "All tasks terminated before QueryGenerator because the generation wrapper "
            "did not export MCP_CONFIG_PATH; raw and sidecar task payloads are incomplete."
            if not static_pass_ids and log_agents.get("QueryGenerator", 0) == 0
            else "See per-task audit ledger."
        ),
    }
    write_json(output / "quality_report.json", quality)
    write_json(
        output / "audit_manifest.json",
        {
            "schema_version": "graph_frontier_rl_v1_audit_manifest",
            "generation_seed": json.loads((source / "generation_run.json").read_text()).get("generation_seed"),
            "generator_models": json.loads((source / "generation_run.json").read_text()).get("model_names"),
            "raw_count": quality["generated_raw_files"],
            "sidecar_count": quality["generated_sidecars"],
            "ledger_sha256": file_sha256(output / "rl_data_v1_audit.jsonl"),
            "quality_report_sha256": file_sha256(output / "quality_report.json"),
            "mapping_warning_report_sha256": file_sha256(output / "mapping_warning_report.json"),
            "contamination_report_sha256": file_sha256(output / "contamination_report.json"),
            "status": quality["dataset_gate"],
        },
    )
    (output / "status").write_text(quality["dataset_gate"] + "\n")
    return quality


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--registry", type=Path, default=Path("configs/mcp_server.json"))
    result.add_argument("--metadata-dir", type=Path, default=Path("envs/metadata"))
    result.add_argument(
        "--frozen-manifest",
        type=Path,
        default=Path(
            "repro_1p7b/results/graph_frontier/confirm_300/frozen/"
            "confirm_300_seed_20260914.jsonl"
        ),
    )
    return result


if __name__ == "__main__":
    report = audit(parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

"""Structure-first Graph-Frontier-rich EnvFactory task generation.

The frozen evaluation set is never loaded by this module.  Graph structure is
selected before QueryGen, and the selected dependency trace is attached while
the live ToolGraph/ToolQueryChain objects still exist.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "repro_1p7b/graph_frontier/preference_generation_v3"
EXPECTED_FROZEN300_SHA256 = "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parameter_name(value: Any) -> str:
    return str(getattr(value, "name", "unknown"))


def parameter_type(value: Any) -> str:
    return str(getattr(value, "data_type", "unknown"))


def is_required(graph: Any, parameter: Any, consumer: Any) -> bool:
    data = graph.graph.get_edge_data(parameter, consumer) or {}
    if isinstance(data.get("required"), bool):
        return data["required"]
    required = consumer.input_schema.get("required", []) or []
    name = parameter_name(parameter)
    return name in required or name.split(".", 1)[0] in required


def producers_for_input(graph: Any, consumer: Any, target: Any) -> list[tuple[Any, Any]]:
    """Return precise ``(producer tool, producer output parameter)`` pairs."""
    from src.graph.tool_node import Parameter, Tool

    result = []
    for predecessor in graph.graph.predecessors(target):
        if isinstance(predecessor, Tool):
            result.append((predecessor, target))
        elif isinstance(predecessor, Parameter):
            for producer in graph.graph.predecessors(predecessor):
                if isinstance(producer, Tool):
                    result.append((producer, predecessor))
    unique = {}
    for producer, source in result:
        if producer is consumer:
            continue
        key = (producer.name, parameter_name(source), parameter_name(target))
        unique[key] = (producer, source)
    return [unique[key] for key in sorted(unique)]


def enumerate_internal_edges(graph: Any) -> list[dict[str, Any]]:
    from src.graph.tool_node import Tool

    edges = []
    tools = sorted((node for node in graph.graph.nodes if isinstance(node, Tool)), key=lambda item: item.name)
    for consumer in tools:
        for target in consumer.input_schema.get("parameters", []) or []:
            if getattr(target, "user_provided", None) is not False:
                continue
            if not is_required(graph, target, consumer):
                continue
            for producer, source in producers_for_input(graph, consumer, target):
                signature = {
                    "producer_environment": producer.server,
                    "producer_tool": producer.name,
                    "producer_output_field": parameter_name(source),
                    "consumer_environment": consumer.server,
                    "consumer_tool": consumer.name,
                    "consumer_argument": parameter_name(target),
                }
                edge_id = sha256_bytes(canonical(signature).encode())[:24]
                edges.append({
                    "edge_id": f"internal-{edge_id}",
                    **signature,
                    "producer_output_type": parameter_type(source),
                    "consumer_argument_type": parameter_type(target),
                    "required": True,
                    "internal_parameter": True,
                })
    return sorted(edges, key=lambda item: item["edge_id"])


def edge_signature(edge: Mapping[str, Any], depth: int | None = None) -> str:
    value = [
        edge["producer_environment"], edge["producer_tool"], edge["producer_output_field"],
        edge["consumer_environment"], edge["consumer_tool"], edge["consumer_argument"],
    ]
    if depth is not None:
        value.append(depth)
    return sha256_bytes(canonical(value).encode())


def enumerate_paths(
    edges: Sequence[Mapping[str, Any]],
    max_depth: int = 4,
    max_paths_per_depth: int = 20_000,
) -> tuple[dict[int, list[list[dict[str, Any]]]], set[int]]:
    outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[str(edge["producer_tool"])].append(edge)
    paths: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
    truncated: set[int] = set()

    def visit(current: list[Mapping[str, Any]], used_tools: set[str]) -> None:
        depth = len(current)
        if len(paths[depth]) >= max_paths_per_depth:
            truncated.add(depth)
            return
        paths[depth].append([dict(edge) for edge in current])
        if len(current) >= max_depth:
            return
        last = str(current[-1]["consumer_tool"])
        for edge in outgoing.get(last, []):
            nxt = str(edge["consumer_tool"])
            if nxt in used_tools:
                continue
            visit(current + [edge], used_tools | {nxt})

    for edge in edges:
        visit([edge], {str(edge["producer_tool"]), str(edge["consumer_tool"])})
    for depth in paths:
        unique = {canonical([edge["edge_id"] for edge in path]): path for path in paths[depth]}
        paths[depth] = [unique[key] for key in sorted(unique)]
    return dict(paths), truncated


def environment_capacity(graph: Any, edges: Sequence[Mapping[str, Any]], paths: Mapping[int, Sequence[Any]]) -> list[dict[str, Any]]:
    from src.graph.tool_node import Parameter, Tool

    servers = sorted(graph.server_to_tools)
    result = []
    for server in servers:
        tools = set(graph.server_to_tools[server])
        relevant_nodes = set(tools)
        for tool in tools:
            relevant_nodes.update(graph.graph.predecessors(tool))
            relevant_nodes.update(graph.graph.successors(tool))
        incident_graph_edges = sum(1 for left, right in graph.graph.edges if left in tools or right in tools)
        internal = [edge for edge in edges if edge["producer_environment"] == server or edge["consumer_environment"] == server]
        depth_counts = {
            str(depth): sum(
                any(edge["producer_environment"] == server or edge["consumer_environment"] == server for edge in path)
                for path in bucket
            )
            for depth, bucket in paths.items()
        }
        reachable_depths = [int(depth) for depth, count in depth_counts.items() if count]
        result.append({
            "environment": server,
            "relevant_nodes": len(relevant_nodes),
            "tool_nodes": len(tools),
            "parameter_nodes": sum(isinstance(node, Parameter) for node in relevant_nodes),
            "incident_graph_edges": incident_graph_edges,
            "internal_compatible_edges": len(internal),
            "max_dependency_depth": max(reachable_depths, default=0),
            "depth1_path_opportunities": depth_counts.get("1", 0),
            "depth2_path_opportunities": depth_counts.get("2", 0),
            "depth3plus_path_opportunities": sum(count for depth, count in depth_counts.items() if int(depth) >= 3),
        })
    return result


def run_capacity(args: argparse.Namespace) -> dict[str, Any]:
    from src.graph.tool_graph import ToolGraph
    from src.graph.tool_node import Parameter, Tool

    graph = ToolGraph.load(args.graph)
    edges = enumerate_internal_edges(graph)
    paths, truncated_depths = enumerate_paths(edges, args.max_depth, args.max_paths_per_depth)
    environments = environment_capacity(graph, edges, paths)
    output = args.output_dir / "audit"
    write_jsonl(output / "internal_edge_inventory.jsonl", edges)
    for depth, bucket in paths.items():
        write_jsonl(output / f"paths_depth{depth}.jsonl", ({"edge_path": path} for path in bucket))
    report = {
        "schema_version": "graph_frontier_rich_toolgraph_capacity_v1",
        "graph_path": str(Path(args.graph).resolve()),
        "graph_sha256": file_sha256(args.graph),
        "tool_nodes": sum(isinstance(node, Tool) for node in graph.graph.nodes),
        "parameter_nodes": sum(isinstance(node, Parameter) for node in graph.graph.nodes),
        "networkx_edges": graph.graph.number_of_edges(),
        "environments": len(graph.server_to_tools),
        "internal_compatible_edge_signatures": len(edges),
        "unique_producer_tools": len({edge["producer_tool"] for edge in edges}),
        "unique_consumer_tools": len({edge["consumer_tool"] for edge in edges}),
        "path_opportunity_lower_bounds": {str(depth): len(bucket) for depth, bucket in sorted(paths.items())},
        "path_enumeration_truncated_depths": sorted(truncated_depths),
        "max_paths_per_depth": args.max_paths_per_depth,
        "max_dependency_depth_within_cutoff": max((depth for depth, bucket in paths.items() if bucket), default=0),
        "environment_capacity": environments,
        "can_support_500_edge_instances_without_signature_reuse": len(edges) >= 500,
        "can_support_depth2plus": sum(len(bucket) for depth, bucket in paths.items() if depth >= 2) >= 200,
        "frozen300_content_loaded": False,
        "frozen300_manifest_sha256_lock": EXPECTED_FROZEN300_SHA256,
    }
    write_json(output / "toolgraph_capacity.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def read_paths(path: Path) -> list[list[dict[str, Any]]]:
    return [json.loads(line)["edge_path"] for line in path.read_text().splitlines() if line.strip()]


def path_environment(path: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(edge["producer_environment"]) for edge in path} | {str(edge["consumer_environment"]) for edge in path}))


def profiler_unambiguous_path(path: Sequence[Mapping[str, Any]]) -> bool:
    """Conservatively exclude scalar fields flattened through a collection."""
    return all("." not in str(edge.get("producer_output_field", "")) for edge in path)


def select_balanced(
    paths: Sequence[list[dict[str, Any]]],
    count: int,
    seed: int,
    edge_counts: Counter[str] | None = None,
    max_edge_reuse: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Round-robin environments and enforce a shared inventory-edge cap."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, ...], list[list[dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        buckets[path_environment(path)].append(path)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    ordered = []
    offsets = Counter()
    while keys:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if offsets[key] < len(bucket):
                ordered.append(bucket[offsets[key]])
                offsets[key] += 1
                progressed = True
        if not progressed:
            break
    selected = []
    counts = edge_counts if edge_counts is not None else Counter()
    for candidate in ordered:
        ids = [str(edge["edge_id"]) for edge in candidate]
        if max_edge_reuse is not None and any(counts[edge_id] >= max_edge_reuse for edge_id in ids):
            continue
        selected.append(candidate)
        counts.update(ids)
        if len(selected) >= count:
            break
    return selected


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    audit = args.output_dir / "audit"
    registry_payload = json.loads(args.registry.read_text())
    registered = set((registry_payload.get("mcpServers") or {}).keys())
    counts = {1: args.depth1, 2: args.depth2, 3: args.depth3}
    plans = []
    shortfalls = {}
    registry_filtered_paths = Counter()
    profiler_ambiguous_paths = Counter()
    inventory_edge_counts: Counter[str] = Counter()
    depth_items = list(counts.items())
    if args.deep_first:
        depth_items.reverse()
    for depth, count in depth_items:
        candidates = read_paths(audit / f"paths_depth{depth}.jsonl")
        before = len(candidates)
        candidates = [
            path for path in candidates
            if all(
                edge["producer_environment"] in registered
                and edge["consumer_environment"] in registered
                for edge in path
            )
        ]
        registry_filtered_paths[str(depth)] = before - len(candidates)
        before_unambiguous = len(candidates)
        if args.profiler_unambiguous_only:
            candidates = [path for path in candidates if profiler_unambiguous_path(path)]
        profiler_ambiguous_paths[str(depth)] = before_unambiguous - len(candidates)
        selected = select_balanced(
            candidates,
            count,
            args.seed + depth * 1009,
            inventory_edge_counts,
            args.max_edge_reuse,
        )
        shortfalls[str(depth)] = max(0, count - len(selected))
        for index, path in enumerate(selected):
            payload = {
                "bucket": f"depth{depth}" if depth < 3 else "depth3plus",
                "target_depth": depth,
                "edge_path": path,
                "tool_sequence": [path[0]["producer_tool"]] + [edge["consumer_tool"] for edge in path],
                "environment_identifiers": list(path_environment(path)),
                "generation_seed": args.seed + depth * 10_000_000 + index * 7919,
            }
            payload["plan_id"] = "gf-rich-" + sha256_bytes(canonical(payload).encode())[:20]
            plans.append(payload)
    write_jsonl(args.output_dir / "manifests" / f"{args.name}.jsonl", plans)
    report = {
        "schema_version": "graph_frontier_rich_plan_v1",
        "name": args.name,
        "requested": {str(key): value for key, value in counts.items()},
        "selected": dict(Counter(str(plan["target_depth"]) for plan in plans)),
        "shortfalls": shortfalls,
        "plans": len(plans),
        "unique_path_signatures": len({canonical([edge["edge_id"] for edge in plan["edge_path"]]) for plan in plans}),
        "internal_edge_instances": sum(len(plan["edge_path"]) for plan in plans),
        "unique_inventory_edge_ids": len(inventory_edge_counts),
        "max_inventory_edge_reuse_config": args.max_edge_reuse,
        "max_inventory_edge_reuse_observed": max(inventory_edge_counts.values(), default=0),
        "registered_environments": len(registered),
        "registry_filtered_path_candidates": dict(registry_filtered_paths),
        "profiler_unambiguous_only": args.profiler_unambiguous_only,
        "profiler_ambiguous_path_candidates_filtered": dict(profiler_ambiguous_paths),
        "environment_distribution": dict(Counter(env for plan in plans for env in plan["environment_identifiers"])),
    }
    write_json(args.output_dir / "manifests" / f"{args.name}.summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def json_container(value: Any) -> Any:
    for _ in range(3):
        if not isinstance(value, str):
            return value
        rendered = value.strip()
        if rendered.startswith("```"):
            rendered = re.sub(r"^```(?:json)?\s*", "", rendered, flags=re.I)
            rendered = re.sub(r"\s*```$", "", rendered)
        try:
            decoded = json.loads(rendered)
        except (TypeError, json.JSONDecodeError):
            starts = [index for index in (rendered.find("{"), rendered.find("[")) if index >= 0]
            if not starts:
                return value
            try:
                decoded, _ = json.JSONDecoder().raw_decode(rendered[min(starts):])
            except json.JSONDecodeError:
                return value
        value = decoded
    return value


def dependency_trace(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for edge in plan["edge_path"]:
        records.append({
            "consumer_tool": edge["consumer_tool"],
            "consumer_input_parameter": edge["consumer_argument"],
            "consumer_input_data_type": edge["consumer_argument_type"],
            "selected_producer_tool": edge["producer_tool"],
            "selected_producer_output_parameter": edge["producer_output_field"],
            "selected_producer_output_data_type": edge["producer_output_type"],
            "required": True,
            "user_provided": False,
            "internal_parameter": True,
            "alternatives": [{
                "producer_tool": edge["producer_tool"],
                "producer_output_parameters": [{
                    "parameter_name": edge["producer_output_field"],
                    "data_type": edge["producer_output_type"],
                }],
            }],
            "alternative_semantics": "edge_first_fixed_target",
            "provenance": "GraphFrontierRichV3.constraint_aware_edge_first",
        })
    return records


def augment_sidecar(path: Path, plan: Mapping[str, Any]) -> None:
    sidecar = json.loads(path.read_text())
    sidecar.update({
        "rich_extension_schema_version": "graph_frontier_rich_v1",
        "generation_seed": plan["generation_seed"],
        "target_bucket": plan["bucket"],
        "target_depth": plan["target_depth"],
        "target_plan_id": plan["plan_id"],
        "target_edge_ids": [edge["edge_id"] for edge in plan["edge_path"]],
        "gold_nodes": copy.deepcopy(sidecar.get("required_tool_nodes", [])),
        "gold_edges": copy.deepcopy(sidecar.get("dependency_edges", [])),
        "max_dependency_depth": sidecar.get("dependency_depth", "unknown"),
        "internal_edge_count": sum(
            edge.get("internal_parameter") is True and edge.get("required") is True
            for edge in sidecar.get("dependency_edges", [])
        ),
        "tool_chain_length": len(sidecar.get("gold_tool_sequence", [])),
        "structure_selected_before_query_generation": True,
        "gold_solution_conditioning": "hidden_structured_reference_not_user_visible",
    })
    write_json(path, sidecar)


def tool_sequence_from_steps(steps: Sequence[Mapping[str, Any]] | None) -> list[str]:
    return [
        str(call.get("name"))
        for step in (steps or [])
        if isinstance(step, Mapping) and step.get("role") == "tool_call"
        for call in (step.get("content") or [])
        if isinstance(call, Mapping) and isinstance(call.get("name"), str)
    ]


def selected_tool_sequence(node: Any) -> list[str]:
    return tool_sequence_from_steps(getattr(node, "steps", None))


def tool_responses_from_steps(steps: Sequence[Mapping[str, Any]] | None) -> list[Any]:
    """Flatten responses paired with structured tool calls in a QueryGen trace."""
    return [
        response
        for step in (steps or [])
        if isinstance(step, Mapping) and step.get("role") == "tool_response"
        for response in (
            step.get("content")
            if isinstance(step.get("content"), list)
            else [step.get("content")]
        )
    ]


def tool_response_failed(response: Any) -> bool:
    """Recognize the explicit failure wrappers emitted by ``MCPManager.call_tool``.

    This deliberately does not classify arbitrary natural-language mentions of
    errors. QueryGen stores one wrapped response per call; the stable wrappers
    below distinguish an executed MCP failure from valid response data.
    """
    if isinstance(response, Mapping):
        is_error = response.get("isError", response.get("is_error"))
        return is_error is True
    if not isinstance(response, str):
        return True
    lowered = response.lower()
    return (
        " failed: error executing tool " in lowered
        or lowered.startswith("invalid tool:")
    )


def trace_tool_execution_success(steps: Sequence[Mapping[str, Any]] | None) -> bool:
    """Require a one-to-one, failure-free response for every structured call."""
    calls = tool_sequence_from_steps(steps)
    responses = tool_responses_from_steps(steps)
    return bool(calls) and len(responses) == len(calls) and not any(
        tool_response_failed(response) for response in responses
    )


def dotted_values(value: Any, path: str) -> list[Any]:
    """Resolve a dotted field through mappings and lists."""
    if not isinstance(path, str) or not path:
        return []
    current = [json_container(value)]
    for component in path.split("."):
        next_values = []
        for item in current:
            if isinstance(item, Mapping) and component in item:
                next_values.append(item[component])
            elif isinstance(item, list):
                next_values.extend(
                    child[component]
                    for child in item
                    if isinstance(child, Mapping) and component in child
                )
        if not next_values:
            return []
        current = next_values
    return current


def trace_follows_plan_dataflow(
    steps: Sequence[Mapping[str, Any]] | None,
    plan: Mapping[str, Any],
) -> bool:
    """Verify every selected edge transfers an observed producer value."""
    calls = [
        call
        for step in (steps or [])
        if isinstance(step, Mapping) and step.get("role") == "tool_call"
        for call in (step.get("content") or [])
        if isinstance(call, Mapping)
    ]
    responses = tool_responses_from_steps(steps)
    if len(calls) != len(responses):
        return False
    events = [
        {"call": call, "response": json_container(response)}
        for call, response in zip(calls, responses)
    ]
    for edge in plan.get("edge_path", []):
        producer = next(
            (
                (index, event)
                for index, event in enumerate(events)
                if event["call"].get("name") == edge.get("producer_tool")
            ),
            None,
        )
        if producer is None:
            return False
        consumer = next(
            (
                event
                for index, event in enumerate(events)
                if index > producer[0]
                and event["call"].get("name") == edge.get("consumer_tool")
            ),
            None,
        )
        if consumer is None:
            return False
        source_values = dotted_values(
            producer[1]["response"], str(edge.get("producer_output_field", ""))
        )
        target_values = dotted_values(
            consumer["call"].get("arguments", {}),
            str(edge.get("consumer_argument", "")),
        )
        if len(target_values) != 1 or not any(
            type(value) is type(target_values[0]) and value == target_values[0]
            for value in source_values
        ):
            return False
    return True


def trace_leaks_plan_values(
    steps: Sequence[Mapping[str, Any]] | None,
    plan: Mapping[str, Any],
    query: str,
) -> bool:
    """Match the final audit's scalar-token leakage rule during selection."""
    responses = tool_responses_from_steps(steps)
    calls = [
        call
        for step in (steps or [])
        if isinstance(step, Mapping) and step.get("role") == "tool_call"
        for call in (step.get("content") or [])
        if isinstance(call, Mapping)
    ]
    if len(calls) != len(responses):
        return False
    events = [
        {"call": call, "response": json_container(response)}
        for call, response in zip(calls, responses)
    ]
    for edge in plan.get("edge_path", []):
        producer = next(
            (
                event
                for event in events
                if event["call"].get("name") == edge.get("producer_tool")
            ),
            None,
        )
        if producer is None:
            continue
        values = dotted_values(
            producer["response"], str(edge.get("producer_output_field", ""))
        )
        if len(values) != 1:
            continue
        value = values[0]
        if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
            continue
        token = str(value).strip()
        if not token:
            continue
        if re.fullmatch(r"[A-Za-z0-9_-]+", token):
            visible = re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])",
                query,
                re.I,
            )
        else:
            visible = token.casefold() in query.casefold()
        if visible:
            return True
    return False


def set_dotted_argument(arguments: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = arguments
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise ValueError(f"cannot set nested argument {path}")
        current = current.setdefault(part, {})
    if not isinstance(current, dict) or not parts[-1]:
        raise ValueError(f"cannot set nested argument {path}")
    current[parts[-1]] = value


def bind_plan_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
    plan: Mapping[str, Any],
    producer_fields: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Bind unambiguous live producer values into one planned consumer call."""
    bound = copy.deepcopy(dict(arguments))
    rebindings = []
    for edge in plan.get("edge_path", []):
        if edge.get("consumer_tool") != tool_name:
            continue
        producer = str(edge.get("producer_tool"))
        if producer not in producer_fields:
            return None
        source_path = str(edge.get("producer_output_field", ""))
        target_path = str(edge.get("consumer_argument", ""))
        source_values = dotted_values(producer_fields[producer], source_path)
        target_values = dotted_values(bound, target_path)
        if not source_values or len(target_values) != 1:
            return None
        old_value = target_values[0]
        if any(type(value) is type(old_value) and value == old_value for value in source_values):
            continue
        unique = {canonical(value): value for value in source_values}
        if len(unique) != 1:
            return None
        value = next(iter(unique.values()))
        if type(value) is not type(old_value):
            return None
        set_dotted_argument(bound, target_path, value)
        rebindings.append({
            "producer_tool": producer,
            "producer_output_parameter": source_path,
            "consumer_input_parameter": target_path,
            "old_value": old_value,
            "runtime_value": value,
        })
    return bound, rebindings


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve every planned edge against live graph, registry and metadata."""
    from src.graph.tool_chain import ToolQueryChain, ToolQueryNode
    from src.graph.tool_graph import ToolGraph
    from src.graph.tool_node import Tool

    from repro_1p7b.graph_frontier.audit_generated_guided_rl import metadata_index, structural_checks
    from repro_1p7b.graph_frontier.gold_sidecar import build_gold_sidecar
    from repro_1p7b.graph_frontier.traceable_sampler import TRACE_ATTRIBUTE

    graph = ToolGraph.load(args.graph)
    tools = {node.name: node for node in graph.graph.nodes if isinstance(node, Tool)}
    plans = read_jsonl(args.plan)
    registry_payload = json.loads(args.registry.read_text())
    registry = registry_payload.get("mcpServers", {})
    metadata = metadata_index(args.metadata_dir)
    results = []
    for plan in plans:
        chain = ToolQueryChain([tools[name] for name in plan["tool_sequence"]], seed=plan["generation_seed"])
        chain.tool_chain = [ToolQueryNode(
            raw_tool_call=list(chain.init_tool_chain),
            query="preflight placeholder",
            initial_scenario={},
            final_scenario={},
            decision=True,
            steps=[{"role": "tool_call", "content": []}],
        )]
        setattr(chain, TRACE_ATTRIBUTE, dependency_trace(plan))
        sidecar = build_gold_sidecar(graph, chain, 0, task_id=plan["plan_id"])
        graph_ok, tool_ok, reasons = structural_checks(sidecar, registry, metadata, ROOT)
        internal = [
            edge for edge in sidecar.get("dependency_edges", [])
            if edge.get("internal_parameter") is True and edge.get("required") is True
        ]
        expected_ids = set(plan["target_edge_ids"] if "target_edge_ids" in plan else [edge["edge_id"] for edge in plan["edge_path"]])
        resolved = {
            edge_signature({
                "producer_environment": edge["producer_tool"]["tool_name"].split("-", 1)[0],
                "producer_tool": edge["producer_tool"]["tool_name"],
                "producer_output_field": edge["producer_output_parameter"]["parameter_name"],
                "consumer_environment": edge["consumer_tool"]["tool_name"].split("-", 1)[0],
                "consumer_tool": edge["consumer_tool"]["tool_name"],
                "consumer_argument": edge["consumer_input_parameter"]["parameter_name"],
            })
            for edge in internal
        }
        planned = {edge_signature(edge) for edge in plan["edge_path"]}
        ok = (
            graph_ok and tool_ok
            and sidecar.get("dependency_resolution", {}).get("status") == "resolved"
            and sidecar.get("dependency_depth") == plan["target_depth"]
            and len(internal) == plan["target_depth"]
            and resolved == planned
        )
        results.append({
            "plan_id": plan["plan_id"],
            "bucket": plan["bucket"],
            "pass": ok,
            "graph_ok": graph_ok,
            "tool_schema_and_mapping_ok": tool_ok,
            "resolved_depth": sidecar.get("dependency_depth"),
            "resolved_internal_edges": len(internal),
            "reasons": reasons,
            "unused_expected_ids": sorted(expected_ids),
        })
    report = {
        "schema_version": "graph_frontier_rich_preflight_v1",
        "plans": len(results),
        "passed": sum(row["pass"] for row in results),
        "failed": sum(not row["pass"] for row in results),
        "by_bucket": {
            bucket: {
                "plans": sum(row["bucket"] == bucket for row in results),
                "passed": sum(row["bucket"] == bucket and row["pass"] for row in results),
            }
            for bucket in sorted({row["bucket"] for row in results})
        },
        "results": results,
    }
    write_json(args.output_dir / "audit" / f"{args.report_name}.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failed"]:
        raise SystemExit(2)
    return report


async def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    """Languageize a precomputed structure plan with the established 14B QueryGen."""
    from src.gen.query_gen import QueryGenConfig, QueryGenState
    from src.gen.query_gen.query_gen_non_conv import QueryGenNonConv
    from src.graph.tool_chain import ToolQueryChain, ToolQueryNode
    from src.graph.tool_graph import ToolGraph
    from src.graph.tool_node import Tool
    from src.manager.mcp_client_manager import MCPManager

    from repro_1p7b.graph_frontier.gold_sidecar import GenerationSidecarCallback
    from repro_1p7b.graph_frontier.guided_rl_generate import _validate_completed_node
    from repro_1p7b.graph_frontier.traceable_sampler import TRACE_ATTRIBUTE

    graph = ToolGraph.load(args.graph)
    if not MCPManager.server_to_path_mapping:
        raise RuntimeError("MCP registry is empty; set MCP_CONFIG_PATH before generation")
    plans = read_jsonl(args.plan)
    if args.limit is not None:
        plans = plans[: args.limit]
    if not plans:
        raise RuntimeError("generation plan is empty")
    seeds = [int(plan["generation_seed"]) for plan in plans]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("generation seeds must be unique")
    plan_by_seed = {int(plan["generation_seed"]): plan for plan in plans}
    tools = {node.name: node for node in graph.graph.nodes if isinstance(node, Tool)}
    missing_tools = sorted({name for plan in plans for name in plan["tool_sequence"] if name not in tools})
    if missing_tools:
        raise RuntimeError(f"planned tools missing from ToolGraph: {missing_tools[:10]}")

    original_load_scenario = MCPManager.load_scenario

    def load_structured_scenario(client_id, scenario=None, check=False):
        return original_load_scenario(client_id=client_id, scenario=json_container(scenario), check=check)

    MCPManager.load_scenario = load_structured_scenario
    output = args.run_dir
    raw_dir, sidecar_dir, log_dir = output / "raw", output / "sidecar", output / "querygen_logs"
    for directory in (raw_dir, sidecar_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    callback = GenerationSidecarCallback(
        sidecar_dir,
        task_id_factory=lambda context: plan_by_seed[int(context.tool_chain.seed)]["plan_id"],
    )

    class RichQueryGen(QueryGenNonConv):
        async def prepare(self, context):
            state = await super().prepare(context)
            if state != QueryGenState.Preparing and context.tool_chain.scenario is not None:
                # The target path is one atomic frontier task.  Do not let the
                # language model or probabilistic fallback split its edges.
                context.tool_chain.tool_chain = [
                    ToolQueryNode(raw_tool_call=list(context.tool_chain.init_tool_chain))
                ]
            return state

        def split_turns(self, init_tool_chain, max_turn=5):
            del max_turn
            return [ToolQueryNode(raw_tool_call=list(init_tool_chain))]

        async def schema_generate(self, context):
            result = await super().schema_generate(context)
            return {server: json_container(value) for server, value in result.items()}

        async def generate(self, context):
            plan = plan_by_seed[int(context.tool_chain.seed)]
            request_id = f"{context.conversation_id}{context.idx}"
            if self.context_manager.get_prompt(self.query_generator.name, request_id) is None:
                self.context_manager.add_prompt(
                    self.query_generator.name,
                    request_id,
                    "Generate one natural user request for exactly these target tools: "
                    f"{' -> '.join(plan['tool_sequence'])}. Every user-provided identifier, "
                    "entity name, credential, status, or other lookup value must be copied from "
                    "the hidden MCP server configuration in your system context. Never invent "
                    "placeholder IDs, generic names, credentials, phone numbers, or references "
                    "to entities absent from that configuration. Values produced by an earlier "
                    "target tool must not be exposed in the user request. The request must make "
                    "every target operation executable in the listed order without extra tools."
                )
            return await super().generate(context)

        async def solve(self, context):
            plan = plan_by_seed[int(context.tool_chain.seed)]
            agent_name = f"{self.query_solver.name}_{context.k}"
            request_id = f"{context.conversation_id}{context.idx}{context.k}"
            if self.context_manager.get_prompt(agent_name, request_id) is None:
                flows = "; ".join(
                    f"use {edge['producer_tool']}.{edge['producer_output_field']} as "
                    f"{edge['consumer_tool']}.{edge['consumer_argument']}"
                    for edge in plan["edge_path"]
                )
                hidden_reference = (
                    f"{context.tool_chain[context.idx].query}\n\n"
                    "[Gold-data construction constraint; never repeat this constraint to the user.] "
                    f"Execute exactly this tool sequence: {' -> '.join(plan['tool_sequence'])}. "
                    f"Preserve real returned values across calls: {flows}. "
                    "For every other argument, copy an exact compatible value from the initial "
                    "environment state below or from the user request; never use placeholders or "
                    "invent entity IDs, names, credentials, phone numbers, or statuses. Stop if a "
                    "tool fails; never claim a failed call succeeded. Do not skip, replace, repeat, "
                    "or invent any tool call or internal value.\n"
                    "Initial environment state (hidden from the user):\n"
                    f"{canonical(context.tool_chain[context.idx].initial_scenario)}"
                )
                self.context_manager.add_prompt(agent_name, request_id, hidden_reference)
            return await super().solve(context)

        def repair_exact_candidate(self, context, trace, plan, selected):
            """Re-execute an exact sequence with live internal-value bindings."""
            node = context.tool_chain[context.idx]
            calls = [
                copy.deepcopy(call)
                for step in trace
                if isinstance(step, Mapping) and step.get("role") == "tool_call"
                for call in (step.get("content") or [])
                if isinstance(call, Mapping)
            ]
            if [call.get("name") for call in calls] != plan["tool_sequence"]:
                return None
            repair_id = (
                f"rich-repair-{context.tool_chain.seed}-{context.idx}-{selected}-"
                f"{sha256_bytes(canonical(calls).encode())[:12]}"
            )
            client_ids = {
                server: f"{server}-{repair_id}"
                for server in node.initial_scenario
            }
            steps = [{"role": "user", "content": node.query}]
            producer_fields: dict[str, Any] = {}
            try:
                for server, scenario in node.initial_scenario.items():
                    MCPManager.load_scenario(
                        client_id=client_ids[server], scenario=scenario, check=True
                    )
                for call in calls:
                    name = call.get("name")
                    arguments = call.get("arguments")
                    if not isinstance(name, str) or not isinstance(arguments, Mapping):
                        return None
                    server = name.split("-", 1)[0]
                    if server not in client_ids:
                        return None
                    binding = bind_plan_arguments(name, arguments, plan, producer_fields)
                    if binding is None:
                        return None
                    arguments, rebindings = binding
                    response = MCPManager.call_tool(
                        client_id=client_ids[server],
                        tool_name=name,
                        tool_args=arguments,
                    )
                    if tool_response_failed(response):
                        return None
                    fields = json_container(response)
                    if isinstance(fields, str):
                        return None
                    producer_fields[name] = fields
                    steps.append({
                        "role": "tool_call",
                        "content": [{"name": name, "arguments": arguments}],
                        "runtime_dependency_rebindings": rebindings,
                    })
                    steps.append({"role": "tool_response", "content": [response]})
                final_scenario = {
                    server: json_container(MCPManager.call_tool(
                        client_id=client_id,
                        tool_name="save_scenario",
                        tool_args={},
                    ))
                    for server, client_id in client_ids.items()
                }
                steps.append({
                    "role": "assistant",
                    "content": "The requested operations were completed successfully.",
                    "think": None,
                })
                if not trace_tool_execution_success(steps):
                    return None
                if not trace_follows_plan_dataflow(steps, plan):
                    return None
                return steps, final_scenario
            except Exception:
                return None
            finally:
                for client_id in client_ids.values():
                    try:
                        MCPManager.close_client(client_id)
                    except Exception:
                        pass

        async def select(self, context):
            plan = plan_by_seed[int(context.tool_chain.seed)]
            node = context.tool_chain[context.idx]
            exact_sequence = [
                index
                for index, trace in sorted(node.pass_k_trace.items())
                if node.pass_k_decision.get(index) is True
                and tool_sequence_from_steps(trace) == plan["tool_sequence"]
            ]
            for selected in exact_sequence:
                trace = node.pass_k_trace[selected]
                if trace_tool_execution_success(trace) and trace_follows_plan_dataflow(trace, plan):
                    final_scenario = node.pass_k_scenario[selected]
                else:
                    repaired = self.repair_exact_candidate(context, trace, plan, selected)
                    if repaired is None:
                        continue
                    trace, final_scenario = repaired
                if trace_leaks_plan_values(trace, plan, str(node.query or "")):
                    continue
                context.tool_chain.update_node(
                    context.idx,
                    decision=True,
                    steps=trace,
                    final_scenario=final_scenario,
                )
                node.accuracy = len(exact_sequence) / self.config.pass_k
                return
            if not exact_sequence:
                raise RuntimeError(
                    "no exact trajectory for target path: "
                    f"planned={plan['tool_sequence']}"
                )
            raise RuntimeError(
                "no exact executable non-leaking dataflow trajectory for target path: "
                f"planned={plan['tool_sequence']}"
            )

        async def terminate(self, context):
            _validate_completed_node(context.tool_chain, context.idx)
            plan = plan_by_seed[int(context.tool_chain.seed)]
            actual = [tool.name for tool in context.tool_chain[context.idx].raw_tool_call]
            if actual != plan["tool_sequence"]:
                raise RuntimeError(f"target path changed: planned={plan['tool_sequence']} actual={actual}")
            selected = selected_tool_sequence(context.tool_chain[context.idx])
            if selected != plan["tool_sequence"]:
                raise RuntimeError(
                    f"selected trajectory does not execute complete target path: "
                    f"planned={plan['tool_sequence']} selected={selected}"
                )
            path = callback.before_save(context)
            augment_sidecar(path, plan)
            # Preserve the live ToolGraph/ToolQueryNode metadata before the
            # official serializer discards raw_tool_call.
            await super().terminate(context)

    model_names = [item.strip() for item in args.model_name.split(",") if item.strip()]
    if not model_names:
        raise ValueError("at least one model name is required")
    configs = {
        name: QueryGenConfig(
            model_name=name,
            pass_k=args.pass_k,
            max_iterations=10,
            max_solve_iterations=15,
            enable_split_turns=False,
            enable_query_refinement=False,
            enable_user_interaction=False,
            enable_user_tool_use=False,
            enable_user_verification=False,
            enable_filteration=False,
            enable_log_thinking_content=True,
            save_folder=str(raw_dir),
            log_folder=str(log_dir),
        )
        for name in model_names
    }
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(position: int, plan: Mapping[str, Any]) -> Any:
        async with semaphore:
            model_name = model_names[position % len(model_names)]
            failures = []
            for attempt in range(args.attempts):
                chain = ToolQueryChain(
                    [tools[name] for name in plan["tool_sequence"]],
                    seed=int(plan["generation_seed"]),
                )
                setattr(chain, TRACE_ATTRIBUTE, dependency_trace(plan))
                try:
                    result = await RichQueryGen(graph, configs[model_name]).gen(chain)
                    if len(result.tool_chain) != 1:
                        raise RuntimeError(f"expected one generated turn, got {len(result.tool_chain)}")
                    _validate_completed_node(result, 0)
                    return result
                except Exception as exc:
                    failures.append(f"attempt={attempt + 1}:{type(exc).__name__}:{exc}")
            raise RuntimeError(" | ".join(failures))

    started = __import__("time").time()
    results = await asyncio.gather(*(one(index, plan) for index, plan in enumerate(plans)), return_exceptions=True)
    failures = [repr(item) for item in results if isinstance(item, BaseException)]
    report = {
        "schema_version": "graph_frontier_rich_generation_run_v1",
        "generator_model": os.environ.get("SGLANG_MODEL", "Qwen2.5-14B-Instruct"),
        "model_names": model_names,
        "requested_plans": len(plans),
        "completed_chains": len(plans) - len(failures),
        "failed_chains": len(failures),
        "failures": failures,
        "raw_chain_files": len(list(raw_dir.glob("*.json"))),
        "sidecar_files": len(list(sidecar_dir.glob("*.gold.json"))),
        "pass_k": args.pass_k,
        "concurrency": args.concurrency,
        "attempts_per_structure": args.attempts,
        "graph_sha256": file_sha256(args.graph),
        "plan_sha256": file_sha256(args.plan),
        "runtime_seconds": __import__("time").time() - started,
        "frozen300_content_loaded": False,
    }
    write_json(output / "generation_run.json", report)
    if len(failures) > args.max_failures:
        raise RuntimeError(f"generation failures exceeded gate: {len(failures)}/{args.max_failures}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    sub = result.add_subparsers(dest="command", required=True)
    capacity = sub.add_parser("capacity")
    capacity.add_argument("--graph", required=True)
    capacity.add_argument("--max-depth", type=int, default=4)
    capacity.add_argument("--max-paths-per-depth", type=int, default=20_000)
    plan = sub.add_parser("plan")
    plan.add_argument("--name", default="smoke_plan")
    plan.add_argument("--depth1", type=int, default=10)
    plan.add_argument("--depth2", type=int, default=10)
    plan.add_argument("--depth3", type=int, default=10)
    plan.add_argument("--seed", type=int, default=20260921)
    plan.add_argument("--max-edge-reuse", type=int, default=2)
    plan.add_argument("--registry", type=Path, default=ROOT / "configs/mcp_server.json")
    plan.add_argument("--deep-first", action="store_true")
    plan.add_argument("--profiler-unambiguous-only", action="store_true")
    generation = sub.add_parser("generate")
    generation.add_argument("--graph", required=True)
    generation.add_argument("--plan", type=Path, required=True)
    generation.add_argument("--run-dir", type=Path, required=True)
    generation.add_argument("--model-name", default="sglang,sglang1")
    generation.add_argument("--pass-k", type=int, default=2)
    generation.add_argument("--concurrency", type=int, default=4)
    generation.add_argument("--attempts", type=int, default=3)
    generation.add_argument("--limit", type=int)
    generation.add_argument("--max-failures", type=int, default=30)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--graph", required=True)
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--registry", type=Path, default=ROOT / "configs/mcp_server.json")
    preflight.add_argument("--metadata-dir", type=Path, default=ROOT / "envs/metadata")
    preflight.add_argument("--report-name", default="smoke_preflight")
    return result


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "capacity":
        run_capacity(args)
    elif args.command == "plan":
        run_plan(args)
    elif args.command == "generate":
        asyncio.run(run_generate(args))
    else:
        run_preflight(args)


if __name__ == "__main__":
    main()

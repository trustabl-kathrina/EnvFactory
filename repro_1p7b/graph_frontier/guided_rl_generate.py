"""EnvFactory-generated executable RL pool with generation-time graph sidecars.

This is a wrapper around the official ToolGraph -> TopologySampler -> QueryGen ->
``convert_to_rl_data`` path.  It does not patch ``src/`` and never reconstructs
gold edges from flattened text.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_FROZEN = ROOT / (
    "repro_1p7b/results/graph_frontier/confirm_300/frozen/"
    "confirm_300_seed_20260914.jsonl"
)
FROZEN_SHA256 = "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_query(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.casefold().strip())


def query_hash(text: Any) -> str:
    return _sha256_bytes(normalized_query(text).encode("utf-8"))


def tool_sequence(sidecar: Mapping[str, Any]) -> list[str]:
    sequence = sidecar.get("gold_tool_sequence", []) or []
    return [
        str(item.get("tool_name"))
        for item in sequence
        if isinstance(item, Mapping) and isinstance(item.get("tool_name"), str)
    ]


def tool_sequence_hash(names: Sequence[str]) -> str:
    return _sha256_bytes(_canonical(list(names)).encode("utf-8"))


def _last_user_query(prompt: Any) -> str:
    prompt = json.loads(prompt) if isinstance(prompt, str) else prompt
    for item in reversed(prompt or []):
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            return item["content"].strip()
    return ""


def _initial_config(row: Mapping[str, Any]) -> Any:
    value = row.get("extra_info", {}).get("mcp_factory_kwargs", {}).get("initial_config", {})
    return json.loads(value) if isinstance(value, str) else value


def _json_container(value: Any) -> Any:
    """Losslessly recover a JSON object/array emitted as a tagged string."""
    original = value
    for _ in range(3):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            break
        if isinstance(value, (dict, list)):
            return value
    return original


def enrich_official_rows(
    rows: Sequence[Mapping[str, Any]], sidecars: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Attach live graph metadata to rows produced by official conversion."""

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sidecar in sidecars:
        by_query[normalized_query(sidecar.get("query", {}).get("text"))].append(dict(sidecar))
    enriched = []
    for raw in rows:
        row = dict(raw)
        query = _last_user_query(row.get("prompt"))
        candidates = by_query.get(normalized_query(query), [])
        initial = _initial_config(row)
        exact = [item for item in candidates if item.get("initial_scenario") == initial]
        selected_pool = exact or candidates
        if len(selected_pool) != 1:
            raise RuntimeError(
                f"sidecar join must be unique for query={query!r}: "
                f"candidates={len(candidates)} exact_initial={len(exact)}"
            )
        sidecar = selected_pool[0]
        candidates.remove(sidecar)
        ground_truth = _json_container(row.get("reward_model", {}).get("ground_truth", []))
        if not isinstance(ground_truth, list):
            raise RuntimeError(f"ground truth is not a list for task {sidecar['task_id']}")
        gold_names = tool_sequence(sidecar)
        normalized_calls = []
        for original_call in ground_truth:
            call = dict(original_call)
            name = call.get("name")
            if isinstance(name, str) and name not in gold_names:
                candidates = [item for item in gold_names if item.endswith("-" + name)]
                if len(candidates) == 1:
                    call["name"] = candidates[0]
            normalized_calls.append(call)
        reward_model = dict(row.get("reward_model", {}))
        reward_model["ground_truth"] = json.dumps(normalized_calls)
        row["reward_model"] = reward_model
        extra = dict(row.get("extra_info", {}))
        extra["graph_frontier"] = sidecar
        extra["task_id"] = sidecar["task_id"]
        extra["generation_seed"] = sidecar.get("seed", "unknown")
        extra["frozen_300_overlap"] = False
        row["extra_info"] = extra
        row["task_id"] = sidecar["task_id"]
        row["source"] = "EnvFactory-generated-guided-rl-v1"
        row["frozen_300_overlap"] = False
        enriched.append(row)
    return enriched


def _frozen_signatures(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result = {"task_ids": set(), "query_hashes": set(), "tool_sequence_hashes": set()}
    for row in rows:
        if isinstance(row.get("task_id"), str):
            result["task_ids"].add(row["task_id"])
        result["query_hashes"].add(query_hash(row.get("query")))
        names = row.get("gold_tools", []) or []
        if names:
            result["tool_sequence_hashes"].add(tool_sequence_hash(names))
    return result


def contamination_audit(
    generated: Sequence[Mapping[str, Any]], frozen: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove exact prompt/task collisions and report structural near matches."""

    signatures = _frozen_signatures(frozen)
    kept, removed, query_hits, task_hits = [], [], 0, 0
    sequence_hits = 0
    seen_generated = set()
    duplicate_generated = 0
    for raw in generated:
        row = dict(raw)
        sidecar = row.get("extra_info", {}).get("graph_frontier", {})
        qhash = query_hash(_last_user_query(row.get("prompt")))
        task_id = str(row.get("task_id", ""))
        shash = tool_sequence_hash(tool_sequence(sidecar))
        query_hit = qhash in signatures["query_hashes"]
        task_hit = task_id in signatures["task_ids"]
        sequence_hits += int(shash in signatures["tool_sequence_hashes"])
        key = (qhash, shash)
        if key in seen_generated:
            duplicate_generated += 1
            removed.append(task_id)
            continue
        seen_generated.add(key)
        query_hits += int(query_hit)
        task_hits += int(task_hit)
        if query_hit or task_hit:
            removed.append(task_id)
            continue
        kept.append(row)
    report = {
        "schema_version": "graph_frontier_rl_v1_contamination_audit",
        "frozen_task_count": len(frozen),
        "generated_input_count": len(generated),
        "kept_count": len(kept),
        "removed_exact_count": len(removed),
        "removed_task_ids": removed,
        "frozen_query_hash_hits": query_hits,
        "frozen_task_id_hits": task_hits,
        "tool_sequence_only_hits_near_duplicate_audit": sequence_hits,
        "within_generated_duplicate_count": duplicate_generated,
        "frozen_300_exact_overlap": 0,
        "exact_removal_rule": "normalized query hash OR task_id; tool sequence alone is audit-only",
    }
    return kept, report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_by_generation_seed(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split whole generated conversations so adjacent turns cannot leak."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        generation_seed = row.get("extra_info", {}).get("generation_seed", "unknown")
        groups[str(generation_seed)].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    validation_group_count = max(1, round(len(keys) * (1.0 - train_ratio))) if len(keys) > 1 else 0
    validation_keys = set(keys[:validation_group_count])
    train = [row for key in keys if key not in validation_keys for row in groups[key]]
    validation = [row for key in keys if key in validation_keys for row in groups[key]]
    train_seeds = {
        str(row.get("extra_info", {}).get("generation_seed", "unknown")) for row in train
    }
    validation_seeds = {
        str(row.get("extra_info", {}).get("generation_seed", "unknown")) for row in validation
    }
    overlap = sorted(train_seeds & validation_seeds)
    if overlap:
        raise RuntimeError(f"generation seed leakage across train/val: {overlap}")
    return train, validation, {
        "group_key": "generation_seed",
        "generation_groups": len(keys),
        "train_generation_groups": len(train_seeds),
        "validation_generation_groups": len(validation_seeds),
        "generation_seed_overlap": overlap,
    }


class LocalBGEEncoder:
    """Drop-in ``LLMClient.encode`` replacement backed by a local BGE model."""

    def __init__(self, model_path: str, device: str, max_length: int = 256) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True
        ).to(self.device).eval()

    def encode(self, texts: str | Sequence[str], disable_progress_bar: bool = False):
        import numpy as np

        del disable_progress_bar
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for begin in range(0, len(texts), 32):
            batch = self.tokenizer(
                list(texts[begin : begin + 32]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                values = self.model(**batch).last_hidden_state[:, 0]
                values = self.torch.nn.functional.normalize(values, p=2, dim=1)
            result.extend(values.float().cpu().numpy())
        return np.asarray(result)


def build_graph(args: argparse.Namespace) -> None:
    from src.graph.tool_graph import ToolGraph
    from src.manager.llm_client_manager import LLMClient

    metadata = [json.loads(path.read_text()) for path in sorted(Path(args.metadata_dir).glob("*.json"))]
    if not metadata:
        raise RuntimeError(f"no metadata found in {args.metadata_dir}")
    encoder = LocalBGEEncoder(args.embedding_model, args.embedding_device)
    LLMClient.encode = encoder.encode
    graph = ToolGraph()
    graph.build_tool_graph(
        metadata=metadata,
        enable_merge=False,
        enable_build_edge_with_llm=not args.no_llm_edges,
    )
    repair_report = repair_unknown_user_provided(
        graph,
        batch_size=args.repair_batch_size,
        max_retry=args.repair_max_retry,
    )
    output = Path(args.output)
    graph.save(output)
    from src.graph.tool_node import Parameter, Tool

    report = {
        "metadata_files": len(metadata),
        "tools": sum(isinstance(node, Tool) for node in graph.graph.nodes),
        "parameters": sum(isinstance(node, Parameter) for node in graph.graph.nodes),
        "edges": graph.graph.number_of_edges(),
        "servers": len(graph.server_to_tools),
        "embedding_model": str(Path(args.embedding_model).resolve()),
        "llm_edges": not args.no_llm_edges,
        "user_provided_repair": repair_report,
        "graph_sha256": file_sha256(output),
    }
    _write_json(output.with_suffix(".audit.json"), report)


def _parameter_to_tool_map(graph: Any) -> dict[Any, Any]:
    """Reconstruct the same parameter context map used by ToolGraph.build_tool_graph."""
    from src.graph.tool_node import Tool

    result: dict[Any, Any] = {}
    for node in graph.graph.nodes:
        if not isinstance(node, Tool):
            continue
        for schema in (node.input_schema, node.output_schema):
            for parameter in schema.get("parameters", []):
                result[parameter] = node
    return result


def repair_unknown_user_provided(
    graph: Any,
    *,
    batch_size: int = 8,
    max_retry: int = 5,
) -> dict[str, Any]:
    """Retry incomplete LLM labels and reject a graph that still contains unknowns.

    The upstream batch helper deliberately skips malformed batches.  Unknown
    labels are unsafe for dependency sampling because ``None`` is treated like
    false, so this wrapper retries only the unresolved parameters in smaller
    batches and turns any remaining unknown into a hard generation gate.
    """
    from src.graph.tool_node import Parameter, batch_get_user_provided

    parameters = [node for node in graph.graph.nodes if isinstance(node, Parameter)]
    unresolved_before = [item for item in parameters if item.user_provided is None]
    if unresolved_before:
        batch_get_user_provided(
            unresolved_before,
            param_to_tool_map=_parameter_to_tool_map(graph),
            batch_size=batch_size,
            max_retry=max_retry,
        )
    unresolved_after = [item for item in parameters if item.user_provided is None]
    report = {
        "parameters": len(parameters),
        "unknown_before": len(unresolved_before),
        "unknown_after": len(unresolved_after),
        "repair_batch_size": batch_size,
        "repair_max_retry": max_retry,
    }
    if unresolved_after:
        sample = [item.name for item in unresolved_after[:10]]
        raise RuntimeError(
            "graph has unresolved user_provided labels after repair: "
            f"{len(unresolved_after)}; sample={sample}"
        )
    return report


def repair_saved_graph(args: argparse.Namespace) -> None:
    """Repair a completed graph from an earlier build without rebuilding edges."""
    from src.graph.tool_graph import ToolGraph

    graph_path = Path(args.graph)
    graph = ToolGraph.load(graph_path)
    report = repair_unknown_user_provided(
        graph,
        batch_size=args.batch_size,
        max_retry=args.max_retry,
    )
    graph.save(graph_path)
    report["graph_sha256"] = file_sha256(graph_path)
    _write_json(graph_path.with_suffix(".repair.json"), report)


def _task_id(context: Any) -> str:
    return f"gf-rl-v1-s{context.tool_chain.seed}-t{context.idx}"


def _validate_completed_node(tool_chain: Any, turn_index: int) -> None:
    node = tool_chain.tool_chain[turn_index]
    issues = []
    if node.decision is not True:
        issues.append("decision_not_true")
    if not isinstance(node.query, str) or not node.query.strip():
        issues.append("query_missing")
    if not isinstance(node.initial_scenario, dict):
        issues.append("initial_scenario_missing")
    else:
        missing = sorted(set(node.mcp_servers) - set(node.initial_scenario))
        if missing:
            issues.append(f"initial_scenario_missing_servers={missing}")
    if not isinstance(node.final_scenario, dict):
        issues.append("final_scenario_missing")
    if not isinstance(node.steps, list) or not node.steps:
        issues.append("steps_missing")
    elif not any(step.get("role") == "tool_call" for step in node.steps):
        issues.append("tool_calls_missing")
    if issues:
        raise RuntimeError(
            f"generated turn failed quality gate seed={tool_chain.seed} "
            f"turn={turn_index}: {issues}"
        )


async def generate(args: argparse.Namespace) -> None:
    from src.gen.query_gen import QueryGenConfig
    from src.gen.query_gen.query_gen_non_conv import QueryGenNonConv
    from src.graph.sampler import TopologySampler
    from src.graph.tool_graph import ToolGraph
    from src.graph.tool_node import Tool
    from src.manager.mcp_client_manager import MCPManager

    from repro_1p7b.graph_frontier.gold_sidecar import GenerationSidecarCallback
    from repro_1p7b.graph_frontier.traceable_sampler import sample_with_dependency_trace

    graph = ToolGraph.load(args.graph)
    if not MCPManager.server_to_path_mapping:
        raise RuntimeError(
            "MCP registry is empty; set MCP_CONFIG_PATH before importing "
            "guided_rl_generate.py"
        )

    # QueryGen's structured-output parser can preserve <schema> JSON as a
    # string, while FastMCP 3.x validates load_scenario as a dictionary.  Keep
    # this compatibility conversion in the wrapper rather than patching src/.
    original_load_scenario = MCPManager.load_scenario

    def load_structured_scenario(client_id, scenario=None, check=False):
        return original_load_scenario(
            client_id=client_id,
            scenario=_json_container(scenario),
            check=check,
        )

    MCPManager.load_scenario = load_structured_scenario
    output = Path(args.output_dir)
    raw_dir, sidecar_dir, log_dir = output / "raw", output / "gold", output / "querygen_logs"
    for directory in (raw_dir, sidecar_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    callback = GenerationSidecarCallback(sidecar_dir, task_id_factory=_task_id)

    class SidecarQueryGen(QueryGenNonConv):
        async def schema_generate(self, context):
            result = await super().schema_generate(context)
            return {server: _json_container(value) for server, value in result.items()}

        async def terminate(self, context):
            await super().terminate(context)
            _validate_completed_node(context.tool_chain, context.idx)
            callback.before_save(context)

    model_names = [item.strip() for item in args.model_name.split(",") if item.strip()]
    if not model_names:
        raise ValueError("at least one model name is required")
    configs = {
        model_name: QueryGenConfig(
            model_name=model_name,
            pass_k=args.pass_k,
            max_iterations=10,
            max_solve_iterations=15,
            enable_split_turns=False,
            enable_query_refinement=False,
            enable_user_interaction=False,
            enable_user_tool_use=False,
            enable_user_verification=False,
            enable_filteration=True,
            enable_log_thinking_content=True,
            save_folder=str(raw_dir),
            log_folder=str(log_dir),
        )
        for model_name in model_names
    }
    tools = sorted(
        (node for node in graph.graph.nodes if isinstance(node, Tool)),
        key=lambda item: item.name,
    )
    rng = random.Random(args.seed)
    seeds = rng.sample(range(1, 2_000_000_000), args.count)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(position: int, seed: int):
        sampler = TopologySampler(max_servers=3, max_recursion_depth=5)
        start = tools[(position * 9973 + seed) % len(tools)]
        max_nodes = (2, 3, 5, 7)[position % 4]
        chain = sample_with_dependency_trace(
            graph, sampler, max_nodes=max_nodes, start_node=start, seed=seed
        )
        async with semaphore:
            model_name = model_names[position % len(model_names)]
            generator = SidecarQueryGen(graph, configs[model_name])
            result = await generator.gen(chain)
            for turn_index in range(len(result.tool_chain)):
                _validate_completed_node(result, turn_index)
            return result

    results = await asyncio.gather(
        *(one(position, seed) for position, seed in enumerate(seeds)),
        return_exceptions=True,
    )
    report = {
        "requested": args.count,
        "completed": sum(not isinstance(item, BaseException) for item in results),
        "failed": sum(isinstance(item, BaseException) for item in results),
        "failures": [repr(item) for item in results if isinstance(item, BaseException)],
        "generation_seed": args.seed,
        "sample_seeds": seeds,
        "pass_k": args.pass_k,
        "concurrency": args.concurrency,
        "graph_sha256": file_sha256(args.graph),
        "model_names": model_names,
        "valid_task_sidecars": len(list(sidecar_dir.glob("*.gold.json"))),
        "raw_chain_files": len(list(raw_dir.glob("*.json"))),
    }
    _write_json(output / "generation_run.json", report)
    if report["valid_task_sidecars"] < args.min_valid_tasks:
        raise RuntimeError(
            f"valid task sidecars below gate: {report['valid_task_sidecars']}"
            f"/{args.min_valid_tasks}"
        )
    if report["failed"] > args.max_failures:
        raise RuntimeError(f"generation failures: {report['failed']}/{args.count}")


def convert_and_audit(args: argparse.Namespace) -> None:
    from src.utils.data_process import convert_to_rl_data, load_tool_chains

    output = Path(args.output_dir)
    frozen_path = Path(args.frozen_manifest)
    if file_sha256(frozen_path) != FROZEN_SHA256:
        raise RuntimeError("frozen-300 manifest SHA256 mismatch")
    chains = load_tool_chains([str(output / "raw")])
    official_path = output / "official_unenriched.json"
    convert_to_rl_data(chains, str(official_path), shuffle=False, seed=args.seed)
    official = json.loads(official_path.read_text())
    sidecars = [json.loads(path.read_text()) for path in sorted((output / "gold").glob("*.gold.json"))]
    enriched = enrich_official_rows(official, sidecars)
    static_filter_removed = []
    if args.audit_ledger:
        ledger_rows = _load_jsonl(Path(args.audit_ledger))
        allowed = {
            str(row["task_id"]) for row in ledger_rows
            if row.get("status") == "STATIC_PASS"
        }
        static_filter_removed = [
            str(row.get("task_id")) for row in enriched
            if str(row.get("task_id")) not in allowed
        ]
        enriched = [
            row for row in enriched if str(row.get("task_id")) in allowed
        ]
    kept, audit = contamination_audit(enriched, _load_jsonl(frozen_path))
    audit["static_audit_filter_removed_count"] = len(static_filter_removed)
    audit["static_audit_filter_removed_task_ids"] = static_filter_removed
    train, val, split_audit = split_by_generation_seed(
        kept,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "rl_val"
    _write_json(output / "train.json", train)
    _write_json(output / "rl_val.json", val)
    _write_json(output / "contamination_audit.json", audit)
    depths = Counter(
        str(row["extra_info"]["graph_frontier"].get("dependency_depth", "unknown"))
        for row in kept
    )
    environments = Counter(
        server
        for row in kept
        for server in row["extra_info"]["graph_frontier"].get("environment_identifiers", [])
    )
    internal_edges = sum(
        edge.get("internal_parameter") is True
        for row in kept
        for edge in row["extra_info"]["graph_frontier"].get("dependency_edges", [])
    )
    manifest = {
        "schema_version": "graph_frontier_guided_rl_v1_dataset_manifest",
        "source": "EnvFactory-generated-guided-rl-v1",
        "released_envfactory_rl_used_as_formal_training_data": False,
        "frozen_300_manifest_sha256": FROZEN_SHA256,
        "frozen_300_train_overlap": 0,
        "generation_seed": args.seed,
        "total_tasks": len(kept),
        "train_tasks": len(train),
        "rl_val_tasks": len(val),
        "split_audit": split_audit,
        "unique_ratio": len({(query_hash(_last_user_query(r["prompt"])), tool_sequence_hash(tool_sequence(r["extra_info"]["graph_frontier"]))) for r in kept}) / max(1, len(kept)),
        "dependency_depth_distribution": dict(sorted(depths.items())),
        "internal_edge_count": internal_edges,
        "environment_distribution": dict(sorted(environments.items())),
        "frontier_sampling_policy": {"targeted": 0.65, "broad_exploration": 0.35},
        "train_sha256": file_sha256(output / "train.json"),
        "rl_val_sha256": file_sha256(output / "rl_val.json"),
        "contamination_audit_sha256": file_sha256(output / "contamination_audit.json"),
    }
    _write_json(output / "dataset_manifest.json", manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    graph = sub.add_parser("build-graph")
    graph.add_argument("--metadata-dir", default=str(ROOT / "envs/metadata"))
    graph.add_argument("--embedding-model", required=True)
    graph.add_argument("--embedding-device", default="cuda:1")
    graph.add_argument("--output", required=True)
    graph.add_argument("--no-llm-edges", action="store_true")
    graph.add_argument("--repair-batch-size", type=int, default=8)
    graph.add_argument("--repair-max-retry", type=int, default=5)
    repair = sub.add_parser("repair-graph")
    repair.add_argument("--graph", required=True)
    repair.add_argument("--batch-size", type=int, default=8)
    repair.add_argument("--max-retry", type=int, default=5)
    generation = sub.add_parser("generate")
    generation.add_argument("--graph", required=True)
    generation.add_argument("--output-dir", required=True)
    generation.add_argument("--count", type=int, default=12)
    generation.add_argument("--seed", type=int, default=20260917)
    generation.add_argument("--pass-k", type=int, default=2)
    generation.add_argument("--concurrency", type=int, default=2)
    generation.add_argument("--model-name", default="sglang")
    generation.add_argument("--max-failures", type=int, default=0)
    generation.add_argument("--min-valid-tasks", type=int, default=0)
    conversion = sub.add_parser("convert-audit")
    conversion.add_argument("--output-dir", required=True)
    conversion.add_argument("--audit-ledger")
    conversion.add_argument("--frozen-manifest", default=str(DEFAULT_FROZEN))
    conversion.add_argument("--seed", type=int, default=20260917)
    conversion.add_argument("--train-ratio", type=float, default=0.90)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-graph":
        build_graph(args)
    elif args.command == "repair-graph":
        repair_saved_graph(args)
    elif args.command == "generate":
        asyncio.run(generate(args))
    else:
        convert_and_audit(args)


if __name__ == "__main__":
    main()

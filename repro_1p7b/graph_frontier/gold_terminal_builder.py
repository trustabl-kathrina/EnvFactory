"""Build replay-locked gold-terminal CE examples without synthesizing a STOP action.

This is a CPU-only data pipeline.  It reads the frozen Rich-v3 source and
queries the repository's MCP tool definitions solely for missing tool schemas.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from repro_1p7b.graph_frontier.first_divergence_v1 import canonical
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import (
    MODEL, SOURCE, reference_for, source_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
ROLLOUTS = ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1"
TRACE_DIR = SOURCE / "replay/traces"
HELDOUT_PLAN = ROLLOUTS / "cycle2_heldout_plan.json"
# Pre-training manual audit: terminal text asserts facts absent from both observations.
MANUAL_REJECT = {
    "gf-rich-4e5587f0e4422d242c03": "empty route_stations but final invents timetable",
    "gf-rich-0a06d52ed385e6019e82": "stock symbols only but final invents prices",
}
BAD_TARGET = re.compile(
    r"<tool_call>|</tool_call>|<\|im_(?:start|end)\|>|"
    r"\{\{[^}]*\}\}|<\.\.\.>|\b(?:UNKNOWN|TODO|placeholder)\b|"
    r"unresolved\s+(?:id|identifier)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_servers(row: Mapping[str, Any]) -> tuple[str, ...]:
    factory = row["extra_info"]["mcp_factory_kwargs"]
    servers = factory["mcp_servers"]
    if isinstance(servers, str):
        servers = json.loads(servers)
    if not isinstance(servers, list) or not servers or any(not isinstance(s, str) for s in servers):
        raise ValueError("invalid server identity")
    return tuple(sorted(servers))


def typed_trace_matches(reference: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    events = trace.get("events")
    actions, responses = reference["actions"], reference["responses"]
    return (
        isinstance(events, list)
        and len(events) == len(actions) == len(responses)
        and all(
            event.get("execution_success") is True
            and event.get("tool_name") == action["name"]
            and canonical(event.get("tool_arguments")) == canonical(action["arguments"])
            and canonical(event.get("tool_response")) == canonical(response)
            for action, response, event in zip(actions, responses, events)
        )
    )


def validate_terminal_steps(steps: list[dict[str, Any]], reference: Mapping[str, Any]) -> tuple[dict, dict, str]:
    """Require one final assistant message immediately after the last response."""
    if len(steps) != 2 * len(reference["actions"]) + 2 or steps[0].get("role") != "user":
        raise ValueError("malformed terminal sequence")
    if steps[-1].get("role") != "assistant" or steps[-2].get("role") != "tool_response":
        raise ValueError("final answer is not after last observation")
    for index in range(len(reference["actions"])):
        if steps[1 + 2 * index].get("role") != "tool_call":
            raise ValueError("non-alternating tool call")
        if steps[2 + 2 * index].get("role") != "tool_response":
            raise ValueError("non-alternating tool observation")
    final = steps[-1].get("content")
    if not isinstance(final, str) or not final.strip() or final != reference["final_text"]:
        raise ValueError("missing or mismatched real final answer")
    call = steps[-2 * 1 - 1]["content"][0]
    response = steps[-2]["content"][0]
    expected = reference["actions"][-1]
    if call.get("name") != expected["name"] or canonical(call.get("arguments")) != canonical(expected["arguments"]):
        raise ValueError("last tool call mismatch")
    if canonical(response) != canonical(reference["responses"][-1]):
        raise ValueError("last observation mismatch")
    return call, response, final


def validate_schema(tools: Any, reference: Mapping[str, Any]) -> bool:
    if not isinstance(tools, list) or not tools:
        return False
    names = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return False
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("parameters"), dict):
            return False
        name = function.get("name")
        if not isinstance(name, str):
            return False
        names.append(name)
    return len(names) == len(set(names)) and all(action["name"] in names for action in reference["actions"])


def existing_schema(task_id: str, reference: Mapping[str, Any]) -> tuple[list[dict] | None, list[str]]:
    paths = sorted(ROLLOUTS.glob("**/" + task_id + ".r0.json"))
    if not paths:
        return None, []
    schemas = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("task_id") != task_id or not validate_schema(payload.get("tools"), reference):
            raise ValueError("same-task persisted schema invalid")
        schemas.append(payload["tools"])
    if len({canonical(s) for s in schemas}) != 1:
        raise ValueError("same-task persisted schemas disagree")
    return schemas[0], [str(p) for p in paths]


def recover_mcp_schemas(server_sets: set[tuple[str, ...]]) -> tuple[dict[tuple[str, ...], list[dict]], dict[str, str]]:
    """Read exact MCP definitions; do not infer parameters from observed calls."""
    if not server_sets:
        return {}, {}
    os.environ["ENVFACTORY_ROOT"] = str(ROOT)
    from src.manager.mcp_client_manager import MCPManager

    required = sorted({server for group in server_sets for server in group})
    hashes = {}
    try:
        for server in required:
            path = ROOT / "envs/tools" / (server + ".py")
            if not path.is_file():
                raise FileNotFoundError(path)
            future = asyncio.run_coroutine_threadsafe(
                MCPManager.register_mcp_server_async(server, str(path), False),
                MCPManager._loop,
            )
            future.result(timeout=120)
            hashes[server] = sha256(path)
        # Match EnvFactoryBatchEnv.reset's sorted server registration order.
        per_server = {server: MCPManager.filter_tools([server]) for server in required}
        return {group: [tool for server in group for tool in per_server[server]]
                for group in server_sets}, hashes
    finally:
        MCPManager.shutdown(timeout=30)


def has_tool_error(value: Any) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if isinstance(value, dict):
        return bool(value.get("error")) or any(has_tool_error(v) for v in value.values())
    if isinstance(value, list):
        return any(has_tool_error(v) for v in value)
    return False


def quality_gate(final: str, trace: Mapping[str, Any]) -> tuple[bool, list[str]]:
    flags = []
    if not final.strip():
        flags.append("empty_final")
    if BAD_TARGET.search(final):
        flags.append("placeholder_or_tool_serialization")
    if any(has_tool_error(e.get("tool_response")) for e in trace["events"]):
        flags.append("tool_error_observed")
    return not flags, flags


def terminal_messages(query: str, reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = [{"role": "user", "content": query}]
    for index, (action, response) in enumerate(zip(reference["actions"], reference["responses"])):
        call_id = "call_gold_" + str(index)
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": action["name"], "arguments": action["arguments"]},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id, "name": action["name"],
            "content": canonical(response),
        })
    return messages


def build_one(
    row: Mapping[str, Any], sidecar: Mapping[str, Any], ledger: Mapping[str, Any],
    raw: Mapping[str, Any], trace: Mapping[str, Any], reference: Mapping[str, Any],
    tools: list[dict[str, Any]] | None, schema_source: str, source_hashes: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if row["task_id"] != sidecar.get("task_id") or row["extra_info"].get("graph_frontier") != sidecar:
        return None, "sidecar_mismatch"
    if ledger.get("replay_ok") is not True or ledger.get("static_pass") is not True:
        return None, "replay_invalid"
    if ledger.get("final_state_match") is not True:
        return None, "final_state_mismatch"
    if trace.get("task_id") != row["task_id"] or not typed_trace_matches(reference, trace):
        return None, "raw_replay_drift"
    if tools is None or not validate_schema(tools, reference):
        return None, "schema_unrecoverable"
    try:
        steps = raw["nodes"][0]["steps"]
        final_call, final_observation, final = validate_terminal_steps(steps, reference)
        query = json.loads(row["prompt"])[-1]["content"]
        if not isinstance(query, str) or query.strip() != steps[0]["content"].strip():
            return None, "query_mismatch"
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None, "terminal_unreadable"
    good, flags = quality_gate(final, trace)
    if not good:
        return None, "quality_" + ",".join(flags)
    messages = terminal_messages(query, reference)
    return {
        "schema_version": "gold_terminal_sample_v2",
        "target_type": "terminal_final",
        "task_id": row["task_id"],
        "env_id": sidecar.get("environment_identifiers"),
        "env_seed": row["extra_info"]["generation_seed"],
        "gold_trajectory_id": reference["gold_trajectory_id"],
        "gold_depth": sidecar.get("dependency_depth"),
        "conversation_prefix": messages,
        "final_tool_call": final_call,
        "final_tool_observation": final_observation,
        "gold_final_response": final,
        "assistant_target": {"role": "assistant", "content": final},
        "tool_schema": tools,
        "schema_source": schema_source,
        "replay_valid": True,
        "final_state_match": True,
        "argument_match": True,
        "observation_match": True,
        "quality_status": "structurally_valid_unverified_semantics",
        "quality_flags": [],
        "loss_mask_metadata": {
            "prompt_tokens": "all_masked",
            "target_tokens": "assistant_final_only",
            "tool_observation_tokens_in_loss": 0,
            "historical_assistant_tokens_in_loss": 0,
        },
        "source_hashes": dict(source_hashes),
    }, "ok"


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen terminal dataset")
    manifest, rows, gold, ledger, raw_by_seed = source_bundle()
    heldout_plan = json.loads(HELDOUT_PLAN.read_text())
    heldout_ids = set(heldout_plan["task_ids"])
    if (len(heldout_ids) != heldout_plan["task_count"]
            or heldout_plan["source_manifest_sha256"] != sha256(SOURCE / "valid/manifest.json")
            or heldout_plan.get("frozen_exact_id_overlap") != 0):
        raise RuntimeError("frozen Rich heldout plan lock failed")
    greedy = json.loads((ROLLOUTS / "cycle3_heldout_greedy_plan.json").read_text())
    if set(greedy["task_ids"]) != heldout_ids:
        raise RuntimeError("Rich heldout protocol task IDs disagree")
    candidates = []
    counts = collections.Counter()
    for task_id, row in rows.items():
        if row["extra_info"].get("graph_frontier") != gold[task_id]:
            continue
        reference, reason = reference_for(row, ledger[task_id], raw_by_seed)
        if reference is None:
            continue
        counts["strict_reference"] += 1
        trace_path = TRACE_DIR / (task_id + ".rollout.json")
        raw_path = Path(reference["raw_file"])
        trace = json.loads(trace_path.read_text())
        if ledger[task_id].get("final_state_match") is not True:
            counts["final_state_mismatch"] += 1
            continue
        if not typed_trace_matches(reference, trace):
            counts["raw_replay_drift"] += 1
            continue
        counts["strict_candidates"] += 1
        if task_id in heldout_ids:
            counts["heldout_candidate_excluded"] += 1
            continue
        counts["strict_train_candidates"] += 1
        if task_id in MANUAL_REJECT:
            counts["manual_quality_excluded"] += 1
            continue
        counts["schema_eligible_candidates"] += 1
        try:
            persisted, paths = existing_schema(task_id, reference)
        except ValueError:
            persisted, paths = None, []
            counts["persisted_schema_conflict"] += 1
        group = source_servers(row)
        candidates.append((task_id, row, gold[task_id], ledger[task_id], reference,
                           json.loads(raw_path.read_text()), trace, raw_path, trace_path,
                           persisted, paths, group))
        if persisted is not None:
            counts["schema_available_original"] += 1
    groups = {c[-1] for c in candidates}
    recovered, tool_definition_hashes = recover_mcp_schemas(groups)
    samples = []
    audited = []
    for (task_id, row, sidecar, cert, reference, raw, trace, raw_path, trace_path,
         persisted, paths, group) in candidates:
        runtime = recovered.get(group)
        if persisted is not None:
            if canonical(persisted) != canonical(runtime):
                counts["schema_source_drift"] += 1
                audited.append({"task_id": task_id, "status": "SKIP", "reason": "schema_source_drift"})
                continue
            tools = persisted
            schema_source = "same_task_persisted_rollout_and_matching_mcp_definition"
        else:
            tools = runtime
            schema_source = "deterministic_mcp_tool_definition"
            if validate_schema(tools, reference):
                counts["schema_recovered_deterministically"] += 1
        hashes = {
            "valid_manifest_sha256": sha256(SOURCE / "valid/manifest.json"),
            "valid_data_sha256": manifest["artifacts"]["valid_data_sha256"],
            "valid_gold_sha256": manifest["artifacts"]["valid_gold_sha256"],
            "final_ledger_sha256": sha256(SOURCE / "audit/final_ledger.jsonl"),
            "raw_sha256": sha256(raw_path),
            "replay_trace_sha256": sha256(trace_path),
            "schema_sha256": digest(tools),
            "mcp_tool_definition_sha256": {s: tool_definition_hashes[s] for s in group},
            "persisted_rollout_sha256": {p: sha256(Path(p)) for p in paths},
        }
        sample, reason = build_one(row, sidecar, cert, raw, trace, reference, tools, schema_source, hashes)
        audited.append({"task_id": task_id, "status": "READY" if sample else "SKIP",
                        "reason": reason, "depth": sidecar.get("dependency_depth"),
                        "schema_source": schema_source})
        counts[reason] += 1
        if sample:
            samples.append(sample)
    counts["schema_unrecoverable"] = counts["schema_eligible_candidates"] - counts["schema_available_original"] - counts["schema_recovered_deterministically"]
    counts["final_terminal_usable"] = len(samples)
    if len(samples) != len({s["task_id"] for s in samples}):
        raise RuntimeError("duplicate terminal task ID")
    output.mkdir(parents=True)
    data_path = output / "dataset.jsonl"
    audit_path = output / "audit.jsonl"
    data_path.write_text("".join(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n" for s in samples))
    audit_path.write_text("".join(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n" for s in audited))
    result = {
        "schema_version": "gold_terminal_dataset_v2",
        "verdict": "DATA_READY" if len(samples) >= 48 else "DATA_NOT_READY",
        "counts": dict(sorted(counts.items())),
        "task_ids": [s["task_id"] for s in samples],
        "dataset_sha256": sha256(data_path),
        "audit_sha256": sha256(audit_path),
        "source_valid_manifest_sha256": sha256(SOURCE / "valid/manifest.json"),
        "heldout_plan_sha256": sha256(HELDOUT_PLAN),
        "heldout_task_overlap": len({s["task_id"] for s in samples} & heldout_ids),
        "manual_quality_exclusions": MANUAL_REJECT,
        "frozen300_exact_id_overlap": 0,
        "quality_claim": "structural only; answer semantics unverified",
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

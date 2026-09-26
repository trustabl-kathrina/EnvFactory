"""CPU-only static audit of the official EnvFactory-RL release.

Does not start an inference server, train a model, or execute task tools.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repro_1p7b/data/envfactory_rl_official/env_factory_rl.json"
OUT = ROOT / "repro_1p7b/graph_frontier/official_rl_audit"
REGISTRY = ROOT / "configs/mcp_server.json"
OFFICIAL_SHA = "0283caf6487f8790df972f192a94cdec7d17b1ccb94aceda59d21efd562105c4"
REVISION = "963ec0607c016b7fafbd81570987daaf8d571153"
FROZEN = ROOT / "repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
RICH24 = ROOT / "repro_1p7b/logs/rich_v3_trajectory_differential_audit/per_task_diff.jsonl"
RICH_TRAIN = ROOT / "repro_1p7b/graph_frontier/preference_generation_v3/pilot_balanced_24chunks_v3/valid/graph_frontier_rich_train_v1.jsonl"
GENERATED_TRAIN = ROOT / "repro_1p7b/data/graph_frontier_rl_v1_frozen_seed20260920/train.json"
SFT = ROOT / "repro_1p7b/datasets/EnvFactory-SFT-FILTERED/mcp_factory_sft_nips.json"


def decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(stable(value).encode()).hexdigest()


def norm_query(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def quantiles(values: list[int | float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def pct(p: float) -> float:
        pos = (len(ordered) - 1) * p
        low = int(pos)
        return round(ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (pos - low), 4)
    return {
        "count": len(values), "mean": round(statistics.mean(values), 4),
        "median": pct(.5), "p25": pct(.25), "p75": pct(.75),
        "p90": pct(.9), "p95": pct(.95), "max": max(values),
    }


def query_from_row(row: dict) -> str:
    for key in ("query", "instruction", "user_query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        graph = extra.get("graph_frontier")
        if isinstance(graph, dict) and isinstance(graph.get("query"), str):
            return graph["query"]
    prompt = row.get("prompt")
    try:
        prompt = decode(prompt)
    except (ValueError, TypeError):
        pass
    if isinstance(prompt, list):
        users = [m.get("content") for m in prompt
                 if isinstance(m, dict) and m.get("role") == "user"]
        return str(users[-1]) if users else ""
    if isinstance(prompt, str):
        return prompt
    return ""


def query_hash(value: Any) -> str | None:
    normalized = norm_query(value)
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def overlap_index(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "available": False, "query_hashes": set(),
                "task_ids": set(), "initial_hashes": set(), "gold_signatures": set()}
    rows = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
    queries, ids, initials, golds = set(), set(), set(), set()
    for row in rows:
        q = query_hash(query_from_row(row))
        if q:
            queries.add(q)
        if row.get("task_id"):
            ids.add(str(row["task_id"]))
        factory = (row.get("extra_info") or {}).get("mcp_factory_kwargs") or {}
        if factory.get("initial_config"):
            try:
                initials.add(digest(decode(factory["initial_config"])))
            except (ValueError, TypeError):
                pass
        if path == FROZEN and row.get("initial_state_path"):
            initials.add(digest(json.loads((ROOT / row["initial_state_path"]).read_text())))
        raw_gold = (row.get("reward_model") or {}).get("ground_truth")
        if raw_gold:
            try:
                golds.add(digest(decode(raw_gold)))
            except (ValueError, TypeError):
                pass
        elif path == RICH24 and row.get("gold_actions"):
            golds.add(digest(row["gold_actions"]))
        elif path == FROZEN and row.get("reference_trace_path"):
            golds.add(digest(json.loads((ROOT / row["reference_trace_path"]).read_text())))
    return {"path": str(path), "available": True, "rows": len(rows),
            "query_hashes": queries, "task_ids": ids,
            "initial_hashes": initials, "gold_signatures": golds}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite audit output: {OUT}")
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_sha != OFFICIAL_SHA:
        raise RuntimeError(f"local dataset differs from official release: {source_sha}")
    rows = json.loads(SOURCE.read_text())
    registry = json.loads(REGISTRY.read_text())["mcpServers"]
    audited: list[dict] = []
    all_calls: list[dict] = []
    schema_counts = collections.Counter()
    error_counts = collections.Counter()
    environments = collections.Counter()
    server_frequency = collections.Counter()
    argument_types = collections.Counter()
    masked_argument_slots = 0
    for index, row in enumerate(rows):
        schema_counts[stable(sorted(row))] += 1
        factory = (row.get("extra_info") or {}).get("mcp_factory_kwargs") or {}
        errors = []
        try:
            prompt = decode(row.get("prompt"))
            query = query_from_row(row)
            if not isinstance(prompt, list) or not query:
                errors.append("prompt_invalid")
        except (ValueError, TypeError):
            query = ""
            errors.append("prompt_invalid")
        try:
            gold = decode((row.get("reward_model") or {}).get("ground_truth"))
            if not isinstance(gold, list) or not gold:
                raise ValueError("gold_not_nonempty_list")
        except (ValueError, TypeError):
            gold = []
            errors.append("ground_truth_malformed")
        try:
            servers = decode(factory.get("mcp_servers"))
            initial = decode(factory.get("initial_config"))
            final = decode(factory.get("final_config"))
            if not isinstance(servers, list) or not all(isinstance(s, str) for s in servers):
                raise ValueError("servers_invalid")
            if not isinstance(initial, dict) or not isinstance(final, dict):
                raise ValueError("scenario_invalid")
        except (ValueError, TypeError):
            servers, initial, final = [], {}, {}
            errors.append("config_malformed")
        env = ",".join(sorted(servers)) or "unknown"
        environments[env] += 1
        server_frequency.update(servers)
        missing_server = [s for s in servers if s not in registry or not (ROOT / registry[s]["tool_path"]).is_file()]
        if missing_server:
            errors.append("server_mapping_missing")
        if any(s not in initial for s in servers):
            errors.append("initial_server_missing")
        if any(s not in final for s in servers):
            errors.append("final_server_missing")
        call_errors = []
        normalized_calls = []
        per_call_args = []
        multi_args = 0
        for call_index, call in enumerate(gold):
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                call_errors.append("gold_call_malformed")
                continue
            name, args = call["name"], call["arguments"]
            server = name.split("-", 1)[0]
            if server not in servers or "-" not in name:
                call_errors.append("gold_call_server_mismatch")
            masked = call.get("masked_arguments") or []
            if not isinstance(masked, list):
                call_errors.append("masked_arguments_malformed")
                masked = []
            masked_argument_slots += len(masked)
            arg_types = {k: type(v).__name__ for k, v in args.items()}
            argument_types.update(arg_types.values())
            per_call_args.append(len(args))
            multi_args += len(args) >= 2
            normalized_calls.append({"name": name, "arguments": args, "masked_arguments": masked})
            all_calls.append({"row_index": index, "call_index": call_index, "name": name,
                              "server": server, "arguments": args, "masked_arguments": masked})
        errors.extend(sorted(set(call_errors)))
        error_counts.update(set(errors))
        qhash = query_hash(query)
        row_id = f"official-rl-{index:04d}-{digest([qhash, servers, initial, normalized_calls])[:12]}"
        audited.append({
            "row_index": index, "audit_id": row_id, "query": query,
            "query_hash": qhash, "query_chars": len(query),
            "environment": env, "servers": servers,
            "initial_hash": digest(initial), "final_hash": digest(final),
            "gold_signature": digest(normalized_calls),
            "gold_tool_name_signature": digest([c["name"] for c in normalized_calls]),
            "gold_calls": normalized_calls,
            "gold_length": len(gold), "gold_args": sum(per_call_args),
            "mean_args_per_tool": statistics.mean(per_call_args) if per_call_args else 0,
            "max_args_per_tool": max(per_call_args, default=0),
            "multi_arg_calls": multi_args, "has_multi_arg": bool(multi_args),
            "masked_slots": sum(len(c["masked_arguments"]) for c in normalized_calls),
            "initial_config": initial, "final_config": final,
            "static_server_mapping_ok": not missing_server,
            "static_config_ok": "config_malformed" not in errors and "initial_server_missing" not in errors,
            "static_gold_ok": not call_errors and bool(normalized_calls),
            "errors": errors,
        })
    if len(audited) != 3092:
        raise RuntimeError("release row count changed")

    def scalars(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set().union(*(scalars(v) for v in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(scalars(v) for v in value)) if value else set()
        if isinstance(value, str) and len(value) >= 4:
            return {stable(value)}
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 10:
            return {stable(value)}
        return set()

    for item in audited:
        prior = scalars(item["initial_config"])
        candidate = 0
        for call_index, call in enumerate(item["gold_calls"]):
            if call_index:
                for value in call["arguments"].values():
                    for encoded in scalars(value):
                        if encoded in prior and norm_query(json.loads(encoded)) not in norm_query(item["query"]):
                            candidate += 1
            prior |= scalars(call["arguments"])
        item["candidate_value_reuse"] = candidate
        item["candidate_internal_dependency"] = candidate > 0 and item["gold_length"] > 1

    references = {
        "Frozen300": overlap_index(FROZEN),
        "Rich24": overlap_index(RICH24),
        "RichTrain24chunks": overlap_index(RICH_TRAIN),
        "Generated1000": overlap_index(GENERATED_TRAIN),
    }
    sft_queries = set()
    if SFT.exists():
        for row in json.loads(SFT.read_text()):
            q = query_hash(query_from_row(row))
            if q:
                sft_queries.add(q)
    references["SFT_FILTERED"] = {
        "path": str(SFT), "available": SFT.exists(), "rows": 26463 if SFT.exists() else 0,
        "query_hashes": sft_queries, "task_ids": set(),
        "initial_hashes": set(), "gold_signatures": set(),
    }
    overlap = {}
    for label, ref in references.items():
        overlap[label] = {
            "source_path": ref["path"], "reference_available": ref["available"],
            "reference_rows": ref.get("rows", 0),
            "query_hash_rows": [a["row_index"] for a in audited if a["query_hash"] in ref["query_hashes"]],
            "task_id_rows": [],  # Official RL release has no task_id field.
            "initial_config_hash_rows": [a["row_index"] for a in audited if a["initial_hash"] in ref["initial_hashes"]],
            "gold_signature_rows": [a["row_index"] for a in audited if a["gold_signature"] in ref["gold_signatures"]],
            "clean_exclusion_rows": [a["row_index"] for a in audited if
                                     a["query_hash"] in ref["query_hashes"] or
                                     (a["initial_hash"] in ref["initial_hashes"] and
                                      a["gold_signature"] in ref["gold_signatures"])],
        }
        for key in ("query_hash_rows", "initial_config_hash_rows",
                    "gold_signature_rows", "clean_exclusion_rows"):
            overlap[label][key + "_count"] = len(overlap[label][key])

    clean_exclusions = set().union(*(set(ref["clean_exclusion_rows"]) for ref in overlap.values()))
    query_counts = collections.Counter(a["query_hash"] for a in audited)
    initial_counts = collections.Counter((a["environment"], a["initial_hash"]) for a in audited)
    gold_counts = collections.Counter(a["gold_signature"] for a in audited)
    tool_name_counts = collections.Counter(a["gold_tool_name_signature"] for a in audited)
    lengths = [a["gold_length"] for a in audited]
    arg_slots = [a["gold_args"] for a in audited]
    call_arg_counts = [len(c["arguments"]) for c in all_calls]
    query_lengths = [a["query_chars"] for a in audited]
    summary = {
        "dataset": {
            "name": "LARK-Lab/EnvFactory-RL", "split": "train",
            "revision": REVISION, "file_upload_commit": "80e0c58",
            "source_path": str(SOURCE), "sha256": source_sha, "rows": len(audited),
            "source_size_bytes": SOURCE.stat().st_size,
            "official_page": "https://huggingface.co/datasets/LARK-Lab/EnvFactory-RL",
        },
        "schema": {
            "top_level_keys": dict(schema_counts),
            "reward_model_keys": sorted(rows[0]["reward_model"]),
            "extra_info_keys": sorted(rows[0]["extra_info"]),
            "factory_keys": sorted(rows[0]["extra_info"]["mcp_factory_kwargs"]),
            "explicit_task_id_rows": sum("task_id" in r for r in rows),
            "explicit_seed_rows": sum("seed" in r for r in rows),
            "explicit_trajectory_rows": sum("trajectory" in r for r in rows),
            "explicit_history_rows": sum("history" in r for r in rows),
            "explicit_tool_observation_rows": sum("tool_responses" in r for r in rows),
            "explicit_final_response_rows": sum(any(k in r for k in ("final_response", "response", "answer")) for r in rows),
            "parseable_gold_sequence_rows": sum(bool(a["gold_calls"]) and
                                                "gold_call_malformed" not in a["errors"] for a in audited),
            "gold_server_consistent_rows": sum(a["static_gold_ok"] for a in audited),
            "exact_argument_structure_rows": sum(bool(a["gold_calls"]) and
                                                 all(isinstance(c["arguments"], dict) for c in a["gold_calls"])
                                                 for a in audited),
            "masked_argument_slots": masked_argument_slots,
            "initial_config_available_rows": sum(bool(a["initial_config"]) for a in audited),
            "final_config_available_rows": sum(bool(a["final_config"]) for a in audited),
            "static_server_mapping_rows": sum(a["static_server_mapping_ok"] for a in audited),
            "static_config_rows": sum(a["static_config_ok"] for a in audited),
            "error_counts": dict(error_counts),
        },
        "coverage": {
            "unique_environment_combinations": len(environments),
            "unique_mcp_servers": len(server_frequency),
            "environment_frequency": dict(environments.most_common()),
            "server_frequency": dict(server_frequency.most_common()),
            "registered_servers": len(registry),
        },
        "richness": {
            "gold_calls": quantiles(lengths), "gold_call_buckets": {
                "1": sum(n == 1 for n in lengths),
                "2": sum(n == 2 for n in lengths),
                "3": sum(n == 3 for n in lengths),
                "4+": sum(n >= 4 for n in lengths),
                "6+": sum(n >= 6 for n in lengths)},
            "query_chars": quantiles(query_lengths),
            "args_per_task": quantiles(arg_slots),
            "args_per_call": quantiles(call_arg_counts),
            "argument_value_types": dict(argument_types),
            "calls_with_2plus_args": sum(n >= 2 for n in call_arg_counts),
            "calls_with_3plus_args": sum(n >= 3 for n in call_arg_counts),
            "calls_with_4plus_args": sum(n >= 4 for n in call_arg_counts),
            "tasks_with_multi_arg_call": sum(a["has_multi_arg"] for a in audited),
            "candidate_only_value_reuse_tasks": sum(a["candidate_internal_dependency"] for a in audited),
            "candidate_only_value_reuse_slots": sum(a["candidate_value_reuse"] for a in audited),
            "verified_internal_dependency_tasks": "unknown_until_replay",
            "dependency_depth": "unknown_without_graph_or_verified_replay",
        },
        "duplicates": {
            "unique_query_hashes": len(query_counts),
            "query_duplicate_rows": sum(n - 1 for n in query_counts.values()),
            "unique_environment_initial_hashes": len(initial_counts),
            "environment_initial_duplicate_rows": sum(n - 1 for n in initial_counts.values()),
            "unique_exact_gold_paths": len(gold_counts),
            "exact_gold_path_duplicate_rows": sum(n - 1 for n in gold_counts.values()),
            "unique_tool_name_sequences": len(tool_name_counts),
            "tool_name_sequence_duplicate_rows": sum(n - 1 for n in tool_name_counts.values()),
            "official_task_id": "unavailable",
        },
        "contamination": {k: {j: v for j, v in ref.items() if not j.endswith("_rows")}
                          for k, ref in overlap.items()},
        "clean_exclusion_count": len(clean_exclusions),
        "clean_candidate_count_before_replay": len(audited) - len(clean_exclusions),
    }

    OUT.mkdir(parents=True)
    parquet_rows = []
    for a in audited:
        parquet_rows.append({
            k: a[k] for k in (
                "row_index", "audit_id", "query_hash", "query_chars",
                "environment", "initial_hash", "final_hash",
                "gold_signature", "gold_tool_name_signature",
                "gold_length", "gold_args", "mean_args_per_tool",
                "max_args_per_tool", "multi_arg_calls", "has_multi_arg",
                "masked_slots", "candidate_value_reuse",
                "candidate_internal_dependency", "static_server_mapping_ok",
                "static_config_ok", "static_gold_ok")
        } | {"servers_json": stable(a["servers"]), "errors_json": stable(a["errors"]),
             "eval_overlap": a["row_index"] in clean_exclusions})
    pq.write_table(pa.Table.from_pylist(parquet_rows), OUT / "row_audit.parquet")
    (OUT / "schema_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "contamination_audit.json").write_text(json.dumps(overlap, indent=2, sort_keys=True) + "\n")
    sample = []
    seen_env = set()
    for a in audited:
        if a["environment"] not in seen_env:
            sample.append(a)
            seen_env.add(a["environment"])
            if len(sample) >= 50:
                break
    if len(sample) < 50:
        sample.extend(a for a in audited if a not in sample)
    sample = sample[:50]
    lines = [
        "# Official EnvFactory-RL schema examples",
        "",
        "50 rows from distinct environment combinations where possible; no raw state, secrets, or long query text.",
        "",
        "| Row | Environment | Query chars | Gold calls | Args/task | Masked slots | Servers |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for a in sample:
        lines.append(f'| {a["row_index"]} | {a["environment"]} | {a["query_chars"]} | '
                     f'{a["gold_length"]} | {a["gold_args"]} | {a["masked_slots"]} | '
                     f'{",".join(a["servers"])} |')
    lines += ["", "The top-level release contains no independent task ID, seed, tool observations, "
              "trajectory/history, or assistant final response. ground_truth is a JSON-encoded "
              "ordered list of tool calls with names, arguments and masked_arguments. "
              "Masked fields are present in JSON but excluded from official matching; "
              "their value fidelity is not independently proven.", ""]
    (OUT / "official_rl_schema_examples.md").write_text("\n".join(lines))
    print(json.dumps({"rows": len(audited), "servers": len(server_frequency),
                      "parseable_gold": summary["schema"]["parseable_gold_sequence_rows"],
                      "clean_before_replay": summary["clean_candidate_count_before_replay"],
                      "mean_gold_calls": summary["richness"]["gold_calls"]["mean"],
                      "mean_args_per_task": summary["richness"]["args_per_task"]["mean"]},
                     sort_keys=True))

if __name__ == "__main__":
    main()


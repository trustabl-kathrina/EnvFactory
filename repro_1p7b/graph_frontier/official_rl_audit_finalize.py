"""Finalize the CPU-only official EnvFactory-RL OPD source audit."""
from __future__ import annotations

import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from repro_1p7b.graph_frontier.official_rl_audit_static import (
    OUT, ROOT, SOURCE, FROZEN, RICH24, RICH_TRAIN, GENERATED_TRAIN, SFT,
    decode, digest, norm_query, query_from_row, query_hash, read_jsonl, stable,
)

PREFERENCE = ROOT / ("repro_1p7b/graph_frontier/preference_generation_v3/"
                     "balanced_effect_pilot_192_v1/graph_pairs_train64.jsonl")
RICH_DISTRIBUTION = ROOT / "repro_1p7b/logs/rich_v3_trajectory_differential_audit/summary.json"
REPORT = ROOT / "repro_1p7b/graph_frontier/reports/official_envfactory_rl_opd_audit.md"
REPORT_JSON = REPORT.with_suffix(".json")


def normalized_calls(value: Any) -> list[dict] | None:
    try:
        calls = decode(value)
    except (ValueError, TypeError):
        return None
    if not isinstance(calls, list) or not calls:
        return None
    result = []
    for call in calls:
        if not isinstance(call, dict):
            return None
        name = call.get("name") or call.get("tool_name")
        args = call.get("arguments")
        if args is None:
            args = call.get("tool_arguments")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        result.append({"name": name, "arguments": args})
    return result


def signature(value: Any) -> str | None:
    calls = normalized_calls(value)
    return digest(calls) if calls else None


def server_names(row: dict) -> list[str]:
    factory = (row.get("extra_info") or {}).get("mcp_factory_kwargs") or {}
    value = factory.get("mcp_servers")
    if value is not None:
        try:
            result = decode(value)
            return result if isinstance(result, list) else []
        except (ValueError, TypeError):
            pass
    env = row.get("environment") or row.get("environment_identifiers")
    if isinstance(env, list):
        return env
    if isinstance(env, str):
        return env.split(",")
    return []


def reference_index(path: Path) -> dict:
    if not path.exists():
        return {"available": False, "rows": 0, "query": set(), "task_id": set(),
                "env_initial": set(), "gold": set()}
    rows = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
    result = {"available": True, "rows": len(rows), "query": set(), "task_id": set(),
              "env_initial": set(), "gold": set()}
    for row in rows:
        q = query_hash(query_from_row(row))
        if q:
            result["query"].add(q)
        if row.get("task_id"):
            result["task_id"].add(str(row["task_id"]))
        servers = server_names(row)
        factory = (row.get("extra_info") or {}).get("mcp_factory_kwargs") or {}
        initial = None
        if factory.get("initial_config") is not None:
            try:
                initial = decode(factory["initial_config"])
            except (ValueError, TypeError):
                pass
        if path == FROZEN and row.get("initial_state_path"):
            initial = json.loads((ROOT / row["initial_state_path"]).read_text())
        if isinstance(initial, dict) and servers:
            key = ",".join(sorted(servers))
            result["env_initial"].add((key, digest(initial)))
            if all(s in initial for s in servers):
                result["env_initial"].add((key, digest({s: initial[s] for s in servers})))
        gold = (row.get("reward_model") or {}).get("ground_truth")
        if path == FROZEN and row.get("reference_trace_path"):
            gold = json.loads((ROOT / row["reference_trace_path"]).read_text())
        if path == RICH24:
            gold = row.get("gold_actions")
        sig = signature(gold)
        if sig:
            result["gold"].add(sig)
    return result


def contamination(raw: list[dict]) -> tuple[dict, set[int]]:
    paths = {"Frozen300": FROZEN, "Rich24": RICH24,
             "RichTrain24chunks": RICH_TRAIN, "Generated1000": GENERATED_TRAIN,
             "PreferencePairs": PREFERENCE}
    indices = {label: reference_index(path) for label, path in paths.items()}
    old = json.loads((OUT / "contamination_audit.json").read_text())
    sft_query_rows = set(old["SFT_FILTERED"]["query_hash_rows"])
    output, exclusions = {}, set(sft_query_rows)
    for label, index in indices.items():
        counts = collections.Counter()
        rows_by_kind = {"query": [], "env_initial": [], "gold": [], "clean_exclusion": []}
        for row_index, row in enumerate(raw):
            query = query_hash(query_from_row(row))
            servers = server_names(row)
            key = ",".join(sorted(servers))
            initial = decode(row["extra_info"]["mcp_factory_kwargs"]["initial_config"])
            gold = signature(row["reward_model"]["ground_truth"])
            qhit = query is not None and query in index["query"]
            ihit = (key, digest(initial)) in index["env_initial"]
            if all(s in initial for s in servers):
                ihit = ihit or (key, digest({s: initial[s] for s in servers})) in index["env_initial"]
            ghit = gold is not None and gold in index["gold"]
            if qhit:
                rows_by_kind["query"].append(row_index)
            if ihit:
                rows_by_kind["env_initial"].append(row_index)
            if ghit:
                rows_by_kind["gold"].append(row_index)
            if qhit or (ihit and ghit):
                rows_by_kind["clean_exclusion"].append(row_index)
                exclusions.add(row_index)
        output[label] = {
            "reference_path": str(paths[label]), "reference_available": index["available"],
            "reference_rows": index["rows"],
            "reference_query_hash_coverage": len(index["query"]),
            "reference_initial_hash_coverage": len(index["env_initial"]),
            "reference_gold_signature_coverage": len(index["gold"]),
            **{name + "_rows": values for name, values in rows_by_kind.items()},
            **{name + "_count": len(values) for name, values in rows_by_kind.items()},
            "task_id_match": "unavailable_official_release_has_no_task_id",
        }
    output["SFT_FILTERED"] = {
        "reference_path": str(SFT), "reference_available": SFT.exists(),
        "query_rows": sorted(sft_query_rows), "query_count": len(sft_query_rows),
        "clean_exclusion_rows": sorted(sft_query_rows),
        "clean_exclusion_count": len(sft_query_rows),
        "terminal_match_status": "query_only_not_stable_composite_key",
    }
    return output, exclusions


def classify_rows(static: list[dict], replay: dict[int, dict],
                  exclusions: set[int]) -> tuple[list[dict], dict]:
    classified = []
    counts = collections.Counter()
    for item in static:
        index = item["row_index"]
        result = replay.get(index)
        structural = item["gold_length"] > 0 and "gold_call_malformed" not in json.loads(item["errors_json"])
        if not structural:
            tier = "D"
        elif result and result["replay_ok"]:
            tier = "B"  # No explicit or deterministically recoverable terminal answer.
        else:
            tier = "C"
        counts[tier] += 1
        flow_edges = result["verified_observed_value_flow_count"] if result else 0
        strict_parameter_rich = (tier == "B" and item["gold_length"] >= 2
                                 and item["has_multi_arg"] and flow_edges >= 1)
        loose_parameter_rich = (tier == "B" and item["gold_length"] >= 2
                                and item["has_multi_arg"] and item["candidate_internal_dependency"])
        depth = "unknown"
        if tier == "B":
            if item["gold_length"] == 1:
                depth = 0
            elif flow_edges:
                depth = result["verified_observed_dependency_depth"]
        classified.append({
            "row_index": index, "audit_id": item["audit_id"],
            "environment": item["environment"], "query_hash": item["query_hash"],
            "gold_length": item["gold_length"], "gold_args": item["gold_args"],
            "has_multi_arg": item["has_multi_arg"],
            "tier": tier, "eval_or_train_overlap": index in exclusions,
            "clean_opd_ready": tier == "B" and index not in exclusions,
            "strict_parameter_rich": strict_parameter_rich,
            "loose_candidate_parameter_rich": loose_parameter_rich,
            "verified_internal_dependency_edges": flow_edges,
            "verified_dependency_depth": str(depth),
            "replay_recorded": result is not None,
            "replay_ok": result["replay_ok"] if result else False,
            "replay_error": result["error_detail"] if result else "not_replayed",
        })
    return classified, dict(counts)

def summarize(raw: list[dict], static: list[dict], replay: dict[int, dict],
              classified: list[dict], tier_counts: dict, overlap: dict) -> dict:
    schema = json.loads((OUT / "schema_summary.json").read_text())
    replay_summary = json.loads((OUT / "replay_summary.json").read_text())
    rich_dist = json.loads(RICH_DISTRIBUTION.read_text())["distribution"]
    clean = [r for r in classified if r["clean_opd_ready"]]
    strict = [r for r in clean if r["strict_parameter_rich"]]
    loose = [r for r in clean if r["loose_candidate_parameter_rich"]]
    depths = collections.Counter(r["verified_dependency_depth"] for r in clean)
    env_rows: dict[str, list[dict]] = collections.defaultdict(list)
    for row in classified:
        env_rows[row["environment"]].append(row)
    env_statistics = {}
    for environment, rows in env_rows.items():
        env_statistics[environment] = {
            "tasks": len(rows),
            "mean_chain_length": round(statistics.mean(r["gold_length"] for r in rows), 4),
            "mean_args_per_task": round(statistics.mean(r["gold_args"] for r in rows), 4),
            "candidate_dependency_rate": round(
                sum(static[r["row_index"]]["candidate_internal_dependency"] for r in rows) / len(rows), 4),
            "verified_dependency_rate": round(
                sum(r["verified_internal_dependency_edges"] > 0 for r in rows) / len(rows), 4),
            "opd_ready_rate": round(sum(r["clean_opd_ready"] for r in rows) / len(rows), 4),
            "strict_parameter_rich_rate": round(sum(r["strict_parameter_rich"] for r in rows) / len(rows), 4),
        }
    official_servers = set(schema["coverage"]["server_frequency"])
    overlaps_by_pool = {}
    for name in ("Rich24", "Frozen300"):
        pool_servers = {
            server for combination in rich_dist[name]["environment_counts"]
            for server in combination.split(",")
        }
        overlaps_by_pool[name] = {
            "pool_server_families": len(pool_servers),
            "shared_server_families": len(pool_servers & official_servers),
            "missing_server_families": sorted(pool_servers - official_servers),
        }
    report = {
        "verdict": "OFFICIAL-RL-PARTIAL",
        "dataset": schema["dataset"],
        "schema": schema["schema"],
        "coverage": schema["coverage"],
        "richness": schema["richness"],
        "duplicates": schema["duplicates"],
        "replay": replay_summary,
        "tier_counts": {name: tier_counts.get(name, 0) for name in "ABCD"},
        "clean_tier_ab": len(clean),
        "terminal_response_classification": {
            "A_explicit": 0, "B_deterministically_reconstructable": 0,
            "C_stable_sft_join": 0, "D_unavailable": len(raw),
            "sft_query_only_overlap": overlap["SFT_FILTERED"]["query_count"],
            "reason": "RL release has no final assistant response or typed gold observation; "
                      "a query-only SFT match cannot certify the same task/configuration.",
        },
        "parameter_rich_clean_strict": len(strict),
        "parameter_rich_clean_candidate_only": len(loose),
        "verified_dependency_depth_clean": dict(depths),
        "verified_flow_edges_all_replay": sum(
            r["verified_observed_value_flow_count"] for r in replay.values()),
        "verified_flow_tasks_all_replay": sum(
            r["verified_observed_value_flow_count"] > 0 for r in replay.values()),
        "contamination": overlap,
        "environment_statistics": env_statistics,
        "distribution_comparison": {
            "OfficialRL": {
                "query_chars_mean": schema["richness"]["query_chars"]["mean"],
                "gold_calls_mean": schema["richness"]["gold_calls"]["mean"],
                "args_per_task_mean": schema["richness"]["args_per_task"]["mean"],
                "depth": "mostly_unknown_without_gold_graph",
                "environment_combinations": schema["coverage"]["unique_environment_combinations"],
                "mcp_servers": schema["coverage"]["unique_mcp_servers"],
            },
            **{
                name: {
                    "query_chars_mean": rich_dist[name]["mean_query_length_characters"],
                    "gold_calls_mean": rich_dist[name]["mean_gold_actions"],
                    "args_per_task_mean": rich_dist[name]["mean_argument_count"],
                    "depth_counts": rich_dist[name]["depth_counts"],
                    **overlaps_by_pool[name],
                }
                for name in ("Rich24", "Frozen300")
            },
        },
        "next_rollout_cost": {
            "clean_student_episodes_at_one_rollout_each": len(clean),
            "relative_to_14b_generation": "reuses existing tasks; no new 14B task synthesis, "
                                           "but still needs one Dynamic-v1 episode per clean task",
            "gpu_hours": "unknown_no_student_runtime_baseline",
            "gold_cpu_replay_observed_seconds": replay_summary["elapsed_seconds"],
        },
        "limitations": [
            "No explicit final assistant response; query-only SFT overlap is not a stable join.",
            "Gold calls mark some argument keys as non-essential (masked_arguments); "
            "presence does not prove precision for every parameter.",
            "Unique typed-value flow is a conservative observation-backed proxy, not a full ToolGraph.",
            "Missing verified flow in multi-tool tasks means unknown depth, not depth zero.",
            "Strict replay requires exact selected-server final state; nonmatching official rows are not FD-ready.",
            "No student on-policy rollout or training was run.",
        ],
    }
    return report


def render_report(report: dict) -> str:
    s = report["schema"]
    rich = report["richness"]
    replay = report["replay"]
    tiers = report["tier_counts"]
    overlap = report["contamination"]
    compare = report["distribution_comparison"]
    lines = [
        "# 官方 EnvFactory-RL 用作下一轮 OPD source 的审计",
        "",
        "**VERDICT = OFFICIAL-RL-PARTIAL。** 全量静态与 executable gold replay 已完成；"
        "该数据可提供一批 FD-only 候选，但没有可验证的 gold final assistant response，"
        "平均参数槽也低于 Rich24，不能直接替代 Rich-like FD+Terminal 1:1 全流程。",
        "",
        "## 来源和协议",
        "",
        f'- 官方数据：LARK-Lab/EnvFactory-RL，train，{report["dataset"]["rows"]} 行；'
        f'主分支修订 {report["dataset"]["revision"]}；本地文件 SHA256 '
        f'{report["dataset"]["sha256"]}，与官方文件页一致。',
        '- 原始文件仅从已有远端副本读取；无下载、模型推理、训练、Git commit 或 push。',
        f'- 真实 MCP 注册 {replay["registered_server_rows"]}/{s["parseable_gold_sequence_rows"]} 行，'
        f'工具存在 {replay["tool_exists_rows"]} 行，参数 schema 有效 '
        f'{replay["arguments_schema_valid_rows"]} 行。',
        "",
        "## Schema 与 terminal 可用性",
        "",
        f'- 3092/3092 行含 prompt/data_source/agent_name/ability/reward_model/extra_info；'
        f'ground_truth 是有序 JSON 工具调用数组，含 name/arguments/masked_arguments。',
        f'- gold JSON 结构可解析 {s["parseable_gold_sequence_rows"]} 行；'
        f'gold 所属 server 与该行激活配置一致 {s["gold_server_consistent_rows"]} 行；'
        f'参数结构完整 {s["exact_argument_structure_rows"]} 行；'
        f'存在 {s["masked_argument_slots"]} 个 masked 参数槽。',
        '- 独立 task_id、seed、tool observation、完整 trajectory/history、final assistant response：'
        '均未出现在发布行中。initial_config 和 final_config 均为 3092/3092。',
        '- Gold-Terminal 分类 A explicit=0、B deterministic=0、C stable SFT join=0、'
        f'D unavailable={report["terminal_response_classification"]["D_unavailable"]}。'
        f'SFT-FILTERED 仅有 {report["terminal_response_classification"]["sft_query_only_overlap"]} '
        '个同 query 命中，缺少配置/轨迹复合键，不能可靠恢复最终回答。',
        "",
        "## Executable replay 与 OPD tiers",
        "",
        f'- 全量已记录 replay {replay["replay_recorded_rows"]}/{report["dataset"]["rows"]}；'
        f'reset 成功 {replay["reset_ok_rows"]}；gold 工具链完全执行 '
        f'{replay["gold_tool_execution_complete_rows"]}；最终状态匹配 '
        f'{replay["final_state_match_rows"]}；严格 replay PASS {replay["replay_pass_rows"]}。',
        f'- 工具缺失/未激活服务 {replay["tool_missing_rows"]} 行；参数 schema 失败 '
        f'{replay["argument_schema_fail_rows"]} 行。7 行 gold 引用了注册表中存在但'
        '未列入自身 mcp_servers 的其他服务，属于配置缺失，不是 benign alias；'
        '不擅自补 server。',
        f'- Tier A FULL {tiers["A"]}；Tier B FD-only {tiers["B"]}；'
        f'Tier C structural {tiers["C"]}；Tier D invalid {tiers["D"]}。'
        f'清洁 Tier A/B {report["clean_tier_ab"]}，仅这些进入 source manifest。',
        f'- replay error types：{replay["error_types"]}。工具执行失败与最终状态不匹配'
        '都不被计为 FD-ready。',
        "",
        "## 参数、深度和环境覆盖",
        "",
        f'- 环境组合 {report["coverage"]["unique_environment_combinations"]}，'
        f'MCP 服务 {report["coverage"]["unique_mcp_servers"]}；'
        f'gold 调用均值/中位数 {rich["gold_calls"]["mean"]}/{rich["gold_calls"]["median"]}，'
        f'p95={rich["gold_calls"]["p95"]}，max={rich["gold_calls"]["max"]}。',
        f'- 每题参数槽均值/中位数 {rich["args_per_task"]["mean"]}/{rich["args_per_task"]["median"]}，'
        f'p75={rich["args_per_task"]["p75"]}，p90={rich["args_per_task"]["p90"]}，'
        f'max={rich["args_per_task"]["max"]}；含多参数调用的任务 '
        f'{rich["tasks_with_multi_arg_call"]}/{report["dataset"]["rows"]}。',
        f'- 每个工具调用参数槽均值/中位数 {rich["args_per_call"]["mean"]}/'
        f'{rich["args_per_call"]["median"]}，p75={rich["args_per_call"]["p75"]}，'
        f'p90={rich["args_per_call"]["p90"]}，max={rich["args_per_call"]["max"]}。',
        f'- query 字符数均值/中位数 {rich["query_chars"]["mean"]}/'
        f'{rich["query_chars"]["median"]}，p75={rich["query_chars"]["p75"]}，'
        f'p90={rich["query_chars"]["p90"]}，max={rich["query_chars"]["max"]}。',
        f'- 调用参数数 >=2/3/4：{rich["calls_with_2plus_args"]}/'
        f'{rich["calls_with_3plus_args"]}/{rich["calls_with_4plus_args"]}；'
        f'静态 candidate-only value reuse 任务 {rich["candidate_only_value_reuse_tasks"]}，'
        f'真实 observation 支持的唯一值流任务 {report["verified_flow_tasks_all_replay"]}。'
        '两者不能混用；多数任务无完整 dependency graph，深度为 unknown。',
        f'- 清洁 Tier B 严格 parameter-rich {report["parameter_rich_clean_strict"]}；'
        f'宽松 candidate-only parameter-rich {report["parameter_rich_clean_candidate_only"]}。'
        f'可验证深度分布 {report["verified_dependency_depth_clean"]}。',
        f'- gold 调用长度桶 1/2/3/4+/6+ = '
        f'{rich["gold_call_buckets"]["1"]}/{rich["gold_call_buckets"]["2"]}/'
        f'{rich["gold_call_buckets"]["3"]}/{rich["gold_call_buckets"]["4+"]}/'
        f'{rich["gold_call_buckets"]["6+"]}；参数值类型统计 '
        f'{rich["argument_value_types"]}。',
        f'- 高频环境组合（前五）：'
        f'{sorted(report["coverage"]["environment_frequency"].items(), key=lambda p: -p[1])[:5]}；'
        f'仅 1 题的长尾环境组合 '
        f'{sum(count == 1 for count in report["coverage"]["environment_frequency"].values())}。'
        '各环境完整指标见 environment_statistics.json。',
        "",
        "## 与 Rich24/Frozen300 的分布对照",
        "",
        "| 指标 | Official RL | Rich24 | Frozen300 |",
        "|---|---:|---:|---:|",
        f'| query 字符均值 | {compare["OfficialRL"]["query_chars_mean"]:.1f} | '
        f'{compare["Rich24"]["query_chars_mean"]:.1f} | {compare["Frozen300"]["query_chars_mean"]:.1f} |',
        f'| gold 工具调用均值 | {compare["OfficialRL"]["gold_calls_mean"]:.2f} | '
        f'{compare["Rich24"]["gold_calls_mean"]:.2f} | {compare["Frozen300"]["gold_calls_mean"]:.2f} |',
        f'| gold 参数槽/题均值 | {compare["OfficialRL"]["args_per_task_mean"]:.2f} | '
        f'{compare["Rich24"]["args_per_task_mean"]:.2f} | {compare["Frozen300"]["args_per_task_mean"]:.2f} |',
        '| dependency depth | 大多数 unknown；仅报告已证实的值流深度 | 1/2/3=8/8/8 | 1/2/3=120/110/70 |',
        f'| 环境家族交集 | 84 MCP servers；与 Rich24 '
        f'{compare["Rich24"]["shared_server_families"]}/{compare["Rich24"]["pool_server_families"]} '
        f'共用；与 Frozen300 {compare["Frozen300"]["shared_server_families"]}/'
        f'{compare["Frozen300"]["pool_server_families"]} 共用 | 20 | 5 |',
        "",
        "query 长度仅作分布代理，不能单独推断难度。官方数据覆盖广，但参数槽均值比 Rich24 低，"
        "且缺少原生 ToolGraph，不能说其依赖轨迹更深。",
        "",
        "## 污染、去重与下一步",
        "",
        f'- Frozen300 exact query overlap {overlap["Frozen300"]["query_count"]}，'
        f'clean exclusion {overlap["Frozen300"]["clean_exclusion_count"]}；'
        f'Rich24 exact query overlap {overlap["Rich24"]["query_count"]}，'
        f'clean exclusion {overlap["Rich24"]["clean_exclusion_count"]}。',
        f'- RichTrain24chunks / Generated1000 / PreferencePairs / SFT-FILTERED 的'
        f'清洁排除数分别为 {overlap["RichTrain24chunks"]["clean_exclusion_count"]}/'
        f'{overlap["Generated1000"]["clean_exclusion_count"]}/'
        f'{overlap["PreferencePairs"]["clean_exclusion_count"]}/'
        f'{overlap["SFT_FILTERED"]["clean_exclusion_count"]}。'
        'query hash 和环境+初始状态+完整工具参数签名分别审计，'
        '单独相同工具序列不等于同一任务。',
        f'- Generated1000 单项 gold 签名相同 {overlap["Generated1000"]["gold_count"]} 行、'
        f'单项环境+初态相同 {overlap["Generated1000"]["env_initial_count"]} 行；'
        '这两种单项命中没有与 query 或彼此组成同一任务的证据，未按确切污染排除。'
        '严谨起见，下轮如用这些行训练还应做更强的近重复检测。',
        f'- 官方内部唯一 query {report["duplicates"]["unique_query_hashes"]}/3092；'
        f'唯一完整 gold path {report["duplicates"]["unique_exact_gold_paths"]}/3092；'
        f'唯一工具名序列 {report["duplicates"]["unique_tool_name_sequences"]}/3092。',
        f'- 若下轮每个清洁 FD-ready task 只做一次 Dynamic-v1 rollout，需要约 '
        f'{report["next_rollout_cost"]["clean_student_episodes_at_one_rollout_each"]} '
        '个环境 episode。现成任务可免去新一轮 14B 任务合成，但没有可靠 student '
        'runtime 基线，不提供虚构 GPU 小时。',
        '- Q1：只能部分替代 14B generation——可替代一批 FD source，不替代 '
        'Rich-like terminal source。Q2：不能证明更丰富，平均参数少于 Rich24、'
        '依赖深度证据稀疏。Q3：PARTIAL；先用清洁 Tier B 做 on-policy FD mining，'
        'Gold-Terminal 必须另找可靠来源，本轮不启动。',
        "",
        "## 审计局限",
        "",
    ]
    lines.extend(f'- {item}' for item in report["limitations"])
    lines += ["", "NO TRAINING · NO COMMIT · NO PUSH · CPU-ONLY", ""]
    return "\n".join(lines)

def main() -> None:
    if REPORT.exists() or REPORT_JSON.exists() or (OUT / "opd_source_manifest.json").exists():
        raise RuntimeError("refusing to overwrite finalized audit artifacts")
    raw = json.loads(SOURCE.read_text())
    static = pq.read_table(OUT / "row_audit.parquet").to_pylist()
    replay_summary = json.loads((OUT / "replay_summary.json").read_text())
    if replay_summary.get("gate") != "FULL_REPLAY_COMPLETE":
        raise RuntimeError("full executable replay gate did not pass")
    if len(raw) != 3092 or len(static) != len(raw):
        raise RuntimeError("official source or static audit row count mismatch")
    replay = {}
    for path in (OUT / "replay_tasks").glob("*.json"):
        result = json.loads(path.read_text())
        index = result["row_index"]
        if index in replay:
            raise RuntimeError(f"duplicate replay row {index}")
        replay[index] = result
    if set(replay) != set(range(len(raw))):
        raise RuntimeError("replay task set is not complete")
    if any(row["row_index"] != index for index, row in enumerate(static)):
        raise RuntimeError("static row order mismatch")
    overlap, exclusions = contamination(raw)
    classified, tier_counts = classify_rows(static, replay, exclusions)
    report = summarize(raw, static, replay, classified, tier_counts, overlap)
    clean = [row for row in classified if row["clean_opd_ready"]]
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_sha != report["dataset"]["sha256"]:
        raise RuntimeError("source hash changed during audit")
    manifest = {
        "name": "official_envfactory_rl_opd_source_v1",
        "source_path": str(SOURCE),
        "source_sha256": source_sha,
        "source_revision": report["dataset"]["revision"],
        "split": "train",
        "selection_rule": "strict gold replay plus zero exact evaluation/training overlap",
        "task_count": len(clean),
        "rows": [{
            "audit_id": row["audit_id"],
            "source_row_index": row["row_index"],
            "query_hash": row["query_hash"],
            "environment": row["environment"],
            "gold_length": row["gold_length"],
            "verified_dependency_depth": row["verified_dependency_depth"],
            "strict_parameter_rich": row["strict_parameter_rich"],
            "loose_candidate_parameter_rich": row["loose_candidate_parameter_rich"],
        } for row in clean],
        "categories": {
            "General": [row["audit_id"] for row in clean],
            "ParameterRichStrict": [row["audit_id"] for row in clean if row["strict_parameter_rich"]],
            "ParameterRichCandidate": [row["audit_id"] for row in clean if row["loose_candidate_parameter_rich"]],
            "DeepVerified": [row["audit_id"] for row in clean if row["verified_dependency_depth"].isdigit()
                             and int(row["verified_dependency_depth"]) >= 2],
            "TerminalReady": [],
        },
        "requires_on_policy_student_rollout_before_fd_training": True,
        "terminal_response_available": False,
        "audit_only_no_training": True,
    }
    if len({row["audit_id"] for row in clean}) != len(clean):
        raise RuntimeError("duplicate audit IDs in source manifest")
    OUT.joinpath("contamination_audit_v2.json").write_text(
        json.dumps(overlap, ensure_ascii=False, indent=2, sort_keys=True))
    pq.write_table(pa.Table.from_pylist(classified), OUT / "opd_tiers.parquet")
    OUT.joinpath("environment_statistics.json").write_text(
        json.dumps(report["environment_statistics"], ensure_ascii=False, indent=2, sort_keys=True))
    OUT.joinpath("opd_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    REPORT.write_text(render_report(report))
    print(json.dumps({
        "verdict": report["verdict"], "tiers": report["tier_counts"],
        "clean_tier_ab": len(clean), "strict_parameter_rich": report["parameter_rich_clean_strict"],
        "overlap_exclusions": len(exclusions),
        "manifest_tasks": manifest["task_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


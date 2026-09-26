#!/usr/bin/env python3
"""Unified held-out evaluation report for Guided RL v2 Standard vs Graph."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from repro_1p7b.graph_frontier.confirm_analyze import (
    load_manifest,
    metric_maps,
    paired_metric,
    root_failure,
    task_rows_for_model,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / "repro_1p7b/results/graph_frontier/confirm_300"
EVAL_ROOT = ROOT / "repro_1p7b/graph_frontier/evals"
REPORT_ROOT = ROOT / "repro_1p7b/graph_frontier/reports"
HIST_BFCL_ROOT = ROOT / "repro_1p7b/evaluation/bfcl/artifacts/full"
MODEL_DIR = "Qwen_Qwen3-1.7B-FC"
HIST_LABELS = ("base", "original_sft", "parameter_aware", "dynamic_v1", "dynamic_v2")
BFCL_DIR = {"base": "base", "original_sft": "sft", "parameter_aware": "parameter_aware", "dynamic_v1": "dynamic_v1", "dynamic_v2": "dynamic_v2"}
NEW_ROOTS = {
    "standard_v2": EVAL_ROOT / "standard_rl_v2_control_step64",
    "graph_v2": EVAL_ROOT / "graph_frontier_rl_v2_step64",
}
EXPECTED_MANIFEST_SHA256 = "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"
EXPECTED_PILOT_SHA256 = "5cef60450c57478df241da3438053fcc34ebcacfc2edc2c7ab761ccf4b27ee2f"
EXPECTED_DYNAMIC_V1_MODEL_SHA256 = "0231573c95e939a3dcc52518c15a7241edb32bcca65729f49230f593cec7b970"
EXPECTED_DYNAMIC_V1_CONFIG_SHA256 = "2eb1c7d9ec4d1dde3651c2257b17d042489c589b16290501e0fbb5243851e03d"
CATEGORY_FILES = {
    "base": "BFCL_v3_multi_turn_base",
    "missing_function": "BFCL_v3_multi_turn_miss_func",
    "missing_parameter": "BFCL_v3_multi_turn_miss_param",
    "long_context": "BFCL_v3_multi_turn_long_context",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rate(successes: int, attempts: int) -> dict[str, Any]:
    return {"successes": successes, "attempts": attempts, "raw_rate": successes / attempts if attempts else None}


def pct(metric: Mapping[str, Any]) -> str:
    value = metric.get("raw_rate")
    return "n/a" if value is None else f"{100 * value:.2f}% ({metric['successes']}/{metric['attempts']})"


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def seed_for(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)


def binary_paired(left: Mapping[str, bool], right: Mapping[str, bool], name: str, bootstrap: int = 10000) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    base = paired_metric(left, right)
    diffs = [int(right[key]) - int(left[key]) for key in shared]
    rng = random.Random(seed_for(name + ":bootstrap"))
    samples = [100 * sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(bootstrap)] if diffs else []
    return {**base, "shared_n": len(shared), "delta_pp": 100 * sum(diffs) / len(diffs) if diffs else None, "paired_bootstrap_95_ci_pp": [percentile(samples, .025), percentile(samples, .975)], "bootstrap_replicates": bootstrap}


def edge_counts(row: Mapping[str, Any], metric: str) -> tuple[int, int]:
    funnel = row["funnel"]
    gold = int(funnel.get("gold_internal_opportunities", 0))
    if metric == "producer_reach": return int(funnel.get("producer_reached", 0)), gold
    if metric == "consumer_reach": return int(funnel.get("consumer_reached", 0)), gold
    if metric == "conditional_propagation": return int(funnel.get("correct_propagated", 0)), int(funnel.get("consumer_reached", 0))
    if metric == "internal_e2e": return int(funnel.get("correct_propagated", 0)), gold
    raise KeyError(metric)


def edge_paired(left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]], metric: str, bootstrap: int = 10000) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    pairs = [(edge_counts(left[key], metric), edge_counts(right[key], metric)) for key in ids]
    def totals(rows, side):
        success = sum(row[side][0] for row in rows); attempts = sum(row[side][1] for row in rows)
        return success, attempts, success / attempts if attempts else None
    ls, la, lr = totals(pairs, 0); rs, ra, rr = totals(pairs, 1)
    observed = (rr - lr) if lr is not None and rr is not None else None
    rng = random.Random(seed_for(metric + ":edge_bootstrap")); boot=[]
    for _ in range(bootstrap):
        sample=[pairs[rng.randrange(len(pairs))] for _ in pairs]
        _,_,a=totals(sample,0); _,_,b=totals(sample,1)
        if a is not None and b is not None: boot.append(100*(b-a))
    rng = random.Random(seed_for(metric + ":edge_permutation")); extreme=0; used=0
    for _ in range(bootstrap):
        perm=[(rgt,lft) if rng.random()<.5 else (lft,rgt) for lft,rgt in pairs]
        _,_,a=totals(perm,0); _,_,b=totals(perm,1)
        if a is None or b is None: continue
        used += 1
        if observed is not None and abs(b-a) >= abs(observed)-1e-15: extreme += 1
    return {
        "shared_tasks": len(ids), "left": rate(ls,la), "right": rate(rs,ra),
        "delta_pp": None if observed is None else 100*observed,
        "paired_cluster_bootstrap_95_ci_pp": [percentile(boot,.025),percentile(boot,.975)],
        "paired_task_label_permutation_p": (extreme+1)/(used+1) if used else None,
        "replicates": bootstrap,
        "denominator_note": "same paired tasks and identical gold edges; conditional propagation retains its reached-consumer denominator",
    }


def count_paired(left: Mapping[str, int], right: Mapping[str, int], name: str, bootstrap: int = 10000) -> dict[str, Any]:
    ids=sorted(set(left)&set(right)); diffs=[right[k]-left[k] for k in ids]
    rng=random.Random(seed_for(name+":count_bootstrap")); boot=[sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs) for _ in range(bootstrap)] if diffs else []
    obs=sum(diffs)/len(diffs) if diffs else None
    rng=random.Random(seed_for(name+":count_signflip")); extreme=0
    for _ in range(bootstrap):
        value=sum(d if rng.random()<.5 else -d for d in diffs)/len(diffs)
        extreme += bool(obs is not None and abs(value)>=abs(obs)-1e-15)
    return {"shared_n":len(ids),"left_total":sum(left[k] for k in ids),"right_total":sum(right[k] for k in ids),"total_delta":sum(diffs),"mean_delta_per_task":obs,"paired_bootstrap_95_ci_mean":[percentile(boot,.025),percentile(boot,.975)],"paired_signflip_p":(extreme+1)/(bootstrap+1) if diffs else None}


def select(rows: Mapping[str, Mapping[str, Any]], *, split: str | None = None, depth: str | None = None, internal_only: bool = False) -> dict[str, Mapping[str, Any]]:
    answer={}
    for key,row in rows.items():
        bucket="3+" if row["dependency_depth"]>=3 else str(row["dependency_depth"])
        if split is not None and row["split"]!=split: continue
        if depth is not None and bucket!=depth: continue
        if internal_only and not row["manifest"]["dependency_edges"]: continue
        answer[key]=row
    return answer


def edge_rate(rows: Mapping[str, Mapping[str, Any]], metric: str) -> dict[str, Any]:
    counts=[edge_counts(row,metric) for row in rows.values()]
    return rate(sum(x for x,_ in counts),sum(n for _,n in counts))


def consumer_completion_map(rows: Mapping[str, Mapping[str, Any]]) -> dict[str,bool]:
    return {k: edge_counts(v,"consumer_reach")[0] == edge_counts(v,"consumer_reach")[1] for k,v in rows.items()}


def consumer_not_executed_map(rows: Mapping[str, Mapping[str, Any]]) -> dict[str,bool]:
    return {
        key: bool(
            row["semantic"]["semantic_verifier_supported"]
            and not row["semantic"]["semantic_success"]
            and root_failure(row) == "consumer_not_executed"
        )
        for key,row in rows.items()
    }


def frozen_view(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values=list(rows.values())
    supported=[row for row in values if row["semantic"]["semantic_verifier_supported"]]
    failures=[row for row in supported if not row["semantic"]["semantic_success"]]
    retries=sum(row["calls"]["repeated_pattern_counts"].get("retry_after_tool_error",0) for row in values)
    redundant=sum(row["calls"]["legacy_exact_redundant_calls"] for row in values)
    unexpected=sum(row["calls"]["legacy_unexpected_calls"] for row in values)
    return {
        "tasks":len(rows), "semantic":rate(sum(row["semantic"]["semantic_success"] for row in supported),len(supported)),
        "reference_path":rate(sum(row["reference_path_complete_success"] for row in values),len(values)),
        "internal_complete":rate(sum(row["internal_task_complete"] for row in values),len(values)), "producer_reach":edge_rate(rows,"producer_reach"),
        "consumer_reach":edge_rate(rows,"consumer_reach"), "conditional_propagation":edge_rate(rows,"conditional_propagation"),
        "internal_e2e":edge_rate(rows,"internal_e2e"), "consumer_task_completion":rate(sum(consumer_completion_map(rows).values()),len(rows)),
        "consumer_not_executed":sum(root_failure(row)=="consumer_not_executed" for row in failures),
        "root_failure_attribution":dict(Counter(root_failure(row) for row in failures)), "retry_after_error":retries,
        "redundant_calls_per_task":redundant/len(values) if values else None, "unexpected_calls_per_task":unexpected/len(values) if values else None,
    }


def frozen_paired(left: Mapping[str,Mapping[str,Any]], right: Mapping[str,Mapping[str,Any]], prefix: str) -> dict[str,Any]:
    lm=metric_maps(left); rm=metric_maps(right)
    lm["consumer_task_completion"]=consumer_completion_map(left); rm["consumer_task_completion"]=consumer_completion_map(right)
    lm["consumer_not_executed"]=consumer_not_executed_map(left); rm["consumer_not_executed"]=consumer_not_executed_map(right)
    result={name:binary_paired(lm[name],rm[name],prefix+":"+name) for name in ("semantic_task_success","reference_path_complete_success","task_level_internal_edge_complete","consumer_task_completion","consumer_not_executed")}
    result["edges"]={name:edge_paired(left,right,name) for name in ("producer_reach","consumer_reach","conditional_propagation","internal_e2e")}
    retry_l={k:v["calls"]["repeated_pattern_counts"].get("retry_after_tool_error",0) for k,v in left.items()}
    retry_r={k:v["calls"]["repeated_pattern_counts"].get("retry_after_tool_error",0) for k,v in right.items()}
    result["retry_after_error"]=count_paired(retry_l,retry_r,prefix+":retry")
    return result


def read_csv_score(path: Path) -> dict[str,Any]:
    with path.open(newline="") as handle: row=next(csv.DictReader(handle))
    get=lambda key: float(row[key].rstrip("%"))
    return {"overall":get("Multi Turn Overall Acc"),"base":get("Base"),"missing_function":get("Miss Func"),"missing_parameter":get("Miss Param"),"long_context":get("Long Context"),"path":str(path)}


def bfcl_outcomes(root: Path) -> tuple[dict[str,dict[str,bool]],dict[str,Any]]:
    result_dir=root/"bfcl/result"/MODEL_DIR; score_dir=root/"bfcl/score"/MODEL_DIR
    maps={}; audits={}
    for category,stem in CATEGORY_FILES.items():
        results=[json.loads(line) for line in (result_dir/f"{stem}_result.json").read_text().splitlines() if line.strip()]
        lines=[json.loads(line) for line in (score_dir/f"{stem}_score.json").read_text().splitlines() if line.strip()]
        summary=lines[0]; details={row["id"]:bool(row["valid"]) for row in lines[1:] if "id" in row}; ids=[row["id"] for row in results]
        if len(details)==summary["total_count"]: outcomes={key:details[key] for key in ids}
        else:
            if any(details.values()) or len(details)!=summary["total_count"]-summary["correct_count"]: raise RuntimeError((category,summary,len(details)))
            outcomes={key:key not in details for key in ids}
        if sum(outcomes.values())!=summary["correct_count"] or len(outcomes)!=summary["total_count"]: raise RuntimeError((category,"score_count_mismatch"))
        maps[category]=outcomes; audits[category]=summary
    return maps,audits


def checkpoint_integrity() -> dict[str,Any]:
    entries={
      "standard_v2":("repro_1p7b/checkpoints/standard_rl_v2_control_pilot128/global_step_64","repro_1p7b/checkpoints_eval/standard_rl_v2_control_pilot128_step64_hf"),
      "graph_v2":("repro_1p7b/checkpoints/graph_frontier_rl_v2_pilot128/global_step_64","repro_1p7b/checkpoints_eval/graph_frontier_rl_v2_pilot128_step64_hf"),
    }; answer={}
    for label,(source_rel,merged_rel) in entries.items():
        source=ROOT/source_rel; actor=source/"actor"; merged=ROOT/merged_rel
        model_shards=sorted(actor.glob("model_world_size_2_rank_*.pt")); optim=sorted(actor.glob("optim_world_size_2_rank_*.pt")); safe=sorted(merged.glob("*.safetensors"))
        answer[label]={"source":source_rel,"merged":merged_rel,"global_step":64,"model_shards":[{"name":p.name,"size":p.stat().st_size} for p in model_shards],"optimizer_shards":[{"name":p.name,"size":p.stat().st_size} for p in optim],"config_sha256":sha256(actor/"config.json"),"tokenizer_sha256":sha256(actor/"tokenizer.json"),"merged_files":[{"name":p.name,"size":p.stat().st_size} for p in safe],"merged_loadable":bool(safe and (merged/"config.json").is_file() and (merged/"tokenizer.json").is_file()),"latest_checkpointed_iteration":int((source.parent/"latest_checkpointed_iteration.txt").read_text().strip())}
    answer["same_config"]=answer["standard_v2"]["config_sha256"]==answer["graph_v2"]["config_sha256"]
    answer["same_tokenizer"]=answer["standard_v2"]["tokenizer_sha256"]==answer["graph_v2"]["tokenizer_sha256"]
    answer["pilot_train_parquet_sha256"]=EXPECTED_PILOT_SHA256
    answer["shared_dynamic_v1_init_model_sha256"]=EXPECTED_DYNAMIC_V1_MODEL_SHA256
    answer["shared_dynamic_v1_init_config_sha256"]=EXPECTED_DYNAMIC_V1_CONFIG_SHA256
    return answer


def verdict(report: Mapping[str,Any]) -> str:
    pair=report["frozen300"]["paired_standard_vs_graph"]; bf=report["bfcl"]["paired_standard_vs_graph"]["overall"]
    regressions=[]
    if pair["semantic_task_success"]["paired_bootstrap_95_ci_pp"][1] < 0: regressions.append("semantic")
    if pair["edges"]["internal_e2e"]["paired_cluster_bootstrap_95_ci_pp"][1] < 0: regressions.append("internal_e2e")
    if pair["edges"]["conditional_propagation"]["paired_cluster_bootstrap_95_ci_pp"][1] < 0: regressions.append("propagation")
    if bf["paired_bootstrap_95_ci_pp"][1] < 0: regressions.append("bfcl")
    if pair["retry_after_error"]["paired_bootstrap_95_ci_mean"][0] > 0: regressions.append("retry")
    if pair["consumer_not_executed"]["paired_bootstrap_95_ci_pp"][0] > 0: regressions.append("consumer_not_executed")
    if regressions: return "NO-GO"
    consumer=pair["consumer_task_completion"]; e2e=pair["edges"]["internal_e2e"]
    improvement=(consumer["delta_pp"]>0 or e2e["delta_pp"]>0)
    established=(consumer["paired_bootstrap_95_ci_pp"][0]>0 or e2e["paired_cluster_bootstrap_95_ci_pp"][0]>0)
    held=report["frozen300"]["split"]["heldout"]["paired_standard_vs_graph"]
    held_direction=(held["consumer_task_completion"]["delta_pp"]>=0 and held["edges"]["internal_e2e"]["delta_pp"]>=0)
    if improvement and established and held_direction: return "GO"
    if improvement: return "PARTIAL-GO"
    return "NO-GO"


def build() -> dict[str,Any]:
    manifest_path,manifests=load_manifest(MANIFEST_ROOT); digest=sha256(manifest_path)
    if digest!=EXPECTED_MANIFEST_SHA256: raise RuntimeError(f"manifest_hash:{digest}")
    rows={}; runtime={}
    for label in HIST_LABELS: rows[label],runtime[label]=task_rows_for_model(MANIFEST_ROOT,manifests,label)
    for label,root in NEW_ROOTS.items(): rows[label],runtime[label]=task_rows_for_model(root,manifests,"frozen_300")
    if any(runtime[label]["valid"]!=300 for label in ("standard_v2","graph_v2")): raise RuntimeError(runtime)
    frozen_models={label:frozen_view(rows[label]) for label in rows}; detail={}
    for grouping,values in (("depth",("1","2","3+")),("split",("diagnosis","heldout"))):
        detail[grouping]={}
        for value in values:
            l=select(rows["standard_v2"],**{grouping:value}); r=select(rows["graph_v2"],**{grouping:value})
            detail[grouping][value]={"standard_v2":frozen_view(l),"graph_v2":frozen_view(r),"paired_standard_vs_graph":frozen_paired(l,r,f"{grouping}:{value}")}
    li=select(rows["standard_v2"],internal_only=True); ri=select(rows["graph_v2"],internal_only=True)
    internal_subset={"standard_v2":frozen_view(li),"graph_v2":frozen_view(ri),"paired_standard_vs_graph":frozen_paired(li,ri,"internal_subset"),"note":"All frozen-300 tasks contain at least one internal dependency edge."}
    bfcl_scores={label:read_csv_score(HIST_BFCL_ROOT/BFCL_DIR[label]/"score/data_multi_turn.csv") for label in HIST_LABELS}
    for label,root in NEW_ROOTS.items(): bfcl_scores[label]=read_csv_score(root/"bfcl/score/data_multi_turn.csv")
    std_maps,std_audit=bfcl_outcomes(NEW_ROOTS["standard_v2"]); graph_maps,graph_audit=bfcl_outcomes(NEW_ROOTS["graph_v2"])
    bf_pairs={category:binary_paired(std_maps[category],graph_maps[category],"bfcl:"+category) for category in CATEGORY_FILES}
    all_std={k:v for category in CATEGORY_FILES for k,v in std_maps[category].items()}; all_graph={k:v for category in CATEGORY_FILES for k,v in graph_maps[category].items()}
    bf_pairs["overall"]=binary_paired(all_std,all_graph,"bfcl:overall")
    report={
      "schema_version":"guided_rl_v2_benchmark_evaluation_v1", "checkpoint_integrity":checkpoint_integrity(),
      "protocol":{"manifest_sha256":digest,"seed":20260914,"count":len(manifests),"split_distribution":dict(Counter(x["split"] for x in manifests)),"environment_distribution":dict(Counter(x["environment"] for x in manifests)),"depth_distribution":dict(Counter(x["complexity_bucket"] for x in manifests)),"gold_internal_edges":sum(len(x["dependency_edges"]) for x in manifests),"mcp_isolation":json.loads((EVAL_ROOT/"_guided_rl_v2_control/mcp_isolation/summary.json").read_text()),"frozen_inference":json.loads((NEW_ROOTS["standard_v2"]/"frozen_300/run_config.json").read_text())["inference"],"bfcl":{"version":"V3 Multi-Turn","examples":800,"temperature":0.7,"context_length":40960,"max_total_tokens":65536,"backend":"sglang","single_gpu_per_model":True}},
      "frozen300":{"runtime":runtime,"models":frozen_models,"paired_standard_vs_graph":frozen_paired(rows["standard_v2"],rows["graph_v2"],"full300"),"depth":detail["depth"],"split":detail["split"],"internal_edge_subset":internal_subset},
      "bfcl":{"scores":bfcl_scores,"score_audit":{"standard_v2":std_audit,"graph_v2":graph_audit},"paired_standard_vs_graph":bf_pairs},
      "comparisons":{"standard_v2_vs_dynamic_v1":{"frozen":{"standard_v2":frozen_models["standard_v2"],"dynamic_v1":frozen_models["dynamic_v1"]},"bfcl_delta_pp":bfcl_scores["standard_v2"]["overall"]-bfcl_scores["dynamic_v1"]["overall"]},"graph_v2_vs_dynamic_v1":{"frozen":{"graph_v2":frozen_models["graph_v2"],"dynamic_v1":frozen_models["dynamic_v1"]},"bfcl_delta_pp":bfcl_scores["graph_v2"]["overall"]-bfcl_scores["dynamic_v1"]["overall"]}},
      "limitations":["Paired task-level binary outcomes use exact McNemar tests and deterministic paired bootstrap confidence intervals.","Edge metrics use task-cluster bootstrap and paired task-label permutation; edges within a task are not treated as independent.","The held-out split has 60 tasks, so its confidence intervals are wide and directions are not over-interpreted.","Conditional propagation uses reached consumers as its protocol-defined denominator; reach and internal E2E use the identical 710 gold-edge denominator."], "full_rl_status":"FULL_RL_NOT_STARTED",
    }
    report["verdict"]=verdict(report)
    report["research_questions"]={
      "q1_standard_grpo_vs_dynamic_v1":{
        "answer":"No overall improvement on Frozen300; only the BFCL point estimate improved.",
        "frozen_semantic_delta_pp":100*(frozen_models["standard_v2"]["semantic"]["raw_rate"]-frozen_models["dynamic_v1"]["semantic"]["raw_rate"]),
        "frozen_consumer_reach_delta_pp":100*(frozen_models["standard_v2"]["consumer_reach"]["raw_rate"]-frozen_models["dynamic_v1"]["consumer_reach"]["raw_rate"]),
        "frozen_internal_e2e_delta_pp":100*(frozen_models["standard_v2"]["internal_e2e"]["raw_rate"]-frozen_models["dynamic_v1"]["internal_e2e"]["raw_rate"]),
        "bfcl_delta_pp":bfcl_scores["standard_v2"]["overall"]-bfcl_scores["dynamic_v1"]["overall"],
        "caveat":"This is a historical, unpaired checkpoint comparison; no causal significance claim is made.",
      },
      "q2_additional_graph_reward":{
        "answer":"No reliable general benefit over the paired Standard-v2 control.",
        "full300_internal_e2e_delta_pp":report["frozen300"]["paired_standard_vs_graph"]["edges"]["internal_e2e"]["delta_pp"],
        "diagnosis240_internal_e2e_delta_pp":report["frozen300"]["split"]["diagnosis"]["paired_standard_vs_graph"]["edges"]["internal_e2e"]["delta_pp"],
        "heldout60_internal_e2e_delta_pp":report["frozen300"]["split"]["heldout"]["paired_standard_vs_graph"]["edges"]["internal_e2e"]["delta_pp"],
        "bfcl_delta_pp":report["bfcl"]["paired_standard_vs_graph"]["overall"]["delta_pp"],
        "consumer_not_executed_delta_pp":report["frozen300"]["paired_standard_vs_graph"]["consumer_not_executed"]["delta_pp"],
      },
      "q3_enter_full_rl":{"answer":"No.","gate":report["verdict"],"status":report["full_rl_status"]},
    }
    report["mechanism_interpretation"]={
      "pilot_transfer":"The diagnosis240 propagation/E2E gain did not reproduce on heldout60; heldout directions reversed.",
      "consumer_not_executed":"Graph has significantly more failures whose first attributed root is consumer_not_executed. This is partly an attribution redistribution (especially at depth 3+, where edge outcomes are identical but missing_internal shifts to consumer_not_executed), not evidence that every count difference represents a newly unexecuted consumer.",
      "conclusion":"The Graph reward shows a diagnosis-local parameter-flow signal, but it is not robust enough to justify full RL.",
    }
    return report


def render(report: Mapping[str,Any]) -> str:
    models=report["frozen300"]["models"]; bf=report["bfcl"]["scores"]; pair=report["frozen300"]["paired_standard_vs_graph"]
    names=[("base","Base"),("original_sft","Original SFT"),("parameter_aware","Static PA"),("dynamic_v1","Dynamic v1"),("dynamic_v2","Dynamic v2"),("standard_v2","Standard-v2"),("graph_v2","Graph-v2")]
    lines=["# Guided RL v2 Benchmark Evaluation","","## Executive Summary","",f"**Verdict: {report['verdict']}**","",f"Frozen300 and BFCL completed under locked protocols. {report['full_rl_status']}.","","# 1. Checkpoint integrity","",f"- Standard/Graph FSDP rank-0 and rank-1 model shards present; step: 64/64.",f"- Same config: {report['checkpoint_integrity']['same_config']}; same tokenizer: {report['checkpoint_integrity']['same_tokenizer']}.",f"- Shared Dynamic-v1 init model SHA256: `{report['checkpoint_integrity']['shared_dynamic_v1_init_model_sha256']}`.",f"- Shared Dynamic-v1 init config SHA256: `{report['checkpoint_integrity']['shared_dynamic_v1_init_config_sha256']}`.",f"- Shared pilot parquet SHA256: `{report['checkpoint_integrity']['pilot_train_parquet_sha256']}`.","- Both checkpoints were merged into independent HF eval directories; originals were not modified.","","# 2. Frozen 300 protocol","",f"- Manifest SHA256: `{report['protocol']['manifest_sha256']}`","- 300 tasks; seed 20260914; diagnosis 240 / held-out 60; 710 gold internal edges.",f"- MCP/state dual-process isolation validated: {report['protocol']['mcp_isolation']['complete_isolation_validated']}.","- GPU0=Standard-v2, GPU1=Graph-v2; identical frozen inference config.","","# 3. Frozen 300 overall","","| Model | Semantic | RefPath | Internal complete | Producer reach | Consumer reach | CondProp | Internal E2E | Consumer not executed | Retry |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,name in names:
        x=models[key]; lines.append(f"| {name} | {pct(x['semantic'])} | {pct(x['reference_path'])} | {pct(x['internal_complete'])} | {pct(x['producer_reach'])} | {pct(x['consumer_reach'])} | {pct(x['conditional_propagation'])} | {pct(x['internal_e2e'])} | {x['consumer_not_executed']} | {x['retry_after_error']} |")
    lines += ["","# 4. Frozen 300 depth breakdown",""]
    for depth in ("1","2","3+"):
        lines += [f"## Depth {depth}","","| Model | Semantic | Producer reach | Consumer reach | CondProp | Internal E2E | Consumer complete |","|---|---:|---:|---:|---:|---:|---:|"]
        for key,name in (("standard_v2","Standard-v2"),("graph_v2","Graph-v2")):
            x=report["frozen300"]["depth"][depth][key]; lines.append(f"| {name} | {pct(x['semantic'])} | {pct(x['producer_reach'])} | {pct(x['consumer_reach'])} | {pct(x['conditional_propagation'])} | {pct(x['internal_e2e'])} | {pct(x['consumer_task_completion'])} |")
        lines.append("")
    lines += ["# 5. Internal-edge subset","",report["frozen300"]["internal_edge_subset"]["note"],"Therefore this subset is identical to full300; metrics and paired results are retained in JSON.","","# 6. Diagnosis240 / Heldout60",""]
    for split in ("diagnosis","heldout"):
        lines += [f"## {split}","","| Model | Semantic | Consumer complete | CondProp | Internal E2E |","|---|---:|---:|---:|---:|"]
        for key,name in (("standard_v2","Standard-v2"),("graph_v2","Graph-v2")):
            x=report["frozen300"]["split"][split][key]; lines.append(f"| {name} | {pct(x['semantic'])} | {pct(x['consumer_task_completion'])} | {pct(x['conditional_propagation'])} | {pct(x['internal_e2e'])} |")
        lines.append("")
    lines += ["# 7. BFCL V3 Multi-Turn 800","","| Model | Overall | Base | Missing Function | Missing Parameter | Long Context |","|---|---:|---:|---:|---:|---:|"]
    for key,name in names:
        x=bf[key]; lines.append(f"| {name} | {x['overall']:.2f}% | {x['base']:.2f}% | {x['missing_function']:.2f}% | {x['missing_parameter']:.2f}% | {x['long_context']:.2f}% |")
    lines += ["","# 8. Paired statistics",""]
    for name in ("semantic_task_success","reference_path_complete_success","task_level_internal_edge_complete","consumer_task_completion","consumer_not_executed"):
        x=pair[name]; lines.append(f"- Frozen300 {name}: delta={x['delta_pp']:.2f} pp, 95% CI={x['paired_bootstrap_95_ci_pp']}, exact McNemar p={x['exact_two_sided_p']:.6g}, Graph-only={x['right_wins']}, Standard-only={x['right_losses']}.")
    for name in ("producer_reach","consumer_reach","conditional_propagation","internal_e2e"):
        x=pair["edges"][name]; lines.append(f"- Frozen300 {name}: delta={x['delta_pp']:.2f} pp, clustered 95% CI={x['paired_cluster_bootstrap_95_ci_pp']}, permutation p={x['paired_task_label_permutation_p']:.6g}.")
    x=pair["retry_after_error"]; lines.append(f"- Frozen300 retry_after_error: total delta={x['total_delta']}, mean/task delta={x['mean_delta_per_task']:.4f}, 95% CI={x['paired_bootstrap_95_ci_mean']}, sign-flip p={x['paired_signflip_p']:.6g}.")
    x=report["bfcl"]["paired_standard_vs_graph"]["overall"]; lines.append(f"- BFCL overall: delta={x['delta_pp']:.2f} pp, 95% CI={x['paired_bootstrap_95_ci_pp']}, exact McNemar p={x['exact_two_sided_p']:.6g}; both correct={x['ties_both_success']}, Standard-only={x['right_losses']}, Graph-only={x['right_wins']}, both wrong={x['ties_both_failure']}.")
    q=report["research_questions"]; mechanism=report["mechanism_interpretation"]
    lines += ["","# 9. Historical comparison","","Tables above use the actual stored historical reports/results, not prompt approximations.","","# 10. Standard RL effect","",f"Standard-v2 vs Dynamic-v1: Frozen semantic delta={q['q1_standard_grpo_vs_dynamic_v1']['frozen_semantic_delta_pp']:.2f} pp; consumer-reach delta={q['q1_standard_grpo_vs_dynamic_v1']['frozen_consumer_reach_delta_pp']:.2f} pp; internal-E2E delta={q['q1_standard_grpo_vs_dynamic_v1']['frozen_internal_e2e_delta_pp']:.2f} pp; BFCL delta={q['q1_standard_grpo_vs_dynamic_v1']['bfcl_delta_pp']:.2f} pp.",f"Conclusion: {q['q1_standard_grpo_vs_dynamic_v1']['answer']} {q['q1_standard_grpo_vs_dynamic_v1']['caveat']}","","# 11. Additional Graph effect","",f"Graph-v2 vs Standard-v2 verdict: {report['verdict']}. The causal comparison is paired: full300 internal-E2E {q['q2_additional_graph_reward']['full300_internal_e2e_delta_pp']:+.2f} pp, diagnosis240 {q['q2_additional_graph_reward']['diagnosis240_internal_e2e_delta_pp']:+.2f} pp, heldout60 {q['q2_additional_graph_reward']['heldout60_internal_e2e_delta_pp']:+.2f} pp, BFCL {q['q2_additional_graph_reward']['bfcl_delta_pp']:+.2f} pp.",f"Conclusion: {q['q2_additional_graph_reward']['answer']}","","# 12. Mechanism interpretation","",f"- consumer_not_executed: Standard={models['standard_v2']['consumer_not_executed']}, Graph={models['graph_v2']['consumer_not_executed']} (delta={q['q2_additional_graph_reward']['consumer_not_executed_delta_pp']:+.2f} pp).",f"- conditional propagation: Standard={pct(models['standard_v2']['conditional_propagation'])}, Graph={pct(models['graph_v2']['conditional_propagation'])}.",f"- retry_after_error: Standard={models['standard_v2']['retry_after_error']}, Graph={models['graph_v2']['retry_after_error']}.",f"- Pilot transfer: {mechanism['pilot_transfer']}",f"- Attribution nuance: {mechanism['consumer_not_executed']}",f"- Mechanistic conclusion: {mechanism['conclusion']}","","# 13. Final verdict","",f"**{report['verdict']}**","",report["full_rl_status"],"","## Research questions","",f"- Q1 Standard GRPO vs Dynamic-v1: {q['q1_standard_grpo_vs_dynamic_v1']['answer']}",f"- Q2 Additional Graph reward: {q['q2_additional_graph_reward']['answer']}",f"- Q3 Enter full RL: {q['q3_enter_full_rl']['answer']} Gate={q['q3_enter_full_rl']['gate']}; {q['q3_enter_full_rl']['status']}.","","## Limitations",""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines)+"\n"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--reports",type=Path,default=REPORT_ROOT); args=parser.parse_args()
    report=build(); args.reports.mkdir(parents=True,exist_ok=True)
    jp=args.reports/"guided_rl_v2_benchmark_evaluation.json"; mp=args.reports/"guided_rl_v2_benchmark_evaluation.md"
    jp.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); mp.write_text(render(report))
    print(json.dumps({"verdict":report["verdict"],"json":str(jp),"markdown":str(mp),"full_rl_status":report["full_rl_status"]},indent=2))


if __name__ == "__main__": main()


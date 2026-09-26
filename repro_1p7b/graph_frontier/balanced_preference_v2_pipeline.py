"""Balanced Conditional Transition Preference v2: frozen-data construction.

Only existing, task-disjoint Rich-v3 TRAIN/preference-val tasks are inputs.
Gold replay and typed frontier states are never inferred from flattened text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import classify_sample
from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action, stable
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256


REPO = Path(__file__).resolve().parents[2]
GF = REPO / "repro_1p7b/graph_frontier"
OLD = GF / "preference_generation_v3/balanced_effect_pilot_192_v1"
SNAP = GF / "preference_generation_v3/pilot_balanced_16chunks_v1"
OUT = GF / "balanced_preference_v2"
MODEL = REPO / "repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b"
SCHEMA = "balanced_conditional_transition_preference_v2"
SEED = 20260925
PROMOTED_STOP_VALIDATION_TASK = "gf-rich-c8f65d5abe7142a34fce"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    if path.exists():
        if rows(path) != values:
            raise FileExistsError(f"existing rows changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values))


def digest(value: Any) -> str:
    return hashlib.sha256(stable(value).encode()).hexdigest()


def frozen() -> dict[str, Any]:
    manifest = read(OLD / "quick_effect_manifest.json")
    replay = read(OLD / "replay_summary.json")
    source = SNAP / "valid/graph_frontier_rich_train_v1.jsonl"
    if manifest["source_data_sha256"] != file_sha256(source):
        raise ValueError("source TRAIN hash changed")
    if manifest["source_gold_sha256"] != file_sha256(SNAP / "valid/graph_frontier_rich_gold_v1.jsonl"):
        raise ValueError("source gold hash changed")
    if manifest["frozen300_exact_overlap"] != 0 or replay["frozen300_exact_overlap"] != 0:
        raise ValueError("Frozen300 contamination")
    ids = manifest["task_ids"]
    if len(set().union(*(set(v) for v in ids.values()))) != sum(map(len, ids.values())):
        raise ValueError("task split leakage")
    if len(ids["train"]) != 150 or len(ids["preference_val"]) != 21 or len(ids["heldout_eval"]) != 21:
        raise ValueError("frozen split changed")
    return manifest


def prepare() -> dict[str, Any]:
    manifest = frozen()
    if (OUT / "preparation.json").exists():
        report = read(OUT / "preparation.json")
        if report["source_data_sha256"] != manifest["source_data_sha256"]:
            raise ValueError("resume source identity changed")
        return report
    prepared = {}
    for split in ("train", "preference_val"):
        ids = set(manifest["task_ids"][split])
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for state in rows(OLD / split / "frontier_states.jsonl"):
            if state["task_id"] not in ids or state["chosen_executable"] is not True:
                raise ValueError("frontier source escaped task split or chosen not executable")
            by_task[state["task_id"]].append(state)
        stop_states = []
        for tid in sorted(ids):
            audit = read(SNAP / "replay/tasks" / f"{tid}.replay.json")
            if not (audit["replay_ok"] is True and audit["final_verifier_computed"] is True
                    and audit["final_state_match"] is True):
                continue
            trace = read(SNAP / "replay/traces" / f"{tid}.rollout.json")
            if trace["task_id"] != tid or not trace["events"] or not all(
                    event["execution_success"] is True for event in trace["events"]):
                raise ValueError(f"gold replay invalid for stop state: {tid}")
            if not by_task[tid]:
                # A replay-valid task may still have no retained frontier state.
                # Do not invent its tool schema; retain only inspectable states.
                continue
            tools = by_task[tid][0]["tools"]
            query = by_task[tid][0]["prompt_messages"][0]
            prompt = [query]
            for index, event in enumerate(trace["events"]):
                call_id = f"gold_stop_{index}"
                prompt.append({"role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id, "type": "function", "function": {
                        "name": event["tool_name"], "arguments": event["tool_arguments"],
                    },
                }]})
                prompt.append({"role": "tool", "tool_call_id": call_id,
                               "name": event["tool_name"], "content": stable(event["tool_response"])})
            signature = digest({"prompt": prompt, "state": trace["final_environment_state"], "tools": tools})
            stop_states.append({
                "schema_version": SCHEMA, "state_id": "stop-" + signature[:24],
                "state_signature": signature, "state_type": "stop_required",
                "task_id": tid, "split": split, "prompt_messages": prompt, "tools": tools,
                "query": query["content"], "gold_tool_names": [e["tool_name"] for e in trace["events"]],
                "last_gold_action": {"name": trace["events"][-1]["tool_name"],
                                     "arguments": trace["events"][-1]["tool_arguments"]},
                "dependency_depth": max((s["dependency_depth"] for s in by_task[tid]), default=0),
                "environment_identifiers": trace["environment_identifiers"],
                "final_state_verifier_pass": True, "gold_replay_success": True,
                "frozen_300_overlap": False,
            })
        if len({s["state_id"] for s in stop_states}) != len(stop_states):
            raise ValueError("duplicate stop state")
        write_rows(OUT / f"stop_states_{split}.jsonl", stop_states)
        prepared[split] = {"strict_stop_states": len(stop_states),
                           "frontier_states": sum(map(len, by_task.values())),
                           "downstream_candidate_states": sum(
                               s["producer_position"] >= 1
                               for states in by_task.values() for s in states)}
    report = {"schema_version": SCHEMA, "source_data_sha256": manifest["source_data_sha256"],
              "source_gold_sha256": manifest["source_gold_sha256"],
              "frozen300_train_overlap": 0, "split_task_ids_sha256": digest(manifest["task_ids"]),
              "prepared": prepared}
    write_json(OUT / "preparation.json", report)
    return report


def sample_stop(split: str, endpoint: str, model: str, shard_count: int, shard_index: int,
                target_k: int, only_missing: bool) -> dict[str, Any]:
    states = rows(OUT / f"stop_states_{split}.jsonl")
    if target_k not in (4, 8) or shard_count < 1 or shard_index >= shard_count:
        raise ValueError("sampling configuration invalid")
    completed = requests = 0
    for index, state in enumerate(states):
        if index % shard_count != shard_index:
            continue
        path = OUT / "stop_samples" / split / f"{state['state_id']}.json"
        cached = read(path) if path.exists() else None
        if cached and (cached["state_signature"] != state["state_signature"] or
                       cached["temperature"] != 0.7 or cached["max_tokens"] != 512):
            raise ValueError("cached sample identity/config changed")
        samples = list(cached["samples"]) if cached else []
        if only_missing and any(s.get("parsed_action", {}).get("kind") == "tool" for s in samples):
            completed += 1
            continue
        for j in range(len(samples), target_k):
            seed = SEED + index * 1009 + j * 9176
            try:
                message, act, usage, finish = generate_action(
                    endpoint, model, state["prompt_messages"], state["tools"],
                    seed=seed, temperature=0.7, max_tokens=512)
                sample = {"sample_index": j, "sampling_seed": seed, "model_message": message,
                          "parsed_action": act, "usage": usage, "finish_reason": finish}
            except Exception as exc:
                sample = {"sample_index": j, "sampling_seed": seed,
                          "parsed_action": {"kind": "request_error"},
                          "error": f"{type(exc).__name__}:{exc}"}
            samples.append(sample)
            requests += 1
        if len(samples) != target_k:
            raise ValueError("sample count changed")
        write_json(path, {"schema_version": SCHEMA, "task_id": state["task_id"],
                          "state_id": state["state_id"], "state_signature": state["state_signature"],
                          "temperature": 0.7, "max_tokens": 512, "samples": samples}, replace=bool(cached))
        completed += 1
        print(f"STOP_SAMPLE {split} shard={shard_index} {completed} requests={requests}", flush=True)
    return {"split": split, "shard_index": shard_index, "states_seen": completed,
            "new_requests": requests, "target_k": target_k, "only_missing": only_missing}


def stop_candidate_count(split: str) -> dict[str, int]:
    states = rows(OUT / f"stop_states_{split}.jsonl")
    found = 0
    for state in states:
        path = OUT / "stop_samples" / split / f"{state['state_id']}.json"
        if path.exists() and any(s.get("parsed_action", {}).get("kind") == "tool"
                                 for s in read(path)["samples"]):
            found += 1
    return {"strict_states": len(states), "states_with_observed_extra_tool": found}


def type_ac_pairs(split: str) -> list[dict[str, Any]]:
    manifest = frozen()
    ids = set(manifest["task_ids"][split])
    pairs = []
    for state in rows(OLD / split / "frontier_states.jsonl"):
        if state["task_id"] not in ids or state["chosen_executable"] is not True or \
                state["chosen_execution"]["execution_success"] is not True:
            raise ValueError("A/C state provenance or chosen execution invalid")
        downstream = state["producer_position"] >= 1
        if downstream and not (state["prefix_events"][-1]["tool_name"] == state["producer_tool"]
                               and state["prefix_events"][-1]["execution_success"] is True):
            raise ValueError("downstream gold consumer was not executed successfully")
        sample_file = OLD / split / "sampled_actions" / f"{state['state_id']}.json"
        if not sample_file.exists():
            raise ValueError("missing frozen Dynamic-v1 frontier samples")
        seen = set()
        candidates = []
        for sample in read(sample_file)["samples"]:
            label = classify_sample(sample, state)
            if label not in {"premature_stop", "wrong_tool", "wrong_value", "useless_retry"}:
                continue
            sig = stable(sample.get("parsed_action"))
            if sig in seen:
                continue
            seen.add(sig)
            if downstream:
                label = {"premature_stop": "downstream_stop", "wrong_tool": "downstream_wrong_tool",
                         "useless_retry": "repeat_tool"}.get(label, label)
            candidates.append((label, sample))
        # Distinct failure types first; at most two real negatives per state.
        candidates.sort(key=lambda x: (x[0], x[1]["sample_index"]))
        kept = []
        for label, sample in candidates:
            if label not in {x[0] for x in kept}:
                kept.append((label, sample))
            if len(kept) == 2:
                break
        for label, sample in candidates:
            if len(kept) == 2:
                break
            if (label, sample) not in kept:
                kept.append((label, sample))
        for label, sample in kept:
            act = sample["parsed_action"]
            pairs.append({
                "schema_version": SCHEMA, "pair_id": f"balanced-v2-{state['state_id']}-{sample['sample_index']}",
                "task_id": state["task_id"], "state_id": state["state_id"],
                "state_type": "downstream_continue" if downstream else "continue_required",
                "prompt_messages": state["prompt_messages"], "prompt": state["prompt_messages"],
                "tools": state["tools"], "chosen": [state["chosen_message"]],
                "rejected": [sample["model_message"]],
                "chosen_action": state["chosen_action"], "rejected_action": act,
                "chosen_action_type": "tool_call", "rejected_action_type":
                    "final_answer" if act["kind"] == "final" else "tool_call",
                "producer": state["producer_tool"], "consumer": state["consumer_tool"],
                "next_gold_node": state["consumer_tool"],
                "dependency_depth": state["dependency_depth"],
                "environment_identifiers": state["environment_identifiers"],
                "failure_type": label, "chosen_valid": True,
                "rejected_from_dynamic_v1": True, "sampling_seed": sample["sampling_seed"],
                "source_state_signature": state["state_signature"], "frozen_300_overlap": False,
            })
    return pairs


def type_b_pairs(split: str) -> list[dict[str, Any]]:
    pairs = []
    for state in rows(OUT / f"stop_states_{split}.jsonl"):
        path = OUT / "stop_samples" / split / f"{state['state_id']}.json"
        if not path.exists():
            raise ValueError("stop state missing Dynamic-v1 K4 sample")
        sample_file = read(path)
        if sample_file["state_signature"] != state["state_signature"]:
            raise ValueError("stop sample state mismatch")
        samples = sample_file["samples"]
        final = next((s for s in samples if s.get("parsed_action", {}).get("kind") == "final"
                      and str(s.get("model_message", {}).get("content") or "").strip()), None)
        if final:
            chosen_message = final["model_message"]
            chosen_source = "Dynamic-v1 observed final at same state"
        else:
            # No language verifier exists. This normalized, non-hallucinating
            # completion is allowed only after exact final-state verification.
            summary = "; ".join(state["gold_tool_names"])
            chosen_message = {"role": "assistant", "content":
                              f"The requested workflow is complete. Successful tools: {summary}."}
            chosen_source = "deterministic normalized final after verifier PASS"
        if not state["final_state_verifier_pass"] or chosen_message.get("tool_calls"):
            raise ValueError("invalid stop chosen")
        used = set()
        kept = []
        for sample in samples:
            act = sample.get("parsed_action") or {}
            if act.get("kind") != "tool":
                continue
            signature = stable(act)
            if signature in used:
                continue
            used.add(signature)
            kept.append(sample)
            if len(kept) == 2:
                break
        for sample in kept:
            act = sample["parsed_action"]
            last = state["last_gold_action"]
            failure = "repeat_tool" if act["name"] == last["name"] else "extra_tool_after_completion"
            if act["name"] == last["name"] and stable(act["arguments"]) == stable(last["arguments"]):
                failure = "useless_retry"
            pairs.append({
                "schema_version": SCHEMA,
                "pair_id": f"balanced-v2-{state['state_id']}-{sample['sample_index']}",
                "task_id": state["task_id"], "state_id": state["state_id"],
                "state_type": "stop_required", "prompt_messages": state["prompt_messages"],
                "prompt": state["prompt_messages"], "tools": state["tools"],
                "chosen": [chosen_message], "rejected": [sample["model_message"]],
                "chosen_action": {"kind": "final", "content": chosen_message["content"]},
                "rejected_action": act, "chosen_action_type": "final_answer",
                "rejected_action_type": "tool_call", "producer": state["last_gold_action"]["name"],
                "consumer": "unknown", "next_gold_node": "final_answer",
                "dependency_depth": state["dependency_depth"],
                "environment_identifiers": state["environment_identifiers"],
                "failure_type": failure, "chosen_valid": True,
                "chosen_source": chosen_source, "final_state_verifier_pass": True,
                "rejected_from_dynamic_v1": True, "sampling_seed": sample["sampling_seed"],
                "source_state_signature": state["state_signature"], "frozen_300_overlap": False,
            })
    return pairs


def spaced_select(pool: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Select without synthetic duplicates; one pair per state before seconds."""
    by_state = defaultdict(list)
    for pair in pool:
        by_state[pair["state_id"]].append(pair)
    order = sorted(by_state, key=lambda s: digest([seed, by_state[s][0]["task_id"], s]))
    chosen = []
    for round_index in range(2):
        for state in order:
            choices = sorted(by_state[state], key=lambda p: digest([seed, p["failure_type"], p["pair_id"]]))
            if len(choices) > round_index and len(chosen) < n:
                chosen.append(choices[round_index])
    return chosen


def build() -> dict[str, Any]:
    manifest = frozen()
    if (OUT / "manifest.json").exists():
        return read(OUT / "manifest.json")
    prior_audit = read(OUT / "audit.json") if (OUT / "audit.json").exists() else None
    if prior_audit and prior_audit["verdict"] != "DATA_GATE_FAILED":
        raise ValueError("unexpected existing audit without frozen manifest")
    if prior_audit:
        write_json(OUT / "audit_initial_gate_failed.json", prior_audit)
    candidates = {}
    for split in ("train", "preference_val"):
        candidates[split] = type_ac_pairs(split) + type_b_pairs(split)
        if len({p["pair_id"] for p in candidates[split]}) != len(candidates[split]):
            raise ValueError("duplicate pair ID")
    # Original preference-val has no observed stop error even at K=8. Move
    # exactly one whole TRAIN task, with one real stop negative, into val.
    # This is locked before any DPO training; independent heldout is unchanged.
    moved = [p for p in candidates["train"] if p["task_id"] == PROMOTED_STOP_VALIDATION_TASK]
    if sum(p["state_type"] == "stop_required" for p in moved) != 1:
        raise ValueError("promoted validation task no longer has exactly one real stop negative")
    candidates["train"] = [p for p in candidates["train"]
                           if p["task_id"] != PROMOTED_STOP_VALIDATION_TASK]
    candidates["preference_val"].extend(moved)
    train_by_type = defaultdict(list)
    for pair in candidates["train"]:
        train_by_type[pair["state_type"]].append(pair)
    b_pool = train_by_type["stop_required"]
    # Exactly 64 distinct selected pairs give two complete balanced epochs at
    # 64 steps x two GPUs x batch one, with no duplicated data-file rows.
    total = 64
    b_target = min(21, len(b_pool))
    a_target = 29
    c_target = total - a_target - b_target
    selected = (spaced_select(train_by_type["continue_required"], a_target, SEED)
                + spaced_select(b_pool, b_target, SEED + 1)
                + spaced_select(train_by_type["downstream_continue"], c_target, SEED + 2))
    if len(selected) != total:
        raise ValueError("insufficient nonduplicated A/B/C candidates")
    # Interleave type-aware strata before the trainer's per-epoch shuffle.
    selected.sort(key=lambda p: digest([SEED, p["pair_id"]]))
    val = sorted(candidates["preference_val"], key=lambda p: digest([SEED, p["pair_id"]]))
    train_types = Counter(p["state_type"] for p in selected)
    val_types = Counter(p["state_type"] for p in val)
    b_states = len({p["state_id"] for p in selected if p["state_type"] == "stop_required"})
    c_states = len({p["state_id"] for p in selected if p["state_type"] == "downstream_continue"})
    unique_states = len({p["state_id"] for p in selected})
    final_ratio = train_types["stop_required"] / total
    chosen_valid = all(p["chosen_valid"] is True for p in selected + val)
    split_tasks = set(p["task_id"] for p in selected) & set(p["task_id"] for p in val)
    valid = (unique_states >= 48 and total >= 64 and all(train_types[t] > 0 for t in
            ("continue_required", "stop_required", "downstream_continue"))
            and (b_states >= 16 or final_ratio >= 0.20)
            and final_ratio > 0.20 and 1 - final_ratio < 0.80 and c_states >= 8
            and chosen_valid and not split_tasks and manifest["frozen300_exact_overlap"] == 0
            and all(val_types[t] > 0 for t in
                    ("continue_required", "stop_required", "downstream_continue")))
    audit = {"schema_version": SCHEMA, "verdict": "DATA_READY" if valid else "DATA_GATE_FAILED",
             "train_candidate_counts": dict(Counter(p["state_type"] for p in candidates["train"])),
             "val_candidate_counts": dict(Counter(p["state_type"] for p in val)),
             "train_selected_counts": dict(train_types), "train_pairs": total,
             "train_unique_states": unique_states, "train_unique_tasks": len({p["task_id"] for p in selected}),
             "stop_unique_states": b_states, "downstream_unique_states": c_states,
             "chosen_tool_call_rate": 1 - final_ratio, "chosen_final_answer_rate": final_ratio,
             "chosen_valid_rate": sum(p["chosen_valid"] is True for p in selected) / total,
             "state_pair_cap": max(Counter(p["state_id"] for p in selected).values()),
             "split_task_overlap": len(split_tasks), "frozen300_train_overlap": 0,
             "validation_promoted_task_ids": [PROMOTED_STOP_VALIDATION_TASK],
             "failure_types": dict(Counter(p["failure_type"] for p in selected)),
             "chosen_sources": dict(Counter(p.get("chosen_source", "gold executable") for p in selected)),
             "stop_sampling": {s: stop_candidate_count(s) for s in ("train", "preference_val")},
             "warning": "downstream below 15% target" if train_types["downstream_continue"] / total < 0.15 else None}
    write_json(OUT / "audit.json", audit, replace=bool(prior_audit))
    if not valid:
        return audit
    write_rows(OUT / "pairs_all.jsonl", candidates["train"])
    write_rows(OUT / "pairs_train.jsonl", selected)
    write_rows(OUT / "pairs_val.jsonl", val)
    for p in selected + val:
        if not (p["rejected_from_dynamic_v1"] is True and p["frozen_300_overlap"] is False):
            raise ValueError("rejected provenance/contamination failure")
    output_manifest = {"schema_version": SCHEMA, "verdict": "DATA_READY",
                       "initialization_model": str(MODEL), "source_data_sha256": manifest["source_data_sha256"],
                       "source_gold_sha256": manifest["source_gold_sha256"],
                       "source_task_split_sha256": digest(manifest["task_ids"]),
                       "selection_seed": SEED, "train_pair_count": len(selected),
                       "validation_promoted_task_ids": [PROMOTED_STOP_VALIDATION_TASK],
                       "validation_pair_count": len(val), "train_type_counts": dict(train_types),
                       "val_type_counts": dict(val_types), "frozen300_train_overlap": 0,
                       "hashes": {name: file_sha256(OUT / name) for name in
                                  ("pairs_all.jsonl", "pairs_train.jsonl", "pairs_val.jsonl")}}
    write_json(OUT / "manifest.json", output_manifest)
    return output_manifest


def materialize() -> dict[str, Any]:
    from transformers import AutoTokenizer
    from repro_1p7b.graph_frontier.materialize_preference_dpo import serialize_pair

    manifest = read(OUT / "manifest.json")
    if manifest["verdict"] != "DATA_READY":
        raise ValueError("data gate did not pass")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    serialized = {}
    drops = Counter()
    for name in ("train", "val"):
        values = []
        for pair in rows(OUT / f"pairs_{name}.jsonl"):
            item, reason = serialize_pair(tokenizer, pair, 8192)
            if reason:
                drops[f"{name}:{reason}"] += 1
            else:
                item["state_type"] = pair["state_type"]
                values.append(item)
        serialized[name] = values
    # Deterministic 8/8/8 smoke, never a random single-type prefix.
    smoke = []
    for t in ("continue_required", "stop_required", "downstream_continue"):
        smoke += [r for r in serialized["train"] if r["state_type"] == t][:8]
    ready = (not drops and len(serialized["train"]) == manifest["train_pair_count"]
             and len(serialized["val"]) == manifest["validation_pair_count"]
             and len(smoke) == 24 and all(Counter(r["state_type"] for r in serialized["val"])[t] > 0
                                   for t in ("continue_required", "stop_required", "downstream_continue")))
    audit = {"schema_version": SCHEMA, "verdict": "DPO_READY" if ready else "SERIALIZATION_GATE_FAILED",
             "train_rows": len(serialized["train"]), "val_rows": len(serialized["val"]),
             "smoke_rows": len(smoke), "smoke_types": dict(Counter(r["state_type"] for r in smoke)),
             "drops": dict(drops), "max_length": 8192}
    write_json(OUT / "serialization_audit.json", audit)
    if ready:
        write_rows(OUT / "dpo_train.jsonl", serialized["train"])
        write_rows(OUT / "dpo_val.jsonl", serialized["val"])
        write_rows(OUT / "dpo_smoke24.jsonl", smoke)
        audit["hashes"] = {name: file_sha256(OUT / name) for name in
                           ("dpo_train.jsonl", "dpo_val.jsonl", "dpo_smoke24.jsonl")}
        write_json(OUT / "serialization_audit.json", audit, replace=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "sample-stop", "count-stop", "build", "materialize"))
    parser.add_argument("--split", choices=("train", "preference_val"), default="train")
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default="dynamic-v1")
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-k", type=int, default=4)
    parser.add_argument("--only-missing", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare()
    elif args.command == "sample-stop":
        if not args.endpoint:
            parser.error("sample-stop requires endpoint")
        report = sample_stop(args.split, args.endpoint, args.model, args.shard_count,
                             args.shard_index, args.target_k, args.only_missing)
    elif args.command == "count-stop":
        report = stop_candidate_count(args.split)
    elif args.command == "build":
        report = build()
    else:
        report = materialize()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

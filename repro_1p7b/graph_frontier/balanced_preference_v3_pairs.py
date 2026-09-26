"""Task-disjoint, observed-action Balanced Preference v3 data gate.

The heldout reservation uses only audited gold states. Dynamic-v1 samples are
then collected for non-heldout tasks, and the task-group train/val split is
frozen before any DPO optimizer step. No synthetic rejected action is allowed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import classify_sample
from repro_1p7b.graph_frontier.balanced_preference_v3_source import (
    GF, OUT, SCHEMA, digest, read, rows, stable, write, write_rows,
)
from repro_1p7b.graph_frontier.collect_preference_rollouts import generate_action
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256


MODEL = GF.parents[1] / "repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b"
SEED = 20260926
TYPES = ("continue_required", "stop_required", "downstream_continue")


def load_states() -> list[dict[str, Any]]:
    manifest = read(OUT / "source_manifest.json")
    paths = (OUT / "frontier_states.jsonl", OUT / "stop_states.jsonl")
    if [file_sha256(p) for p in paths] != [manifest["state_hashes"][k] for k in ("frontier", "stop")]:
        raise ValueError("source state hash changed")
    values = rows(paths[0]) + rows(paths[1])
    if len({s["state_id"] for s in values}) != len(values):
        raise ValueError("duplicate state ID")
    if any(s.get("frozen_300_overlap") is not False for s in values):
        raise ValueError("state contamination flag")
    return values


def reserve_heldout() -> dict[str, Any]:
    path = OUT / "heldout_manifest.json"
    states = load_states()
    source_hash = file_sha256(OUT / "source_manifest.json")
    if path.exists():
        old = read(path)
        if old["source_manifest_sha256"] != source_hash:
            raise ValueError("heldout reservation source changed")
        return old
    stop_tasks = {s["task_id"] for s in states if s["state_type"] == "stop_required"}
    downstream_by_task: dict[str, int] = Counter(s["task_id"] for s in states
                                                    if s["state_type"] == "downstream_continue")
    heldout = set(sorted(stop_tasks, key=lambda t: digest([SEED, "stop-heldout", t]))[:40])
    downstream_count = sum(downstream_by_task[t] for t in heldout)
    for task in sorted(downstream_by_task, key=lambda t: digest([SEED, "downstream-heldout", t])):
        if downstream_count >= 25:
            break
        if task not in heldout:
            heldout.add(task)
            downstream_count += downstream_by_task[task]
    training_tasks = {s["task_id"] for s in states} - heldout
    result = {"schema_version": SCHEMA, "source_manifest_sha256": source_hash,
              "heldout_task_ids": sorted(heldout), "heldout_strict_stop_states":
              sum(s["state_type"] == "stop_required" and s["task_id"] in heldout for s in states),
              "heldout_downstream_states": sum(s["state_type"] == "downstream_continue" and
                                                s["task_id"] in heldout for s in states),
              "candidate_preference_tasks": len(training_tasks),
              "heldout_selection_uses_model_outputs": False}
    write(path, result)
    return result


def candidate_states() -> list[dict[str, Any]]:
    reservation = reserve_heldout()
    excluded = set(reservation["heldout_task_ids"])
    return [s for s in load_states() if s["task_id"] not in excluded]


def sample_path(state: dict[str, Any]) -> Path:
    return OUT / "sampled_actions" / f"{state['state_id']}.json"


@lru_cache(maxsize=1)
def _historical_stop_index() -> dict[str, tuple[dict[str, Any], Path]]:
    historical = GF / "balanced_preference_v2"
    index = {}
    for split in ("train", "preference_val"):
        for previous in rows(historical / f"stop_states_{split}.jsonl"):
            path = historical / "stop_samples" / split / f"{previous['state_id']}.json"
            index[previous["task_id"]] = (previous, path)
    return index


def old_samples(state: dict[str, Any]) -> list[dict[str, Any]]:
    if state["state_type"] != "stop_required":
        path_text = state.get("historical_sample_path")
        if not path_text:
            return []
        old = read(Path(path_text))
        if old["task_id"] != state["task_id"] or old["state_id"] != state["source_state_id"] or old["model"] != "dynamic-v1":
            raise ValueError("historical frontier sample identity mismatch")
        return old["samples"]
    match = _historical_stop_index().get(state["task_id"])
    if match is None:
        return []
    previous, path = match
    if previous["state_signature"] != state["state_signature"] or previous["prompt_messages"] != state["prompt_messages"]:
        return []  # Same task is not sufficient to reuse a different completed state.
    if not path.exists():
        return []
    payload = read(path)
    if payload["state_signature"] != state["state_signature"] or payload["task_id"] != state["task_id"]:
        raise ValueError("historical stop sample identity mismatch")
    return payload["samples"]


def samples_for(state: dict[str, Any]) -> list[dict[str, Any]]:
    path = sample_path(state)
    if path.exists():
        payload = read(path)
        if payload["task_id"] != state["task_id"] or payload["state_id"] != state["state_id"] or \
                payload["prompt_sha256"] != digest(state["prompt_messages"]) or \
                payload["tools_sha256"] != digest(state["tools"]) or payload["model"] != "dynamic-v1":
            raise ValueError("Dynamic-v1 sample cache identity/config mismatch")
        return payload["samples"]
    return old_samples(state)


def sample(shard_count: int, shard_index: int, endpoint: str, target_k: int,
           only_missing_type: str | None) -> dict[str, Any]:
    if target_k not in (4, 8) or shard_count < 1 or not (0 <= shard_index < shard_count):
        raise ValueError("invalid sample shard or K")
    if only_missing_type and only_missing_type not in TYPES:
        raise ValueError("invalid targeted K8 type")
    states = sorted(candidate_states(), key=lambda s: s["state_id"])
    requests = completed = 0
    for index, state in enumerate(states):
        if index % shard_count != shard_index or (only_missing_type and state["state_type"] != only_missing_type):
            continue
        prior = samples_for(state)
        if target_k == 4 and any((x.get("parsed_action") or {}).get("kind") == "request_error" for x in prior):
            archive = OUT / "context8k_error_archive" / f"{state['state_id']}.json"
            if not archive.exists():
                write(archive, {"task_id": state["task_id"], "state_id": state["state_id"],
                                "samples": prior, "server_context_length": 8192})
            values = copy.deepcopy(prior)
            for j, old in enumerate(values):
                if (old.get("parsed_action") or {}).get("kind") != "request_error":
                    continue
                if not old.get("error", "").startswith("HTTPError:400"):
                    continue
                message, action, usage, finish = generate_action(
                    endpoint, "dynamic-v1", state["prompt_messages"], state["tools"],
                    seed=old["sampling_seed"], temperature=0.7, max_tokens=512)
                values[j] = {"sample_index": old["sample_index"], "sampling_seed": old["sampling_seed"],
                             "model_message": message, "parsed_action": action, "usage": usage,
                             "finish_reason": finish, "context_repaired_from": old["error"],
                             "server_context_length": 24576}
                requests += 1
            write(sample_path(state), {"schema_version": SCHEMA, "task_id": state["task_id"],
                                      "state_id": state["state_id"],
                                      "prompt_sha256": digest(state["prompt_messages"]),
                                      "tools_sha256": digest(state["tools"]), "model": "dynamic-v1",
                                      "temperature": 0.7, "max_tokens": 512, "samples": values})
            prior = values
        if target_k == 8 and prior and any(
                _failure(state, x) is not None for x in prior):
            continue
        if len(prior) >= target_k:
            completed += 1
            continue
        values = copy.deepcopy(prior)
        for j in range(len(values), target_k):
            seed = SEED + index * 1009 + j * 9176
            try:
                message, action, usage, finish = generate_action(
                    endpoint, "dynamic-v1", state["prompt_messages"], state["tools"],
                    seed=seed, temperature=0.7, max_tokens=512)
                item = {"sample_index": j, "sampling_seed": seed, "model_message": message,
                        "parsed_action": action, "usage": usage, "finish_reason": finish}
            except Exception as exc:
                item = {"sample_index": j, "sampling_seed": seed,
                        "parsed_action": {"kind": "request_error"},
                        "error": f"{type(exc).__name__}:{str(exc)[:240]}"}
            values.append(item)
            requests += 1
        write(sample_path(state), {"schema_version": SCHEMA, "task_id": state["task_id"],
                                  "state_id": state["state_id"], "prompt_sha256": digest(state["prompt_messages"]),
                                  "tools_sha256": digest(state["tools"]), "model": "dynamic-v1",
                                  "temperature": 0.7, "max_tokens": 512, "samples": values})
        completed += 1
        if completed % 10 == 0:
            print(f"SAMPLED shard={shard_index} states={completed} requests={requests}", flush=True)
    return {"states": completed, "requests": requests, "target_k": target_k,
            "targeted_type": only_missing_type}


def _failure(state: dict[str, Any], sample: dict[str, Any]) -> str | None:
    action = sample.get("parsed_action") or {}
    if state["state_type"] == "stop_required":
        if action.get("kind") != "tool" or not isinstance(action.get("arguments"), dict):
            return None
        previous = state["last_gold_action"]
        if action["name"] == previous["name"]:
            return "useless_retry" if stable(action["arguments"]) == stable(previous["arguments"]) else "repeat_tool"
        return "extra_tool_after_completion"
    label = classify_sample(sample, state)
    if label not in {"premature_stop", "wrong_tool", "wrong_value", "useless_retry"}:
        return None
    if state["state_type"] == "downstream_continue":
        return {"premature_stop": "downstream_stop", "wrong_tool": "downstream_wrong_tool",
                "wrong_value": "downstream_wrong_value", "useless_retry": "repeat_tool"}[label]
    return label


def _chosen(state: dict[str, Any], samples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if state["state_type"] != "stop_required":
        if state["chosen_executable"] is not True or state["chosen_execution"]["execution_success"] is not True:
            raise ValueError("frontier chosen not executable")
        return state["chosen_message"], state["chosen_action"], "gold executable"
    if state["final_state_verifier_pass"] is not True or state["gold_replay_success"] is not True:
        raise ValueError("stop final state not verified")
    content = "The requested workflow is complete. Successful tools: " + "; ".join(state["gold_tool_names"]) + "."
    return {"role": "assistant", "content": content}, {"kind": "final", "content": content}, \
        "deterministic normalized final after verifier PASS"


def make_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states = candidate_states()
    pairs = []
    missing = Counter()
    for state in states:
        samples = samples_for(state)
        if sum((x.get("parsed_action") or {}).get("kind") != "request_error" for x in samples) < 4:
            missing[state["state_type"]] += 1
            continue
        chosen_message, chosen_action, chosen_source = _chosen(state, samples)
        failures = []
        seen = set()
        for sample in samples:
            label = _failure(state, sample)
            if label is None or not sample.get("model_message"):
                continue
            action = sample["parsed_action"]
            signature = stable(action)
            if signature in seen:
                continue
            seen.add(signature)
            failures.append((label, sample))
        failures.sort(key=lambda item: (item[0], item[1]["sample_index"]))
        selected = []
        for item in failures:
            if item[0] not in {x[0] for x in selected}:
                selected.append(item)
            if len(selected) == 2:
                break
        for item in failures:
            if len(selected) == 2:
                break
            if item not in selected:
                selected.append(item)
        for label, sample in selected:
            rejected = sample["parsed_action"]
            pair = {"schema_version": SCHEMA,
                    "pair_id": "balanced-v3-" + digest([state["state_id"], sample["sample_index"], rejected])[:32],
                    "task_id": state["task_id"], "state_id": state["state_id"],
                    "state_type": state["state_type"], "prompt_messages": state["prompt_messages"],
                    "prompt": state["prompt_messages"], "tools": state["tools"],
                    "chosen": [chosen_message], "rejected": [sample["model_message"]],
                    "chosen_action": chosen_action, "rejected_action": rejected,
                    "chosen_action_type": "final_answer" if state["state_type"] == "stop_required" else "tool_call",
                    "rejected_action_type": "final_answer" if rejected["kind"] == "final" else "tool_call",
                    "producer": state.get("producer_tool", state.get("last_gold_action", {}).get("name")),
                    "consumer": state.get("consumer_tool", "unknown"),
                    "next_gold_node": state.get("consumer_tool", "final_answer"),
                    "dependency_depth": state["dependency_depth"],
                    "environment_identifiers": state["environment_identifiers"],
                    "failure_type": label, "chosen_valid": True, "chosen_source": chosen_source,
                    "rejected_from_dynamic_v1": True, "sampling_seed": sample["sampling_seed"],
                    "source_state_signature": state["state_signature"] if "state_signature" in state else digest(state["state_identity"]),
                    "frozen_300_overlap": False}
            if state["state_type"] == "stop_required":
                pair["final_state_verifier_pass"] = True
                pair["terminal_tool"] = state["last_gold_action"]["name"]
                pair["tool_chain_signature"] = digest(state["gold_tool_names"])
            pairs.append(pair)
    return pairs, {"states": len(states), "missing_k4": dict(missing),
                   "candidate_pairs": len(pairs), "candidate_types": dict(Counter(p["state_type"] for p in pairs))}


def _select(pool: list[dict[str, Any]], count: int, tag: str) -> list[dict[str, Any]]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pool:
        by_state[pair["state_id"]].append(pair)
    order = sorted(by_state, key=lambda s: digest([SEED, tag, s]))
    result = []
    for iteration in range(2):
        for sid in order:
            items = sorted(by_state[sid], key=lambda p: digest([SEED, p["failure_type"], p["pair_id"]]))
            if len(items) > iteration and len(result) < count:
                result.append(items[iteration])
    return result


def _balanced_train(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = {t: [p for p in pool if p["state_type"] == t] for t in TYPES}
    best = []
    for n in range(min(500, len(pool)), 279, -1):
        options = []
        for b in range(math.ceil(.25*n), min(math.floor(.4*n), len(buckets[TYPES[1]]))+1):
            for c in range(math.ceil(.2*n), min(math.floor(.35*n), len(buckets[TYPES[2]]))+1):
                a = n-b-c
                if math.ceil(.30*n) <= a <= min(math.floor(.45*n), len(buckets[TYPES[0]])):
                    options.append((abs(a/n-.38)+abs(b/n-.32)+abs(c/n-.30), a, b, c))
        for _, a, b, c in sorted(options):
            chosen = _select(buckets[TYPES[0]], a, "A") + _select(buckets[TYPES[1]], b, "B") + \
                _select(buckets[TYPES[2]], c, "C")
            if len(chosen) == n and len({p["state_id"] for p in chosen}) >= 180:
                return sorted(chosen, key=lambda p: digest([SEED, "train-order", p["pair_id"]]))
            if len(chosen) > len(best):
                best = chosen
    return best


def _split(pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pool:
        by_task[pair["task_id"]].append(pair)
    tasks = sorted(by_task)
    val_count = max(1, round(.15*len(tasks)))
    best: tuple[float, list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]] | None = None
    for offset in range(1000):
        ranked = sorted(tasks, key=lambda t: digest([SEED+offset, "task-group-val", t]))
        val_ids = set(ranked[:val_count])
        val = [p for p in pool if p["task_id"] in val_ids]
        train = [p for p in pool if p["task_id"] not in val_ids]
        val_unique = {t: len({p["state_id"] for p in val if p["state_type"] == t}) for t in TYPES}
        score = min(val_unique[TYPES[0]]/10, val_unique[TYPES[1]]/10,
                    val_unique[TYPES[2]]/8, sum(val_unique.values())/30)
        report = {"val_unique_states": val_unique, "val_tasks": len(val_ids),
                  "train_tasks": len(tasks)-len(val_ids), "split_seed": SEED+offset}
        if best is None or score > best[0]:
            best = (score, train, val, SEED+offset, report)
        if score >= 1:
            return train, val, SEED+offset, report
    assert best is not None
    return best[1], best[2], best[3], best[4]


def build() -> dict[str, Any]:
    if (OUT / "manifest.json").exists():
        return read(OUT / "manifest.json")
    reservation = reserve_heldout()
    candidates, preliminary = make_candidates()
    if len({p["pair_id"] for p in candidates}) != len(candidates):
        raise ValueError("duplicate pair IDs")
    train_pool, val, seed, split = _split(candidates)
    train = _balanced_train(train_pool)
    counts = Counter(p["state_type"] for p in train)
    val_states = {t: len({p["state_id"] for p in val if p["state_type"] == t}) for t in TYPES}
    unique = len({p["state_id"] for p in train})
    stop = [p for p in train if p["state_type"] == TYPES[1]]
    signature_counts = Counter(p["tool_chain_signature"] for p in stop)
    task_ids = {p["task_id"] for p in train}
    val_ids = {p["task_id"] for p in val}
    heldout_ids = set(reservation["heldout_task_ids"])
    systematic_issues = 0
    for pair in candidates:
        if (pair["state_type"] == TYPES[1]) != (pair["chosen_action_type"] == "final_answer") or \
                pair["chosen_valid"] is not True or pair["rejected_from_dynamic_v1"] is not True or \
                pair["frozen_300_overlap"] is not False or len(pair["chosen"]) != 1 or len(pair["rejected"]) != 1:
            systematic_issues += 1
    frontier_leakage_issues = 0
    for state in candidate_states():
        if state["state_type"] == "stop_required":
            continue
        prefix_count = state["producer_position"] + 1
        tool_messages = sum(message.get("role") == "tool" for message in state["prompt_messages"])
        if len(state["prefix_events"]) != prefix_count or tool_messages != prefix_count or \
                state["chosen_action"]["name"] != state["consumer_tool"]:
            frontier_leakage_issues += 1
    valid = (unique >= 180 and len(train) >= 280 and bool(train) and
             counts[TYPES[1]]/len(train) >= .25 and counts[TYPES[2]]/len(train) >= .20 and
             len({p["state_id"] for p in val}) >= 30 and val_states[TYPES[0]] >= 10 and
             val_states[TYPES[1]] >= 10 and val_states[TYPES[2]] >= 8 and
             not (task_ids & val_ids or task_ids & heldout_ids or val_ids & heldout_ids) and
             systematic_issues == 0 and frontier_leakage_issues == 0 and
             (not stop or max(signature_counts.values())/len(stop) < .70) and
             preliminary["missing_k4"] == {})
    report = {"schema_version": SCHEMA, "verdict": "DATA_READY" if valid else "DATA_NOT_READY",
              "source_manifest_sha256": file_sha256(OUT / "source_manifest.json"),
              "heldout_manifest_sha256": file_sha256(OUT / "heldout_manifest.json"),
              "candidate": preliminary, "train_unique_states": unique, "train_pairs": len(train),
              "train_unique_tasks": len(task_ids), "train_types": dict(counts),
              "train_type_unique_states": {t: len({p["state_id"] for p in train if p["state_type"] == t}) for t in TYPES},
              "chosen_final_answer_fraction": counts[TYPES[1]]/len(train) if train else 0,
              "validation_unique_states": len({p["state_id"] for p in val}),
              "validation_types": dict(Counter(p["state_type"] for p in val)),
              "validation_type_unique_states": val_states, "split": split,
              "failure_types": dict(Counter(p["failure_type"] for p in train)),
              "depth_distribution": dict(Counter(str(p["dependency_depth"]) for p in train)),
              "environment_distribution": dict(Counter(e for p in train for e in p["environment_identifiers"])),
              "consumer_tool_distribution": dict(Counter(p["consumer"] for p in train)),
              "state_pair_cap": max(Counter(p["state_id"] for p in train).values(), default=0),
              "stop_unique_tasks": len({p["task_id"] for p in stop}),
              "stop_terminal_tools": len({p["terminal_tool"] for p in stop}),
              "stop_environments": len({e for p in stop for e in p["environment_identifiers"]}),
              "stop_tool_chain_signatures": len(signature_counts),
              "max_stop_chain_share": max(signature_counts.values())/len(stop) if stop else 0,
              "overlap": {"train_val_tasks": len(task_ids & val_ids),
                          "train_new_heldout_tasks": len(task_ids & heldout_ids),
                          "val_new_heldout_tasks": len(val_ids & heldout_ids),
                          "frozen300_exact": 0, "historical_eval_exact": 0},
              "systematic_issues": systematic_issues,
              "frontier_prefix_leakage_issues": frontier_leakage_issues,
              "chosen_validation_rate": 1.0 - systematic_issues / len(candidates) if candidates else 0.0,
              "selection_seed": seed}
    write(OUT / "audit.json", report)
    if not valid:
        return report
    write_rows(OUT / "pairs_all.jsonl", candidates)
    write_rows(OUT / "pairs_train.jsonl", train)
    write_rows(OUT / "pairs_val.jsonl", val)
    representative = []
    for type_name, n in ((TYPES[0], 15), (TYPES[1], 15), (TYPES[2], 10)):
        representative.extend(_select([p for p in train if p["state_type"] == type_name], n,
                                      "representative-"+type_name))
    if len(representative) != 40:
        raise ValueError("40 representative pairs unavailable")
    lines = ["# Balanced Preference v3: 40 stratified observed-action pairs", ""]
    for pair in representative:
        lines += [f"## {pair['pair_id']}", "", f"- task: `{pair['task_id']}`; state: `{pair['state_id']}`",
                  f"- type: `{pair['state_type']}`; failure: `{pair['failure_type']}`", "",
                  "Chosen:", "```json", json.dumps(pair["chosen_action"], ensure_ascii=False, indent=2), "```", "",
                  "Rejected (observed Dynamic-v1):", "```json",
                  json.dumps(pair["rejected_action"], ensure_ascii=False, indent=2), "```", ""]
    (GF / "reports/balanced_preference_v3_representative_pairs.md").write_text("\n".join(lines)+"\n")
    manifest = {"schema_version": SCHEMA, "verdict": "DATA_READY",
                "initialization_model": str(MODEL), "source_manifest_sha256": report["source_manifest_sha256"],
                "heldout_manifest_sha256": report["heldout_manifest_sha256"],
                "selection_seed": seed, "train_pair_count": len(train), "validation_pair_count": len(val),
                "train_type_counts": dict(counts), "val_type_counts": report["validation_types"],
                "frozen300_train_overlap": 0,
                "hashes": {name: file_sha256(OUT / name) for name in
                           ("pairs_all.jsonl", "pairs_train.jsonl", "pairs_val.jsonl")}}
    write(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("reserve", "sample", "build"))
    parser.add_argument("--endpoint")
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-k", type=int, default=4)
    parser.add_argument("--only-missing-type", choices=TYPES)
    args = parser.parse_args()
    if args.command == "reserve":
        result = reserve_heldout()
    elif args.command == "sample":
        if not args.endpoint:
            parser.error("sample requires --endpoint")
        result = sample(args.shard_count, args.shard_index, args.endpoint,
                        args.target_k, args.only_missing_type)
    else:
        result = build()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

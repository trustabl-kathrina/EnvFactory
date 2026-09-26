"""Same clean-task rollout/verifier for the final fixed-gold SFT policy."""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics

from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import atomic_json, immutable_json
from repro_1p7b.graph_frontier.official_rl_audit_static import OUT, SOURCE
from repro_1p7b.graph_frontier.recursive_opd_v1.mine import classify, result
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import CONFIG, PROTOCOL, RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN, collect_one

EVAL = RUN / "static/evaluation"
MODEL = RUN / "static/cycle3/checkpoint"


def manifest() -> dict:
    protocol = check()
    state = json.loads((MODEL / "trainer_state.json").read_text())
    weight = sha256(MODEL / "model.safetensors")
    if state["status"] != "TRAINED" or state["static_phase"] != 2 or state["model_sha256"] != weight:
        raise RuntimeError("static pi3 checkpoint invalid")
    value = {"schema_version": "recursive_opd_static_eval_v1",
             "model_path": str(MODEL), "model_sha256": weight,
             "protocol_sha256": sha256(PROTOCOL), "config": CONFIG,
             "served_model": "recursive-opd-static-pi3",
             "task_count": len(protocol["train_task_ids"])+len(protocol["heldout_task_ids"])}
    path = EVAL / "rollout_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("static eval manifest drift")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stable(value)+"\n")
    return value


def worker(shard: int, endpoint: str) -> dict:
    if shard not in (0, 1):
        raise ValueError("two static eval shards required")
    lock = manifest()
    protocol = check()
    source = json.loads(SOURCE.read_text())
    by_id = {r["audit_id"]: r for r in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    ids = protocol["train_task_ids"]+protocol["heldout_task_ids"]
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for position, task in enumerate(ids):
        if position % 2 != shard:
            continue
        path = EVAL / "rollouts" / f"{task}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["model_sha256"] != lock["model_sha256"] or old["task_id"] != task:
                raise RuntimeError("static persisted identity drift")
            counts["skipped"] += 1
            continue
        entry = by_id[task]
        raw = source[entry["source_row_index"]]
        gold_source = GOLD_RUN / "episodes" / f"{task}.json"
        gold_episode = json.loads(gold_source.read_text())
        source_kind = "historical"
        if not valid_gold(entry, raw, gold_episode):
            cache_path = RUN / "gold_cache" / f"{task}.json"
            cache = json.loads(cache_path.read_text())
            if cache["task_id"] != task or not isinstance(cache["schemas"], list):
                raise RuntimeError("static cached gold identity drift")
            gold_episode = {**gold_episode, "gold": cache["gold"]}
            if not valid_gold(entry, raw, gold_episode):
                raise RuntimeError("static cached gold replay invalid")
            schemas = cache["schemas"]
            schema_sha = immutable_json(RUN / "schemas", schemas)
            gold_source, source_kind = cache_path, "independent_replay_cache"
        else:
            schemas = schema_for(GOLD_RUN, gold_episode)
            schema_sha = gold_episode["schema_sha256"]
        episode = {"task_id": task, "pool": "train" if task in set(protocol["train_task_ids"]) else "heldout",
                   "model_sha256": lock["model_sha256"],
                   "manifest_sha256": sha256(EVAL / "rollout_manifest.json"),
                   "gold_source_episode_sha256": sha256(gold_source),
                   "gold_source_kind": source_kind, "schema_sha256": schema_sha,
                   "gold": gold_episode["gold"], "student": None, "student_error": None}
        try:
            episode["student"] = asyncio.run(collect_one(entry, raw, lock["served_model"], endpoint,
                                                         gold_episode, schemas))
        except Exception as exc:
            episode["student_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            counts["failed"] += 1
        atomic_json(path, episode)
        counts["completed"] += 1
        if counts["completed"] % 10 == 0 or episode["student_error"]:
            print(stable({"task_id": task, "shard": shard, "counts": counts}), flush=True)
    return counts


def summarize() -> dict:
    lock = manifest()
    protocol = check()
    source = json.loads(SOURCE.read_text())
    by_id = {r["audit_id"]: r for r in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    ids = protocol["train_task_ids"]+protocol["heldout_task_ids"]
    records = []
    for position, task in enumerate(ids):
        episode = json.loads((EVAL / "rollouts" / f"{task}.json").read_text())
        entry = by_id[task]
        raw = source[entry["source_row_index"]]
        gold_source = (RUN / "gold_cache" / f"{task}.json" if
                       episode.get("gold_source_kind") == "independent_replay_cache"
                       else GOLD_RUN / "episodes" / f"{task}.json")
        if (episode["model_sha256"] != lock["model_sha256"]
                or episode["manifest_sha256"] != sha256(EVAL / "rollout_manifest.json")
                or episode["gold_source_episode_sha256"] != sha256(gold_source)
                or episode["pool"] != ("train" if position < 1726 else "heldout")
                or not valid_gold(entry, raw, episode)):
            raise RuntimeError(f"static eval source/gold drift: {task}")
        schema = schema_for(RUN if episode.get("gold_source_kind") == "independent_replay_cache"
                            else GOLD_RUN, episode)
        if episode["student_error"] or not episode["student"]:
            record = result("UNCERTAIN", 0, len(episode["gold"]["actions"]), reason="student runtime failure")
        else:
            try:
                record = asyncio.run(classify(episode, entry, raw, schema))
            except Exception as exc:
                record = result("UNCERTAIN", 0, len(episode["gold"]["actions"]),
                                reason=f"verifier exception:{type(exc).__name__}:{str(exc)[:200]}")
        record.update(task_id=task, pool=episode["pool"],
                      termination_reason=(episode.get("student") or {}).get("termination_reason"))
        records.append(record)
        if (position + 1) % 100 == 0:
            print(stable({"static_mined": position + 1, "of": 2158}), flush=True)
    output = EVAL / "metrics"
    if output.exists():
        raise RuntimeError("refusing to overwrite static metrics")
    metrics = {}
    for name in ("train", "heldout"):
        group = [r for r in records if r["pool"] == name]
        metrics[name] = {"tasks": len(group),
                         "success": sum(r["status"] == "SUCCESS" for r in group),
                         "first_action_accuracy": sum(r["frontier_depth"] > 0 for r in group)/len(group),
                         "mean_frontier_depth": statistics.mean(r["frontier_depth"] for r in group),
                         "median_frontier_depth": statistics.median(r["frontier_depth"] for r in group),
                         "mean_normalized_frontier": statistics.mean(r["normalized_frontier"] for r in group),
                         "start_error": sum(r["error_regime"] == "START" for r in group),
                         "advance_error": sum(r["error_regime"] == "ADVANCE" for r in group),
                         "terminate_error": sum(r["error_regime"] == "TERMINATE" for r in group),
                         "correct_termination": sum(r["status"] == "SUCCESS" for r in group),
                         "uncertain": sum(r["status"] == "UNCERTAIN" for r in group),
                         "max_turn_exhaustion": sum(r["termination_reason"] == "MAX_TURNS" for r in group)}
    if metrics["train"]["tasks"] != 1726 or metrics["heldout"]["tasks"] != 432:
        raise RuntimeError("static split drift")
    output.mkdir(parents=True)
    (output / "per_task.jsonl").write_text("".join(stable(r)+"\n" for r in records))
    answer = {"model_sha256": lock["model_sha256"], "metrics": metrics,
              "per_task_sha256": sha256(output / "per_task.jsonl"),
              "rollout_manifest_sha256": sha256(EVAL / "rollout_manifest.json")}
    (output / "summary.json").write_text(stable(answer)+"\n")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("manifest", "worker", "summarize"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.command == "manifest":
        answer = manifest()
    elif args.command == "worker":
        if args.shard is None or args.endpoint is None:
            raise ValueError("worker requires shard and endpoint")
        answer = worker(args.shard, args.endpoint)
    else:
        answer = summarize()
    print(stable(answer), flush=True)


if __name__ == "__main__":
    main()

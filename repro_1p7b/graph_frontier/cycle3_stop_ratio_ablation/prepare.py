"""Freeze nested 2:1 and 4:1 exposure schedules without changing sources.

Exactly 64 steps and two samples/step leave 128 sample slots; 85:43 and
102:26 are the nearest integer-exposure schedules to 2:1 and 4:1 while
holding total tokens/samples per update type and step budget fixed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_pilot.prepare import (
    FD_SOURCE, MODEL, OUT as DATA, RUN, TERM_SOURCE, _pick, read_jsonl,
)
from repro_1p7b.graph_frontier.cycle3_pilot.train import load_plan as old_load_plan
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "repro_1p7b/graph_frontier/cycle3_stop_ratio_ablation"
PROBE = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/evaluation/next_action_probe"
SEED = 20260926


def _extra(rows: list[dict], used: set[str], count: tuple[int, int, int],
           rng: random.Random) -> list[dict]:
    wrong = [r for r in rows if r["failure_type"] == "WRONG_ARGUMENT"]
    deep = [r for r in rows if r["gold_step"] >= 2]
    broad = [r for r in rows if r["failure_type"] != "WRONG_ARGUMENT"
             and r["gold_step"] == 1]
    chosen = (_pick(wrong, count[0], used, rng)
              + _pick(deep, count[1], used, rng)
              + _pick(broad, count[2], used, rng))
    rng.shuffle(chosen)
    return chosen


def _bit_reverse(value: int) -> int:
    return int(f"{value:06b}"[::-1], 2)


def freeze(output: Path = OUT) -> dict:
    if any((output / f"schedule_{ratio}.json").exists() for ratio in (2, 4)):
        raise RuntimeError("ratio schedule already frozen")
    original_pairs, original_plan = old_load_plan(DATA, MODEL, "formal")
    if (len(original_pairs) != 64
            or original_plan["initialization_sha256"] != "0231573c95e939a3dcc52518c15a7241edb32bcca65729f49230f593cec7b970"):
        raise RuntimeError("original Cycle-3 plan changed")
    old_result = json.loads((ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/formal_train/result.json").read_text())
    if old_result["status"] != "TRAINED" or old_result["steps"] != 64:
        raise RuntimeError("original EOS 1:1 result unavailable")
    probe_manifest = json.loads((PROBE / "manifest.json").read_text())
    if (probe_manifest["task_count"] != 80 or probe_manifest["train_task_overlap"] != 0
            or probe_manifest["states_sha256"] != sha256(PROBE / "states.jsonl")):
        raise RuntimeError("heldout probe changed")
    heldout = {r["task_id"] for r in read_jsonl(PROBE / "states.jsonl")}
    if len(heldout) != 80:
        raise RuntimeError("heldout probe task duplication")
    original_fd = [a["task_id"] for a, _ in original_pairs]
    original_term = [b["task_id"] for _, b in original_pairs]
    used = set(original_fd) | set(original_term) | heldout
    encoded = read_jsonl(DATA / "fd_ce_dataset.jsonl")
    if len(encoded) != 989 or sha256(DATA / "fd_ce_dataset.jsonl") != original_plan["fd_ce_sha256"]:
        raise RuntimeError("FD source changed")
    rng = random.Random(SEED + 29)
    extra = (_extra(encoded, used, (11, 5, 5), rng)
             + _extra(encoded, used, (8, 5, 4), rng))
    if len(extra) != 38 or len({r["task_id"] for r in extra}) != 38:
        raise RuntimeError("extra FD selection count/identity failed")
    fd_src = {r["task_id"]: r for r in read_jsonl(FD_SOURCE)
              if r.get("keep_for_training") is True}
    terminal = {r["task_id"]: r for r in read_jsonl(TERM_SOURCE)}
    if len(fd_src) != 989 or len(terminal) != 2154:
        raise RuntimeError("verified source counts changed")
    for row in extra:
        validate_mask(row)
        if row["task_id"] not in fd_src:
            raise RuntimeError("extra FD absent from verified source")
    # Bit-reversal gives a nested, time-spread replacement priority: all 2:1
    # replacements remain identical in 4:1; only 17 further slots change.
    priority = sorted(range(64), key=_bit_reverse)
    manifests = {}
    for ratio, replace_count in ((2, 21), (4, 38)):
        replacement = {step: extra[index] for index, step in enumerate(priority[:replace_count])}
        rows = []
        for i, (old_fd, old_term) in enumerate(original_pairs):
            if i in replacement:
                rank1 = {"kind": "FD", "task_id": replacement[i]["task_id"]}
            else:
                rank1 = {"kind": "terminal_stop", "task_id": old_term["task_id"]}
            rows.append({"step": i + 1,
                         "rank0": {"kind": "FD", "task_id": old_fd["task_id"]},
                         "rank1": rank1})
        fd_ids = [row[rank]["task_id"] for row in rows for rank in ("rank0", "rank1")
                  if row[rank]["kind"] == "FD"]
        term_ids = [row[rank]["task_id"] for row in rows for rank in ("rank0", "rank1")
                    if row[rank]["kind"] == "terminal_stop"]
        counts = {"FD": len(fd_ids), "terminal_stop": len(term_ids),
                  "WRONG_ARGUMENT": sum(fd_src[x]["new_failure_type"] == "WRONG_ARGUMENT" for x in fd_ids),
                  "depth_ge2": sum(fd_src[x]["gold_step"] >= 2 for x in fd_ids),
                  "other_fd": sum(fd_src[x]["new_failure_type"] != "WRONG_ARGUMENT" for x in fd_ids),
                  "terminal_environments": len({terminal[x]["environment"] for x in term_ids})}
        if (counts["FD"] != (85 if ratio == 2 else 102)
                or counts["terminal_stop"] != (43 if ratio == 2 else 26)
                or len(set(fd_ids)) != len(fd_ids)
                or len(set(term_ids)) != len(term_ids)
                or (set(fd_ids) | set(term_ids)) & heldout
                or counts["WRONG_ARGUMENT"] / counts["FD"] < 413 / 989
                or counts["depth_ge2"] / counts["FD"] < 200 / 989
                or counts["other_fd"] == 0):
            raise RuntimeError(f"ratio {ratio} exposure/holdout gate failed: {counts}")
        manifest = {"schema_version": "cycle3_eos_ratio_schedule_v1",
                    "label": f"eos_{ratio}to1", "target_ratio": f"{ratio}:1",
                    "actual_ratio": counts["FD"] / counts["terminal_stop"],
                    "steps": 64, "world_size": 2, "seed": SEED,
                    "source_train_manifest_sha256": sha256(DATA / "train_manifest.json"),
                    "source_sampling_manifest_sha256": sha256(DATA / "sampling_manifest.json"),
                    "source_fd_sha256": sha256(FD_SOURCE),
                    "source_terminal_sha256": sha256(TERM_SOURCE),
                    "fd_ce_sha256": sha256(DATA / "fd_ce_dataset.jsonl"),
                    "dynamic_v1_model_sha256": sha256(MODEL / "model.safetensors"),
                    "heldout_probe_manifest_sha256": sha256(PROBE / "manifest.json"),
                    "heldout_probe_states_sha256": sha256(PROBE / "states.jsonl"),
                    "train_task_probe_overlap": 0, "counts": counts,
                    "replacement_priority": priority[:replace_count],
                    "schedule": rows}
        manifests[ratio] = manifest
    output.mkdir(parents=True, exist_ok=True)
    for ratio, manifest in manifests.items():
        (output / f"schedule_{ratio}.json").write_text(stable(manifest) + "\n")
    summary = {"status": "FROZEN", "schema_version": "cycle3_eos_ratio_ablation_v1",
               "ratios": {str(r): {"counts": m["counts"], "actual_ratio": m["actual_ratio"],
                                   "schedule_sha256": sha256(output / f"schedule_{r}.json")}
                          for r, m in manifests.items()},
               "source_train_manifest_sha256": sha256(DATA / "train_manifest.json"),
               "source_sampling_manifest_sha256": sha256(DATA / "sampling_manifest.json"),
               "probe_manifest_sha256": sha256(PROBE / "manifest.json")}
    (output / "ablation_manifest.json").write_text(stable(summary) + "\n")
    return summary


if __name__ == "__main__":
    print(stable(freeze()), flush=True)

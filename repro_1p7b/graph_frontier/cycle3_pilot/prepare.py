"""CPU-only immutable data gate and deterministic 64-step exposure schedule.

This reads saved Official-RL episodes; it does not call a model or generate a
new task. The previous FD50/natural-language-terminal protocol is untouched.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.cycle3_official_rl.audit import (
    make_fd, schema_for, valid_gold,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import typed_equal
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import (
    MODEL, RUN, checked_plan, sha256, stable,
)
from repro_1p7b.graph_frontier.official_rl_audit_static import (
    FROZEN, RICH24, overlap_index,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/data_execfd_stop_64_v1"
FD_SOURCE = RUN / "exec_verified_v2/first_divergence_exec_verified_v2.jsonl"
TERM_SOURCE = RUN / "audit_v1/terminal_stop_all.jsonl"
SEED = 20260926
STEPS = 64


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(stable(r) + "\n" for r in rows))


def _pick(pool: list[dict], count: int, selected: set[str], rng: random.Random) -> list[dict]:
    options = [r for r in pool if r["task_id"] not in selected]
    options.sort(key=lambda r: r["task_id"])
    rng.shuffle(options)
    if len(options) < count:
        raise RuntimeError("FD stratification pool too small")
    answer = options[:count]
    selected.update(r["task_id"] for r in answer)
    return answer


def select_fd(rows: list[dict], count: int = STEPS, seed: int = SEED) -> list[dict]:
    """32 parameter, 16 deep, 16 broad; overlap is reported, never duplicated."""
    if count != 64:
        raise ValueError("only fixed 64-step pilot exposure is supported")
    rng, selected = random.Random(seed), set()
    wrong = [r for r in rows if r["failure_type"] == "WRONG_ARGUMENT"]
    deep = [r for r in rows if r["gold_step"] >= 2]
    broad = [r for r in rows if r["failure_type"] != "WRONG_ARGUMENT"
             and r["gold_step"] == 1]
    chosen = (_pick(wrong, 32, selected, rng) + _pick(deep, 16, selected, rng)
              + _pick(broad, 16, selected, rng))
    if (len(chosen) != 64 or len(selected) != 64
            or sum(r["failure_type"] == "WRONG_ARGUMENT" for r in chosen) < 32
            or sum(r["gold_step"] >= 2 for r in chosen) < 16):
        raise RuntimeError("FD exposure quota failed")
    rng.shuffle(chosen)
    return chosen


def select_terminal(rows: list[dict], count: int = STEPS, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed + 1)
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["environment"]].append(row)
    groups = sorted(grouped)
    rng.shuffle(groups)
    for group in groups:
        grouped[group].sort(key=lambda r: r["task_id"])
        rng.shuffle(grouped[group])
    chosen = []
    while len(chosen) < count:
        progressed = False
        for group in groups:
            if grouped[group] and len(chosen) < count:
                chosen.append(grouped[group].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("terminal pool exhausted")
    if len({r["task_id"] for r in chosen}) != count:
        raise RuntimeError("terminal sampling duplicated a task")
    rng.shuffle(chosen)
    return chosen


def _overlap_gate(fd: list[dict], term: list[dict]) -> dict:
    findings = {}
    for name, path, expected in (("Frozen300", FROZEN, 300), ("Rich24", RICH24, 24)):
        index = overlap_index(path)
        if (not index["available"] or index["rows"] != expected
                or len(index["task_ids"]) != expected
                or len(index["query_hashes"]) != expected):
            raise RuntimeError(f"{name} overlap index incomplete")
        counts = {
            "fd_task_ids": len({r["task_id"] for r in fd} & index["task_ids"]),
            "fd_queries": len({r["query_hash"] for r in fd} & index["query_hashes"]),
            "terminal_task_ids": len({r["task_id"] for r in term} & index["task_ids"]),
            "terminal_queries": len({r["query_hash"] for r in term} & index["query_hashes"]),
        }
        if any(counts.values()):
            raise RuntimeError(f"{name} exact ID/query overlap: {counts}")
        findings[name] = {"rows": expected, "source_sha256": sha256(path), **counts}
    return findings


def prepare(output: Path = OUT) -> dict:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite pilot data: {output}")
    plan, raw = checked_plan()
    summary = json.loads((RUN / "exec_verified_v2/summary.json").read_text())
    if (summary.get("new_fd_total") != 998
            or summary.get("new_conservative_usable_fd") != 989
            or summary.get("previous_uncertain") != 895
            or summary.get("multicall_status", {}).get("UNCERTAIN") != 615
            or not summary.get("regression", {}).get("gate_pass")
            or summary.get("source_fd_sha256") != sha256(RUN / "audit_v1/first_divergence_all.jsonl")
            or summary.get("source_terminal_sha256") != sha256(TERM_SOURCE)):
        raise RuntimeError("exec-verified v2 / terminal source lock changed")
    all_fd, term = read_jsonl(FD_SOURCE), read_jsonl(TERM_SOURCE)
    fd = [r for r in all_fd if r.get("keep_for_training") is True]
    if (len(all_fd) != 998 or len(fd) != 989 or len(term) != 2154
            or len({r["task_id"] for r in fd}) != 989
            or len({r["task_id"] for r in term}) != 2154
            or any(r.get("alignment_version") != "exec_verified_v2" for r in fd)
            or any(r.get("new_failure_type") in ("UNCERTAIN", "PARSER_ARTIFACT",
                                                     "GENERATION_LENGTH_TRUNCATED") for r in fd)):
        raise RuntimeError("FD/terminal row count or quality gate failed")
    overlap = _overlap_gate(fd, term)
    for row in term:
        validate_mask(row)
        p = row["prompt_tokens"]
        if (row.get("target_type") != "terminal_stop"
                or row.get("assistant_end_token_id") != 151645
                or row.get("target_tokens") != 1
                or row["labels"][p] != 151645
                or row["labels"].count(151645) != 1
                or row.get("tool_observation_tokens_in_loss") != 0
                or row.get("historical_assistant_tokens_in_loss") != 0):
            raise RuntimeError("terminal EOS-only active-label gate failed")
    entries = {entry["task_id"]: entry for entry in plan["tasks"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("Dynamic-v1 tokenizer EOS ID changed")
    encoded = []
    for i, row in enumerate(fd):
        task_id = row["task_id"]
        entry = entries.get(task_id)
        if not entry or row["official_row_index"] != entry["source_row_index"]:
            raise RuntimeError(f"FD task identity unrecoverable: {task_id}")
        episode_path = RUN / "episodes" / f"{task_id}.json"
        episode = json.loads(episode_path.read_text())
        if not valid_gold(entry, raw[entry["source_row_index"]], episode):
            raise RuntimeError(f"FD gold executable replay lock failed: {task_id}")
        index = row["gold_step"] - 1
        expected_before = (episode["gold"]["initial_state"] if index == 0
                           else episode["gold"]["states_after"][index - 1])
        target = row.get("gold_next_action")
        if (index < 0 or index >= len(episode["gold"]["actions"])
                or not typed_equal(target, episode["gold"]["actions"][index])
                or not typed_equal(row["student_state"]["environment_state"], expected_before)
                or not isinstance(target.get("arguments"), dict)):
            raise RuntimeError(f"FD local gold target/state mismatch: {task_id}")
        schema = schema_for(RUN, episode)
        if target["name"] not in {tool["function"]["name"] for tool in schema}:
            raise RuntimeError(f"FD gold target absent from tool schema: {task_id}")
        result = make_fd(tokenizer, entry, row, schema)
        if result["target_tokens"] <= 0:
            raise RuntimeError(f"empty FD target: {task_id}")
        encoded.append(result)
        if (i + 1) % 200 == 0:
            print(stable({"fd_encoded": i + 1, "of": len(fd)}), flush=True)
    chosen_fd = select_fd(encoded)
    chosen_term = select_terminal(term)
    fd_by_id = {r["task_id"]: r for r in fd}
    wrong = sum(fd_by_id[r["task_id"]]["new_failure_type"] == "WRONG_ARGUMENT"
                for r in chosen_fd)
    deep = sum(fd_by_id[r["task_id"]]["gold_step"] >= 2 for r in chosen_fd)
    other = sum(fd_by_id[r["task_id"]]["new_failure_type"] != "WRONG_ARGUMENT"
                for r in chosen_fd)
    natural_wrong = sum(r["new_failure_type"] == "WRONG_ARGUMENT" for r in fd) / len(fd)
    natural_deep = sum(r["gold_step"] >= 2 for r in fd) / len(fd)
    if wrong / 64 < natural_wrong or deep / 64 < natural_deep:
        raise RuntimeError("actual FD exposure below natural bucket frequency")
    schedule = [{"step": i + 1,
                 "fd_task_id": chosen_fd[i]["task_id"],
                 "terminal_task_id": chosen_term[i]["task_id"],
                 "fd_tokens": chosen_fd[i]["total_tokens"],
                 "terminal_tokens": chosen_term[i]["total_tokens"],
                 "fd_failure_type": fd_by_id[chosen_fd[i]["task_id"]]["new_failure_type"],
                 "fd_gold_step": fd_by_id[chosen_fd[i]["task_id"]]["gold_step"]}
                for i in range(STEPS)]
    # The two-step sanity must cover the longest scheduled context of each
    # type. Formal training always reloads original Dynamic-v1, not smoke.
    smoke = [{"step": i + 1, "fd_task_id": f["task_id"],
              "terminal_task_id": t["task_id"]}
             for i, (f, t) in enumerate(zip(
                 sorted(chosen_fd, key=lambda r: r["total_tokens"], reverse=True)[:2],
                 sorted(chosen_term, key=lambda r: r["total_tokens"], reverse=True)[:2]))]
    manifest = {
        "schema_version": "cycle3_execfd_eos_pilot_data_v1",
        "status": "DATA_READY", "seed": SEED, "steps": STEPS,
        "source_plan_sha256": sha256(RUN / "rollout_manifest.json"),
        "source_fd_sha256": sha256(FD_SOURCE),
        "source_terminal_sha256": sha256(TERM_SOURCE),
        "source_exec_summary_sha256": sha256(RUN / "exec_verified_v2/summary.json"),
        "dynamic_v1_model_sha256": sha256(MODEL / "model.safetensors"),
        "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
        "fd_available": len(encoded), "terminal_available": len(term),
        "fd_exposure": len(chosen_fd), "terminal_exposure": len(chosen_term),
        "wrong_argument_exposure": wrong, "deep_exposure": deep,
        "other_fd_exposure": other,
        "fd_natural_wrong_argument_fraction": natural_wrong,
        "fd_natural_deep_fraction": natural_deep,
        "environment_count_terminal_exposure": len({r["environment"] for r in chosen_term}),
        "overlap": overlap,
        "longest_scheduled_fd_tokens": max(r["total_tokens"] for r in chosen_fd),
        "longest_scheduled_terminal_tokens": max(r["total_tokens"] for r in chosen_term),
    }
    output.mkdir(parents=True)
    write_jsonl(output / "fd_ce_dataset.jsonl", encoded)
    (output / "train_manifest.json").write_text(stable({
        **manifest, "fd_ce_sha256": sha256(output / "fd_ce_dataset.jsonl")}) + "\n")
    (output / "sampling_manifest.json").write_text(stable({
        "schema_version": "cycle3_execfd_eos_exposure_v1",
        "seed": SEED, "formal_schedule": schedule, "memory_smoke_schedule": smoke,
        "counts": {"FD": 64, "terminal_stop": 64, "WRONG_ARGUMENT": wrong,
                   "depth_ge2": deep, "other_fd": other},
    }) + "\n")
    print(stable(manifest), flush=True)
    return manifest


if __name__ == "__main__":
    prepare()

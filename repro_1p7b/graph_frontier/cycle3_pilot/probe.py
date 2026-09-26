"""Task-disjoint next-action probes from saved Cycle-3 states (no new tasks).

These are diagnostic greedy HF continuations, not EnvFactory executable scores.
Only complete EOS/tool-call evidence is classified; truncated outputs stay unknown.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_pilot.prepare import (
    FD_SOURCE, MODEL, OUT as DATA, TERM_SOURCE, read_jsonl,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/evaluation/next_action_probe"
TRAINED = ROOT / "repro_1p7b/checkpoints/cycle3_execfd_terminalstop_pilot"
SEED = 20260928
TOOL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def parse_action(text: str, eos: bool) -> dict:
    match = TOOL.search(text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
                return {"kind": "tool", "name": obj["name"], "arguments": obj["arguments"]}
        except json.JSONDecodeError:
            pass
        return {"kind": "invalid_tool"}
    if "<tool_call>" in text:
        return {"kind": "incomplete_tool" if not eos else "invalid_tool"}
    if eos:
        return {"kind": "no_tool_termination"}
    return {"kind": "incomplete"}


def prepare(output: Path = OUT) -> dict:
    if output.exists():
        raise RuntimeError("probe output already exists")
    train = json.loads((DATA / "train_manifest.json").read_text())
    schedule = json.loads((DATA / "sampling_manifest.json").read_text())["formal_schedule"]
    used = {x["fd_task_id"] for x in schedule} | {x["terminal_task_id"] for x in schedule}
    if (len(schedule) != 64 or train["fd_available"] != 989
            or train["terminal_available"] != 2154
            or train["source_fd_sha256"] != sha256(FD_SOURCE)
            or train["source_terminal_sha256"] != sha256(TERM_SOURCE)):
        raise RuntimeError("probe training/source lock failed")
    fd_source = {r["task_id"]: r for r in read_jsonl(FD_SOURCE) if r.get("keep_for_training") is True}
    encoded = {r["task_id"]: r for r in read_jsonl(DATA / "fd_ce_dataset.jsonl")}
    terminal = read_jsonl(TERM_SOURCE)
    if len(fd_source) != 989 or len(encoded) != 989 or len(terminal) != 2154:
        raise RuntimeError("probe source count drift")
    rng = random.Random(SEED)
    chosen_fd: list[dict] = []
    fd_used = set(used)
    for count, predicate in (
        (20, lambda r: r["new_failure_type"] == "WRONG_ARGUMENT"),
        (8, lambda r: r["gold_step"] >= 2),
        (12, lambda r: r["new_failure_type"] != "WRONG_ARGUMENT"),
    ):
        options = [r for r in fd_source.values() if r["task_id"] not in fd_used and predicate(r)]
        options.sort(key=lambda r: r["task_id"])
        rng.shuffle(options)
        if len(options) < count:
            raise RuntimeError("probe FD strata depleted")
        chosen_fd.extend(options[:count])
        fd_used.update(r["task_id"] for r in options[:count])
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in terminal:
        if row["task_id"] not in fd_used:
            grouped[row["environment"]].append(row)
    groups = sorted(grouped)
    rng.shuffle(groups)
    for group in groups:
        grouped[group].sort(key=lambda r: r["task_id"])
        rng.shuffle(grouped[group])
    chosen_term = []
    while len(chosen_term) < 40:
        progress = False
        for group in groups:
            if grouped[group] and len(chosen_term) < 40:
                chosen_term.append(grouped[group].pop())
                progress = True
        if not progress:
            raise RuntimeError("probe terminal strata depleted")
    rows = []
    for row in chosen_fd:
        item = encoded[row["task_id"]]
        rows.append({"task_id": row["task_id"], "kind": "unfinished_fd",
                     "environment": row["environment"], "failure_type": row["new_failure_type"],
                     "gold_step": row["gold_step"], "expected_action": row["gold_next_action"],
                     "prompt_ids": item["input_ids"][:item["prompt_tokens"]]})
    for row in chosen_term:
        rows.append({"task_id": row["task_id"], "kind": "terminal_stop",
                     "environment": row["environment"], "prompt_ids": row["input_ids"][:row["prompt_tokens"]]})
    if (len(rows) != 80 or len({r["task_id"] for r in rows}) != 80
            or any(r["task_id"] in used for r in rows)):
        raise RuntimeError("probe task-disjoint gate failed")
    output.mkdir(parents=True)
    (output / "states.jsonl").write_text("".join(stable(r) + "\n" for r in rows))
    manifest = {"schema_version": "cycle3_task_disjoint_probe_v1",
                "status": "FROZEN", "seed": SEED, "terminal_count": 40,
                "unfinished_count": 40, "task_count": 80,
                "train_task_overlap": 0,
                "fd_source_sha256": sha256(FD_SOURCE),
                "terminal_source_sha256": sha256(TERM_SOURCE),
                "training_schedule_sha256": sha256(DATA / "sampling_manifest.json"),
                "states_sha256": sha256(output / "states.jsonl"),
                "base_model_sha256": sha256(MODEL / "model.safetensors"),
                "cycle3_model_sha256": sha256(TRAINED / "model.safetensors"),
                "max_new_tokens": 256, "decoding": "greedy_hf_sdpa",
                "limitation": "stateless next-action probe; not executable trajectory evaluation"}
    (output / "manifest.json").write_text(stable(manifest) + "\n")
    return manifest


def collect(output: Path, label: str, gpu: int) -> dict:
    if label not in ("base", "cycle3") or gpu not in (0, 1):
        raise ValueError("invalid label/GPU")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["states_sha256"] != sha256(output / "states.jsonl"):
        raise RuntimeError("probe states changed")
    model_path = MODEL if label == "base" else TRAINED
    expected_hash = manifest["base_model_sha256" if label == "base" else "cycle3_model_sha256"]
    if sha256(model_path / "model.safetensors") != expected_hash:
        raise RuntimeError("probe model changed")
    dest = output / f"{label}.jsonl"
    if dest.exists():
        raise RuntimeError("refusing to overwrite probe result")
    torch.cuda.set_device(gpu)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True).to(f"cuda:{gpu}").eval()
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("probe EOS token mismatch")
    states = read_jsonl(output / "states.jsonl")
    records = []
    with torch.inference_mode():
        for index, row in enumerate(states):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=f"cuda:{gpu}")
            continuation = model.generate(ids, attention_mask=torch.ones_like(ids),
                                          max_new_tokens=256, do_sample=False,
                                          eos_token_id=151645, pad_token_id=151645)[0, ids.shape[1]:].tolist()
            text = tokenizer.decode(continuation, skip_special_tokens=False)
            action = parse_action(text, bool(continuation and continuation[-1] == 151645))
            prior_calls = TOOL.findall(tokenizer.decode(row["prompt_ids"], skip_special_tokens=False))
            prior_action = None
            if prior_calls:
                try:
                    prior_action = json.loads(prior_calls[-1])
                except json.JSONDecodeError:
                    pass
            repeat_previous_exact = bool(action["kind"] == "tool" and isinstance(prior_action, dict)
                                         and action["name"] == prior_action.get("name")
                                         and canonical(action["arguments"]) == canonical(prior_action.get("arguments")))
            if row["kind"] == "terminal_stop":
                category = ("stop_no_tool" if action["kind"] == "no_tool_termination"
                            else "extra_tool" if action["kind"] == "tool"
                            else "invalid_or_incomplete")
            else:
                gold = row["expected_action"]
                if action["kind"] == "tool":
                    category = ("correct_tool_and_arguments" if action["name"] == gold["name"]
                                and canonical(action["arguments"]) == canonical(gold["arguments"])
                                else "wrong_argument" if action["name"] == gold["name"]
                                else "wrong_tool")
                elif action["kind"] == "no_tool_termination":
                    category = "premature_stop"
                else:
                    category = "invalid_or_incomplete"
            records.append({"task_id": row["task_id"], "kind": row["kind"],
                            "category": category, "action": action,
                            "repeat_previous_exact": repeat_previous_exact,
                            "generated_token_count": len(continuation),
                            "eos_observed": bool(continuation and continuation[-1] == 151645)})
            if (index + 1) % 10 == 0:
                print(stable({"label": label, "completed": index + 1, "total": len(states)}), flush=True)
    dest.write_text("".join(stable(r) + "\n" for r in records))
    return {"label": label, "rows": len(records),
            "categories": dict(collections.Counter(r["kind"] + ":" + r["category"] for r in records)),
            "sha256": sha256(dest)}


def summarize(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    a, b = read_jsonl(output / "base.jsonl"), read_jsonl(output / "cycle3.jsonl")
    if len(a) != 80 or len(b) != 80 or [r["task_id"] for r in a] != [r["task_id"] for r in b]:
        raise RuntimeError("probe pairing mismatch")
    result = {"schema_version": "cycle3_task_disjoint_probe_result_v1",
              "manifest_sha256": sha256(output / "manifest.json"),
              "base_sha256": sha256(output / "base.jsonl"),
              "cycle3_sha256": sha256(output / "cycle3.jsonl"),
              "task_count": 80, "terminal_count": 40, "unfinished_count": 40,
              "train_task_overlap": manifest["train_task_overlap"], "models": {}}
    for label, rows in (("base", a), ("cycle3", b)):
        result["models"][label] = {
            "terminal_categories": dict(collections.Counter(r["category"] for r in rows if r["kind"] == "terminal_stop")),
            "terminal_repeat_previous_exact": sum(r["repeat_previous_exact"] for r in rows if r["kind"] == "terminal_stop"),
            "unfinished_categories": dict(collections.Counter(r["category"] for r in rows if r["kind"] == "unfinished_fd")),
        }
    (output / "summary.json").write_text(stable(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "collect", "summarize"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--label", choices=("base", "cycle3"))
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    result = (prepare(args.output) if args.command == "prepare"
              else collect(args.output, args.label, args.gpu) if args.command == "collect"
              else summarize(args.output))
    print(stable(result), flush=True)


if __name__ == "__main__":
    main()

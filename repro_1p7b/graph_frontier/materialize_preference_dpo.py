"""Serialize audited preference pairs with Dynamic-v1's exact tool template.

TRL accepts plain string prompt/chosen/rejected columns.  We intentionally
serialize each row here because every EnvFactory task has a different tool
schema; TRL's DPOConfig only accepts one global ``tools`` value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def serialize_pair(tokenizer, pair: dict[str, Any], max_length: int) -> tuple[dict[str, Any] | None, str | None]:
    kwargs = {
        "tools": pair["tools"],
        "tokenize": False,
        "enable_thinking": False,
    }
    # Keep the prompt at a completed user/tool turn.  The completion then owns
    # the assistant header.  Qwen3's generation-only prefix is not present
    # when serializing an already completed assistant message, so using
    # add_generation_prompt=True would break exact-prefix equivalence.
    prompt = tokenizer.apply_chat_template(pair["prompt"], add_generation_prompt=False, **kwargs)
    chosen_full = tokenizer.apply_chat_template(pair["prompt"] + pair["chosen"], add_generation_prompt=False, **kwargs)
    rejected_full = tokenizer.apply_chat_template(pair["prompt"] + pair["rejected"], add_generation_prompt=False, **kwargs)
    if not chosen_full.startswith(prompt) or not rejected_full.startswith(prompt):
        return None, "serialized_prompt_not_exact_prefix"
    chosen = chosen_full[len(prompt):]
    rejected = rejected_full[len(prompt):]
    if not chosen or not rejected:
        return None, "empty_serialized_completion"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]
    if prompt_ids + chosen_ids != tokenizer(chosen_full, add_special_tokens=False)["input_ids"]:
        return None, "chosen_token_boundary_mismatch"
    if prompt_ids + rejected_ids != tokenizer(rejected_full, add_special_tokens=False)["input_ids"]:
        return None, "rejected_token_boundary_mismatch"
    prompt_tokens = len(prompt_ids)
    chosen_tokens = len(chosen_ids)
    rejected_tokens = len(rejected_ids)
    chosen_length = prompt_tokens + chosen_tokens
    rejected_length = prompt_tokens + rejected_tokens
    if chosen_length > max_length or rejected_length > max_length:
        return None, "over_max_length"
    result = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "pair_id": pair["pair_id"],
        "task_id": pair["task_id"],
        "pair_type": "graph" if pair["schema_version"].startswith("graph_") else "standard",
        "prompt_tokens": prompt_tokens,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        # DPO evaluates prompt twice, once with each preference completion.
        "preference_tokens": 2 * prompt_tokens + chosen_tokens + rejected_tokens,
        "max_sequence_tokens": max(chosen_length, rejected_length),
        "failure_type": pair.get("failure_type", "official_trajectory_ordering"),
    }
    return result, None


def deterministic_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).hexdigest())


def closest_prefix(rows: list[dict[str, Any]], target: int, minimum: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    cumulative = 0
    candidates: list[tuple[int, int]] = []
    for index, row in enumerate(rows, 1):
        cumulative += int(row["preference_tokens"])
        if index >= min(minimum, len(rows)):
            candidates.append((abs(cumulative - target), index))
    count = min(candidates)[1]
    return rows[:count]


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "unique_tasks": len({row["task_id"] for row in rows}),
        "preference_tokens": sum(row["preference_tokens"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "chosen_tokens": sum(row["chosen_tokens"] for row in rows),
        "rejected_tokens": sum(row["rejected_tokens"] for row in rows),
        "max_sequence_tokens": max((row["max_sequence_tokens"] for row in rows), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--smoke-pairs", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = {
        "graph_train": load_jsonl(args.pair_dir / "graph_targeted_pairs_train.jsonl"),
        "graph_val": load_jsonl(args.pair_dir / "graph_targeted_pairs_val.jsonl"),
        "standard_train": load_jsonl(args.pair_dir / "standard_dpo_pairs_train.jsonl"),
        "standard_val": load_jsonl(args.pair_dir / "standard_dpo_pairs_val.jsonl"),
    }
    serialized: dict[str, list[dict[str, Any]]] = {}
    drops: dict[str, dict[str, int]] = {}
    for name, rows in raw.items():
        kept = []
        reasons: dict[str, int] = {}
        for pair in rows:
            row, reason = serialize_pair(tokenizer, pair, args.max_length)
            if reason is None:
                kept.append(row)
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        serialized[name] = deterministic_order(kept, args.seed)
        drops[name] = reasons

    graph_pool = serialized["graph_train"][: args.max_pairs]
    standard_pool = serialized["standard_train"][: args.max_pairs]
    shared_target = min(
        sum(row["preference_tokens"] for row in graph_pool),
        sum(row["preference_tokens"] for row in standard_pool),
    )
    graph_pilot = closest_prefix(graph_pool, shared_target, args.smoke_pairs)
    standard_pilot = closest_prefix(standard_pool, shared_target, args.smoke_pairs)

    outputs = {
        "graph_pilot_train.jsonl": graph_pilot,
        "standard_pilot_train.jsonl": standard_pilot,
        "graph_smoke_train.jsonl": graph_pilot[: args.smoke_pairs],
        "standard_smoke_train.jsonl": standard_pilot[: args.smoke_pairs],
        "graph_val.jsonl": serialized["graph_val"],
        "standard_val.jsonl": serialized["standard_val"],
    }
    for name, rows in outputs.items():
        write_jsonl(args.output_dir / name, rows)

    report = {
        "schema_version": "envfactory_preference_dpo_materialization_v1",
        "model": str(args.model),
        "seed": args.seed,
        "max_length": args.max_length,
        "max_pairs": args.max_pairs,
        "smoke_pairs_required": args.smoke_pairs,
        "serialization": "per-row Dynamic-v1 chat template with row-specific tools; plain TRL strings",
        "drops": drops,
        "raw_counts": {name: len(rows) for name, rows in raw.items()},
        "serialized_counts": {name: len(rows) for name, rows in serialized.items()},
        "shared_target_preference_tokens": shared_target,
        "graph_pilot": stats(graph_pilot),
        "standard_pilot": stats(standard_pilot),
        "graph_smoke": stats(outputs["graph_smoke_train.jsonl"]),
        "standard_smoke": stats(outputs["standard_smoke_train.jsonl"]),
        "hashes": {name: sha256(args.output_dir / name) for name in outputs},
    }
    (args.output_dir / "materialization_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if len(graph_pilot) < args.smoke_pairs or len(standard_pilot) < args.smoke_pairs:
        raise SystemExit("insufficient clean pairs for deterministic 16-pair smoke")


if __name__ == "__main__":
    main()


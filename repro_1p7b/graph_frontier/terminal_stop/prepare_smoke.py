"""Build a small, replay-backed Official-RL terminal-stop / verified FD smoke set.

This script executes only gold tools in isolated MCP clients. It does not run
student inference. The output directory is immutable and excluded from Git.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_terminal_stop
from repro_1p7b.graph_frontier.freeze_terminal_ce_v2 import MODEL, validate_fd
from repro_1p7b.graph_frontier.official_rl_audit_static import (
    FROZEN, OUT, REGISTRY, RICH24, RICH_TRAIN, SOURCE, decode, digest,
    norm_query, overlap_index, query_from_row, query_hash, stable,
)
from repro_1p7b.graph_frontier.official_rl_audit_replay import raw_call, saved
from repro_1p7b.graph_frontier.fastmcp_adapter import adapt_fastmcp_result
from repro_1p7b.graph_frontier.state_verifier import compare_final_states
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_overlap(raw: dict, task_id: str, indexes: list[dict]) -> bool:
    factory = raw["extra_info"]["mcp_factory_kwargs"]
    servers = decode(factory["mcp_servers"])
    initial = decode(factory["initial_config"])
    query = query_hash(query_from_row(raw))
    gold = digest(decode(raw["reward_model"]["ground_truth"]))
    initial_hash = digest(initial)
    return all(
        task_id not in ix["task_ids"]
        and query not in ix["query_hashes"]
        and initial_hash not in ix["initial_hashes"]
        and gold not in ix["gold_signatures"]
        for ix in indexes
    )


def replay_terminal(index: int, raw: dict, audit: dict, manager, tokenizer) -> tuple[dict, dict]:
    factory = raw["extra_info"]["mcp_factory_kwargs"]
    servers = decode(factory["mcp_servers"])
    initial = decode(factory["initial_config"])
    expected = decode(factory["final_config"])
    gold = decode(raw["reward_model"]["ground_truth"])
    replay_lock = json.loads((OUT / "replay_tasks" / f"{index:04d}.json").read_text())
    if (replay_lock.get("replay_ok") is not True
            or replay_lock.get("final_state_match") is not True
            or replay_lock.get("audit_id") != audit["audit_id"]):
        raise RuntimeError("persisted gold replay lock failed")
    client_ids = {server: f"{server}-terminalstop-{index}-{os.getpid()}" for server in servers}
    try:
        for server in servers:
            manager.load_scenario(client_ids[server], initial[server], check=True)
        if stable(saved(manager, client_ids)) != stable({s: initial[s] for s in servers}):
            raise RuntimeError("initial state mismatch")
        messages = [{"role": "user", "content": query_from_row(raw)}]
        for step, call in enumerate(gold):
            name, args = call["name"], call["arguments"]
            response = raw_call(manager, client_ids[name.split("-", 1)[0]], name, args)
            fields, success = adapt_fastmcp_result(response)
            if success is not True:
                raise RuntimeError(f"gold tool failed at {step}")
            call_id = f"gold_{step}"
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": call_id, "type": "function",
                                "function": {"name": name, "arguments": args}}],
            })
            messages.append({
                "role": "tool", "tool_call_id": call_id, "name": name,
                "content": stable(fields),
            })
        final = saved(manager, client_ids)
        if compare_final_states(final, {s: expected[s] for s in servers}) is not True:
            raise RuntimeError("final state verifier failed")
        sample = {
            "task_id": audit["audit_id"],
            "state_identity": audit["audit_id"] + ":terminal_stop",
            "environment": audit["environment"],
            "conversation_prefix": messages,
            "tool_schema": manager.filter_tools(servers),
            "replay_valid": True, "final_state_match": True,
            "source_row_index": index,
        }
        encoded = encode_terminal_stop(tokenizer, sample)
        validate_mask(encoded)
        encoded["environment"] = audit["environment"]
        return sample, encoded
    finally:
        for client_id in client_ids.values():
            try:
                manager.close_client(client_id)
            except Exception:
                pass


def collator_audit(rows: list[dict], pad_token_id: int) -> dict:
    import torch
    max_length = max(row["total_tokens"] for row in rows)
    ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        validate_mask(row)
        length = row["total_tokens"]
        ids[i, :length] = torch.tensor(row["input_ids"])
        labels[i, :length] = torch.tensor(row["labels"])
        mask[i, :length] = 1
        if row["target_type"] == "terminal_stop":
            assert int((labels[i] != -100).sum()) == 1
            assert int(labels[i, row["prompt_tokens"]]) == row["assistant_end_token_id"]
        assert int(mask[i].sum()) == length
        assert torch.all(labels[i, length:] == -100)
    return {"shape": list(ids.shape), "attention_tokens": int(mask.sum()),
            "active_labels": int((labels != -100).sum()), "padding_masked": True,
            "truncated": False}


def build(output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite smoke dataset")
    raw_rows = json.loads(SOURCE.read_text())
    if sha(SOURCE) != "0283caf6487f8790df972f192a94cdec7d17b1ccb94aceda59d21efd562105c4":
        raise RuntimeError("Official-RL source hash drift")
    tier_rows = pq.read_table(OUT / "opd_tiers.parquet").to_pylist()
    fd_rows, fd_lock = validate_fd()
    fd_ids = {r["task_id"] for r in fd_rows}
    rich_source = [json.loads(line) for line in RICH_TRAIN.read_text().splitlines() if line.strip()]
    fd_environment = {}
    for row in rich_source:
        if row["task_id"] in fd_ids:
            factory = row["extra_info"]["mcp_factory_kwargs"]
            fd_environment[row["task_id"]] = ",".join(sorted(decode(factory["mcp_servers"])))
    if fd_ids - fd_environment.keys():
        raise RuntimeError("FD environment mapping incomplete")
    by_type = collections.defaultdict(list)
    for row in fd_rows:
        by_type[row["divergence_type"]].append(row)
    quotas = {"wrong_arguments": 6, "wrong_tool": 5, "premature_stop": 5}
    if any(len(by_type[key]) < count for key, count in quotas.items()):
        raise RuntimeError("verified FD type quota unavailable")
    fd_selected = []
    for kind, count in quotas.items():
        fd_selected.extend(sorted(by_type[kind], key=lambda r: (r["total_tokens"], r["task_id"]))[:count])
    fd_selected = [{**row, "target_type": "first_divergence_tool",
                    "environment": fd_environment[row["task_id"]]} for row in fd_selected]
    indexes = [overlap_index(FROZEN), overlap_index(RICH24)]
    if not all(ix["available"] for ix in indexes):
        raise RuntimeError("Frozen300/Rich24 overlap source unavailable")
    if any(row["task_id"] in ix["task_ids"] for row in fd_selected for ix in indexes):
        raise RuntimeError("FD heldout overlap")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.eos_token != "<|im_end|>" or tokenizer.eos_token_id != 151645:
        raise RuntimeError("Qwen3 assistant end token drift")
    os.environ["MCP_CONFIG_PATH"] = str(REGISTRY.resolve())
    from src.manager.mcp_client_manager import MCPManager
    candidates = sorted(
        (r for r in tier_rows if r["clean_opd_ready"] and r["replay_ok"]
         and not r["eval_or_train_overlap"] and 2 <= r["gold_length"] <= 3
         and len(r["environment"].split(",")) == 1),
        key=lambda r: (r["gold_length"], r["environment"], r["row_index"]),
    )
    terminal_states, terminal_rows, errors, used_envs = [], [], [], set()
    try:
        for audit in candidates:
            if len(terminal_rows) == 16:
                break
            env = audit["environment"]
            if env in used_envs:
                continue
            raw = raw_rows[audit["row_index"]]
            if not no_overlap(raw, audit["audit_id"], indexes):
                continue
            try:
                sample, encoded = replay_terminal(audit["row_index"], raw, audit, MCPManager, tokenizer)
                terminal_states.append(sample)
                terminal_rows.append(encoded)
                used_envs.add(env)
                print(f"terminal {len(terminal_rows)}/16 row={audit['row_index']} env={env}", flush=True)
            except Exception as exc:
                errors.append({"row_index": audit["row_index"], "error": str(exc)[:200]})
            if len(errors) >= 48:
                break
    finally:
        MCPManager.shutdown(timeout=30)
    if len(terminal_rows) != 16 or len(fd_selected) != 16:
        raise RuntimeError(f"data gate failed FD={len(fd_selected)} stop={len(terminal_rows)} errors={errors[:8]}")
    rows = fd_selected + terminal_rows
    if len({r["task_id"] for r in rows}) != 32:
        raise RuntimeError("task IDs not unique")
    audit = collator_audit(rows, tokenizer.pad_token_id)
    token_audit = [{
        "task_id": r["task_id"], "type": r["target_type"],
        "tail_ids": r["input_ids"][-8:], "tail_labels": r["labels"][-8:],
    } for r in fd_selected[:3] + terminal_rows[:3]]
    output.mkdir(parents=True)
    data = output / "dataset.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    (output / "terminal_states.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n" for s in terminal_states))
    result = {
        "schema_version": "terminal_stop_smoke_data_v1", "status": "DATA_READY",
        "dataset_sha256": sha(data), "fd_source_sha256": fd_lock["dataset_sha256"],
        "tokenizer_json_sha256": sha(MODEL / "tokenizer.json"),
        "source_official_sha256": sha(SOURCE),
        "fd_samples": 16, "terminal_stop_samples": 16, "unique_tasks": 32,
        "fd_types": dict(collections.Counter(r["divergence_type"] for r in fd_selected)),
        "environment_distribution": dict(collections.Counter(r["environment"] for r in rows)),
        "terminal_environment_count": len(used_envs),
        "frozen300_overlap": 0, "rich24_overlap": 0,
        "collator": audit, "token_audit": token_audit, "replay_errors": errors,
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

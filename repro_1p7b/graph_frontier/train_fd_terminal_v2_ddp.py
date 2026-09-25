"""Matched two-GPU full-parameter assistant-target CE for FD + terminal v2.

RUN-A and RUN-B always initialize from original Dynamic-v1.  Data manifests,
heldout exclusion, tokenizer identity, masks, and deterministic exposure quotas
are checked before any GPU allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from repro_1p7b.graph_frontier.freeze_terminal_ce_v2 import FD_SOURCE, MODEL

LR = 2.0e-6
SEED = 20260927
WORLD_SIZE = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_mask(row: dict) -> None:
    p = row["prompt_tokens"]
    if (
        len(row["input_ids"]) != len(row["labels"])
        or row["total_tokens"] != len(row["input_ids"])
        or p <= 0 or p >= len(row["input_ids"])
        or row["labels"][:p] != [-100] * p
        or row["labels"][p:] != row["input_ids"][p:]
        or len(row["input_ids"]) > 16384
    ):
        raise RuntimeError("assistant-only CE mask invalid: " + row["task_id"])


def build_schedule(fd_rows: list[dict], terminal_rows: list[dict], *,
                   ratio: int, steps: int, seed: int = SEED) -> tuple[list[dict], dict]:
    if ratio not in (1, 2) or steps not in (16, 64):
        raise ValueError("only frozen 1:1/1:2 and 16/64-step protocols")
    if len(fd_rows) != 50 or len(terminal_rows) < 48:
        raise RuntimeError("data gate failed")
    slots = steps * WORLD_SIZE
    # Nearest integer quota; 64-step 1:2 is 43 FD + 85 terminal.
    fd_slots = round(slots / (1 + ratio))
    types = ["first_divergence_tool"] * fd_slots + ["terminal_final"] * (slots - fd_slots)
    rng = random.Random(seed + 100 * ratio + steps)
    rng.shuffle(types)
    pools = {
        "first_divergence_tool": list(fd_rows),
        "terminal_final": list(terminal_rows),
    }
    for pool in pools.values():
        rng.shuffle(pool)
    cursors = {key: 0 for key in pools}
    selected = []
    for kind in types:
        pool = pools[kind]
        index = cursors[kind]
        if index and index % len(pool) == 0:
            rng.shuffle(pool)
        selected.append(pool[index % len(pool)])
        cursors[kind] = index + 1
    if steps == 16:
        # Smoke must exercise the longest terminal context, not only the
        # random subset that happens to be selected.
        longest = max(terminal_rows, key=lambda row: row["total_tokens"])
        first_terminal = types.index("terminal_final")
        if longest in selected:
            other = selected.index(longest)
            selected[first_terminal], selected[other] = selected[other], selected[first_terminal]
        else:
            selected[first_terminal] = longest
    counts = {"first_divergence_tool": fd_slots, "terminal_final": slots - fd_slots}
    return selected, counts


def load_and_schedule(frozen_dir: Path, model_path: Path, *,
                      ratio: int, steps: int) -> tuple[list[dict], dict]:
    lock = json.loads((frozen_dir / "manifest.json").read_text())
    fd_rows = jsonl(FD_SOURCE / "dataset.jsonl")
    terminal_rows = jsonl(frozen_dir / "terminal_ce_dataset.jsonl")
    if (
        model_path.resolve() != MODEL.resolve()
        or lock.get("schema_version") != "first_divergence_terminal_v2_frozen_data"
        or lock.get("verdict") != "FROZEN_READY"
        or lock.get("fd_source_manifest_sha256") != sha256(FD_SOURCE / "manifest.json")
        or lock.get("fd_source_dataset_sha256") != sha256(FD_SOURCE / "dataset.jsonl")
        or lock.get("terminal_ce_dataset_sha256") != sha256(frozen_dir / "terminal_ce_dataset.jsonl")
        or lock.get("tokenizer_json_sha256") != sha256(MODEL / "tokenizer.json")
        or lock.get("fd_examples") != len(fd_rows) or len(fd_rows) != 50
        or lock.get("terminal_examples") != len(terminal_rows)
        or lock.get("heldout_task_overlap") != 0 or lock.get("frozen300_exact_id_overlap") != 0
        or any(r.get("target_type") != "terminal_final" for r in terminal_rows)
        or len({r["state_identity"] for r in terminal_rows}) != len(terminal_rows)
    ):
        raise RuntimeError("frozen v2 data/model gate failed")
    fd_rows = [{**r, "target_type": "first_divergence_tool"} for r in fd_rows]
    for row in fd_rows + terminal_rows:
        validate_mask(row)
    selected, counts = build_schedule(fd_rows, terminal_rows, ratio=ratio, steps=steps)
    plan = {
        "schema_version": "first_divergence_terminal_v2_ddp_plan",
        "initialization_checkpoint": str(MODEL),
        "initialization_weights_sha256": sha256(MODEL / "model.safetensors"),
        "frozen_manifest_sha256": sha256(frozen_dir / "manifest.json"),
        "fd_dataset_sha256": lock["fd_source_dataset_sha256"],
        "terminal_ce_dataset_sha256": lock["terminal_ce_dataset_sha256"],
        "ratio_target": f"1:{ratio}",
        "exposure_counts": counts,
        "global_steps": steps,
        "world_size": WORLD_SIZE,
        "per_rank_batch_size": 1,
        "learning_rate": LR,
        "optimizer": "AdamW",
        "weight_decay": 0,
        "precision": "bfloat16",
        "gradient_checkpointing": True,
        "seed": SEED,
        "schedule": [{
            "step": step + 1,
            "rank0_task": selected[step * 2]["task_id"],
            "rank1_task": selected[step * 2 + 1]["task_id"],
            "rank0_target_type": selected[step * 2]["target_type"],
            "rank1_target_type": selected[step * 2 + 1]["target_type"],
            "rank0_tokens": selected[step * 2]["total_tokens"],
            "rank1_tokens": selected[step * 2 + 1]["total_tokens"],
        } for step in range(steps)],
    }
    return selected, plan


def train(frozen_dir: Path, model_path: Path, output: Path, logs: Path, *,
          ratio: int, steps: int) -> dict | None:
    os.environ.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    os.environ["PATH"] = str(Path(os.environ["CUDA_HOME"]) / "bin") + ":" + os.environ.get("PATH", "")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE or local_rank not in (0, 1):
        raise RuntimeError("requires exactly two local GPUs")
    if output.exists() or (rank == 0 and logs.exists()):
        raise RuntimeError("refusing to overwrite training output/logs")
    selected, plan = load_and_schedule(frozen_dir, model_path, ratio=ratio, steps=steps)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dist.init_process_group("nccl")
    if rank == 0:
        logs.mkdir(parents=True)
        (logs / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(f"cuda:{local_rank}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
              find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=LR, weight_decay=0.0)
    records = []
    for step in range(steps):
        row = selected[step * 2 + rank]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=local_rank)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=local_rank)
        mask = torch.ones_like(ids)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids=ids, attention_mask=mask, labels=labels).loss
        finite = torch.tensor([int(torch.isfinite(loss).item())], device=local_rank)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(grad_norm).item())], device=local_rank)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite gradient at step {step + 1}")
        optimizer.step()
        torch.cuda.synchronize(local_rank)
        losses = loss.detach().float().clone()
        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        losses /= world_size
        if rank == 0:
            record = {
                "step": step + 1, "mean_loss": float(losses.cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "rank0_task": selected[step * 2]["task_id"],
                "rank1_task": selected[step * 2 + 1]["task_id"],
                "rank0_target_type": selected[step * 2]["target_type"],
                "rank1_target_type": selected[step * 2 + 1]["target_type"],
                "rank0_tokens": selected[step * 2]["total_tokens"],
                "rank1_tokens": selected[step * 2 + 1]["total_tokens"],
                "rank0_peak_gb": round(torch.cuda.max_memory_allocated(local_rank)/(1024**3), 3),
            }
            records.append(record)
            with (logs / "metrics.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
    peak = torch.tensor([torch.cuda.max_memory_allocated(local_rank)], dtype=torch.float64,
                        device=local_rank)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank == 0:
        model.eval()
        output.mkdir(parents=True)
        model.save_pretrained(output, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output)
        result = {
            "schema_version": "first_divergence_terminal_v2_ddp_result",
            "status": "TRAINED", "global_steps": steps,
            "ratio_target": f"1:{ratio}",
            "world_size": world_size, "full_parameter": True,
            "all_loss_finite": True, "all_grad_finite": True,
            "first_mean_loss": records[0]["mean_loss"],
            "last_mean_loss": records[-1]["mean_loss"],
            "peak_gpu_gb": round(peak.item()/(1024**3), 3),
            "initialization_weights_sha256": plan["initialization_weights_sha256"],
            "fd_dataset_sha256": plan["fd_dataset_sha256"],
            "terminal_ce_dataset_sha256": plan["terminal_ce_dataset_sha256"],
            "model_path": str(output),
            "model_weights_sha256": sha256(output / "model.safetensors"),
        }
        (logs / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    else:
        result = None
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--ratio", type=int, choices=(1, 2), required=True)
    parser.add_argument("--steps", type=int, choices=(16, 64), required=True)
    args = parser.parse_args()
    train(args.frozen_dir, args.model_path, args.output, args.logs,
          ratio=args.ratio, steps=args.steps)


if __name__ == "__main__":
    main()

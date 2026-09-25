"""Two-GPU bounded full-parameter CE on cumulative on-policy first-divergence states."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256

STEPS = 64
LR = 2.0e-6
SEED = 20260927


def load_and_schedule(dataset_dir: Path, model_path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (dataset_dir / "dataset.jsonl").read_text().splitlines()]
    if (
        manifest.get("schema_version") != "first_divergence_cumulative_ce_v1"
        or manifest.get("dataset_sha256") != sha256(dataset_dir / "dataset.jsonl")
        or manifest.get("example_count") != len(rows)
        or manifest.get("new_examples", 0) < 20
        or len(rows) < 48
        or manifest.get("heldout_task_overlap") != 0
        or manifest.get("frozen_exact_id_overlap") != 0
        or manifest.get("tool_observation_tokens_in_loss") != 0
        or len({row["state_identity"] for row in rows}) != len(rows)
    ):
        raise RuntimeError("cumulative CE dataset gate failed")
    for row in rows:
        p = row["prompt_tokens"]
        if (
            len(row["input_ids"]) != len(row["labels"])
            or row["labels"][:p] != [-100] * p
            or row["labels"][p:] != row["input_ids"][p:]
            or len(row["input_ids"]) > 16384
        ):
            raise RuntimeError(f"assistant-only mask failed: {row['task_id']}")
    rng = random.Random(SEED)
    order = []
    while len(order) < STEPS * 2:
        cycle = list(range(len(rows)))
        rng.shuffle(cycle)
        order.extend(cycle)
    order = order[:STEPS * 2]
    by_length = sorted(range(len(rows)), key=lambda i: rows[i]["total_tokens"])
    order[:4] = [by_length[0], by_length[len(rows)//2], by_length[-1], by_length[len(rows)//3]]
    plan = {
        "schema_version": "first_divergence_ddp64_plan_v1",
        "initialization_checkpoint": str(model_path),
        "initialization_weights_sha256": sha256(model_path / "model.safetensors"),
        "dataset_sha256": manifest["dataset_sha256"],
        "global_steps": STEPS, "world_size": 2, "per_rank_batch_size": 1,
        "learning_rate": LR, "optimizer": "AdamW", "weight_decay": 0,
        "full_parameter": True, "precision": "bfloat16",
        "gradient_checkpointing": True, "seed": SEED,
        "schedule": [{
            "step": step + 1,
            "rank0_task": rows[order[step*2]]["task_id"],
            "rank1_task": rows[order[step*2+1]]["task_id"],
            "rank0_tokens": rows[order[step*2]]["total_tokens"],
            "rank1_tokens": rows[order[step*2+1]]["total_tokens"],
        } for step in range(STEPS)],
    }
    return [rows[i] for i in order], plan


def train(dataset_dir: Path, model_path: Path, output: Path, logs: Path) -> dict | None:
    os.environ.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    os.environ["PATH"] = str(Path(os.environ["CUDA_HOME"]) / "bin") + ":" + os.environ.get("PATH", "")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or local_rank not in (0, 1):
        raise RuntimeError("requires exactly two local GPUs")
    if output.exists() or (rank == 0 and logs.exists()):
        raise RuntimeError("refusing to overwrite training output/logs")
    selected, plan = load_and_schedule(dataset_dir, model_path)
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
    for step in range(STEPS):
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
                "rank0_task": selected[step*2]["task_id"],
                "rank1_task": selected[step*2+1]["task_id"],
                "rank0_tokens": selected[step*2]["total_tokens"],
                "rank1_tokens": selected[step*2+1]["total_tokens"],
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
            "schema_version": "first_divergence_ddp64_result_v1",
            "status": "TRAINED", "global_steps": STEPS,
            "world_size": world_size, "full_parameter": True,
            "all_loss_finite": True, "all_grad_finite": True,
            "first_mean_loss": records[0]["mean_loss"],
            "last_mean_loss": records[-1]["mean_loss"],
            "peak_gpu_gb": round(peak.item()/(1024**3), 3),
            "initialization_weights_sha256": plan["initialization_weights_sha256"],
            "dataset_sha256": plan["dataset_sha256"],
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
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    train(args.dataset_dir, args.model_path, args.output, args.logs)


if __name__ == "__main__":
    main()

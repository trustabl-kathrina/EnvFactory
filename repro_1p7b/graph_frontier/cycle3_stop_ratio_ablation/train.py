"""Same full-parameter 64-step BF16 DDP CE trainer; only row exposure changes."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_pilot.prepare import (
    FD_SOURCE, MODEL, OUT as DATA, RUN, TERM_SOURCE, read_jsonl,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.prepare import OUT, PROBE
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

LR = 2e-6
SEED = 20260927


def load_plan(ratio: int) -> tuple[list[tuple[dict, dict]], dict]:
    if ratio not in (2, 4):
        raise ValueError("only frozen 2:1 and 4:1 ablations are allowed")
    manifest = json.loads((OUT / f"schedule_{ratio}.json").read_text())
    source = json.loads((DATA / "train_manifest.json").read_text())
    fd = read_jsonl(DATA / "fd_ce_dataset.jsonl")
    term = read_jsonl(TERM_SOURCE)
    if (manifest.get("schema_version") != "cycle3_eos_ratio_schedule_v1"
            or manifest.get("target_ratio") != f"{ratio}:1"
            or manifest.get("steps") != 64 or manifest.get("world_size") != 2
            or manifest.get("source_train_manifest_sha256") != sha256(DATA / "train_manifest.json")
            or manifest.get("source_sampling_manifest_sha256") != sha256(DATA / "sampling_manifest.json")
            or manifest.get("fd_ce_sha256") != sha256(DATA / "fd_ce_dataset.jsonl")
            or manifest.get("source_fd_sha256") != sha256(FD_SOURCE)
            or manifest.get("source_terminal_sha256") != sha256(TERM_SOURCE)
            or manifest.get("dynamic_v1_model_sha256") != sha256(MODEL / "model.safetensors")
            or manifest.get("heldout_probe_manifest_sha256") != sha256(PROBE / "manifest.json")
            or manifest.get("heldout_probe_states_sha256") != sha256(PROBE / "states.jsonl")
            or source.get("fd_available") != 989 or source.get("terminal_available") != 2154
            or len(fd) != 989 or len(term) != 2154):
        raise RuntimeError("ablation source/model/probe lock failed")
    for row in fd + term:
        validate_mask(row)
    fd_by, term_by = ({r["task_id"]: r for r in fd},
                      {r["task_id"]: r for r in term})
    if len(fd_by) != 989 or len(term_by) != 2154:
        raise RuntimeError("ablation duplicate source identity")
    schedule = manifest["schedule"]
    if len(schedule) != 64:
        raise RuntimeError("ablation must be exactly 64 steps")
    pairs = []
    exposure = {"FD": 0, "terminal_stop": 0}
    for index, item in enumerate(schedule):
        if item["step"] != index + 1 or item["rank0"]["kind"] != "FD":
            raise RuntimeError("ablation schedule order/kind drift")
        rows = []
        for rank in ("rank0", "rank1"):
            entry = item[rank]
            kind = entry["kind"]
            row = (fd_by if kind == "FD" else term_by if kind == "terminal_stop" else {})[entry["task_id"]]
            if row["target_type"] != ("first_divergence_tool" if kind == "FD" else "terminal_stop"):
                raise RuntimeError("ablation target type drift")
            exposure[kind] += 1
            rows.append(row)
        pairs.append(tuple(rows))
    if (exposure != {"FD": 85, "terminal_stop": 43} if ratio == 2
            else exposure != {"FD": 102, "terminal_stop": 26}):
        raise RuntimeError("actual exposure ratio drift")
    if any(exposure[k] != manifest["counts"][k] for k in exposure):
        raise RuntimeError("manifest exposure mismatch")
    plan = {"schema_version": "cycle3_eos_ratio_ddp_plan_v1",
            "ratio": ratio, "actual_ratio": exposure["FD"] / exposure["terminal_stop"],
            "steps": 64, "world_size": 2, "per_rank_batch_size": 1,
            "initialization": str(MODEL), "initialization_sha256": sha256(MODEL / "model.safetensors"),
            "precision": "bfloat16", "full_parameter": True, "optimizer": "AdamW",
            "scheduler": "none_constant_lr_as_prior_CE_trainer",
            "learning_rate": LR, "weight_decay": 0.0,
            "gradient_checkpointing": True, "seed": SEED,
            "schedule_sha256": sha256(OUT / f"schedule_{ratio}.json"),
            "source_train_manifest_sha256": sha256(DATA / "train_manifest.json"),
            "source_sampling_manifest_sha256": sha256(DATA / "sampling_manifest.json"),
            "exposures": exposure,
            "selected_input_tokens": sum(a["total_tokens"] + b["total_tokens"] for a, b in pairs),
            "selected_target_tokens": sum(a["target_tokens"] + b["target_tokens"] for a, b in pairs)}
    return pairs, plan


def train(ratio: int, output: Path, logs: Path) -> None:
    os.environ.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    os.environ["PATH"] = str(Path(os.environ["CUDA_HOME"]) / "bin") + ":" + os.environ.get("PATH", "")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or local_rank not in (0, 1):
        raise RuntimeError("ablation requires exactly two GPUs")
    if output.exists() or logs.exists():
        raise RuntimeError("refusing to overwrite ablation training")
    pairs, plan = load_plan(ratio)
    torch.cuda.set_device(local_rank)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dist.init_process_group("nccl")
    if rank == 0:
        logs.mkdir(parents=True)
        (logs / "plan.json").write_text(stable(plan) + "\n")
    dist.barrier()
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True).to(f"cuda:{local_rank}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
              find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=LR, weight_decay=0.0)
    records = []
    for step, (a, b) in enumerate(pairs, start=1):
        row = a if rank == 0 else b
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=local_rank)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=local_rank)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        finite = torch.tensor([int(torch.isfinite(loss).item())], device=local_rank)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(grad).item())], device=local_rank)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite grad norm at step {step}")
        optimizer.step()
        losses = [torch.zeros((), dtype=torch.float32, device=local_rank) for _ in range(2)]
        dist.all_gather(losses, loss.detach().float())
        torch.cuda.synchronize(local_rank)
        if rank == 0:
            kinds = ("FD", "FD" if b["target_type"] == "first_divergence_tool" else "terminal_stop")
            record = {"step": step, "rank0_kind": kinds[0], "rank1_kind": kinds[1],
                      "rank0_task_id": a["task_id"], "rank1_task_id": b["task_id"],
                      "rank0_loss": float(losses[0].cpu()),
                      "rank1_loss": float(losses[1].cpu()),
                      "fd_loss_mean": float(((losses[0] + losses[1]) / 2).cpu()) if kinds[1] == "FD" else float(losses[0].cpu()),
                      "terminal_stop_loss": float(losses[1].cpu()) if kinds[1] == "terminal_stop" else None,
                      "mean_loss": float(((losses[0] + losses[1]) / 2).cpu()),
                      "grad_norm": float(grad.detach().cpu()),
                      "learning_rate": optimizer.param_groups[0]["lr"],
                      "rank0_input_tokens": a["total_tokens"],
                      "rank1_input_tokens": b["total_tokens"],
                      "rank0_target_tokens": a["target_tokens"],
                      "rank1_target_tokens": b["target_tokens"],
                      "rank0_peak_gb": round(torch.cuda.max_memory_allocated(local_rank) / 1024**3, 3),
                      "elapsed_seconds": round(time.monotonic() - start, 2)}
            records.append(record)
            with (logs / "training_metrics.jsonl").open("a") as stream:
                stream.write(stable(record) + "\n")
            print(stable(record), flush=True)
    peak = torch.tensor([torch.cuda.max_memory_allocated(local_rank)],
                        dtype=torch.float64, device=local_rank)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank == 0:
        output.mkdir(parents=True)
        model.eval()
        model.save_pretrained(output, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True,
                                                  local_files_only=True)
        tokenizer.save_pretrained(output)
        torch.save(optimizer.state_dict(), output / "optimizer_state.pt")
        (output / "scheduler_state.json").write_text(stable({
            "family": "none", "constant_learning_rate": LR,
            "last_completed_step": len(pairs)}) + "\n")
        result = {"schema_version": "cycle3_eos_ratio_ddp_result_v1",
                  "status": "TRAINED", "ratio": ratio, "actual_ratio": plan["actual_ratio"],
                  "steps": 64, "world_size": 2, "full_parameter": True,
                  "precision": "bfloat16", "all_loss_finite": all(r["mean_loss"] == r["mean_loss"] for r in records),
                  "all_grad_finite": all(r["grad_norm"] == r["grad_norm"] for r in records),
                  "exposures": plan["exposures"],
                  "wrong_argument_exposures": json.loads((OUT / f"schedule_{ratio}.json").read_text())["counts"]["WRONG_ARGUMENT"],
                  "depth_ge2_exposures": json.loads((OUT / f"schedule_{ratio}.json").read_text())["counts"]["depth_ge2"],
                  "other_fd_exposures": json.loads((OUT / f"schedule_{ratio}.json").read_text())["counts"]["other_fd"],
                  "trained_input_tokens": plan["selected_input_tokens"],
                  "trained_target_tokens": plan["selected_target_tokens"],
                  "initialization_sha256": plan["initialization_sha256"],
                  "final_loss": records[-1]["mean_loss"],
                  "final_grad_norm": records[-1]["grad_norm"],
                  "peak_gpu_gb": round(peak.item() / 1024**3, 3),
                  "wall_clock_seconds": round(time.monotonic() - start, 2),
                  "steps_per_second": round(len(pairs) / (time.monotonic() - start), 5),
                  "model_sha256": sha256(output / "model.safetensors"),
                  "optimizer_sha256": sha256(output / "optimizer_state.pt"),
                  "schedule_sha256": plan["schedule_sha256"]}
        (output / "trainer_state.json").write_text(stable(result) + "\n")
        (logs / "result.json").write_text(stable(result) + "\n")
        print(stable(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int, choices=(2, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    train(args.ratio, args.output, args.logs)


if __name__ == "__main__":
    main()

"""Full-parameter Dynamic-v1 DPO entry point for matched preference pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {"train": str(args.train_file)}
    if args.eval_file is not None and args.eval_file.exists() and args.eval_file.stat().st_size:
        data_files["validation"] = str(args.eval_file)
    dataset = load_dataset("json", data_files=data_files)
    if len(dataset["train"]) == 0:
        raise RuntimeError("empty DPO training dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "local_files_only": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    ref_model.config.use_cache = False
    ref_model.requires_grad_(False)

    has_eval = "validation" in dataset and len(dataset["validation"]) > 0
    config = DPOConfig(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=has_eval,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.max_steps if has_eval else None,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=2,
        save_only_model=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        deepspeed=str(args.deepspeed),
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=True,
        include_num_input_tokens_seen=True,
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_length,
        max_prompt_length=args.max_length,
        max_completion_length=args.max_length,
        truncation_mode="keep_end",
        beta=args.beta,
        loss_type="sigmoid",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_eval else None,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "final_model"))
    trainer.save_state()
    if has_eval:
        eval_metrics = trainer.evaluate()
    else:
        eval_metrics = {}

    if rank == 0:
        run_manifest = {
            "schema_version": "envfactory_preference_dpo_run_v1",
            "run_name": args.run_name,
            "initialization_model": args.model,
            "reference_model": args.model,
            "full_parameter": True,
            "beta": args.beta,
            "loss_type": "sigmoid",
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "seed": args.seed,
            "train_file": str(args.train_file),
            "train_sha256": sha256(args.train_file),
            "eval_file": str(args.eval_file) if has_eval else None,
            "eval_sha256": sha256(args.eval_file) if has_eval else None,
            "train_examples": len(dataset["train"]),
            "eval_examples": len(dataset["validation"]) if has_eval else 0,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "log_history": trainer.state.log_history,
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(json.dumps(run_manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()


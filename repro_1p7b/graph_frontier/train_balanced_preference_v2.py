"""Bounded full-parameter Dynamic-v1 DPO with state-type validation."""
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, required=True)
    p.add_argument("--preference-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--deepspeed", type=Path, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    frozen = json.loads(args.preference_manifest.read_text())
    if frozen["verdict"] != "DATA_READY" or frozen["initialization_model"] != args.model:
        raise ValueError("frozen preference manifest/model mismatch")
    files = {"train": args.train_file, "validation": args.eval_file}
    known_hashes = json.loads((args.preference_manifest.parent / "serialization_audit.json").read_text())["hashes"]
    for key, path in files.items():
        expected = known_hashes[path.name]
        if digest(path) != expected:
            raise ValueError(f"serialized {key} hash changed")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("json", data_files={key: str(path) for key, path in files.items()})
    expected_types = {"continue_required", "stop_required", "downstream_continue"}
    for split in ds:
        if set(ds[split]["state_type"]) != expected_types:
            raise ValueError(f"{split} missing preference type")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    reference = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.use_cache = reference.config.use_cache = False
    reference.requires_grad_(False)
    config = DPOConfig(
        output_dir=str(args.output_dir), run_name=args.run_name, overwrite_output_dir=False,
        do_train=True, do_eval=True, eval_strategy="no",
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-6, max_steps=args.max_steps,
        lr_scheduler_type="constant", warmup_steps=0,
        logging_strategy="steps", logging_steps=1, logging_first_step=True,
        save_strategy="steps", save_steps=args.max_steps, save_total_limit=2, save_only_model=False,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_grad_norm=1.0,
        optim="adamw_torch_fused", deepspeed=str(args.deepspeed), ddp_find_unused_parameters=False,
        report_to="none", remove_unused_columns=True, include_num_input_tokens_seen=True,
        seed=args.seed, data_seed=args.seed, max_length=12288, max_prompt_length=12288,
        max_completion_length=12288, truncation_mode="keep_end", beta=0.1, loss_type="sigmoid",
    )
    trainer = DPOTrainer(model=model, ref_model=reference, args=config,
                         train_dataset=ds["train"], eval_dataset=ds["validation"],
                         processing_class=tokenizer)
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_model"))
    tokenizer.save_pretrained(str(args.output_dir / "final_model"))
    trainer.save_state()
    type_metrics = {}
    for state_type in sorted(expected_types):
        subset = trainer.eval_dataset.filter(lambda row: row["state_type"] == state_type)
        if len(subset) == 0:
            raise ValueError(f"empty processed validation type: {state_type}")
        type_metrics[state_type] = trainer.evaluate(eval_dataset=subset, metric_key_prefix=state_type)
    eval_metrics = trainer.evaluate(metric_key_prefix="overall")
    if rank == 0:
        manifest = {
            "schema_version": "balanced_preference_v2_training_v1", "run_name": args.run_name,
            "initialization_model": args.model, "reference_model": args.model,
            "full_parameter": True, "gpus": [0, 1], "max_steps": args.max_steps,
            "seed": args.seed, "beta": 0.1, "learning_rate": 1e-6,
            "preference_manifest_sha256": digest(args.preference_manifest),
            "train_sha256": digest(args.train_file), "eval_sha256": digest(args.eval_file),
            "train_examples": len(ds["train"]), "eval_examples": len(ds["validation"]),
            "train_metrics": result.metrics, "eval_metrics": eval_metrics,
            "type_eval_metrics": type_metrics, "log_history": trainer.state.log_history,
        }
        (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps(manifest, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()

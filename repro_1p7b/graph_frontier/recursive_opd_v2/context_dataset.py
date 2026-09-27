"""Immutable, context-accounted D0 builder; CPU only and no prompt truncation."""
from __future__ import annotations

import collections
import json

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, check
from repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics import (
    LIMIT, account_context, context_class, distribution, sample_lengths,
)
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import (
    REPORT, RUN, digest, encode_row, load_cycle, load_verified_episode,
    natural_stats, source_policy_gate,
)

OUTPUT = RUN / "d0_context16k_v2"
FORENSICS = REPORT / "recursive_opd_v2_context_forensics.json"
VERSION = "recursive_opd_v2_context_accounted_1"


def build() -> dict:
    from transformers import AutoTokenizer

    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite context-accounted D0: {OUTPUT}")
    forensic = json.loads(FORENSICS.read_text())
    if (forensic.get("inference_case") != "B"
            or forensic.get("server_side_truncation") != "NO"
            or forensic.get("protocol_decision") != "SINGLE_CONTEXT_OVERFLOW"
            or forensic.get("semantic_eligible_count") != 1115
            or forensic.get("context_compatible_count") != 1114
            or forensic.get("context_overflow_count") != 1):
        raise RuntimeError("16k context protocol cannot be adopted from inconclusive forensics")
    protocol = check()
    rows, rollout_manifest, source_hashes = load_cycle(0, protocol)
    source_policy_gate(0, rollout_manifest, protocol)
    source = json.loads(SOURCE.read_text())
    entries = {entry["audit_id"]: entry for entry in
               json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645 or sha256(MODEL / "tokenizer.json") != protocol["tokenizer_sha256"]:
        raise RuntimeError("original tokenizer identity drift")
    encoded, overflow, gold_sources, full_lengths = [], [], {}, []
    eligible = 0
    for record in rows:
        if record["pool"] != "train" or record["status"] == "UNCERTAIN":
            continue
        eligible += 1
        episode, schema, gold_sha = load_verified_episode(
            record, rollout_manifest, source_hashes["manifest_sha256"], source, entries)
        from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import checked_prefix
        prefix = checked_prefix(record, episode)
        prompt_len, full_len = sample_lengths(tokenizer, prefix, schema, record)
        full_lengths.append(full_len)
        gold_sources[record["task_id"]] = gold_sha
        if context_class(full_len, LIMIT) == "CONTEXT_OVERFLOW":
            overflow.append({"task_id": record["task_id"], "query_hash": record["query_hash"],
                             "kind": record["error_regime"] if record["status"] == "FIRST_ERROR"
                             else "SUCCESS_TERMINAL_POSITIVE",
                             "semantic_eligible": True, "execution_verified": True,
                             "train_context_compatible": False, "reason": "CONTEXT_OVERFLOW",
                             "prompt_tokens_local": prompt_len, "full_tokens": full_len,
                             "gold_depth": record["gold_tool_count"]})
            continue
        item = encode_row(tokenizer, record, episode, schema, gold_sha)
        if len(item["input_ids"]) != full_len or len(item["input_ids"]) > LIMIT:
            raise RuntimeError("context length differs between audit and encoded target")
        item["semantic_eligible"] = True
        item["execution_verified"] = True
        item["train_context_compatible"] = True
        encoded.append(item)
        if eligible % 250 == 0:
            print(stable({"eligible_inspected": eligible, "encoded": len(encoded)}), flush=True)
    account_context(semantic_eligible=eligible, encoded=len(encoded), overflow=overflow)
    expected = natural_stats(rows, 0)["train"]
    forensic_overflow = forensic["context_overflow_records"]
    if (eligible != expected["v2_natural_rows"] or eligible != 1115
            or len(overflow) != 1 or overflow[0]["task_id"] != forensic_overflow[0]["task_id"]
            or overflow[0]["full_tokens"] != forensic_overflow[0]["full_tokens"]
            or distribution(full_lengths) != forensic["length_distribution"]["ALL"]
            or len({row["task_id"] for row in encoded}) != len(encoded)
            or len({row["query_hash"] for row in encoded}) != len(encoded)
            or {row["task_id"] for row in encoded} & set(protocol["heldout_task_ids"])):
        raise RuntimeError("D0 context, split, or forensic identity gate failed")
    count = collections.Counter(row["correction_kind"] for row in encoded)
    OUTPUT.mkdir(parents=True)
    dataset = OUTPUT / "dataset.jsonl"
    dataset.write_text("".join(stable(row) + "\n" for row in encoded))
    manifest = {
        "schema_version": VERSION, "status": "READY_FOR_GPU_TRAINING", "cycle": 0,
        "usage": "TRAINABLE_V2_D0", "builder_version": VERSION,
        "context_protocol": "only exact-prefix, context-compatible samples are trainable",
        "max_train_context": LIMIT,
        "semantic_eligible_count": eligible, "context_compatible_count": len(encoded),
        "trainable_count": len(encoded), "encoded_rows": len(encoded),
        "context_overflow_count": len(overflow), "context_overflow_records": overflow,
        "skipped_uncertain_count": expected["UNCERTAIN"],
        "heldout_training_rows": 0, "uncertain_training_rows": 0,
        "sample_count": len(encoded), "eligible_task_count": eligible,
        "class_counts": dict(count), "tool_targets": count["START"] + count["ADVANCE"],
        "eos_targets": count["TERMINATE"] + count["SUCCESS_TERMINAL_POSITIVE"],
        "natural_eos_fraction": expected["v2_eos_fraction"],
        "source_pi0_model_sha256": rollout_manifest["model_sha256"],
        "source_rollout_manifest_sha256": source_hashes["manifest_sha256"],
        "source_v1_metrics_sha256": source_hashes["metrics_sha256"],
        "gold_source_sha256": digest(gold_sources),
        "source_clean_dataset_sha256": sha256(SOURCE),
        "task_split_sha256": digest({"train_task_ids": protocol["train_task_ids"],
                                     "heldout_task_ids": protocol["heldout_task_ids"]}),
        "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
        "protocol_sha256": sha256(PROTOCOL),
        "forensic_sha256": sha256(FORENSICS),
        "dataset_sha256": sha256(dataset),
        "gates": {name: "PASS" for name in (
            "frozen_pi0_source", "v1_verifier_metrics_hashes", "gold_source_identity",
            "student_exact_predecision_prefix", "success_verified_terminal_and_stop",
            "terminate_verified_overcontinuation", "gold_next_tool_pointer",
            "one_task_one_target", "zero_task_overlap", "zero_query_hash_overlap",
            "heldout_zero_rows", "no_uncertain_rows", "tokenizer_eos_identity",
            "assistant_only_loss_mask", "eos_only_single_label", "dataset_hash",
            "context_accounting", "single_explicit_context_overflow",
            "case_b_no_server_truncation", "all_context_compatible_rows_encoded")},
    }
    (OUTPUT / "manifest.json").write_text(stable(manifest) + "\n")
    print(stable({"status": manifest["status"], "eligible": eligible,
                  "encoded": len(encoded), "overflow": len(overflow),
                  "dataset_sha256": manifest["dataset_sha256"]}), flush=True)
    return manifest


if __name__ == "__main__":
    build()

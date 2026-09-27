"""Read-only CPU audit of all saved V1 verdicts and V2 D0 token targets."""
from __future__ import annotations

import collections
import json
import random

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, check
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import (
    REPORT, RUN, checked_prefix, digest, encode_row, load_cycle,
    load_verified_episode,
)
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


def audit(*, context_accounted: bool = False) -> dict:
    from transformers import AutoTokenizer

    protocol = check()
    source = json.loads(SOURCE.read_text())
    entries = {row["audit_id"]: row for row in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    retrospective = []
    for cycle in range(4):
        rows, manifest, identity = load_cycle(cycle, protocol)
        counts = collections.Counter()
        for i, record in enumerate(rows):
            if record["status"] == "UNCERTAIN":
                counts["UNCERTAIN"] += 1
                continue
            episode, _, _ = load_verified_episode(record, manifest, identity["manifest_sha256"], source, entries)
            prefix = checked_prefix(record, episode)
            if not prefix:
                raise RuntimeError("empty verified student prefix")
            counts[record["status"]] += 1
            if i % 500 == 0:
                print(stable({"cycle": cycle, "inspected": i + 1}), flush=True)
        retrospective.append({"cycle": cycle, "usage": "TRAINABLE_V2_D0" if cycle == 0 else "RETROSPECTIVE_ONLY",
                              "saved_verdicts": len(rows), "checked": dict(counts)})
    dataset_dir = RUN / ("d0_context16k_v2" if context_accounted else "d0")
    dataset = dataset_dir / "dataset.jsonl"
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    if manifest["dataset_sha256"] != sha256(dataset):
        raise RuntimeError("V2 candidate dataset hash drift")
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    if context_accounted:
        from repro_1p7b.graph_frontier.recursive_opd_v2.train import validate_context_manifest
        if validate_context_manifest(manifest, rows, cycle=0) != 1:
            raise RuntimeError("independent context overflow accounting audit failed")
    train, heldout = set(protocol["train_task_ids"]), set(protocol["heldout_task_ids"])
    if (len(rows) != manifest["sample_count"] or len({r["task_id"] for r in rows}) != len(rows)
            or any(r["task_id"] not in train or r["task_id"] in heldout for r in rows)):
        raise RuntimeError("D0 rows/split/uniqueness drift")
    count = collections.Counter()
    for row in rows:
        validate_mask(row)
        if (row["labels"][:row["prompt_tokens"]] != [-100] * row["prompt_tokens"]
                or row["tool_observation_tokens_in_loss"] != 0
                or row["historical_assistant_tokens_in_loss"] != 0):
            raise RuntimeError("assistant-only full-dataset token mask failed")
        if row["target_type"] == "terminal_stop":
            if sum(label != -100 for label in row["labels"]) != 1:
                raise RuntimeError("EOS target has more than one supervised token")
        elif row["target_type"] != "first_divergence_tool":
            raise RuntimeError("unknown V2 target type")
        count[row["correction_kind"]] += 1
    if dict(count) != manifest["class_counts"]:
        raise RuntimeError("D0 class-count drift")
    if context_accounted and (manifest["semantic_eligible_count"] != 1115
                              or manifest["context_compatible_count"] != 1114
                              or manifest["sample_count"] != 1114):
        raise RuntimeError("context-accounted D0 population drift")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("tokenizer EOS drift")
    rng = random.Random(20260927)
    sampled = []
    for kind in sorted(count):
        group = [r for r in rows if r["correction_kind"] == kind]
        sampled.extend(rng.sample(group, min(16, len(group))))
    cycle0_rows, cycle0_manifest, cycle0_identity = load_cycle(0, protocol)
    by_task = {r["task_id"]: r for r in cycle0_rows}
    for row in sampled:
        record = by_task[row["task_id"]]
        episode, schema, gold_sha = load_verified_episode(record, cycle0_manifest,
                                                           cycle0_identity["manifest_sha256"], source, entries)
        fresh = encode_row(tokenizer, record, episode, schema, gold_sha)
        if (fresh["input_ids"] != row["input_ids"] or fresh["labels"] != row["labels"]
                or fresh["student_prefix_sha256"] != row["student_prefix_sha256"]
                or digest(checked_prefix(record, episode)) != row["student_prefix_sha256"]):
            raise RuntimeError("random token-level re-encoding differs from immutable D0 row")
    result = {"status": "AUDITED", "verdict": manifest["status"],
              "all_saved_verdicts_checked": sum(x["saved_verdicts"] for x in retrospective),
              "execution_verified_prefixes_checked": sum(x["checked"].get("FIRST_ERROR", 0)
                                                         + x["checked"].get("SUCCESS", 0) for x in retrospective),
              "retrospective": retrospective, "d0_rows_full_mask_checked": len(rows),
              "random_reencoded": len(sampled), "random_reencoded_by_kind": dict(collections.Counter(
                  r["correction_kind"] for r in sampled)),
              "candidate_dataset_sha256": manifest["dataset_sha256"],
              "blocking_encoding_failures": manifest.get("encoding_failures", []),
              "semantic_eligible_count": manifest.get("semantic_eligible_count"),
              "context_overflow_count": manifest.get("context_overflow_count", 0),
              "gates": {"all_saved_episode_hashes": "PASS", "all_eligible_student_prefixes": "PASS",
                        "full_d0_assistant_only_mask": "PASS", "random_token_reencode": "PASS",
                        "all_train_context_compatible_encoded": manifest["gates"][
                            "all_context_compatible_rows_encoded" if context_accounted
                            else "all_eligible_rows_encoded"]}}
    path = REPORT / ("recursive_opd_v2_context_token_audit.json" if context_accounted
                     else "recursive_opd_v2_token_audit.json")
    if path.exists():
        raise RuntimeError("refusing to overwrite V2 token audit")
    path.write_text(stable(result) + "\n")
    return result


if __name__ == "__main__":
    import sys
    result = audit(context_accounted="--context-accounted" in sys.argv)
    print(stable({"status": result["status"], "verdict": result["verdict"],
                  "prefixes": result["execution_verified_prefixes_checked"],
                  "reencoded": result["random_reencoded"]}), flush=True)

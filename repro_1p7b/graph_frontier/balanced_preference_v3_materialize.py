"""Serialize the immutable v3 preference gate with the proven v2 template."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from repro_1p7b.graph_frontier.balanced_preference_v3_pairs import MODEL, OUT, TYPES
from repro_1p7b.graph_frontier.balanced_preference_v3_source import read, rows, write
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256


def _immutable_rows(path: Path, values: list[dict]) -> None:
    payload = "".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"frozen serialized rows changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def materialize() -> dict:
    from transformers import AutoTokenizer
    from repro_1p7b.graph_frontier.materialize_preference_dpo import serialize_pair

    manifest = read(OUT / "manifest.json")
    audit = read(OUT / "audit.json")
    if manifest["verdict"] != "DATA_READY" or audit["verdict"] != "DATA_READY" or \
            manifest["initialization_model"] != str(MODEL):
        raise ValueError("v3 preference gate/model mismatch")
    for name in ("pairs_all.jsonl", "pairs_train.jsonl", "pairs_val.jsonl"):
        if file_sha256(OUT / name) != manifest["hashes"][name]:
            raise ValueError(f"frozen preference hash changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    serialized = {}
    drops = Counter()
    for split in ("train", "val"):
        values = []
        for pair in rows(OUT / f"pairs_{split}.jsonl"):
            item, reason = serialize_pair(tokenizer, pair, 8192)
            if reason:
                drops[f"{split}:{reason}"] += 1
            else:
                item["state_type"] = pair["state_type"]
                values.append(item)
        serialized[split] = values
    smoke = []
    for state_type in TYPES:
        smoke += [r for r in serialized["train"] if r["state_type"] == state_type][:8]
    val_types = Counter(r["state_type"] for r in serialized["val"])
    ready = (not drops and len(serialized["train"]) == manifest["train_pair_count"] and
             len(serialized["val"]) == manifest["validation_pair_count"] and
             len(smoke) == 24 and set(val_types) == set(TYPES))
    report = {"verdict": "DPO_READY" if ready else "SERIALIZATION_GATE_FAILED",
              "train_rows": len(serialized["train"]), "val_rows": len(serialized["val"]),
              "smoke_rows": len(smoke), "smoke_types": dict(Counter(r["state_type"] for r in smoke)),
              "validation_types": dict(val_types), "drops": dict(drops), "max_length": 8192,
              "template_reused": "materialize_preference_dpo.serialize_pair"}
    if ready:
        _immutable_rows(OUT / "dpo_train.jsonl", serialized["train"])
        _immutable_rows(OUT / "dpo_val.jsonl", serialized["val"])
        _immutable_rows(OUT / "dpo_smoke24.jsonl", smoke)
        report["hashes"] = {name: file_sha256(OUT / name) for name in
                            ("dpo_train.jsonl", "dpo_val.jsonl", "dpo_smoke24.jsonl")}
    write(OUT / "serialization_audit.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(materialize(), sort_keys=True))

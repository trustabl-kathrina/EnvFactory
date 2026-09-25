"""Read-only Frozen300 evaluation shard for a new checkpoint."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from repro_1p7b.graph_frontier import confirm_300 as frozen

EXPECTED_MANIFEST_SHA256 = "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.shard_index not in (0, 1):
        raise ValueError("shard index must be 0 or 1")
    model = args.model_path.resolve()
    manifest = args.manifest.resolve()
    required = [model / "config.json", model / "tokenizer_config.json",
                model / "tokenizer.json", model / "model.safetensors"]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete checkpoint: {model}")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Frozen300 manifest hash mismatch")
    frozen.PATHS[args.label] = str(model)
    frozen.pilot.register()
    try:
        summary = asyncio.run(frozen.run(
            manifest, args.label, args.output_dir.resolve(),
            shard_count=2, shard_index=args.shard_index,
        ))
    finally:
        frozen.MCPManager.shutdown()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

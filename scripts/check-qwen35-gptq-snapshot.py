#!/usr/bin/env python3
"""Verify local Qwen3.5-122B-A10B-GPTQ-Int4 snapshot vs Hugging Face (tokenizer + weight shards)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import list_repo_files

REPO = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
REQUIRED_TOKENIZER = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root",
        nargs="?",
        default="vllm-models/Qwen3.5-122B-A10B-GPTQ-Int4",
        help="Local model directory (default: vllm-models/Qwen3.5-122B-A10B-GPTQ-Int4)",
    )
    args = p.parse_args()
    root = Path(args.root).resolve()
    print(f"Local dir: {root}")
    print(f"Hub repo:  {REPO}")
    print()

    hub_files = list_repo_files(repo_id=REPO)
    hub_shards = sorted(f for f in hub_files if f.startswith("model.safetensors-") and f.endswith(".safetensors"))
    print(f"Weight shards on Hub: {len(hub_shards)}")
    if root.is_dir():
        local_shards = sorted(root.glob("model.safetensors-*.safetensors"))
        print(f"Weight shards locally: {len(local_shards)}")
        if len(local_shards) != len(hub_shards):
            print(
                "  WARNING: shard count mismatch — weights incomplete or wrong folder; "
                "vLLM may fail or behave oddly until snapshot_download finishes.",
            )
    else:
        print("  (local directory missing)")
    print()

    print("Tokenizer / text assets:")
    bad = False
    for name in REQUIRED_TOKENIZER:
        path = root / name
        ok = path.is_file() and path.stat().st_size > 0
        status = "OK" if ok else "MISSING"
        if not ok:
            bad = True
        extra = f" ({path.stat().st_size} bytes)" if ok else ""
        print(f"  {status:7} {name}{extra}")

    print()
    if bad:
        print(
            "Fix: run model-download again (compose now fetches tokenizer files if config exists), or:\n"
            "  python3 -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='{REPO}', local_dir='{root}', "
            "local_dir_use_symlinks=False, allow_patterns=['tokenizer.json','tokenizer_config.json','vocab.json'])\"",
        )
        return 1
    print("Tokenizer files look present. Prefer --tokenizer pointing at this same directory as the model weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

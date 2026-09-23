"""Temporary helper: download a HF repo into a flat ``models/<basename>`` layout.

This reproduces the existing checkpoint layout (config.json, model.safetensors,
tokenizer files at the top level, no symlink/hub-cache clutter) so that
``CausalLM._resolve_local_checkpoint`` picks it up directly.

Usage: python scripts/_dl_model.py REPO_ID
Example: python scripts/_dl_model.py Qwen/Qwen2.5-3B
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

_IGNORED_HUB_ARTIFACTS = {".cache", ".no_exist", ".gitattributes", "README.md"}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    repo_id = sys.argv[1]
    target = root / "models" / os.path.basename(repo_id)
    target.mkdir(parents=True, exist_ok=True)

    tmp = target.with_suffix(".dl")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"[dl] downloading {repo_id} -> {tmp}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(tmp),
    )

    # Flatten: move everything out of any subfolders (trusted repos keep files flat).
    for sub in list(tmp.iterdir()):
        if sub.is_dir() and sub.name not in _IGNORED_HUB_ARTIFACTS:
            for f in sub.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(tmp / f.name))
            shutil.rmtree(sub)

    for artifact in _IGNORED_HUB_ARTIFACTS:
        p = tmp / artifact
        if p.exists():
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()

    has_weights = (tmp / "model.safetensors").exists() or list(
        tmp.glob("model-*-of-*.safetensors")
    )
    if not has_weights:
        print("[dl] ERROR: no model.safetensors / weight shards found", flush=True)
        return 1

    # Atomically replace target with the fresh download.
    if target.exists():
        shutil.rmtree(target)
    os.rename(str(tmp), str(target))
    print(f"[dl] flattened checkpoint ready at {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
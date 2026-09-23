"""Temporary helper: download a HF-compatible repo from ModelScope (domestic mirror).

The weights are byte-identical to HuggingFace (``model-0000X-of-0000Y.safetensors``
shards plus ``config.json`` and tokenizer files), so they drop straight into the
flat ``models/<basename>`` layout that ``_resolve_local_checkpoint`` already
recognises (shard support was added for Qwen2.5-3B).

Usage: python scripts/_dl_modelscope.py Qwen/Qwen2.5-3B
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from modelscope import snapshot_download

_IGNORED = {".cache", ".no_exist", ".gitattributes", "README.md", "LICENSE.txt"}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    repo = sys.argv[1]
    target = root / "models" / os.path.basename(repo)
    target.mkdir(parents=True, exist_ok=True)

    print(f"[ms] downloading {repo} -> {target}", flush=True)
    snapshot_download(repo, local_dir=str(target))

    for artifact in _IGNORED:
        p = target / artifact
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    has_weights = (target / "model.safetensors").exists() or list(
        target.glob("model-*-of-*.safetensors")
    )
    if not has_weights:
        print("[ms] ERROR: no weights after download", flush=True)
        return 1
    print(f"[ms] ready: {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
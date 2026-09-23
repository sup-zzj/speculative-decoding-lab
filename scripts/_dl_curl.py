"""Download Qwen2.5-3B weights straight from ModelScope with curl.exe.

The ``modelscope`` pip package crashes on Python 3.8 (`list[int]` is a 3.9+
annotation), so we bypass the library entirely and fetch the flat repo files
with curl, which handles retries and ``-C -`` resume for the two ~3 GB shards.

Files are placed in ``models/Qwen2.5-3B/``, the same flat layout the trainer
already resolves (single ``model.safetensors`` or ``model-*-of-*.safetensors``
shards).
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = "https://modelscope.cn/models/Qwen/Qwen2.5-3B/resolve/master"
BASE_NAME = "Qwen2.5-3B"
FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def resolve_curl() -> str:
    path = shutil.which("curl.exe") or shutil.which("curl")
    if not path:
        raise SystemExit("curl.exe not found on PATH")
    return path


def fetch_one(curl: str, url: str, dest: Path) -> None:
    cmd = [
        curl,
        "-L", "-C", "-",
        "--retry", "5", "--retry-delay", "2", "--retry-all-errors",
        "-s", "-S",
        "-o", str(dest), url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed for {dest.name}: {proc.stderr[-400:]}")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    target = root / "models" / BASE_NAME
    target.mkdir(parents=True, exist_ok=True)
    curl = resolve_curl()

    small = [f for f in FILES if "safetensors" not in f or f.startswith("model.safetensors.index")]
    shards = sorted(f for f in FILES if f.endswith(".safetensors") and "index" not in f)

    # Shards first (the slow part), parallel.
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {
            pool.submit(fetch_one, curl, f"{BASE}/{name}", target / name): name
            for name in shards
        }
        for fut in cf.as_completed(futs):
            fut.result()  # re-raise on failure
    for name in small:
        fetch_one(curl, f"{BASE}/{name}", target / name)
        print(f"[curl] ok {name} ({target/name})", flush=True)

    has = (target / "model.safetensors").exists() or list(
        target.glob("model-*-of-*.safetensors")
    )
    if not has:
        print("[curl] ERROR: no weights after download", flush=True)
        return 1
    total = sum(p.stat().st_size for p in target.iterdir() if p.is_file())
    print(f"[curl] DONE {round(total / 1e9, 2)} GB in {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
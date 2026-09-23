"""Shared argparse groups so every script exposes the same flags."""

from __future__ import annotations

import argparse
import os
from typing import Optional

DEFAULT_OUTPUT_DIR = "results"
DEFAULT_CACHE_DIR = "models"


def add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("models")
    group.add_argument(
        "--draft-model",
        default="Qwen/Qwen2.5-0.5B",
        help="small model that proposes draft tokens (token-level path requires the same "
        "tokenizer; different tokenizers fall back to text-level verification)",
    )
    group.add_argument(
        "--target-model",
        default="Qwen/Qwen2.5-1.5B",
        help="large model whose distribution must be reproduced",
    )
    group.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        help="auto -> cuda when available",
    )
    group.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="auto -> float16 on GPU, float32 on CPU",
    )
    group.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="where HuggingFace checkpoints are downloaded",
    )


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument("--temperature", type=float, default=0.0, help="<=0 means greedy")
    group.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    group.add_argument("--top-p", type=float, default=1.0, help="1.0 disables nucleus")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime")
    group.add_argument("--seed", type=int, default=20260920)
    group.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    group.add_argument("--prompt", default=None, help="single prompt or a .txt file path")


def validate_args(args: argparse.Namespace) -> None:
    """Fail fast with exit code 2 on obviously invalid combinations."""
    problems = []
    if getattr(args, "temperature", 0.0) < 0:
        problems.append("--temperature must be >= 0")
    if getattr(args, "top_k", 0) < 0:
        problems.append("--top-k must be >= 0")
    top_p = getattr(args, "top_p", 1.0)
    if not 0.0 < top_p <= 1.0:
        problems.append("--top-p must be in (0, 1]")
    gamma = getattr(args, "gamma", 1)
    if gamma is not None and (gamma < 1 or gamma > 32):
        problems.append("--gamma must be within [1, 32]")
    if problems:
        raise SystemExit("invalid arguments:\n  - " + "\n  - ".join(problems))


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sampling_from_args(args: argparse.Namespace):
    from specdec.config import SamplingConfig

    return SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
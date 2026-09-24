"""Seeded, deterministic generation for the serving engine.

The engine borrows the Phase-1 lab's CPU generator (see
``specdec.sampler.build_generator``) so that request arrival times, prompts and
sampled tokens are all reproducible from a single seed -- the same reason the
speculative-decoding experiments give every configuration an explicit ``seed``.
"""

from __future__ import annotations

from typing import Optional

import torch

from specdec.sampler import build_generator as _build_generator

__all__ = ["build_generator"]


def build_generator(
    seed: int, device: Optional[torch.device] = None
) -> torch.Generator:
    """Deterministic CPU generator seeded with ``seed`` (interface kept thin)."""
    return _build_generator(seed, device)
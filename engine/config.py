"""Typed configuration for the continuous-batching / paged-attention engine.

Phase 2 of the speculative-decoding lab: a from-scratch, vLLM-style serving
engine. The engine is *single-model* (no draft/verify phase here): it schedules a
dynamic set of heterogeneous requests over one ``PagedLM``, packs them into
packed paged-attention forwards, and manages KV in fixed-size physical blocks.

This module mirrors ``specdec/config.py``: every config object is a frozen
dataclass with a ``to_dict()`` so experiment scripts can record the full
configuration in their JSON report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class EngineConfig:
    """Model + paged KV parameters for a single serving engine."""

    model_id: str = "Qwen/Qwen2.5-1.5B"   # or a local checkpoint path
    device: str = "auto"
    dtype: str = "auto"
    cache_dir: str = "models"
    block_size: int = 16                    # tokens per physical KV block
    max_num_blocks: int = 256               # physical block capacity
    max_batch: int = 16                     # soft cap on sequences per step

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SamplingConfig:
    """Decoding temperature. ``temperature <= 0`` selects greedy decoding.

    Reuses the same semantic as ``specdec.config.SamplingConfig``: greedy is a
    one-hot distribution on the argmax token, so the scheduler can be verified
    against the autoregressive baseline token-for-token.
    """

    temperature: float = 0.0

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

    def describe(self) -> str:
        """Human-readable decoding config (logged into every ``SchedulerResult``)."""
        if self.is_greedy:
            return "greedy"
        return "temperature={}".format(self.temperature)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "mode": "greedy" if self.is_greedy else "sampling",
        }


@dataclass
class WorkloadRequest:
    """One generation request fed to the engine.

    ``arrive_time`` is expressed in *scheduler steps* (not wall-clock): the
    continuous-batching scheduler admits requests whose ``arrive_time <= current
    step``, which lets a workload model staggered arrivals without coupling to
    absolute timing.
    """

    prompt: str
    max_new_tokens: int
    arrive_time: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadConfig:
    """How to synthesise a batch of diverse requests for a serving experiment.

    ``num_requests`` requests with prompts sampled from ``prompts``, max-new-token
    lengths drawn in ``[min_tokens, max_tokens]`` (seeded), and arrivals spread
    over ``arrival_window`` steps. The heterogenous lengths *and* staggered
    arrivals are what make continuous batching win over static batching.
    """

    prompts: Sequence[str]
    num_requests: int = 8
    min_tokens: int = 8
    max_tokens: int = 32
    arrival_window: int = 4
    seed: int = 20260920

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["prompts"] = list(self.prompts)
        return payload


DEFAULT_PROMPTS: Tuple[str, ...] = (
    "The key idea behind a paged attention server is",
    "In a large-scale inference system, continuous batching",
    "The GPU is most underutilised during the decode phase",
    "A KV cache must be re-layouted to fit dynamic request",
    "Scheduling matters more than raw FLOPs for serving",
    "When one request finishes early its slot should be reused",
)


def config_from_namespace(args: Any, base: Optional[EngineConfig] = None) -> EngineConfig:
    """Fold an argparse namespace into an :class:`EngineConfig` (reuse pattern)."""
    config = base or EngineConfig()
    config.model_id = getattr(args, "model", config.model_id)
    config.device = getattr(args, "device", config.device)
    config.dtype = getattr(args, "dtype", config.dtype)
    config.cache_dir = getattr(args, "cache_dir", config.cache_dir)
    config.block_size = getattr(args, "block_size", config.block_size)
    config.max_num_blocks = getattr(args, "max_num_blocks", config.max_num_blocks)
    config.max_batch = getattr(args, "max_batch", config.max_batch)
    return config


def build_workload(
    cfg: WorkloadConfig,
    generator: Any,
) -> List[WorkloadRequest]:
    """Sample a heterogeneous request set from ``cfg`` (seeded via ``generator``).

    Each request gets a prompt cycled from ``cfg.prompts``, a max-new-token budget
    drawn uniformly in ``[min_tokens, max_tokens]``, and a tick ``arrive_time`` in
    ``[0, arrival_window]``. Using an explicit RNG keeps the workload reproducible
    across CPU/GPU runs.
    """
    requests: List[WorkloadRequest] = []
    for index in range(cfg.num_requests):
        prompt = cfg.prompts[index % len(cfg.prompts)]
        budget = int(torch.randint(cfg.min_tokens, cfg.max_tokens + 1, (1,), generator=generator).item())
        arrive = int(torch.randint(0, cfg.arrival_window + 1, (1,), generator=generator).item())
        requests.append(
            WorkloadRequest(prompt=prompt, max_new_tokens=budget, arrive_time=arrive)
        )
    return requests
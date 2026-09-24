"""Phase 2: a from-scratch continuous-batching / PagedAttention serving engine.

Extends the speculative-decoding lab with a serving engine: a dynamic set of
heterogeneous requests is scheduled over one ``PagedLM`` whose K/V lives in
fixed-size physical blocks (``PagedKVCache``), so sequences of different lengths
coexist and can be dropped/rewound independently. A toy model provides the
correctness harness; a Qwen2.5 adapter runs real weights through the same engine.
"""

from __future__ import annotations

from .attention import full_attention, gather_ctx, paged_attention_decode, paged_attention_prefill_chunk
from .benchmark import compare_schedulers, run_serving_experiment
from .config import EngineConfig, SamplingConfig, WorkloadConfig, WorkloadRequest, build_workload
from .kv import BlockAllocator, BlockExhaustedError, PagedKVCache
from .model import RMSNorm, ToyConfig, ToyPagedLM, ToyTokenizer, apply_rotary, make_rotary_cache, rotate_half
from .qwen import QwenPagedModel, build_paged_model
from .scheduler import (
    PagedScheduler,
    SchedulerResult,
    Sequence,
    build_kv,
)

__version__ = "0.2.0"

__all__ = [
    "BlockAllocator",
    "BlockExhaustedError",
    "EngineConfig",
    "PagedKVCache",
    "PagedScheduler",
    "QwenPagedModel",
    "RMSNorm",
    "SchedulerResult",
    "SamplingConfig",
    "Sequence",
    "ToyConfig",
    "ToyPagedLM",
    "ToyTokenizer",
    "WorkloadConfig",
    "WorkloadRequest",
    "apply_rotary",
    "build_kv",
    "build_paged_model",
    "build_workload",
    "compare_schedulers",
    "full_attention",
    "gather_ctx",
    "make_rotary_cache",
    "paged_attention_decode",
    "paged_attention_prefill_chunk",
    "rotate_half",
    "run_serving_experiment",
]
"""Speculative scheduler correctness: greedy spec == plain continuous greedy.

The decisive Phase-3 property is that adding a draft/verify round must NOT change
the emitted tokens under greedy decoding -- the cache rollback and per-sequence
bookkeeping must be exact. We run the plain ``PagedScheduler`` and a
``SpeculativePagedScheduler`` (target = draft toy models, same shared vocabulary)
over the same heterogeneous workload and assert every sequence matches. A
re-run also verifies determinism.
"""

from __future__ import annotations

import torch
import pytest

from engine.config import (
    DEFAULT_PROMPTS,
    EngineConfig,
    SamplingConfig,
    WorkloadConfig,
    build_workload,
)
from engine.model import ToyConfig, ToyPagedLM
from engine.sampler import build_generator
from engine.scheduler import PagedScheduler, build_kv
from engine.spec_scheduler import SpeculativePagedScheduler

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _workload(seed: int = 0):
    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS[:3],
        num_requests=4,
        min_tokens=4,
        max_tokens=10,
        arrival_window=2,
        seed=seed,
    )
    return build_workload(workload, build_generator(seed))


def test_spec_greedy_equals_plain_continuous() -> None:
    target = ToyPagedLM(ToyConfig(seed=0), _DEVICE)
    draft = ToyPagedLM(ToyConfig(seed=7), _DEVICE)  # different weights -> rejections
    cfg = EngineConfig(block_size=4, max_num_blocks=128)
    greedy = SamplingConfig(temperature=0.0)
    requests = _workload(seed=1)
    gamma = 2

    # Plain continuous batching reference.
    kv_base = build_kv(target, cfg, _DEVICE)
    base = PagedScheduler(target, kv_base, cfg, greedy).run_continuous(requests, seed=1)

    # Speculative engine: same target, independent caches, a distinct draft.
    kv_spec = build_kv(target, cfg, _DEVICE)
    dkv_spec = build_kv(draft, cfg, _DEVICE)
    sched = SpeculativePagedScheduler(
        target, kv_spec, draft, dkv_spec, cfg, greedy, gamma
    )
    spec = sched.run_spec_continuous(requests, seed=1)

    assert spec.mode == "spec-continuous"
    base_gen = sorted((s.max_new_tokens, tuple(s.generated)) for s in base.sequences)
    spec_gen = sorted((s.max_new_tokens, tuple(s.generated)) for s in spec.sequences)
    assert base_gen == spec_gen
    assert all(s.finished for s in spec.sequences)


def test_spec_is_deterministic() -> None:
    draft = ToyPagedLM(ToyConfig(seed=7), _DEVICE)
    cfg = EngineConfig(block_size=4, max_num_blocks=128)
    greedy = SamplingConfig(temperature=0.0)
    requests = _workload(seed=5)

    def _run() -> list[tuple[int, tuple]]:
        target = ToyPagedLM(ToyConfig(seed=0), _DEVICE)
        kv = build_kv(target, cfg, _DEVICE)
        dkv = build_kv(draft, cfg, _DEVICE)
        sched = SpeculativePagedScheduler(
            target, kv, draft, dkv, cfg, greedy, gamma=2
        )
        res = sched.run_spec_continuous(requests, seed=5)
        return sorted((s.max_new_tokens, tuple(s.generated)) for s in res.sequences)

    first = _run()
    second = _run()
    assert first == second
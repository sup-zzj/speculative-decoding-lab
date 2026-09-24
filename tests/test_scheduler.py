"""Tests for the continuous-batching scheduler.

The key correctness property of a serving engine is **greedy equivalence**: the
tokens the engine emits for a request must be identical to what a sole,
single-sequence auto-regressive decode would have produced. Because the engine's
paged attention is per-sequence (no cross-request interference), batching and
staggered admission must not change the sampled tokens.

Also checked is **determinism**: re-running the same seeded workload yields the
byte-identical token streams.
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
from engine.kv import PagedKVCache
from engine.model import ToyConfig, ToyPagedLM
from engine.sampler import build_generator
from engine.scheduler import PagedScheduler, build_kv

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model() -> ToyPagedLM:
    torch.manual_seed(0)
    return ToyPagedLM(ToyConfig(seed=0), _DEVICE)


@pytest.fixture
def kv_and_engine(model) -> None:
    pass


def _encode_text(model: ToyPagedLM, text: str) -> torch.Tensor:
    return torch.tensor([model.encode(text)], dtype=torch.long, device=_DEVICE)


def _single_sequence_reference(
    model: ToyPagedLM,
    kv: PagedKVCache,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    """Independent greedy reference using one sequence (no batching)."""
    sampled: list[int] = []
    logits = model.prefill(prompt_ids, kv, [0], [int(prompt_ids.shape[1])])
    probs = logits[0]
    sampled.append(int(probs.argmax(dim=-1).item()))
    for _ in range(max_new_tokens - 1):
        ctx = [int(kv.seq_lengths[0])]
        ids = torch.tensor([sampled[-1]], dtype=torch.long, device=_DEVICE)
        logits = model.decode(ids, kv, [0], ctx)
        sampled.append(int(logits[0].argmax(dim=-1).item()))
    return sampled


def test_continuous_greedy_equals_single_sequence_reference() -> None:
    model = ToyPagedLM(ToyConfig(seed=0), _DEVICE)
    cfg = EngineConfig(block_size=4, max_num_blocks=64)
    kv = build_kv(model, cfg, _DEVICE)
    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS[:3],
        num_requests=4,
        min_tokens=4,
        max_tokens=8,
        arrival_window=2,
        seed=0,
    )
    requests = build_workload(workload, build_generator(0))
    scheduler = PagedScheduler(model, kv, cfg, SamplingConfig(temperature=0.0))
    result = scheduler.run_continuous(requests, seed=0)

    assert result.mode == "continuous"
    # Compare each scheduler sequence to a fresh single-sequence reference.
    for seq in result.sequences:
        kv.reset()
        kv.add_sequence(0)
        prompt_ids = torch.tensor([seq.prompt_ids], dtype=torch.long, device=_DEVICE)
        reference = _single_sequence_reference(model, kv, prompt_ids, seq.max_new_tokens)
        # Greedy equivalence: scheduler token stream == independent reference.
        assert seq.generated == reference
    # Every sequence should have finished (finished or hit its budget).
    assert all(s.finished for s in result.sequences)


def test_continuous_is_deterministic() -> None:
    model = ToyPagedLM(ToyConfig(seed=0), _DEVICE)
    cfg = EngineConfig(block_size=4, max_num_blocks=64)
    kv = build_kv(model, cfg, _DEVICE)
    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS[:3],
        num_requests=4,
        min_tokens=4,
        max_tokens=8,
        arrival_window=2,
        seed=5,
    )
    requests = build_workload(workload, build_generator(5))

    def _tokens() -> list[list[int]]:
        kv.reset()
        scheduler = PagedScheduler(model, kv, cfg, SamplingConfig(temperature=0.0))
        result = scheduler.run_continuous(requests, seed=5)
        return [list(s.generated) for s in result.sequences]

    first = _tokens()
    second = _tokens()
    assert first == second


def test_static_greedy_matches_continuous() -> None:
    """Both schedulers must emit the same per-request greedy tokens.

    Continuous assigns sequence ids by *admission* order (arrival time) while
    static assigns them by the input request order, so ids do not line up
    between the two runs. What must be identical is the *multiset* of outputs --
    each request, regardless of when it is admitted, generates the very same
    tokens under greedy decoding.
    """
    model = ToyPagedLM(ToyConfig(seed=0), _DEVICE)
    cfg = EngineConfig(block_size=4, max_num_blocks=64)
    kv = build_kv(model, cfg, _DEVICE)
    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS[:4],
        num_requests=5,
        min_tokens=3,
        max_tokens=6,
        arrival_window=2,
        seed=1,
    )
    requests = build_workload(workload, build_generator(1))
    sampling = SamplingConfig(temperature=0.0)

    def _fingerprint(res) -> list[tuple[int, tuple]]:
        return sorted((s.max_new_tokens, tuple(s.generated)) for s in res.sequences)

    kv.reset()
    cont = PagedScheduler(model, kv, cfg, sampling).run_continuous(requests, seed=1)
    kv.reset()
    stat = PagedScheduler(model, kv, cfg, sampling).run_static(requests, seed=1)

    assert _fingerprint(cont) == _fingerprint(stat)


def test_continuous_reports_metrics() -> None:
    model = ToyPagedLM(ToyConfig(seed=0), _DEVICE)
    cfg = EngineConfig(block_size=4, max_num_blocks=64)
    kv = build_kv(model, cfg, _DEVICE)
    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS[:2],
        num_requests=3,
        min_tokens=3,
        max_tokens=5,
        arrival_window=2,
        seed=3,
    )
    requests = build_workload(workload, build_generator(3))
    scheduler = PagedScheduler(model, kv, cfg, SamplingConfig(temperature=0.0))
    result = scheduler.run_continuous(requests, seed=3)
    payload = result.to_dict()
    assert payload["generated_tokens"] > 0
    assert payload["makespan_ms"] >= 0
    assert payload["steps"] >= 1
    assert payload["throughput_tokens_per_s"] is not None
    # Idle-slot utilisation must be within (0, 1].
    assert 0.0 < payload["slot_utilization"] <= 1.0
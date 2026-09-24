"""Continuous-batching scheduler and a static-batching baseline.

The scheduler owns the interaction between a dynamic set of *requests* and one
``PagedLM`` over a :class:`PagedKVCache`. Every step it:

1. admits requests whose ``arrive_time`` has passed (prefill their prompts into
   physical blocks);
2. packs all *running* sequences' current tokens into one paged decode forward;
3. samples each next token, caches its K/V, and drops finished sequences (freeing
   their blocks).

Because KV is paged per-sequence, the batch size changes every step with zero
cost -- the continuous-batching property. The static baseline instead forces a
fixed batch (admit everything at step 0, keep idle slots until the longest
request finishes), which wastes capacity on heterogeneous workloads and is the
thing continuous batching improves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import EngineConfig, SamplingConfig, WorkloadRequest
from .kv import PagedKVCache
from .utils import Stopwatch, mean_std, setup_logger, sync_device

logger = setup_logger(__name__)

_MISSING = object()


@dataclass
class Sequence:
    """One request plus its generation progress/telemetry."""

    sid: int
    prompt_ids: List[int]
    max_new_tokens: int
    arrive_time: int
    generated: List[int] = field(default_factory=list)
    prefilled: bool = False
    finished: bool = False
    finish_step: int = -1
    admit_step: int = -1
    ttft_ms: Optional[float] = None
    eos_hit: bool = False

    @property
    def done(self) -> bool:
        budget = 0 if self.finished else (self.max_new_tokens - len(self.generated))
        return budget <= 0


@dataclass
class SchedulerResult:
    """Aggregate telemetry for one scheduling run."""

    mode: str = ""
    sequences: List[Sequence] = field(default_factory=list)
    steps: int = 0
    makespan_ms: float = 0.0
    generated_tokens: int = 0
    total_forwards: int = 0
    batch_hist: List[int] = field(default_factory=list)
    block_util_hist: List[float] = field(default_factory=list)
    kv_peak_blocks: int = 0
    kv_peak_bytes: int = 0
    peak_memory_mb: Optional[float] = None
    sampling: str = "greedy"

    @property
    def throughput_tokens_per_s(self) -> Optional[float]:
        if self.makespan_ms <= 0:
            return None
        return self.generated_tokens / (self.makespan_ms / 1000.0)

    @property
    def avg_active_batch(self) -> float:
        return sum(self.batch_hist) / len(self.batch_hist) if self.batch_hist else 0.0

    @property
    def slot_utilization(self) -> float:
        """Useful decode tokens / (steps * peak batch). Idle slots lower it.

        ``peak batch`` is the maximum concurrent batch over the run (the final
        histogram entry can be 0 once every sequence has finished).
        """
        peak = max(self.batch_hist) if self.batch_hist else 0.0
        if peak <= 0:
            return 0.0
        return self.generated_tokens / (self.steps * peak)

    def ttft_list(self) -> List[Optional[float]]:
        return [seq.ttft_ms for seq in self.sequences if seq.ttft_ms is not None]

    def to_dict(self) -> Dict[str, Any]:
        ttft = mean_std([v for v in self.ttft_list() if v is not None])
        return {
            "mode": self.mode,
            "sampling": self.sampling,
            "num_requests": len(self.sequences),
            "steps": self.steps,
            "makespan_ms": round(self.makespan_ms, 3),
            "generated_tokens": self.generated_tokens,
            "total_forwards": self.total_forwards,
            "throughput_tokens_per_s": (
                round(self.throughput_tokens_per_s, 3)
                if self.throughput_tokens_per_s is not None
                else None
            ),
            "avg_active_batch": round(self.avg_active_batch, 3),
            "slot_utilization": round(self.slot_utilization, 4),
            "ttft_ms_avg": ttft["mean"],
            "ttft_ms_p50": _p50(ttft),
            "kv_peak_blocks": self.kv_peak_blocks,
            "kv_peak_bytes": self.kv_peak_bytes,
            "peak_memory_mb": self.peak_memory_mb,
            "block_util_final": (
                round(self.block_util_hist[-1], 4) if self.block_util_hist else None
            ),
            "batch_hist": list(self.batch_hist),
            "block_util_hist": [round(v, 4) for v in self.block_util_hist],
        }


def _p50(stats: Dict[str, Optional[float]]) -> Optional[float]:
    return stats["mean"]  # mean used where exact percentiles are not retained


class PagedScheduler:
    """Drives ``model`` over ``kv`` to serve a list of requests under a policy."""

    def __init__(
        self,
        model: Any,
        kv: PagedKVCache,
        engine_cfg: EngineConfig,
        sampling: SamplingConfig,
    ) -> None:
        self.model = model
        self.kv = kv
        self.engine_cfg = engine_cfg
        self.sampling = sampling
        self._eos = getattr(getattr(model, "tokenizer", None), "eos_token_id", None)
        self._next_sid = 0

    # ------------------------------------------------------------ admission
    def _flush_arrivals(
        self,
        arrivals: List[WorkloadRequest],
        when: int,
        result: SchedulerResult,
        generator: torch.Generator,
    ) -> List[Sequence]:
        """Prefill every not-yet-started request due at step ``when``."""
        admitted: List[Sequence] = []
        for request in arrivals:
            seq = Sequence(
                sid=self._next_sid,
                prompt_ids=self.model.encode(request.prompt),
                max_new_tokens=request.max_new_tokens,
                arrive_time=request.arrive_time,
                admit_step=when,
            )
            self._next_sid += 1
            self.kv.add_sequence(seq.sid)
            ttft = self._prefill_one(seq, result, generator)
            seq.ttft_ms = ttft
            admitted.append(seq)
        result.sequences.extend(admitted)
        return admitted

    def _prefill_one(
        self, seq: Sequence, result: SchedulerResult, generator: torch.Generator
    ) -> Optional[float]:
        watch = Stopwatch()
        with watch:
            ids = torch.tensor(
                [list(seq.prompt_ids)], dtype=torch.long, device=self.model.device
            )
            logits = self.model.prefill(
                ids, self.kv, [seq.sid], [len(seq.prompt_ids)]
            )
            probs = self._probs(logits[0])
            token = self._sample(probs, generator)
            seq.generated.append(token)
        seq.prefilled = True
        return watch.elapsed_ms

    # ------------------------------------------------------------ decode step
    def _decode_active(
        self,
        running: Sequence,
        result: SchedulerResult,
        generator: torch.Generator,
    ) -> None:
        model = self.model
        device = model.device
        ids = [seq.generated[-1] for seq in running]
        ctx = [self.kv.seq_lengths[seq.sid] for seq in running]
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        logits = model.decode(input_ids, self.kv, [s.sid for s in running], ctx)
        probs = self._probs(logits)          # (B, vocab)
        for b, seq in enumerate(running):
            token = self._sample(probs[b], generator)
            if self._eos is not None and token == self._eos:
                seq.eos_hit = True
                seq.finished = True
                seq.finish_step = result.steps
            else:
                seq.generated.append(token)
            if len(seq.generated) >= seq.max_new_tokens:
                seq.finished = True
                seq.finish_step = result.steps
            if seq.finished:
                self.kv.clear(seq.sid)

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sampling.is_greedy:
            flat = logits.reshape(-1, logits.shape[-1])
            return flat.argmax(dim=-1, keepdim=False)
        # sampling mode: temperature softmax over the vocab.
        scaled = logits / max(self.sampling.temperature, 1e-9)
        return torch.softmax(scaled.float(), dim=-1)

    def _sample(self, probs, generator) -> int:
        if self.sampling.is_greedy:
            return int(probs)
        flat = probs.detach().reshape(-1).to("cpu", dtype=torch.float32)
        return int(torch.multinomial(flat, num_samples=1, generator=generator).item())

    # -------------------------------------------------------------- policies
    def run_continuous(
        self, requests: Sequence[WorkloadRequest], seed: int, measure: bool = True
    ) -> SchedulerResult:
        result = SchedulerResult(mode="continuous", sampling=self.sampling.describe())
        generator = torch.Generator().manual_seed(seed)
        device = self.model.device
        pending = sorted(requests, key=lambda r: r.arrive_time)
        arrivals = list(pending)
        step = 0
        if measure:
            sync_device(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        stopwatch = Stopwatch()
        with stopwatch:
            due = [r for r in arrivals if r.arrive_time <= step]
            pending = [r for r in arrivals if r.arrive_time > step]
            self._flush_arrivals(due, step, result, generator)
            running = [s for s in result.sequences if not s.finished]
            while running or pending:
                running = [s for s in result.sequences if not s.finished]
                if running:
                    self._decode_active(running, result, generator)
                    result.total_forwards += 1
                step += 1
                due = [r for r in pending if r.arrive_time <= step]
                pending = [r for r in pending if r.arrive_time > step]
                if due:
                    self._flush_arrivals(due, step, result, generator)
                running = [s for s in result.sequences if not s.finished]
                result.batch_hist.append(len(running))
                result.block_util_hist.append(self.kv.block_utilization)
                if self.kv.num_allocated_blocks > result.kv_peak_blocks:
                    result.kv_peak_blocks = self.kv.num_allocated_blocks
            result.steps = step
        result.makespan_ms = stopwatch.elapsed_ms
        result.generated_tokens = sum(len(s.generated) for s in result.sequences)
        result.kv_peak_bytes = result.kv_peak_blocks * self._bytes_per_block()
        if measure and device.type == "cuda":
            result.peak_memory_mb = round(
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2
            )
        return result

    def run_static(
        self, requests: Sequence[WorkloadRequest], seed: int, measure: bool = True
    ) -> SchedulerResult:
        """Fixed-batch baseline: everything starts at step 0, idle slots until done.

        The batch size stays at the request count for the whole run; sequences that
        finish early remain in the batch as *idle slots* (we pad their decode but
        count no token), which is the capacity waste continuous batching removes.
        """
        result = SchedulerResult(mode="static", sampling=self.sampling.describe())
        generator = torch.Generator().manual_seed(seed)
        device = self.model.device
        if measure:
            sync_device(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        stopwatch = Stopwatch()
        with stopwatch:
            self._flush_arrivals(list(requests), 0, result, generator)
            all_seqs = list(result.sequences)
            fixed = len(all_seqs)
            step = 0
            while any(not s.done for s in all_seqs):
                # Decode the *whole* fixed batch; idle (done) rows get a pad token
                # so the forward stays width ``fixed`` and wastes capacity.
                living = [s for s in all_seqs if not s.done]
                if living:
                    ids = []
                    ctx = []
                    for s in all_seqs:
                        # idle (done) rows re-feed their last token: a wasted slot.
                        ids.append(s.generated[-1] if s.generated else 0)
                        ctx.append(self.kv.seq_lengths[s.sid])
                    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
                    logits = self.model.decode(
                        input_ids, self.kv, [s.sid for s in all_seqs], ctx
                    )
                    result.total_forwards += 1
                    for b, s in enumerate(all_seqs):
                        if s.done:
                            continue
                        token = self._sample(self._probs(logits[b]), generator)
                        if self._eos is not None and token == self._eos:
                            s.eos_hit = True
                            s.finished = True
                        else:
                            s.generated.append(token)
                        if len(s.generated) >= s.max_new_tokens:
                            s.finished = True
                        s.finish_step = step
                step += 1
                result.steps = step
                result.batch_hist.append(fixed)
                result.block_util_hist.append(self.kv.block_utilization)
                if self.kv.num_allocated_blocks > result.kv_peak_blocks:
                    result.kv_peak_blocks = self.kv.num_allocated_blocks
        result.makespan_ms = stopwatch.elapsed_ms
        result.generated_tokens = sum(len(s.generated) for s in result.sequences)
        result.kv_peak_bytes = result.kv_peak_blocks * self._bytes_per_block()
        if measure and device.type == "cuda":
            result.peak_memory_mb = round(
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2
            )
        return result

    def _bytes_per_block(self) -> int:
        """Bytes occupied by one physical block across all layers (K+V)."""
        d = self.model.head_dim
        kv_heads = self.model.num_kv_heads
        return (
            2
            * self.model.num_layers
            * self.engine_cfg.block_size
            * kv_heads
            * d
            * self.model.dtype.itemsize
        )


def build_kv(model: Any, engine_cfg: EngineConfig, device: torch.device) -> PagedKVCache:
    """Construct a sized paged cache from a model's shape and the engine config."""
    return PagedKVCache(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        capacity_blocks=engine_cfg.max_num_blocks,
        block_size=engine_cfg.block_size,
        device=device,
        dtype=model.dtype,
    )
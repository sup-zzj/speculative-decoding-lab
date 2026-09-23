"""Same-sequence batched speculative decoding.

This is a *minimal scientific probe* of the bundle-size hypothesis: holding the
batch to identical copies of one prompt keeps every stream in lockstep, so the
accept/reject logic collapses to the single-sample rule and the *only* thing that
changes is the batch dimension ``B`` of both forward passes. That lets us cleanly
attribute any wall-clock change to kernel utilisation (the draft--target cost
ratio ``c`` as ``B`` grows) rather than to per-sample bookkeeping.

Correctness guarantee
---------------------
Because every row of the batch is fed the identical token block, ``batch = 1``
here reduces *exactly* to :func:`specdec.speculative.speculative_decode`. For
``B > 1`` the committed token stream is identical to the single-sample result and
therefore identical to :func:`specdec.decoding.baseline_decode` under greedy
decoding -- the batch dimension is purely a performance knob, never a correctness
one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import SamplingConfig, validate_gamma
from .models import CausalLM, PastType, past_length, truncate_past
from .sampler import logits_to_probs, residual_distribution, sample_token
from .utils import Stopwatch, sync_device


@dataclass
class BatchSpeculativeStats:
    """Decode statistics for the batched (same-sequence) path."""

    batch_size: int = 0
    gamma: int = 0
    rounds: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    tokens_per_round: List[int] = field(default_factory=list)
    target_forward_calls: int = 0
    draft_forward_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "rounds": self.rounds,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_round_hist": self.tokens_per_round,
            "target_forward_calls": self.target_forward_calls,
            "draft_forward_calls": self.draft_forward_calls,
        }


@dataclass
class BatchSpeculativeOutput:
    """Token stream plus instrumentation; ``token_ids`` are the per-slot stream."""

    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    latency_ms: float = 0.0
    peak_memory_mb: Optional[float] = None
    stats: BatchSpeculativeStats = field(default_factory=BatchSpeculativeStats)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.stats.to_dict()
        payload.update(
            {
                "tokens_per_slot": len(self.token_ids),
                "tokens_per_slot_per_second": (
                    round(len(self.token_ids) / (self.latency_ms / 1000.0), 2)
                    if self.latency_ms > 0
                    else None
                ),
                "latency_ms": round(self.latency_ms, 3),
                "peak_memory_mb": self.peak_memory_mb,
                "text": self.text,
            }
        )
        return payload


@dataclass
class BatchBaselineOutput:
    """Wall-clock output of the batched autoregressive reference decoder."""

    latency_ms: float = 0.0
    tokens_per_slot: int = 0
    peak_memory_mb: Optional[float] = None

    @property
    def ms_per_token_slot(self) -> Optional[float]:
        if self.latency_ms > 0 and self.tokens_per_slot > 0:
            return self.latency_ms / self.tokens_per_slot
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": round(self.latency_ms, 3),
            "tokens_per_slot": self.tokens_per_slot,
            "ms_per_token_slot": (
                round(self.ms_per_token_slot, 4)
                if self.ms_per_token_slot is not None
                else None
            ),
            "tokens_per_slot_per_second": (
                round(1000.0 / self.ms_per_token_slot, 2)
                if self.ms_per_token_slot and self.ms_per_token_slot > 0
                else None
            ),
            "peak_memory_mb": self.peak_memory_mb,
        }


def _broadcast_2d(values: Sequence[int], batch_size: int, device: torch.device) -> torch.Tensor:
    """Replicate one token block across ``B`` rows of a ``(B, L)`` tensor."""
    return torch.as_tensor(
        [list(values)] * batch_size, dtype=torch.long, device=device
    )


def batch_baseline_decode(
    model: CausalLM,
    prompt_ids: Sequence[int],
    batch_size: int,
    max_new_tokens: int,
    sampling: SamplingConfig,
    measure: bool = True,
) -> BatchBaselineOutput:
    """Reference autoregressive decode with ``B`` identical slots.

    All ``B`` rows are fed the same greedy token each step, so the token stream is
    identical across slots; the forward passes are genuinely batched, which is
    what makes this a fair baseline for the bundle hypothesis. ``ms_per_token_slot``
    is per-slot (total wall-clock / new tokens per slot), so it is directly
    comparable across ``B``.
    """
    output = BatchBaselineOutput()
    device = model.device
    model.reset_counters()
    if measure:
        sync_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    stopwatch = Stopwatch()
    with stopwatch:
        past, logits = model.prefill_batch([list(prompt_ids)] * batch_size)
        token = int(logits.argmax(dim=-1)[0].item())
        model.reset_counters()  # drop the prefill from step-time accounting
        for _ in range(max_new_tokens):
            block = _broadcast_2d([token], batch_size, device)
            past, logits = model.extend_batch(block, past)
            token = int(logits.argmax(dim=-1)[0].item())

    output.latency_ms = stopwatch.elapsed_ms
    output.tokens_per_slot = max_new_tokens
    if measure and device.type == "cuda":
        output.peak_memory_mb = round(
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2
        )
    return output


def measure_batch_cost_ratio(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    batch_size: int,
    max_new_tokens: int,
    sampling: SamplingConfig,
) -> Dict[str, Any]:
    """Empirical ``c(B) = draft per-slot step cost / target per-slot step cost``.

    Steps are measured on genuinely batched forwards (``batch_size`` copies), so
    ``c`` captures how much the draft's kernel utilisation gap closes as the
    bundle grows -- the quantity the bundle hypothesis predicts should fall.
    """
    draft_baseline = batch_baseline_decode(
        draft, prompt_ids, batch_size, max_new_tokens, sampling, measure=True
    )
    target_baseline = batch_baseline_decode(
        target, prompt_ids, batch_size, max_new_tokens, sampling, measure=True
    )
    draft_ms = draft_baseline.ms_per_token_slot
    target_ms = target_baseline.ms_per_token_slot
    return {
        "draft_ms_per_token_slot": draft_ms,
        "target_ms_per_token_slot": target_ms,
        "cost_ratio": (draft_ms / target_ms) if target_ms and target_ms > 0 else float("nan"),
    }


def speculative_decode_batch(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    sampling: SamplingConfig,
    batch_size: int = 1,
    gamma: int = 6,
    measure: bool = True,
    eos_token_id: Optional[int] = None,
) -> BatchSpeculativeOutput:
    """Batched speculative decoding of ``batch_size`` identical slots.

    The batch caches share one KV tensor with a batch dimension; every forward
    uses ``extend_batch`` so the true batched kernel cost is what gets measured.
    Accept/reject follows the greedy argmax rule on row ``0`` (valid because all
    rows are identical), which reduces exactly to ``baseline_decode`` -- no RNG is
    drawn, keeping every run bit-deterministic.
    """
    validate_gamma(gamma)
    if draft.vocab_size != target.vocab_size:
        raise ValueError("batch speculative decoding requires a shared vocabulary")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not sampling.is_greedy:
        raise NotImplementedError(
            "same-sequence batch decoding is only verified for greedy sampling"
        )

    B = batch_size
    stats = BatchSpeculativeStats(batch_size=B, gamma=gamma)
    output = BatchSpeculativeOutput(stats=stats)
    device = target.device

    target.reset_counters()
    draft.reset_counters()
    if measure:
        sync_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    stopwatch = Stopwatch()
    with stopwatch:
        # ---- prefill both batched caches, then rewind to the invariant position
        t_past, _ = target.prefill_batch([list(prompt_ids)] * B)
        t_past = truncate_past(t_past, past_length(t_past) - 1)
        d_past, _ = draft.prefill_batch([list(prompt_ids)] * B)
        d_past = truncate_past(d_past, past_length(d_past) - 1)
        pending_t = int(prompt_ids[-1])
        pending_d = int(prompt_ids[-1])
        sequence: List[int] = list(prompt_ids)

        generated = 0
        while generated < max_new_tokens:
            # ---- 1. draft proposes `gamma` tokens at batch B (greedy, identical rows)
            proposals: List[int] = []
            token = pending_d
            for _ in range(gamma):
                d_past, logits = draft.extend_batch(
                    _broadcast_2d([token], B, device), d_past
                )
                token = int(logits.argmax(dim=-1)[0].item())
                proposals.append(token)
            stats.proposed_tokens += len(proposals)

            # ---- 2. target verifies the whole block in one batched forward
            pass_tokens = [pending_t] + proposals
            t_past, block_logits = target.extend_batch(
                _broadcast_2d(pass_tokens, B, device), t_past
            )
            p_all = logits_to_probs(block_logits[0], sampling)  # (gamma+1, vocab)

            # ---- 3. greedy accept/reject on row 0 (identical across rows)
            accepted: List[int] = []
            rejected_at: Optional[int] = None
            for index, token in enumerate(proposals):
                if int(p_all[index].argmax().item()) == token:
                    accepted.append(token)
                else:
                    rejected_at = index
                    corrected = sample_token(residual_distribution(p_all[index], _one_hot(p_all[index], token)))
                    accepted.append(corrected)
                    break

            if rejected_at is None:
                bonus = sample_token(p_all[gamma])
                accepted.append(bonus)
                stats.accepted_tokens += gamma
            else:
                stats.accepted_tokens += rejected_at

            stats.rounds += 1
            stats.tokens_per_round.append(len(accepted))

            # ---- 4. commit (truncating anything the caller asked not to keep)
            remaining = max_new_tokens - generated
            accepted = accepted[:remaining]
            sequence.extend(accepted)
            generated += len(accepted)

            # ---- 5. re-establish the invariant for the next round
            committed_len = len(sequence)
            keep = committed_len - 1
            t_past = truncate_past(t_past, keep)
            pending_t = sequence[-1]
            d_past = truncate_past(d_past, min(past_length(d_past), keep))
            while past_length(d_past) < keep:
                feed_index = past_length(d_past)
                d_past, _ = draft.extend_batch(
                    _broadcast_2d([sequence[feed_index]], B, device), d_past
                )
            pending_d = sequence[-1]

            if eos_token_id is not None and sequence[-1] == eos_token_id:
                break

    stats.target_forward_calls = target.forward_calls
    stats.draft_forward_calls = draft.forward_calls
    output.latency_ms = stopwatch.elapsed_ms
    output.token_ids = sequence[len(prompt_ids):]
    output.text = target.decode(output.token_ids)
    if measure and device.type == "cuda":
        output.peak_memory_mb = round(
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2
        )
    return output


def _one_hot(probs: torch.Tensor, token: int) -> torch.Tensor:
    """One-hot vector matching the greedy draft proposal for residual resampling."""
    one_hot = torch.zeros_like(probs)
    one_hot[token] = 1.0
    return one_hot
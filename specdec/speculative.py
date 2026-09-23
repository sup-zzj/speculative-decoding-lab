"""Speculative decoding: a small draft model proposes, the target verifies.

Implementation notes
--------------------
The interesting part is not the accept/reject rule but the *cache bookkeeping*.
Feeding a draft block and then rewinding the target cache is the step that naive
implementations get wrong, so both models here share one invariant:

    after every round, ``past`` holds the KV of positions ``0 .. L-2`` and the
    final committed token ``sequence[L-1]`` is left *un-fed* as a "pending" token.

A round then consists of exactly one batched target forward pass over
``[pending] + draft_tokens`` (``gamma + 1`` positions). Those positions yield the
``gamma + 1`` distributions needed to verify the draft block *and* to sample the
bonus token, without any extra synchronisation forward. On a rejection the cache
is truncated to ``L' - 1`` so the rejected token and its successors disappear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import SamplingConfig, validate_gamma
from .models import CausalLM, PastType, past_length, truncate_past
from .sampler import (
    acceptance_probability,
    logits_to_probs,
    residual_distribution,
    sample_token,
)
from .utils import Stopwatch, sync_device

if TYPE_CHECKING:
    from .graph_draft import GraphDraft


@dataclass
class SpeculativeStats:
    """Everything needed to explain *why* a speed-up did (or did not) happen."""

    gamma: int = 0
    rounds: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    tokens_per_round: List[int] = field(default_factory=list)
    rejection_positions: List[int] = field(default_factory=list)
    target_forward_calls: int = 0
    draft_forward_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    @property
    def mean_tokens_per_round(self) -> float:
        if not self.tokens_per_round:
            return 0.0
        return sum(self.tokens_per_round) / len(self.tokens_per_round)

    @property
    def mean_target_tokens_per_call(self) -> float:
        """Target tokens committed per *target forward call* (the real efficiency axis)."""
        if self.target_forward_calls == 0:
            return 0.0
        return sum(self.tokens_per_round) / self.target_forward_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "rounds": self.rounds,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "mean_tokens_per_round": round(self.mean_tokens_per_round, 4),
            "mean_target_tokens_per_target_call": round(
                self.mean_target_tokens_per_call, 4
            ),
            "target_forward_calls": self.target_forward_calls,
            "draft_forward_calls": self.draft_forward_calls,
            "tokens_per_round_hist": self.tokens_per_round,
            "rejection_positions_hist": self.rejection_positions,
        }


@dataclass
class SpeculativeOutput:
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    latency_ms: float = 0.0
    peak_memory_mb: Optional[float] = None
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)
    #: logits actually used to accept/reject, kept for offline analysis
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.stats.to_dict()
        payload.update(
            {
                "latency_ms": round(self.latency_ms, 3),
                "tokens_per_second": (
                    round(len(self.token_ids) / (self.latency_ms / 1000.0), 2)
                    if self.latency_ms > 0
                    else None
                ),
                "peak_memory_mb": self.peak_memory_mb,
                "text": self.text,
            }
        )
        return payload


def _draw_uniform(generator: Optional[torch.Generator]) -> float:
    """Deterministic uniform draw used for the accept/reject coin flip."""
    if generator is None:
        return float(torch.rand(1).item())
    return float(torch.rand(1, generator=generator).item())


def _generate_draft_block(
    draft: CausalLM,
    draft_past: PastType,
    pending_token: int,
    gamma: int,
    sampling: SamplingConfig,
    generator: Optional[torch.Generator],
) -> Tuple[PastType, List[int], List[torch.Tensor]]:
    """Propose ``gamma`` tokens, returning their probabilities under the draft."""
    proposals: List[int] = []
    rows: List[torch.Tensor] = []
    past = draft_past
    token = pending_token

    for _ in range(gamma):
        past, logits = draft.logits_for_next(token, past)
        q = logits_to_probs(logits, sampling)
        rows.append(q)
        token = sample_token(q, generator)
        proposals.append(token)
    return past, proposals, rows


def _verify_block(
    target: CausalLM,
    target_past: PastType,
    pending_token: int,
    draft_tokens: Sequence[int],
    sampling: SamplingConfig,
) -> Tuple[PastType, torch.Tensor]:
    """One target forward pass over the pending token plus the whole draft block.

    The pass yields ``gamma + 1`` distributions ``p_0 .. p_gamma``: ``p_i``
    verifies draft token ``i`` and ``p_gamma`` is the distribution of the bonus
    token, so no additional synchronisation forward is required.
    """
    pass_input = [pending_token] + list(draft_tokens)
    new_past, block_logits = target.extend(pass_input, target_past)
    p_all = logits_to_probs(block_logits, sampling)
    return new_past, p_all


def speculative_decode(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    sampling: SamplingConfig,
    gamma: int = 6,
    generator: Optional[torch.Generator] = None,
    measure: bool = True,
    collect_trace: bool = False,
    eos_token_id: Optional[int] = None,
    draft_engine: Optional[GraphDraft] = None,
) -> SpeculativeOutput:
    """Decode ``max_new_tokens`` tokens with draft/verify speculative sampling.

    The output distribution is provably identical to :func:`baseline_decode`
    under greedy decoding, and identical under sampling up to the exactness of
    the residual-resampling step (see ``specdec.correctness``).
    """
    validate_gamma(gamma)
    if draft.vocab_size != target.vocab_size:
        raise ValueError("draft and target must share a vocabulary")

    output = SpeculativeOutput()
    stats = SpeculativeStats(gamma=gamma)
    output.stats = stats
    device = target.device

    target.reset_counters()
    draft.reset_counters()
    if measure:
        sync_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    draft_gen = generator
    target_gen = generator

    stopwatch = Stopwatch()
    with stopwatch:
        # ---- prefill both models, then rewind one position to keep the invariant
        t_past, _ = target.prefill(prompt_ids)
        t_past = truncate_past(t_past, past_length(t_past) - 1)
        if draft_engine is None:
            d_past, _ = draft.prefill(prompt_ids)
            d_past = truncate_past(d_past, past_length(d_past) - 1)
        else:
            draft_engine.prefill(prompt_ids)
            draft_engine.rewind(len(prompt_ids) - 1)
        pending_t = int(prompt_ids[-1])
        pending_d = int(prompt_ids[-1])
        sequence: List[int] = list(prompt_ids)

        generated = 0
        while generated < max_new_tokens:
            # ---- 1. draft proposes a block
            if draft_engine is None:
                d_past, proposals, q_rows = _generate_draft_block(
                    draft, d_past, pending_d, gamma, sampling, draft_gen
                )
            else:
                proposals, q_rows = draft_engine.propose(
                    pending_d, gamma, sampling, draft_gen
                )
            stats.proposed_tokens += len(proposals)

            # ---- 2. target verifies the block in a single forward
            t_past, p_all = _verify_block(
                target, t_past, pending_t, proposals, sampling
            )

            # ---- 3. accept / reject with residual resampling
            accepted: List[int] = []
            rejected_at: Optional[int] = None
            for index, token in enumerate(proposals):
                p_i = p_all[index]
                q_i = q_rows[index]
                if _draw_uniform(generator) < acceptance_probability(p_i, q_i, token):
                    accepted.append(token)
                else:
                    rejected_at = index
                    corrected = sample_token(residual_distribution(p_i, q_i), generator)
                    accepted.append(corrected)
                    break

            if rejected_at is None:
                # full acceptance -> draw one extra token from the target itself
                bonus = sample_token(p_all[gamma], generator)
                accepted.append(bonus)
                stats.accepted_tokens += gamma
                stats.rejection_positions.append(gamma)
            else:
                stats.accepted_tokens += rejected_at
                stats.rejection_positions.append(rejected_at)

            stats.rounds += 1
            stats.tokens_per_round.append(len(accepted))

            # ---- 4. commit (truncating anything the caller asked not to keep)
            remaining = max_new_tokens - generated
            accepted = accepted[:remaining]
            sequence.extend(accepted)
            generated += len(accepted)

            # ---- 5. re-establish the cache invariant for the next round
            committed_len = len(sequence)
            keep = committed_len - 1
            t_past = truncate_past(t_past, keep)
            pending_t = sequence[-1]

            if draft_engine is None:
                d_past = truncate_past(d_past, min(past_length(d_past), keep))
                while past_length(d_past) < keep:
                    feed_index = past_length(d_past)
                    d_past, _ = draft.logits_for_next(sequence[feed_index], d_past)
            else:
                if draft_engine.cache_len() > keep:
                    draft_engine.rewind(keep)
                while draft_engine.cache_len() < keep:
                    feed_index = draft_engine.cache_len()
                    draft_engine.step_token(sequence[feed_index])
            pending_d = sequence[-1]

            if collect_trace:
                output.trace.append(
                    {
                        "round": stats.rounds,
                        "proposals": proposals,
                        "accepted": accepted,
                        "rejected_at": rejected_at,
                        "acceptance_rate": (
                            rejected_at / len(proposals)
                            if rejected_at is not None and proposals
                            else 1.0
                        ),
                    }
                )

            if eos_token_id is not None and sequence and sequence[-1] == eos_token_id:
                break

    stats.target_forward_calls = target.forward_calls
    stats.draft_forward_calls = (
        draft_engine.forward_calls if draft_engine is not None else draft.forward_calls
    )
    output.latency_ms = stopwatch.elapsed_ms
    output.token_ids = sequence[len(prompt_ids):]
    output.text = target.decode(output.token_ids)
    if measure and device.type == "cuda":
        output.peak_memory_mb = round(
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2
        )
    return output


def expected_tokens_per_round(acceptance_rate: float, gamma: int) -> float:
    """Closed form ``E[tokens per round] = (1 - alpha^(gamma+1)) / (1 - alpha)``.

    ``alpha`` is the probability that a single draft token survives verification;
    the "+1" is the bonus token drawn from the target on full acceptance.
    """
    alpha = float(min(max(acceptance_rate, 0.0), 1.0))
    if alpha >= 1.0 - 1e-9:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def predicted_speedup(
    acceptance_rate: float, gamma: int, draft_cost_ratio: float
) -> float:
    """Analytic speed-up.

    ``draft_cost_ratio`` is the wall-clock cost of one draft step divided by one
    target step (``c`` in the literature). The denominator ``1 + gamma * c``
    compares against plain autoregressive decoding, where one target call yields
    exactly one token.
    """
    tokens = expected_tokens_per_round(acceptance_rate, gamma)
    cost = 1.0 + gamma * float(draft_cost_ratio)
    if cost <= 0:
        return float("nan")
    return tokens / cost
"""Cross-tokenizer (text-level) speculative decoding.

The token-level scheme in ``speculative.py`` requires the draft and the target
to share one vocabulary, which rules out pairing a large target with a much
smaller draft model that ships its own tokenizer (e.g. Qwen2.5-1.5B with
SmolLM2-135M).  This module relaxes that constraint at the cost of accepting
*text* instead of *tokens*.

Algorithm
---------
1. The draft proposes ``gamma`` tokens in its own vocabulary; they are decoded
   into a text delta ``d``.
2. ``committed_text + d`` is re-encoded with the *target* tokenizer, yielding a
   candidate sequence of target tokens ``c``.  If the BPE segmentations do not
   line up (the new encoding is no longer an extension of the committed
   tokens), the round degrades to a single target step -- correctness is
   preserved, only the speculation is dropped.
3. The target verifies the new tokens in one batched forward pass.  Under
   greedy decoding a proposed token is accepted iff it equals the target's
   argmax at that position, which reproduces the target's greedy path exactly.

Correctness notes
-----------------
* Only greedy decoding is supported: with two different vocabularies the
  ``(p - q)_+`` residual resampling of ``sampler.residual_distribution`` is not
  defined (``q`` lives in the draft's vocab, ``p`` in the target's), so there
  is no exactness-preserving rejection sampling across vocabularies.
* ``candidate_text`` is committed via the *target* tokenizer, so the returned
  ``token_ids`` are always target tokens and can be compared directly against
  ``baseline_decode`` output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import torch

from .config import SamplingConfig, validate_gamma
from .models import CausalLM, PastType, past_length, truncate_past
from .sampler import logits_to_probs
from .speculative import SpeculativeOutput, SpeculativeStats
from .utils import Stopwatch, sync_device

if TYPE_CHECKING:  # pragma: no cover
    pass


def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


def _encode_no_special(model: CausalLM, text: str) -> List[int]:
    """Encode with the model's tokenizer, never adding BOS/special tokens."""
    return list(model.tokenizer(text, add_special_tokens=False).input_ids)


def text_speculative_decode(
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
) -> SpeculativeOutput:
    """Cross-tokenizer speculative decoding (greedy only).

    ``prompt_ids`` are encoded with the *target* tokenizer; ``token_ids`` in the
    returned output are likewise target tokens, so the result is comparable
    with :func:`specdec.decoding.baseline_decode` bit for bit.
    """
    validate_gamma(gamma)
    if not sampling.is_greedy:
        raise ValueError(
            "text-level speculative decoding supports greedy sampling only "
            "(cross-tokenizer residual resampling is not defined)"
        )
    if draft.vocab_size == target.vocab_size:
        raise ValueError(
            "draft and target share a vocabulary; use speculative_decode instead"
        )

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

    # Committed text lives in the *target* tokenizer's token space.
    committed_text: str = target.decode(list(prompt_ids))
    committed_tokens: List[int] = list(prompt_ids)

    stopwatch = Stopwatch()
    with stopwatch:
        # ---- prefill both models, then rewind one position (pending invariant)
        t_past, _ = target.prefill(prompt_ids)
        t_past = truncate_past(t_past, past_length(t_past) - 1)

        # Draft history is re-encoded from the committed text on every round;
        # the seed of that history is the draft encoding of the prompt.
        draft_ids = _encode_no_special(draft, committed_text)
        d_past, _ = draft.prefill(draft_ids)
        d_past = truncate_past(d_past, past_length(d_past) - 1)
        draft_fed: List[int] = list(draft_ids[:-1])
        pending_t = int(committed_tokens[-1])
        pending_d = int(draft_ids[-1])

        sequence: List[int] = list(prompt_ids)
        generated = 0

        while generated < max_new_tokens:
            # ---- 1. draft proposes ``gamma`` tokens in its own vocabulary
            proposals_d: List[int] = []
            past = d_past
            token = pending_d
            for _ in range(gamma):
                past, logits = draft.logits_for_next(token, past)
                q = logits_to_probs(logits, sampling)
                token = int(q.argmax(dim=-1).item())
                proposals_d.append(token)
            d_past = past
            delta = draft.decode(proposals_d)

            # ---- 2. re-encode the extended text with the target tokenizer
            candidate_text = committed_text + delta
            cand = _encode_no_special(target, candidate_text)

            # BPE segmentation must not rewrite already committed tokens.
            if cand[: len(committed_tokens)] == committed_tokens:
                proposals_t: List[int] = cand[len(committed_tokens):]
            else:
                proposals_t = []  # degradation: verify nothing, step the target

            # ---- 3. target verifies in one batched forward pass
            verify_input = [pending_t] + proposals_t
            t_past, block_logits = target.extend(verify_input, t_past)
            p_all = logits_to_probs(block_logits, sampling)

            # ---- 4. greedy accept / reject on *target* tokens
            accepted: List[int] = []
            rejected_at: Optional[int] = None
            for index, token in enumerate(proposals_t):
                if int(p_all[index].argmax(dim=-1).item()) == token:
                    accepted.append(token)
                else:
                    rejected_at = index
                    corrected = int(p_all[index].argmax(dim=-1).item())
                    accepted.append(corrected)
                    break

            if rejected_at is None:
                bonus = int(p_all[-1].argmax(dim=-1).item())
                accepted.append(bonus)
                stats.accepted_tokens += len(proposals_t)
                stats.rejection_positions.append(len(proposals_t))
            else:
                stats.accepted_tokens += rejected_at
                stats.rejection_positions.append(rejected_at)

            stats.proposed_tokens += gamma
            stats.rounds += 1
            stats.tokens_per_round.append(len(accepted))

            # ---- 5. commit (truncating anything the caller asked not to keep)
            remaining = max_new_tokens - generated
            accepted = accepted[:remaining]
            sequence.extend(accepted)
            generated += len(accepted)
            if accepted:
                committed_tokens = list(sequence)
                committed_text = target.decode(committed_tokens)

            # ---- 6. re-establish the cache invariant for both models
            committed_len = len(sequence)
            keep = committed_len - 1
            t_past = truncate_past(t_past, keep)
            pending_t = sequence[-1]

            # Draft KV must follow the committed text re-encoded into the
            # draft's vocabulary.  Align to the longest common prefix of the
            # draft history and the fresh encoding, then extend what is new.
            fresh = _encode_no_special(draft, committed_text)
            common = _lcp(draft_fed, fresh[:-1])
            d_past = truncate_past(d_past, common)
            while past_length(d_past) < len(fresh) - 1:
                feed_index = past_length(d_past)
                d_past, _ = draft.logits_for_next(fresh[feed_index], d_past)
            draft_fed = list(fresh[:-1])
            pending_d = int(fresh[-1])

            if collect_trace:
                output.trace.append(
                    {
                        "round": stats.rounds,
                        "draft_tokens": proposals_d,
                        "delta_text": delta,
                        "target_tokens": proposals_t,
                        "accepted": accepted,
                        "rejected_at": rejected_at,
                    }
                )

            if eos_token_id is not None and sequence and sequence[-1] == eos_token_id:
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

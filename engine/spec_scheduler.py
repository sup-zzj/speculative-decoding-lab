"""Speculative decoding fused into the continuous-batching engine.

Phase 3 of the speculative-decoding lab: a ``SpeculativePagedScheduler`` that
extends :class:`PagedScheduler` with a draft model, so every decode round becomes
a *draft-then-verify* round over the running batch. The target and draft each own
their own :class:`PagedKVCache`; both share the single sequence bookkeeping (same
``sid``/``generated``), so the batch stays heterogeneous exactly as in Phase 2.

Cache invariant (one per model), consistent with ``specdec.speculative``:
    ``kv.seq_lengths[sid] == len(prompt) + len(generated) - 1``

i.e. the full prompt plus every committed generated token live in the cache, and
the *last* generated token (the "pending" token) is deliberately not written.
Each round the draft proposes ``gamma`` tokens from the pending token, the target
verifies ``[pending] + proposals`` (``gamma + 1`` positions) in a single batched
``decode_block``, accept/reject decides how many of those actually commit, then
both caches are rewound to ``len(prompt) + len(generated) - 1`` so the invariant
re-holds. Under greedy decoding this reproduces the plain engine's token stream
bit-for-bit, which :mod:`tests.test_spec_scheduler` verifies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from specdec.sampler import (
    acceptance_probability,
    logits_to_probs,
    residual_distribution,
    sample_token,
)
from .config import EngineConfig, SamplingConfig, WorkloadRequest
from .scheduler import PagedScheduler, SchedulerResult, Sequence
from .utils import Stopwatch, setup_logger, sync_device

logger = setup_logger(__name__)


def _draw_uniform(generator: torch.Generator) -> float:
    """Deterministic uniform coin used for each accept/reject in the batch."""
    return float(torch.rand(1, generator=generator).item())


def batch_accept(
    p_by_sid: Dict[int, torch.Tensor],
    q_by_sid: Dict[int, Sequence[torch.Tensor]],
    proposals_by_sid: Dict[int, Sequence[int]],
    generator: torch.Generator,
) -> Dict[int, List[int]]:
    """Accept/reject each sequence's draft block, returning the committed tokens.

    ``p_by_sid[sid]`` is the target's ``(gamma + 1, vocab)`` distribution block
    (row ``j`` verifies proposal ``j``, row ``gamma`` is the bonus distribution).
    ``q_by_sid[sid]`` is the ``gamma`` draft rows as a list of ``(vocab,)``
    vectors; ``proposals_by_sid[sid]`` the ``gamma`` proposed token ids.

    The rule matches ``specdec.speculative``: scan proposals in order, accept an
    item with ``min(1, p/q)``; on the first rejection resample from the residual
    ``(p - q)_+`` and stop; if everything is accepted, append one bonus token
    drawn from ``p``'s bonus row. Returns the ordered accepted token list per
    sequence (length `1..gamma+1`). For greedy (one-hot) distributions every coin
    is exact, making the result deterministic.
    """
    accepted_by_sid: Dict[int, List[int]] = {}
    for sid, p_all in p_by_sid.items():
        q_rows = q_by_sid[sid]
        proposals = proposals_by_sid[sid]
        gamma = len(proposals)
        accepted: List[int] = []
        rejected = False
        for j in range(gamma):
            proposal = proposals[j]
            if _draw_uniform(generator) < acceptance_probability(
                p_all[j], q_rows[j], proposal
            ):
                accepted.append(proposal)
            else:
                residual = residual_distribution(p_all[j], q_rows[j])
                accepted.append(sample_token(residual, generator))
                rejected = True
                break
        if not rejected:
            bonus = sample_token(p_all[gamma], generator)
            accepted.append(bonus)
        accepted_by_sid[sid] = accepted
    return accepted_by_sid


class SpeculativePagedScheduler(PagedScheduler):
    """Continuous-batching scheduler with an integrated draft/verify round."""

    def __init__(
        self,
        model: Any,
        kv: PagedKVCache,
        draft_model: Any,
        draft_kv: PagedKVCache,
        engine_cfg: EngineConfig,
        sampling: SamplingConfig,
        gamma: int,
    ) -> None:
        super().__init__(model, kv, engine_cfg, sampling)
        self.draft_model = draft_model
        self.draft_kv = draft_kv
        self.gamma = int(gamma)

    # ------------------------------------------------------------ admission
    def _flush_arrivals(
        self,
        arrivals: List[WorkloadRequest],
        when: int,
        result: SchedulerResult,
        generator: torch.Generator,
    ) -> List[Sequence]:
        """Prefill both caches for every request due at step ``when``.

        Rewritten so each new sequence is registered in *both* caches and both
        models prefill the same prompt. The whole prompt's K/V stays in cache (we
        do NOT rewind it): the pending token (``generated[-1]``) is never written,
        so the decoded token stream matches the plain ``PagedScheduler`` exactly --
        the property the spec-equivalence test relies on.
        """
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
            self.draft_kv.add_sequence(seq.sid)
            ttft = self._prefill_spec_one(seq, result, generator)
            seq.ttft_ms = ttft
            admitted.append(seq)
        result.sequences.extend(admitted)
        return admitted

    def _prefill_spec_one(
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
            # Draft sees the identical prompt (assumed shared vocabulary).
            self.draft_model.prefill(
                ids, self.draft_kv, [seq.sid], [len(seq.prompt_ids)]
            )
            # No rewind: the full prompt stays in cache, pending = generated[-1].
        seq.prefilled = True
        return watch.elapsed_ms

    # ------------------------------------------------------------ spec step
    def run_spec_continuous(
        self, requests: Sequence[WorkloadRequest], seed: int, measure: bool = True
    ) -> SchedulerResult:
        result = SchedulerResult(
            mode="spec-continuous", sampling=self.sampling.describe()
        )
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
                    self._spec_step(running, result, generator)
                    # ``total_forwards`` keeps Phase-2 semantics: one target
                    # ``decode_block`` per round == 1. Draft cost is separate.
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
                if self.draft_kv.num_allocated_blocks > result.draft_kv_peak_blocks:
                    result.draft_kv_peak_blocks = self.draft_kv.num_allocated_blocks
            result.steps = step
        result.makespan_ms = stopwatch.elapsed_ms
        result.generated_tokens = sum(len(s.generated) for s in result.sequences)
        result.kv_peak_bytes = result.kv_peak_blocks * self._bytes_per_block()
        if result.tokens_per_round:
            result.mean_tokens_per_round = (
                sum(result.tokens_per_round) / len(result.tokens_per_round)
            )
        if result.tokens_per_round:
            result.mean_acceptance = (sum(result.tokens_per_round)) / (
                len(result.tokens_per_round) * (self.gamma + 1)
            )
        if measure and device.type == "cuda":
            result.peak_memory_mb = round(
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2
            )
        return result

    def _spec_step(
        self,
        running: Sequence[Sequence],
        result: SchedulerResult,
        generator: torch.Generator,
    ) -> None:
        """One draft-propose / target-verify / commit round over ``running``."""
        model = self.model
        device = model.device
        draft = self.draft_model
        gamma = self.gamma
        chunk = gamma + 1
        sids = [s.sid for s in running]
        B = len(running)

        # ---- 1. draft proposes ``gamma`` tokens per sequence (batched)
        proposals: Dict[int, List[int]] = {sid: [] for sid in sids}
        q_rows: Dict[int, List[torch.Tensor]] = {sid: [] for sid in sids}
        cur_ids = [s.generated[-1] for s in running]
        for _ in range(gamma):
            ids_t = torch.tensor(cur_ids, dtype=torch.long, device=device)
            bctx = [self.draft_kv.seq_lengths[sid] for sid in sids]
            q_logits = draft.decode(ids_t, self.draft_kv, sids, bctx)
            result.draft_forward_calls += 1
            for b, s in enumerate(running):
                q_probs = logits_to_probs(q_logits[b], self.sampling)
                q_rows[s.sid].append(q_probs)
                proposals[s.sid].append(sample_token(q_probs, generator))
            cur_ids = [proposals[s.sid][-1] for s in running]

        # ---- 2. target verifies the whole block in one batched forward
        start_positions = [self.kv.seq_lengths[sid] for sid in sids]
        rows = [[s.generated[-1]] + proposals[s.sid] for s in running]
        ids_2d = torch.tensor(rows, dtype=torch.long, device=device)  # (B, chunk)
        p_all_batch = model.decode_block(
            ids_2d, self.kv, sids, start_positions
        )  # (B, chunk, V); forward_calls += 1 inside
        p_by: Dict[int, torch.Tensor] = {
            sids[b]: logits_to_probs(p_all_batch[b], self.sampling) for b in range(B)
        }
        q_by: Dict[int, List[torch.Tensor]] = {sid: q_rows[sid] for sid in sids}

        # ---- 3. accept/reject -> per-sequence committed token list
        accepted_by = batch_accept(p_by, q_by, proposals, generator)

        # ---- 4. per-sequence commit + re-establish the cache invariant
        for b, s in enumerate(running):
            sid = s.sid
            prompt_len = len(s.prompt_ids)
            acc = list(accepted_by[sid])
            # respect the remaining budget (non-spec stops at max_new_tokens)
            remaining = s.max_new_tokens - len(s.generated)
            if len(acc) > remaining:
                acc = acc[:remaining]
            # eos terminates a sequence; the eos token itself is not kept
            if self._eos is not None:
                cut = None
                for i, tok in enumerate(acc):
                    if tok == self._eos:
                        cut = i
                        break
                if cut is not None:
                    acc = acc[:cut]
                    s.eos_hit = True
                    s.finished = True
                    s.finish_step = result.steps
            k = len(acc)
            result.tokens_per_round.append(k)
            if k == 0:
                # nothing committed (e.g. eos first) -> finish.
                s.finished = True
                s.finish_step = result.steps
            else:
                s.generated.extend(acc)
                if len(s.generated) >= s.max_new_tokens:
                    s.finished = True
                    s.finish_step = result.steps
                # invariant: cache holds prompt + all but the last generated token
                keep = prompt_len + len(s.generated) - 1
                self.kv.truncate(sid, keep)
                # align the draft to the same new committed prefix
                d_len = self.draft_kv.seq_lengths[sid]
                if d_len > keep:
                    self.draft_kv.truncate(sid, keep)
                elif d_len < keep:
                    p = d_len
                    while p < keep:
                        tok = s.generated[p - prompt_len]
                        self.draft_model.decode(
                            torch.tensor([[tok]], dtype=torch.long, device=device),
                            self.draft_kv,
                            [sid],
                            [self.draft_kv.seq_lengths[sid]],
                        )
                        result.draft_forward_calls += 1
                        p += 1
            if s.finished:
                self.kv.clear(sid)
                self.draft_kv.clear(sid)
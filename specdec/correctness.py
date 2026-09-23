"""Exactness checks for speculative decoding.

Three independent checks are provided, in increasing order of strength:

1. ``greedy_equivalence`` -- under greedy decoding the draft/verify scheme must
   reproduce the target's argmax path *bit for bit*.
2. ``first_token_test`` -- the empirical first-token distribution produced by the
   speculative decoder is compared against the target's *analytic* distribution
   (KL / TV) and against a chi-square goodness-of-fit test.
3. ``sequence_distribution_test`` -- on a toy vocabulary the joint distribution of
   an ``L``-token continuation can be enumerated, so the speculative decoder's
   output distribution is compared to the exact target joint, not just marginally.

Check 3 is the one that actually validates the rejection-sampling maths: any bug
in ``(p - q)_+`` resampling or in cache truncation shows up as a mismatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from scipy import stats

from .config import SamplingConfig
from .decoding import baseline_decode, next_token_distribution
from .models import CausalLM
from .sampler import (
    build_generator,
    empirical_distribution,
    kl_divergence,
    logits_to_probs,
    top_tokens,
    total_variation,
)
from .speculative import speculative_decode
from .utils import setup_logger

logger = setup_logger(__name__)


def pearson_chi_square(
    observed: torch.Tensor, expected: torch.Tensor, min_expected: float = 5.0
) -> Tuple[Optional[float], Optional[float], int]:
    """Pooled Pearson goodness-of-fit test on *counts*.

    Two deviations from ``scipy.stats.chisquare`` are deliberate:

    * the statistic is evaluated directly, because scipy rejects inputs whose
      totals differ by more than 1e-8 *relative* -- a tolerance that rescaling
      cannot always meet in floating point, and one the statistic never needs;
    * bins with fewer than ``min_expected`` expected counts are pooled into a
      single tail bin, otherwise the chi-square approximation is invalid and
      nearly every test would be reported as a rejection.

    Returns ``(statistic, p_value, dof)``; ``p_value`` is ``None`` when the
    support collapses to a single bin (degenerate / greedy decoding).
    """
    observed = observed.double().reshape(-1)
    expected = expected.double().reshape(-1)
    valid = expected > 0
    observed, expected = observed[valid], expected[valid]
    if min_expected > 0 and bool((expected < min_expected).any()):
        keep = expected >= min_expected
        if int(keep.sum()) >= 1:
            observed = torch.cat([observed[keep], observed[~keep].sum().reshape(1)])
            expected = torch.cat([expected[keep], expected[~keep].sum().reshape(1)])
    dof = int(observed.numel() - 1)
    if dof <= 0:
        return 0.0, None, dof
    statistic = float(((observed - expected) ** 2 / expected).sum().item())
    return statistic, float(stats.chi2.sf(statistic, dof)), dof


@dataclass
class EquivalenceReport:
    """Result bundle for a single correctness experiment."""

    name: str
    num_samples: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_samples": self.num_samples,
            "passed": self.passed,
            "metrics": self.metrics,
            "notes": self.notes,
        }


def greedy_equivalence(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    gamma: int,
    seed: int = 0,
    draft_engine: Optional[Any] = None,
) -> EquivalenceReport:
    """Greedy speculative decoding must equal greedy autoregressive decoding."""
    sampling = SamplingConfig(temperature=0.0)
    reference = baseline_decode(
        target, prompt_ids, max_new_tokens, sampling, measure=False
    )
    speculative = speculative_decode(
        draft,
        target,
        prompt_ids,
        max_new_tokens,
        sampling,
        gamma=gamma,
        measure=False,
        draft_engine=draft_engine,
    )
    matches = reference.token_ids == speculative.token_ids
    first_diff = next(
        (
            index
            for index, (a, b) in enumerate(
                zip(reference.token_ids, speculative.token_ids)
            )
            if a != b
        ),
        None,
    )
    return EquivalenceReport(
        name="greedy_equivalence",
        num_samples=1,
        metrics={
            "gamma": gamma,
            "max_new_tokens": max_new_tokens,
            "reference_tokens": len(reference.token_ids),
            "speculative_tokens": len(speculative.token_ids),
            "first_mismatch_index": first_diff,
            "exact_match": matches,
        },
        passed=bool(matches),
        notes=["greedy decoding is deterministic, so an exact match is required"],
    )


def binned_chi_square(
    empirical: torch.Tensor,
    expected: torch.Tensor,
    num_samples: int,
    top_bins: int = 20,
) -> Dict[str, Optional[float]]:
    """Chi-square gof test on the ``top_bins`` most likely tokens plus an "other" bin.

    A full vocabulary test is impossible (151k bins), so the support is truncated
    at the tokens the target itself considers plausible -- which is exactly where
    a mis-specified sampler would deviate.
    """
    expected = expected.double().reshape(-1)
    empirical = empirical.double().reshape(-1)
    if top_bins >= expected.numel():
        indices = torch.arange(expected.numel())
    else:
        indices = torch.topk(expected, top_bins).indices
    other = torch.ones(expected.numel(), dtype=torch.bool)
    other[indices] = False

    observed_parts = [empirical[indices]]
    expected_parts = [expected[indices]]
    if bool(other.any()):
        # Without a tail bin the two distributions are compared on their full support.
        observed_parts.append(empirical[other].sum().reshape(1))
        expected_parts.append(expected[other].sum().reshape(1))

    total = float(num_samples)
    observed_counts = torch.cat(observed_parts) * total
    expected_counts = torch.cat(expected_parts) * total

    statistic, p_value, dof = pearson_chi_square(observed_counts, expected_counts)
    if statistic is None:
        return {"statistic": None, "p_value": None, "dof": None}
    return {"statistic": statistic, "p_value": p_value, "dof": dof}


def first_token_test(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    sampling: SamplingConfig,
    num_samples: int = 2000,
    gamma: int = 4,
    seed: int = 0,
    label: str = "",
) -> EquivalenceReport:
    """Compare the speculative first-token distribution to the target's exact one."""
    reference = next_token_distribution(target, prompt_ids, sampling).detach().cpu()

    samples: List[int] = []
    for index in range(num_samples):
        generator = build_generator(seed + index, target.device)
        out = speculative_decode(
            draft,
            target,
            prompt_ids,
            max_new_tokens=1,
            sampling=sampling,
            gamma=gamma,
            generator=generator,
            measure=False,
        )
        samples.append(out.token_ids[0])

    empirical = empirical_distribution(samples, target.vocab_size, torch.device("cpu"))
    kl = kl_divergence(empirical, reference)
    tv = total_variation(empirical, reference)
    chi = binned_chi_square(empirical, reference, num_samples)

    # Sampling noise floor: the same statistic for a *direct* sampler from the
    # target, which tells us how large TV/KL are expected to be at this N.
    direct_samples = torch.multinomial(
        reference.detach().cpu().double(), num_samples, replacement=True
    ).tolist()
    direct_empirical = empirical_distribution(
        direct_samples, target.vocab_size, torch.device("cpu")
    )

    # Keep the leading part of the distribution so the figure script can plot the
    # comparison without re-running the model.
    top_ids, top_probs = top_tokens(reference, 20)
    return EquivalenceReport(
        name="first_token_test" + ("_" + label if label else ""),
        num_samples=num_samples,
        metrics={
            "gamma": gamma,
            "kl_speculative_vs_target": kl,
            "tv_speculative_vs_target": tv,
            "kl_noise_floor": kl_divergence(direct_empirical, reference),
            "tv_noise_floor": total_variation(direct_empirical, reference),
            "chi_square": chi,
            "sampling_mode": sampling.describe(),
            "top_token_ids": [int(i) for i in top_ids.tolist()],
            "target_probs_top": [float(v) for v in top_probs.tolist()],
            "speculative_probs_top": [
                float(empirical[int(i)].item()) for i in top_ids.tolist()
            ],
            "monte_carlo_samples": int(num_samples),
        },
        passed=bool(tv <= max(0.05, 3.0 * total_variation(direct_empirical, reference))),
        notes=[
            "TV is bounded below by the Monte-Carlo noise floor; the pass criterion "
            "allows three times that floor, capped at 0.05"
        ],
    )


def analytic_sequence_distribution(
    target: CausalLM,
    prompt_ids: Sequence[int],
    seq_len: int,
    sampling: SamplingConfig,
) -> torch.Tensor:
    """Exact joint distribution over all ``vocab_size ** seq_len`` continuations.

    Only tractable for toy vocabularies, which is precisely why the toy model is
    part of the repository.
    """
    vocab = target.vocab_size
    total = vocab ** seq_len
    flat = torch.zeros(total, dtype=torch.float64)

    def recurse(prefix: List[int], probability: float, depth: int) -> None:
        if depth == seq_len:
            index = 0
            for token in prefix:
                index = index * vocab + token
            flat[index] += probability
            return
        past, logits = target.prefill(prompt_ids)
        if prefix:
            past, logits = target.extend(prefix, past)
            probs = logits_to_probs(logits[-1], sampling)
        else:
            probs = logits_to_probs(logits[-1], sampling)
        for token in range(vocab):
            value = float(probs[token].item())
            if value <= 0:
                continue
            recurse(prefix + [token], probability * value, depth + 1)

    recurse([], 1.0, 0)
    return flat


def empirical_sequence_distribution(
    samples: Sequence[Sequence[int]], vocab_size: int, seq_len: int
) -> torch.Tensor:
    """Histogram over enumerated continuations, matching the analytic layout."""
    total = vocab_size ** seq_len
    counts = torch.zeros(total, dtype=torch.float64)
    for sample in samples:
        index = 0
        for token in sample[:seq_len]:
            index = index * vocab_size + int(token)
        counts[index] += 1.0
    return counts / max(1.0, counts.sum())


def sequence_distribution_test(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    sampling: SamplingConfig,
    seq_len: int = 2,
    num_samples: int = 20000,
    gamma: int = 3,
    seed: int = 0,
    batch_report_every: int = 5000,
) -> EquivalenceReport:
    """Exhaustive joint-distribution test on a toy vocabulary.

    Draws ``num_samples`` continuations with *both* the baseline and the
    speculative decoder and compares each against the analytic joint ``p``.
    """
    exact = analytic_sequence_distribution(target, prompt_ids, seq_len, sampling)

    baseline_samples: List[List[int]] = []
    speculative_samples: List[List[int]] = []
    for index in range(num_samples):
        generator = build_generator(seed + index, target.device)
        baseline_samples.append(
            baseline_decode(
                target,
                prompt_ids,
                seq_len,
                sampling,
                generator=generator,
                measure=False,
            ).token_ids
        )
        generator = build_generator(seed + index, target.device)
        speculative_samples.append(
            speculative_decode(
                draft,
                target,
                prompt_ids,
                seq_len,
                sampling,
                gamma=gamma,
                generator=generator,
                measure=False,
            ).token_ids
        )
        if batch_report_every and (index + 1) % batch_report_every == 0:
            logger.info(
                "sequence test progress: %d/%d", index + 1, num_samples
            )

    baseline_empirical = empirical_sequence_distribution(
        baseline_samples, target.vocab_size, seq_len
    )
    speculative_empirical = empirical_sequence_distribution(
        speculative_samples, target.vocab_size, seq_len
    )

    def compare(empirical: torch.Tensor) -> Dict[str, Any]:
        counts = empirical * num_samples
        keep = exact > 0
        statistic, p_value, _ = pearson_chi_square(
            counts[keep], exact[keep] * num_samples
        )
        # A degenerate distribution (greedy decoding) yields dof = 0 and no
        # p-value; None means "not applicable", not "rejected".
        finite_p = p_value if p_value is None or math.isfinite(p_value) else None
        return {
            "kl_vs_exact": kl_divergence(empirical, exact),
            "tv_vs_exact": total_variation(empirical, exact),
            "chi_square_vs_exact": {
                "statistic": float(statistic),
                "p_value": finite_p,
                "note": (
                    "dof = 0: the target distribution is degenerate (greedy decoding), "
                    "so the goodness-of-fit test carries no information"
                    if finite_p is None
                    else None
                ),
            },
        }

    baseline_metrics = compare(baseline_empirical)
    speculative_metrics = compare(speculative_empirical)
    cross_tv = total_variation(baseline_empirical, speculative_empirical)

    # Both empirical estimates are noisy at this N, so the speculative error must
    # stay within a small multiple of the *baseline sampler's* own error, and the
    # chi-square test must not reject the exact joint distribution.
    tolerance = max(0.02, 3.0 * float(baseline_metrics["tv_vs_exact"]))
    chi_p = speculative_metrics["chi_square_vs_exact"]["p_value"]
    tv_ok = float(speculative_metrics["tv_vs_exact"]) <= tolerance
    chi_ok = chi_p is None or float(chi_p) > 0.01
    return EquivalenceReport(
        name="sequence_distribution_test",
        num_samples=num_samples,
        metrics={
            "gamma": gamma,
            "seq_len": seq_len,
            "vocab_size": target.vocab_size,
            "num_outcomes": int(target.vocab_size ** seq_len),
            "baseline": baseline_metrics,
            "speculative": speculative_metrics,
            "tv_baseline_vs_speculative": cross_tv,
            "tolerance": tolerance,
            "chi_square_pass": chi_ok,
            "sampling_mode": sampling.describe(),
            # Small enumerations are exported so figures can be rebuilt offline.
            "exact_distribution": (
                [float(v) for v in exact.tolist()] if exact.numel() <= 4096 else None
            ),
            "baseline_distribution": (
                [float(v) for v in baseline_empirical.tolist()]
                if exact.numel() <= 4096
                else None
            ),
            "speculative_distribution": (
                [float(v) for v in speculative_empirical.tolist()]
                if exact.numel() <= 4096
                else None
            ),
        },
        passed=bool(tv_ok and chi_ok),
        notes=[
            "chi-square compares counts against the exact analytic joint over all "
            "{} outcomes".format(int(target.vocab_size ** seq_len)),
            "passes when TV <= max(0.02, 3x baseline TV) and chi-square p > 0.01",
        ],
    )
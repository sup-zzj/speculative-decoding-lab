"""Tests for the decoding loop: equivalence, cache bookkeeping, token accounting."""

from __future__ import annotations

import pytest
import torch

from specdec.config import SamplingConfig
from specdec.correctness import sequence_distribution_test
from specdec.decoding import baseline_decode
from specdec.models import build_toy_pair
from specdec.sampler import build_generator
from specdec.speculative import speculative_decode

TOY_KWARGS = {"vocab_size": 8, "hidden_size": 32, "num_layers": 2, "seed": 0}


@pytest.fixture(scope="module")
def toy_pair():
    target, draft, tokenizer = build_toy_pair(noise=0.05, **TOY_KWARGS)
    prompt_ids = [0, 1, 2, 0, 1]
    return target, draft, tokenizer, prompt_ids


def test_greedy_speculative_matches_autoregressive(toy_pair) -> None:
    target, draft, _, prompt_ids = toy_pair
    sampling = SamplingConfig(temperature=0.0)
    reference = baseline_decode(target, prompt_ids, 24, sampling, measure=False)
    speculative = speculative_decode(
        draft, target, prompt_ids, 24, sampling, gamma=4, measure=False
    )
    assert speculative.token_ids == reference.token_ids


def test_identical_models_accept_every_token(toy_pair) -> None:
    target, _, tokenizer, prompt_ids = toy_pair
    identical_target, identical_draft, _ = build_toy_pair(
        noise=0.0, **TOY_KWARGS
    )
    sampling = SamplingConfig(temperature=0.0)
    output = speculative_decode(
        identical_draft, identical_target, prompt_ids, 12, sampling, gamma=5, measure=False
    )
    assert output.stats.acceptance_rate == 1.0
    assert output.stats.mean_tokens_per_round == 6.0
    # gamma + 1 tokens per round means ceil(12 / 6) = 2 rounds.
    assert output.stats.rounds == 2
    # target: 1 prefill + 1 round per batch of commits (12 / 6 = 2 rounds).
    assert output.stats.target_forward_calls == 3


def test_token_budget_is_exact(toy_pair) -> None:
    target, draft, _, prompt_ids = toy_pair
    sampling = SamplingConfig(temperature=0.0)
    for max_new_tokens in (1, 3, 9, 17):
        output = speculative_decode(
            draft, target, prompt_ids, max_new_tokens, sampling, gamma=4, measure=False
        )
        assert len(output.token_ids) == max_new_tokens


def test_one_target_forward_per_round(toy_pair) -> None:
    """Cache bookkeeping must never need an extra synchronisation forward."""
    target, draft, _, prompt_ids = toy_pair
    sampling = SamplingConfig(temperature=1.0)
    output = speculative_decode(
        draft,
        target,
        prompt_ids,
        32,
        sampling,
        gamma=3,
        generator=build_generator(11),
        measure=False,
    )
    stats = output.stats
    assert stats.target_forward_calls == stats.rounds + 1  # +1 for the prompt prefill
    assert stats.proposed_tokens == stats.rounds * 3
    assert len(stats.tokens_per_round) == stats.rounds


def test_acceptance_rate_is_bounded(toy_pair) -> None:
    target, draft, _, prompt_ids = toy_pair
    output = speculative_decode(
        draft,
        target,
        prompt_ids,
        16,
        SamplingConfig(temperature=1.0),
        gamma=3,
        generator=build_generator(5),
        measure=False,
    )
    assert 0.0 <= output.stats.acceptance_rate <= 1.0
    assert all(1 <= value <= 4 for value in output.stats.tokens_per_round)


def test_divergent_draft_still_reproduces_target(toy_pair) -> None:
    """A badly mismatched draft must lower speed but never bias the output."""
    target, _, _, prompt_ids = toy_pair
    _, poor_draft, _ = build_toy_pair(noise=0.6, **TOY_KWARGS)
    report = sequence_distribution_test(
        poor_draft,
        target,
        prompt_ids,
        SamplingConfig(temperature=1.0),
        seq_len=2,
        num_samples=400,
        gamma=4,
        seed=1,
    )
    metrics = report.metrics
    # With a poor draft the acceptance rate collapses, yet the joint stays correct.
    assert metrics["speculative"]["tv_vs_exact"] < 0.35
    assert metrics["baseline"]["tv_vs_exact"] < 0.35


def test_speculative_requires_shared_vocabulary(toy_pair) -> None:
    target, _, _, prompt_ids = toy_pair
    _, other_draft, _ = build_toy_pair(
        vocab_size=4, hidden_size=32, num_layers=2, seed=0
    )
    with pytest.raises(ValueError):
        speculative_decode(
            other_draft,
            target,
            prompt_ids,
            4,
            SamplingConfig(temperature=0.0),
            gamma=2,
            measure=False,
        )


def test_gamma_validation(toy_pair) -> None:
    target, draft, _, prompt_ids = toy_pair
    with pytest.raises(ValueError):
        speculative_decode(
            draft,
            target,
            prompt_ids,
            4,
            SamplingConfig(temperature=0.0),
            gamma=0,
            measure=False,
        )
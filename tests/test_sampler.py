"""Unit tests for the logit post-processing and distribution utilities."""

from __future__ import annotations

import torch

from specdec.config import SamplingConfig
from specdec.sampler import (
    apply_top_k,
    apply_top_p,
    kl_divergence,
    logits_to_probs,
    residual_distribution,
    total_variation,
)


def test_greedy_mode_is_one_hot() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.5]])
    probs = logits_to_probs(logits, SamplingConfig(temperature=0.0))
    assert probs.shape == logits.shape
    assert float(probs.sum()) == 1.0
    assert int(probs.argmax()) == 1


def test_temperature_sharpens_distribution() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    cold = logits_to_probs(logits, SamplingConfig(temperature=0.5))
    warm = logits_to_probs(logits, SamplingConfig(temperature=2.0))
    assert float(cold.max()) > float(warm.max())
    assert abs(float(cold.sum()) - 1.0) < 1e-6


def test_top_k_keeps_exactly_k_tokens() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    filtered = apply_top_k(logits, 2)
    kept = torch.isfinite(filtered).sum().item()
    assert kept == 2
    assert float(filtered[0, 0]) == 5.0


def test_top_p_truncates_tail() -> None:
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    filtered = apply_top_p(logits, 0.8)
    probs = torch.softmax(filtered, dim=-1)
    # The smallest token must be gone, and the kept mass must be >= top_p.
    assert float(probs[0, 3]) == 0.0
    assert float(probs.sum()) >= 0.8 - 1e-6


def test_residual_distribution_is_normalised_and_non_negative() -> None:
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.1, 0.4, 0.5])
    residual = residual_distribution(p, q)
    assert abs(float(residual.sum()) - 1.0) < 1e-6
    assert float(residual.min()) >= 0.0
    # Token 2 has q > p, so the residual there must vanish.
    assert float(residual[2]) == 0.0


def test_residual_falls_back_to_p_when_identical() -> None:
    p = torch.tensor([0.2, 0.8])
    residual = residual_distribution(p, p.clone())
    assert torch.allclose(residual, p, atol=1e-6)


def test_kl_and_tv_properties() -> None:
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.25, 0.75])
    assert kl_divergence(p, p.clone()) == 0.0
    assert kl_divergence(p, q) > 0.0
    assert 0.0 < total_variation(p, q) <= 1.0
    assert total_variation(p, p.clone()) == 0.0


def test_top_p_ignored_when_one() -> None:
    logits = torch.randn(1, 8)
    assert torch.allclose(apply_top_p(logits, 1.0), logits)
"""Logit post-processing, token sampling and distribution distances.

The speculative decoder is only *exact* if the draft and the target are compared
under the very same post-processed distribution, so every transform here is
applied identically to both models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .config import SamplingConfig

_EPS = 1e-12


def logits_to_probs(logits: torch.Tensor, sampling: SamplingConfig) -> torch.Tensor:
    """Convert raw logits of shape (..., vocab) into a probability distribution.

    Greedy mode (``temperature <= 0``) collapses to a one-hot vector so that the
    rejection-sampling rule degenerates into the standard "accept while argmax
    agrees" rule -- the two variants share a single code path on purpose.
    """
    logits = logits.float()
    if sampling.is_greedy:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs

    scaled = logits / max(sampling.temperature, _EPS)
    filtered = apply_top_k(scaled, sampling.top_k)
    filtered = apply_top_p(filtered, sampling.top_p)
    probs = torch.softmax(filtered, dim=-1)
    return probs


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out every logit outside the top-k (kept logits pass through)."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set whose cumulative mass exceeds ``top_p``."""
    if top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Shift right so that the token crossing the threshold is itself retained.
    mask = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    return sorted_logits.scatter(-1, sorted_index, sorted_logits)


def sample_token(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> int:
    """Draw a single token id from a probability vector.

    Sampling always happens on CPU: it keeps the CPU and CUDA code paths
    bit-comparable (one generator, one algorithm) and costs at most a few
    hundred kilobytes per drawn token, which is negligible next to a forward pass.
    """
    flat = probs.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    if generator is None:
        return int(torch.multinomial(flat, num_samples=1).item())
    return int(torch.multinomial(flat, num_samples=1, generator=generator).item())


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalised positive part of ``p - q``, used when a draft token is rejected.

    Sampling from ``(p - q)_+`` is what makes the accept/reject scheme produce
    *exactly* the target distribution (Leviathan et al., 2023, Theorem 1).
    """
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if float(total) <= _EPS:
        # Can only happen when p == q; any sample from p is then also exact.
        return p / p.sum().clamp_min(_EPS)
    return residual / total


def acceptance_probability(p: torch.Tensor, q: torch.Tensor, token_id: int) -> float:
    """``min(1, p(x) / q(x))`` for the proposed token."""
    q_prob = float(q[token_id].item())
    p_prob = float(p[token_id].item())
    if q_prob <= _EPS:
        return 1.0 if p_prob > _EPS else 0.0
    return float(min(1.0, p_prob / q_prob))


def build_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    """Deterministic CPU generator (see :func:`sample_token` for the rationale)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """KL(p || q) in nats, evaluated on the support of ``p``."""
    p = p.double().reshape(-1)
    q = q.double().reshape(-1)
    support = p > 0
    if support.sum() == 0:
        return 0.0
    p_support = p[support]
    q_support = torch.clamp(q[support], min=eps)
    return float(torch.sum(p_support * torch.log(p_support / q_support)).item())


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total variation distance ``0.5 * sum |p - q|`` (in [0, 1])."""
    p = p.double().reshape(-1)
    q = q.double().reshape(-1)
    return float(0.5 * torch.abs(p - q).sum().item())


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence in nats (symmetric, always finite)."""
    p = p.double().reshape(-1)
    q = q.double().reshape(-1)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def empirical_distribution(token_ids, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Histogram of sampled token ids, normalised into a probability vector."""
    counts = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    for token_id in token_ids:
        counts[token_id] += 1.0
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def assert_shared_vocab(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> None:
    """Speculative decoding is only defined when both heads cover one vocabulary."""
    if draft_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError(
            "draft/target vocab mismatch: {} vs {}. Pick two models that share a "
            "tokenizer (e.g. Qwen2.5-0.5B and Qwen2.5-1.5B).".format(
                draft_logits.shape[-1], target_logits.shape[-1]
            )
        )


def top_tokens(probs: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the top-k token ids and their probabilities (for qualitative reports)."""
    values, indices = torch.topk(probs.reshape(-1).float(), min(k, probs.numel()))
    return indices, values
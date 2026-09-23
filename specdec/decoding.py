"""Reference autoregressive decoder.

Every speed-up number in this repository is expressed relative to this
implementation, so it deliberately uses the same cache machinery as the
speculative decoder: the only difference is that it draws one token per forward
pass instead of verifying a whole draft block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import SamplingConfig
from .models import CausalLM
from .sampler import logits_to_probs, sample_token
from .utils import Stopwatch, sync_device


@dataclass
class DecodingOutput:
    """Token ids plus the instrumentation the benchmark reports on."""

    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    steps: int = 0
    forward_calls: int = 0
    latency_ms: float = 0.0
    mean_step_ms: float = 0.0
    peak_memory_mb: Optional[float] = None
    per_step_ms: List[float] = field(default_factory=list)
    #: full distributions are only retained when an experiment asks for them
    probabilities: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "forward_calls": self.forward_calls,
            "latency_ms": round(self.latency_ms, 3),
            "mean_step_ms": round(self.mean_step_ms, 4),
            "tokens_per_second": (
                round(len(self.token_ids) / (self.latency_ms / 1000.0), 2)
                if self.latency_ms > 0
                else None
            ),
            "peak_memory_mb": self.peak_memory_mb,
            "text": self.text,
        }


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2)


def baseline_decode(
    model: CausalLM,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    sampling: SamplingConfig,
    generator: Optional[torch.Generator] = None,
    keep_probabilities: bool = False,
    measure: bool = True,
) -> DecodingOutput:
    """Standard single-token-per-step decoding with a KV cache."""
    output = DecodingOutput()
    device = model.device

    model.reset_counters()
    if measure:
        sync_device(device)
        reset_peak_memory(device)

    stopwatch = Stopwatch()
    with stopwatch:
        past, logits = model.prefill(prompt_ids)
        probs = logits_to_probs(logits, sampling)
        if keep_probabilities:
            output.probabilities.append(probs)

        for _ in range(max_new_tokens):
            token_id = sample_token(probs, generator)
            output.token_ids.append(token_id)
            past, logits = model.logits_for_next(token_id, past)
            probs = logits_to_probs(logits, sampling)
            if keep_probabilities:
                output.probabilities.append(probs)

    output.steps = max_new_tokens
    output.forward_calls = model.forward_calls
    output.latency_ms = stopwatch.elapsed_ms
    output.per_step_ms = [stopwatch.elapsed_ms / max(1, max_new_tokens)]
    output.mean_step_ms = stopwatch.elapsed_ms / max(1, max_new_tokens)
    output.peak_memory_mb = peak_memory_mb(device) if measure else None
    output.text = model.decode(output.token_ids)
    return output


def next_token_distribution(
    model: CausalLM,
    prompt_ids: Sequence[int],
    sampling: SamplingConfig,
) -> torch.Tensor:
    """The model's exact distribution over the first generated token (shape ``(vocab,)``)."""
    _, logits = model.prefill(prompt_ids)
    return logits_to_probs(logits, sampling).reshape(-1)


def distribution_over_prefix(
    model: CausalLM,
    prompt_ids: Sequence[int],
    prefix: Sequence[int],
    sampling: SamplingConfig,
) -> torch.Tensor:
    """Distribution over the token following an explicit prefix.

    Used by the correctness suite to compare the target's *analytic* next-token
    distribution against the empirical one produced by the speculative decoder.
    """
    past, logits = model.prefill(prompt_ids)
    if len(prefix) == 0:
        return logits_to_probs(logits, sampling).reshape(-1)
    past, logits = model.extend(prefix, past)
    return logits_to_probs(logits[-1], sampling).reshape(-1)
"""Speculative decoding lab: exact draft/verify decoding, correctness proofs, benchmarks."""

from __future__ import annotations

from .config import ExperimentConfig, ModelConfig, SamplingConfig
from .correctness import (
    EquivalenceReport,
    first_token_test,
    greedy_equivalence,
    sequence_distribution_test,
)
from .decoding import DecodingOutput, baseline_decode, next_token_distribution
from .models import CausalLM, build_toy_pair
from .speculative import (
    SpeculativeOutput,
    SpeculativeStats,
    expected_tokens_per_round,
    predicted_speedup,
    speculative_decode,
)

__version__ = "0.1.0"

__all__ = [
    "CausalLM",
    "DecodingOutput",
    "EquivalenceReport",
    "ExperimentConfig",
    "ModelConfig",
    "SamplingConfig",
    "SpeculativeOutput",
    "SpeculativeStats",
    "baseline_decode",
    "build_toy_pair",
    "expected_tokens_per_round",
    "first_token_test",
    "greedy_equivalence",
    "next_token_distribution",
    "predicted_speedup",
    "sequence_distribution_test",
    "speculative_decode",
]
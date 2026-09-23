"""Typed configuration objects shared by every experiment script."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Prompts are deliberately short: speculative decoding measures *decoding* cost,
# so a long prefill would dilute the numbers we care about.
DEFAULT_PROMPTS: Tuple[str, ...] = (
    "The key idea behind speculative decoding is",
    "In a distributed system, exactly-once semantics requires",
    "Photosynthesis converts light energy into chemical energy by",
)

DEFAULT_GAMMAS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12)


@dataclass
class SamplingConfig:
    """Post-processing applied to raw logits before drawing a token.

    When ``temperature <= 0`` the decoder runs in greedy mode and the
    distribution is a one-hot vector on the argmax token.
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

    def describe(self) -> str:
        if self.is_greedy:
            return "greedy"
        parts = ["T={:.2f}".format(self.temperature)]
        if self.top_k > 0:
            parts.append("top_k={}".format(self.top_k))
        if self.top_p < 1.0:
            parts.append("top_p={:.2f}".format(self.top_p))
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = "greedy" if self.is_greedy else "sampling"
        return payload


@dataclass
class ModelConfig:
    """Which model pair to load and how to place it on the hardware."""

    draft_model: str = "Qwen/Qwen2.5-0.5B"
    target_model: str = "Qwen/Qwen2.5-1.5B"
    device: str = "auto"
    dtype: str = "auto"
    cache_dir: str = "models"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Everything a benchmark / correctness run needs to be reproducible."""

    models: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prompts: Sequence[str] = DEFAULT_PROMPTS
    max_new_tokens: int = 64
    gamma: int = 6
    gammas: Sequence[int] = DEFAULT_GAMMAS
    repeats: int = 3
    warmup_runs: int = 1
    seed: int = 20260920
    output_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["prompts"] = list(self.prompts)
        payload["gammas"] = list(self.gammas)
        payload["sampling"] = self.sampling.to_dict()
        return payload


def config_from_namespace(args: Any, base: Optional[ExperimentConfig] = None) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from an argparse namespace."""
    config = base or ExperimentConfig()
    config.models.draft_model = getattr(args, "draft_model", config.models.draft_model)
    config.models.target_model = getattr(args, "target_model", config.models.target_model)
    config.models.device = getattr(args, "device", config.models.device)
    config.models.dtype = getattr(args, "dtype", config.models.dtype)
    config.models.cache_dir = getattr(args, "cache_dir", config.models.cache_dir)

    config.sampling.temperature = getattr(args, "temperature", config.sampling.temperature)
    config.sampling.top_k = getattr(args, "top_k", config.sampling.top_k)
    config.sampling.top_p = getattr(args, "top_p", config.sampling.top_p)

    config.max_new_tokens = getattr(args, "max_new_tokens", config.max_new_tokens)
    config.gamma = getattr(args, "gamma", config.gamma)
    config.repeats = getattr(args, "repeats", config.repeats)
    config.seed = getattr(args, "seed", config.seed)
    config.output_dir = getattr(args, "output_dir", config.output_dir)
    return config


def validate_gamma(gamma: int) -> int:
    """Guard against meaningless draft lengths."""
    if gamma < 1:
        raise ValueError("gamma must be >= 1, got {}".format(gamma))
    if gamma > 32:
        raise ValueError("gamma > 32 is not supported by this implementation")
    return gamma


def parse_prompts(raw: Optional[str]) -> List[str]:
    """Accept either ``--prompt-file`` contents or comma-free free text."""
    if not raw:
        return list(DEFAULT_PROMPTS)
    if raw.endswith(".txt"):
        with open(raw, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    return [raw]
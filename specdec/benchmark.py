"""Wall-clock measurement harness.

Every configuration is warmed up, repeated and synchronised before timing, so the
reported means are not polluted by CUDA lazy initialisation or by the first
allocation of the KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import ExperimentConfig, SamplingConfig
from .decoding import DecodingOutput, baseline_decode
from .graph_draft import GraphDraft
from .models import CausalLM
from .sampler import build_generator
from .speculative import (
    SpeculativeOutput,
    expected_tokens_per_round,
    predicted_speedup,
    speculative_decode,
)
from .text_speculative import text_speculative_decode
from .utils import mean_std, setup_logger

logger = setup_logger(__name__)


@dataclass
class Timing:
    """Aggregated timing for one (prompt, configuration) pair."""

    latency_ms: List[float] = field(default_factory=list)
    peak_memory_mb: Optional[float] = None
    tokens: int = 0

    def add(self, output: Any) -> None:
        self.latency_ms.append(float(output.latency_ms))
        if output.peak_memory_mb is not None:
            self.peak_memory_mb = max(self.peak_memory_mb or 0.0, float(output.peak_memory_mb))
        self.tokens = len(output.token_ids)

    def summary(self) -> Dict[str, Any]:
        stats = mean_std(self.latency_ms)
        mean_latency = stats["mean"] or 0.0
        return {
            "latency_ms": stats,
            "tokens": self.tokens,
            "tokens_per_second": (
                round(self.tokens / (mean_latency / 1000.0), 2) if mean_latency > 0 else None
            ),
            "ms_per_token": (
                round(mean_latency / self.tokens, 4) if self.tokens > 0 else None
            ),
            "peak_memory_mb": self.peak_memory_mb,
        }


def measure_baseline(
    model: CausalLM,
    prompt_ids: Sequence[int],
    sampling: SamplingConfig,
    max_new_tokens: int,
    repeats: int,
    warmup_runs: int,
    seed: int,
) -> Timing:
    """Time plain autoregressive decoding of ``model``."""
    timing = Timing()
    for _ in range(warmup_runs):
        baseline_decode(model, prompt_ids, max_new_tokens, sampling, measure=True)
    for repeat in range(repeats):
        generator = build_generator(seed + repeat, model.device)
        output: DecodingOutput = baseline_decode(
            model, prompt_ids, max_new_tokens, sampling, generator=generator, measure=True
        )
        timing.add(output)
    return timing


def build_graph_engine(
    draft: CausalLM, prompt_ids: Sequence[int], max_new_tokens: int, gammas: Sequence[int]
) -> Optional[GraphDraft]:
    """A CUDA-Graph draft engine sized to the worst-case cache footprint.

    The graph writes at most ``prompt_len + max_new_tokens - 1 + gamma - 1``
    distinct positions, so the static cache must leave headroom for the largest
    ``gamma`` evaluated in this run.
    """
    max_cache_len = len(prompt_ids) + int(max_new_tokens) + int(max(gammas or (1,))) + 8
    if draft.device.type != "cuda":
        return None
    return GraphDraft(draft, max_cache_len=max_cache_len)


def measure_speculative(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    sampling: SamplingConfig,
    max_new_tokens: int,
    gamma: int,
    repeats: int,
    warmup_runs: int,
    seed: int,
    draft_engine: Optional[GraphDraft] = None,
    text_level: bool = False,
) -> Dict[str, Any]:
    """Time speculative decoding and aggregate the draft/verify statistics."""
    if text_level:
        # Text-level decoding drives the draft cache from re-encoded text, so a
        # captured CUDA graph (fixed token positions) cannot be replayed.
        if draft_engine is not None:
            raise ValueError("--use-cuda-graph is incompatible with text-level decoding")
        if not sampling.is_greedy:
            raise ValueError("text-level decoding requires greedy sampling")

    decode = text_speculative_decode if text_level else speculative_decode
    # Text-level decoding has no CUDA-graph engine (see the guard above).
    decode_kwargs: Dict[str, Any] = {} if text_level else {"draft_engine": draft_engine}
    timings: List[float] = []
    peak_memory: Optional[float] = None
    token_count = 0
    acceptance_rates: List[float] = []
    tokens_per_round: List[int] = []
    target_calls: List[int] = []
    draft_calls: List[int] = []
    sample_text = ""

    for _ in range(warmup_runs):
        decode(
            draft,
            target,
            prompt_ids,
            max_new_tokens,
            sampling,
            gamma=gamma,
            measure=True,
            **decode_kwargs,
        )
    for repeat in range(repeats):
        generator = build_generator(seed + repeat, target.device)
        output: SpeculativeOutput = decode(
            draft,
            target,
            prompt_ids,
            max_new_tokens,
            sampling,
            gamma=gamma,
            generator=generator,
            measure=True,
            **decode_kwargs,
        )
        timings.append(float(output.latency_ms))
        if output.peak_memory_mb is not None:
            peak_memory = max(peak_memory or 0.0, float(output.peak_memory_mb))
        token_count = len(output.token_ids)
        acceptance_rates.append(output.stats.acceptance_rate)
        tokens_per_round.extend(output.stats.tokens_per_round)
        target_calls.append(output.stats.target_forward_calls)
        draft_calls.append(output.stats.draft_forward_calls)
        sample_text = output.text

    latency = mean_std(timings)
    mean_latency = latency["mean"] or 0.0
    return {
        "gamma": gamma,
        "latency_ms": latency,
        "tokens": token_count,
        "tokens_per_second": (
            round(token_count / (mean_latency / 1000.0), 2) if mean_latency > 0 else None
        ),
        "ms_per_token": round(mean_latency / token_count, 4) if token_count else None,
        "peak_memory_mb": peak_memory,
        "acceptance_rate": mean_std(acceptance_rates),
        "mean_tokens_per_round": mean_std([float(v) for v in tokens_per_round]),
        "expected_tokens_per_round_theory": expected_tokens_per_round(
            acceptance_rates[0] if acceptance_rates else 0.0, gamma
        ),
        "target_forward_calls": mean_std([float(v) for v in target_calls]),
        "draft_forward_calls": mean_std([float(v) for v in draft_calls]),
        "tokens_per_round_hist": tokens_per_round,
        "sample_text": sample_text,
    }


def measure_draft_cost_ratio(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: Sequence[int],
    sampling: SamplingConfig,
    max_new_tokens: int,
    repeats: int,
    seed: int,
    draft_prompt_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Empirical ``c = draft step cost / target step cost``.

    This single number is what the analytic speed-up model needs, and measuring it
    (rather than assuming 0.1 or 0.3 from a paper) is what makes the predicted
    curve comparable to the measured one.

    ``prompt_ids`` lives in the target's token space; when the draft uses a
    different tokenizer (text-level mode) ``draft_prompt_ids`` must be supplied so
    the draft baseline never sees out-of-vocabulary token ids.
    """
    if draft_prompt_ids is None:
        draft_prompt_ids = prompt_ids
    draft_timing = measure_baseline(
        draft, draft_prompt_ids, sampling, max_new_tokens, repeats, warmup_runs=1, seed=seed
    )
    target_timing = measure_baseline(
        target, prompt_ids, sampling, max_new_tokens, repeats, warmup_runs=1, seed=seed
    )
    draft_ms = (draft_timing.summary()["ms_per_token"] or 0.0)
    target_ms = (target_timing.summary()["ms_per_token"] or 0.0)
    ratio = draft_ms / target_ms if target_ms > 0 else float("nan")
    return {
        "draft_ms_per_token": draft_ms,
        "target_ms_per_token": target_ms,
        "cost_ratio": ratio,
    }


def measure_engine_draft_cost(
    engine: GraphDraft,
    prompt_ids: Sequence[int],
    sampling: SamplingConfig,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Per-token draft cost of a CUDA-Graph engine, in ms.

    The engine advances one token per ``propose(..., 1)`` replay, which mirrors
    the real speculative loop (including the sampling step) rather than an
    artificial bare-forward micro-benchmark.
    """
    engine.prefill(prompt_ids)
    engine.rewind(len(prompt_ids) - 1)
    token = int(prompt_ids[-1])

    def advance() -> None:
        nonlocal token
        proposals, _ = engine.propose(token, 1, sampling, None)
        token = proposals[0]

    for _ in range(8):  # first call captures the graph; then it is just replays
        advance()
    torch.cuda.synchronize()
    steps = max(64, int(max_new_tokens))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        advance()
    end.record()
    torch.cuda.synchronize()
    ms_per_token = start.elapsed_time(end) / steps
    return {"draft_ms_per_token": ms_per_token, "cost_ratio": ms_per_token}


def run_benchmark(
    config: ExperimentConfig, use_cuda_graph: bool = False
) -> Dict[str, Any]:
    """Full baseline-vs-speculative benchmark over every prompt in the config."""
    from .models import resolve_device, resolve_dtype

    device = resolve_device(config.models.device)
    logger.info("device=%s dtype=%s", device, resolve_dtype(config.models.dtype, device))

    draft = CausalLM.from_pretrained(config.models, name="draft")
    target = CausalLM.from_pretrained(config.models, name="target")
    logger.info("draft:  %s", draft.describe())
    logger.info("target: %s", target.describe())

    text_level = draft.vocab_size != target.vocab_size
    if text_level:
        logger.info(
            "draft/target vocabularies differ (%d vs %d): using text-level decoding "
            "(greedy only)",
            draft.vocab_size,
            target.vocab_size,
        )
        if use_cuda_graph:
            logger.warning("--use-cuda-graph ignored: incompatible with text-level decoding")

    per_prompt: List[Dict[str, Any]] = []
    for prompt_index, prompt in enumerate(config.prompts):
        prompt_ids = target.encode(prompt)
        logger.info("prompt %d/%d: %r (%d tokens)", prompt_index + 1, len(config.prompts), prompt[:48], len(prompt_ids))

        baseline_timing = measure_baseline(
            target,
            prompt_ids,
            config.sampling,
            config.max_new_tokens,
            config.repeats,
            config.warmup_runs,
            config.seed,
        )
        graph_engine = (
            build_graph_engine(draft, prompt_ids, config.max_new_tokens, config.gammas)
            if (use_cuda_graph and not text_level)
            else None
        )
        if use_cuda_graph and not text_level and graph_engine is None:
            logger.warning("--use-cuda-graph requested but draft is not on CUDA; falling back to eager")
        speculative = measure_speculative(
            draft,
            target,
            prompt_ids,
            config.sampling,
            config.max_new_tokens,
            config.gamma,
            config.repeats,
            config.warmup_runs,
            config.seed,
            draft_engine=graph_engine,
            text_level=text_level,
        )
        baseline_ms = baseline_timing.summary()["latency_ms"]["mean"] or 0.0
        speculative_ms = speculative["latency_ms"]["mean"] or 0.0

        record = {
            "prompt": prompt,
            "prompt_tokens": len(prompt_ids),
            "baseline": baseline_timing.summary(),
            "speculative": speculative,
            "speedup": round(baseline_ms / speculative_ms, 4) if speculative_ms > 0 else None,
        }
        per_prompt.append(record)
        logger.info(
            "  baseline %.1f ms | speculative %.1f ms | speedup %.2fx | acceptance %.3f",
            baseline_ms,
            speculative_ms,
            record["speedup"] or float("nan"),
            speculative["acceptance_rate"]["mean"] or 0.0,
        )

    speedups = [r["speedup"] for r in per_prompt if r["speedup"]]
    mode = "text_level" if text_level else ("cuda_graph" if use_cuda_graph else "eager")
    return {
        "kind": "benchmark",
        "mode": mode,
        "config": config.to_dict(),
        "models": {"draft": draft.describe(), "target": target.describe()},
        "per_prompt": per_prompt,
        "aggregate": {
            "speedup": mean_std([float(v) for v in speedups]),
            "acceptance_rate": mean_std(
                [
                    float(r["speculative"]["acceptance_rate"]["mean"] or 0.0)
                    for r in per_prompt
                ]
            ),
            "baseline_latency_ms": mean_std(
                [
                    float(r["baseline"]["latency_ms"]["mean"] or 0.0)
                    for r in per_prompt
                ]
            ),
            "speculative_latency_ms": mean_std(
                [
                    float(r["speculative"]["latency_ms"]["mean"] or 0.0)
                    for r in per_prompt
                ]
            ),
        },
    }


def run_gamma_sweep(
    config: ExperimentConfig, use_cuda_graph: bool = False
) -> Dict[str, Any]:
    """Measure speed-up for every ``gamma`` and confront it with theory."""
    from .models import resolve_device, resolve_dtype

    device = resolve_device(config.models.device)
    logger.info("gamma sweep on %s (%s)", device, resolve_dtype(config.models.dtype, device))

    draft = CausalLM.from_pretrained(config.models, name="draft")
    target = CausalLM.from_pretrained(config.models, name="target")

    text_level = draft.vocab_size != target.vocab_size
    if text_level:
        logger.info(
            "draft/target vocabularies differ (%d vs %d): using text-level decoding "
            "(greedy only)",
            draft.vocab_size,
            target.vocab_size,
        )
        if use_cuda_graph:
            logger.warning("--use-cuda-graph ignored: incompatible with text-level decoding")

    prompt_ids = target.encode(config.prompts[0])
    graph_engine = (
        build_graph_engine(draft, prompt_ids, config.max_new_tokens, config.gammas)
        if (use_cuda_graph and not text_level)
        else None
    )
    if use_cuda_graph and not text_level and graph_engine is None:
        logger.warning("--use-cuda-graph requested but draft is not on CUDA; falling back to eager")
    baseline_timing = measure_baseline(
        target,
        prompt_ids,
        config.sampling,
        config.max_new_tokens,
        config.repeats,
        config.warmup_runs,
        config.seed,
    )
    baseline_ms = baseline_timing.summary()["latency_ms"]["mean"] or 0.0
    if graph_engine is not None:
        engine_cost = measure_engine_draft_cost(
            graph_engine, prompt_ids, config.sampling, config.max_new_tokens
        )
        cost = {"cost_ratio": engine_cost["cost_ratio"] / (baseline_timing.summary()["ms_per_token"] or 0.0)}
        cost["draft_ms_per_token"] = engine_cost["draft_ms_per_token"]
        cost["target_ms_per_token"] = baseline_timing.summary()["ms_per_token"]
    else:
        cost = measure_draft_cost_ratio(
            draft,
            target,
            prompt_ids,
            config.sampling,
            config.max_new_tokens,
            repeats=max(2, config.repeats),
            seed=config.seed,
            draft_prompt_ids=draft.encode(config.prompts[0]) if text_level else None,
        )
    c = cost["cost_ratio"]

    rows: List[Dict[str, Any]] = []
    for gamma in config.gammas:
        logger.info("gamma=%d ...", gamma)
        measured = measure_speculative(
            draft,
            target,
            prompt_ids,
            config.sampling,
            config.max_new_tokens,
            gamma,
            config.repeats,
            config.warmup_runs,
            config.seed,
            draft_engine=graph_engine,
            text_level=text_level,
        )
        latency = measured["latency_ms"]["mean"] or 0.0
        alpha = float(measured["acceptance_rate"]["mean"] or 0.0)
        rows.append(
            {
                "gamma": gamma,
                "acceptance_rate": measured["acceptance_rate"],
                "measured_speedup": round(baseline_ms / latency, 4) if latency > 0 else None,
                "predicted_speedup": predicted_speedup(alpha, gamma, c),
                "expected_tokens_per_round_measured": measured["mean_tokens_per_round"],
                "expected_tokens_per_round_theory": expected_tokens_per_round(alpha, gamma),
                "tokens_per_second": measured["tokens_per_second"],
                "ms_per_token": measured["ms_per_token"],
                "target_forward_calls": measured["target_forward_calls"],
                "latency_ms": measured["latency_ms"],
                "peak_memory_mb": measured["peak_memory_mb"],
                "tokens_per_round_hist": measured["tokens_per_round_hist"],
                "accepted_tokens_per_target_call": (
                    (measured["mean_tokens_per_round"]["mean"] or 0.0)
                    / (measured["target_forward_calls"]["mean"] or 1.0)
                ),
            }
        )
        logger.info(
            "  measured %.2fx | predicted %.2fx | alpha %.3f | tokens/round %.2f",
            rows[-1]["measured_speedup"] or float("nan"),
            rows[-1]["predicted_speedup"],
            alpha,
            rows[-1]["expected_tokens_per_round_measured"]["mean"] or 0.0,
        )

    mode = "text_level" if text_level else ("cuda_graph" if use_cuda_graph else "eager")
    return {
        "kind": "gamma_sweep",
        "mode": mode,
        "config": config.to_dict(),
        "models": {"draft": draft.describe(), "target": target.describe()},
        "cost_ratio": cost,
        "baseline_ms_per_token": baseline_timing.summary()["ms_per_token"],
        "baseline_latency_ms": baseline_timing.summary()["latency_ms"],
        "rows": rows,
    }
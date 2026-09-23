"""Sweep the batch (bundle) size B and test whether the draft cost ratio falls.

Core hypothesis: making ``B`` identical copies of one prompt share the forward
passes lets the draft model's under-utilised kernels catch up with the target, so
``c(B)`` (draft step cost / target step cost) should drop and per-slot throughput
should improve. This script measures that with a *same-sequence* batch (all
streams in lockstep) and reports ``c(B)`` and the batch-baseline speed-up.

Scope note: this is the minimal bundle-size probe. Heterogeneous batches with
per-sample rollback, batched gamma sweeps and batched CUDA Graph engines are
explicitly out of scope (see the design document).

Usage::

    python scripts/run_batch_experiment.py --batches 1,2,4 --output-dir results/gpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from _cli import add_model_args, add_runtime_args, ensure_output_dir, validate_args  # noqa: E402
from make_plots import _PALETTE, _apply_paper_style, _save, _title  # noqa: E402

from specdec.batch_speculative import (  # noqa: E402
    batch_baseline_decode,
    measure_batch_cost_ratio,
    speculative_decode_batch,
)
from specdec.config import ExperimentConfig, ModelConfig, SamplingConfig, parse_prompts  # noqa: E402
from specdec.models import CausalLM  # noqa: E402
from specdec.utils import save_json, set_seed, setup_logger, timestamp, mean_std  # noqa: E402

logger = setup_logger("run_batch_experiment")

DEFAULT_BATCHES = "1,2,4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the draft cost ratio c(B) and per-slot speed-up vs the "
            "autoregressive baseline for same-sequence batch sizes B"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--batches",
        default=DEFAULT_BATCHES,
        help="comma separated batch sizes to evaluate",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=6)
    parser.add_argument("--log-level", default="INFO")
    return parser


def parse_batches(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 1:
            raise SystemExit("--batches must be >= 1: {}".format(value))
        values.append(value)
    if not values:
        raise SystemExit("no valid batch sizes supplied")
    return sorted(set(values))


def _is_oom(error: Exception) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError)


def _mean_ms_per_slot(values: List[float], tokens_per_slot: int) -> Dict[str, Any]:
    stats = mean_std([v / tokens_per_slot for v in values]) if tokens_per_slot > 0 else {"mean": None, "std": None, "n": 0}
    return stats


def measure_row(
    draft: CausalLM,
    target: CausalLM,
    prompt_ids: List[int],
    sampling: SamplingConfig,
    batch_size: int,
    max_new_tokens: int,
    gamma: int,
    repeats: int,
    warmup_runs: int,
) -> Dict[str, Any]:
    """Time one batch size: cost ratio c(B), baseline and speculative latencies."""
    cost = measure_batch_cost_ratio(
        draft, target, prompt_ids, batch_size, max_new_tokens, sampling
    )

    baseline_latencies: List[float] = []
    for _ in range(warmup_runs):
        batch_baseline_decode(target, prompt_ids, batch_size, max_new_tokens, sampling, measure=True)
    for _ in range(repeats):
        baseline_output = batch_baseline_decode(
            target, prompt_ids, batch_size, max_new_tokens, sampling, measure=True
        )
        baseline_latencies.append(baseline_output.latency_ms)
    baseline_ms_per_slot = _mean_ms_per_slot(baseline_latencies, max_new_tokens)

    spec_latencies: List[float] = []
    spec_peak = 0.0
    acceptance_rates: List[float] = []
    rounds_list: List[int] = []
    for _ in range(warmup_runs):
        speculative_decode_batch(
            draft, target, prompt_ids, max_new_tokens, sampling,
            batch_size=batch_size, gamma=gamma, measure=True,
        )
    for _ in range(repeats):
        spec_output = speculative_decode_batch(
            draft, target, prompt_ids, max_new_tokens, sampling,
            batch_size=batch_size, gamma=gamma, measure=True,
        )
        spec_latencies.append(spec_output.latency_ms)
        if spec_output.peak_memory_mb is not None:
            spec_peak = max(spec_peak, spec_output.peak_memory_mb)
        acceptance_rates.append(spec_output.stats.acceptance_rate)
        rounds_list.append(spec_output.stats.rounds)
    spec_stats = mean_std(spec_latencies)
    spec_mean = spec_stats["mean"] or 0.0

    base_ms = baseline_ms_per_slot["mean"] or 0.0
    spec_ms_per_slot = spec_mean / max_new_tokens if max_new_tokens > 0 else None
    speedup = base_ms / spec_ms_per_slot if base_ms and spec_ms_per_slot and spec_ms_per_slot > 0 else None

    row: Dict[str, Any] = {
        "batch_size": batch_size,
        "cost_ratio": cost["cost_ratio"],
        "draft_ms_per_token_slot": cost["draft_ms_per_token_slot"],
        "target_ms_per_token_slot": cost["target_ms_per_token_slot"],
        "baseline_ms_per_token_slot": baseline_ms_per_slot,
        "speculative_latency_ms": spec_stats,
        "speculative_ms_per_token_slot": (
            round(spec_ms_per_slot, 4) if spec_ms_per_slot is not None else None
        ),
        "speedup": round(speedup, 4) if speedup is not None else None,
        "per_slot_throughput_spec": (
            round(1000.0 / spec_ms_per_slot, 2) if spec_ms_per_slot and spec_ms_per_slot > 0 else None
        ),
        "baseline_per_slot_throughput": (
            round(1000.0 / base_ms, 2) if base_ms else None
        ),
        "acceptance_rate": mean_std(acceptance_rates),
        "rounds": mean_std([float(v) for v in rounds_list]),
        "peak_memory_mb": spec_peak or None,
    }
    logger.info(
        "B=%d c=%.3f | speedup=%.2fx | spec %s ms/slot | accept %.3f | peak %.1f MB",
        batch_size,
        cost["cost_ratio"],
        row["speedup"] or float("nan"),
        ("%.2f" % spec_ms_per_slot) if spec_ms_per_slot else "n/a",
        row["acceptance_rate"]["mean"] or 0.0,
        spec_peak,
    )
    return row


def plot_batch_c_speedup(payload: Dict[str, Any], outdir: str, device: str) -> List[str]:
    """Twin-axis figure: left ``c(B)``, right per-slot speed-up vs ``B``."""
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    batches = [row["batch_size"] for row in rows]
    costs = [row["cost_ratio"] or float("nan") for row in rows]
    speedups = [row["speedup"] or float("nan") for row in rows]

    figure, left = plt.subplots(figsize=(7.0, 4.3))
    right = left.twinx()

    left.plot(batches, costs, marker="o", color=_PALETTE["blue"], linestyle="-", label="draft cost ratio $c(B)$")
    left.set_ylabel("draft / target per-slot cost  $c(B)$")
    left.set_xlabel("batch size $B$ (identical slots)")

    right.plot(batches, speedups, marker="s", color=_PALETTE["orange"], linestyle="--", markersize=5, label="per-slot speed-up")
    right.set_ylabel("per-slot speed-up vs. batch baseline")
    right.axhline(1.0, color=_PALETTE["accent"], linewidth=1.0, linestyle=":")

    left.set_xticks(batches)
    left.set_title(_title(device, "Bundle size: draft cost ratio and per-slot speed-up"))
    handles_l, labels_l = left.get_legend_handles_labels()
    handles_r, labels_r = right.get_legend_handles_labels()
    left.legend(handles_l + handles_r, labels_l + labels_r, loc="upper right", fontsize=8.5)
    left.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "batch_c_speedup_vs_batch")


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.gamma is not None and not (1 <= args.gamma <= 32):
        raise SystemExit("--gamma must be within [1, 32]")
    logging.getLogger("specdec").setLevel(getattr(logging, args.log_level.upper()))
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    config = ExperimentConfig(
        models=ModelConfig(
            draft_model=args.draft_model,
            target_model=args.target_model,
            device=args.device,
            dtype=args.dtype,
            cache_dir=args.cache_dir,
        ),
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    sampling = SamplingConfig(temperature=args.temperature if hasattr(args, "temperature") else 0.0)

    try:
        draft = CausalLM.from_pretrained(config.models, name="draft")
        target = CausalLM.from_pretrained(config.models, name="target")
    except ValueError as error:
        logger.error("model loading failed: %s", error)
        return 2
    logger.info("draft:  %s", draft.describe())
    logger.info("target: %s", target.describe())

    if draft.vocab_size != target.vocab_size:
        logger.error(
            "batch speculative decoding requires a shared vocabulary "
            "(draft=%d target=%d); pick a same-tokenizer pair like Qwen2.5-0.5B -> Qwen2.5-1.5B",
            draft.vocab_size,
            target.vocab_size,
        )
        return 2

    prompt = config.prompts[0]
    prompt_ids: List[int] = target.encode(prompt)
    logger.info("prompt %r (%d tokens)", prompt[:48], len(prompt_ids))

    batches = parse_batches(args.batches)
    rows: List[Dict[str, Any]] = []
    for batch_size in batches:
        try:
            rows.append(
                measure_row(
                    draft,
                    target,
                    prompt_ids,
                    sampling,
                    batch_size,
                    config.max_new_tokens,
                    config.gamma,
                    config.repeats,
                    config.warmup_runs,
                )
            )
        except Exception as error:  # noqa: BLE001 - any OOM/engine error terminates the sweep
            if _is_oom(error):
                logger.warning("B=%d out of memory; stopping (larger batches unavailable)", batch_size)
                break
            logger.error("B=%d failed: %s", batch_size, error)
            return 1
    if not rows:
        logger.error("no rows were measured (first batch OOM?)")
        return 1

    payload: Dict[str, Any] = {
        "kind": "batch_experiment",
        "mode": "same_sequence_batch",
        "config": config.to_dict(),
        "models": {"draft": draft.describe(), "target": target.describe()},
        "gamma": config.gamma,
        "max_new_tokens": config.max_new_tokens,
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "rows": rows,
        "timestamp": timestamp(),
    }
    path = os.path.join(args.output_dir, "batch_experiment_{}.json".format(timestamp()))
    save_json(payload, path)
    logger.info("report written to %s", os.path.abspath(path))

    # Regenerate the bundle-size figure into <output-dir>/figures.
    _apply_paper_style()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    device_label = os.path.basename(os.path.normpath(args.output_dir)).upper()
    figures = plot_batch_c_speedup(payload, fig_dir, device_label if device_label in ("CPU", "GPU") else "")
    for figure in figures:
        logger.info("figure: %s", figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
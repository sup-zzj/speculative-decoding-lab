"""Serving benchmark: continuous batching vs static batching on a paged engine.

Phase 2 of the lab. The engine is *single-model* (no draft/verify step): it
schedules a heterogeneous request set with staggered arrivals over one paged
model and, per step, packs every running sequence into a single paged-attention
decode forward. The static baseline admits everything at step 0 and keeps idle
slots until the longest request finishes (capacity waste). The comparison that
follows -- makespan, throughput, slot utilisation, peak KV blocks, peak GPU
memory -- is the claim continuous batching improves on, and it is measured on
the *same* seeded workload and the same physical block pool so the only
difference is the scheduling policy.

Two backends mirror the correctness script:

* ``--mode toy`` -- a deterministic 2-layer model; validates the whole
  continuous-vs-static pipeline with no download (fast, any machine).
* ``--mode real`` -- real Qwen2.5 weights; the meaningful serving numbers.

Results are written as structured JSON under ``--output-dir`` (use
``results/cpu`` or ``results/gpu`` to honour the hardware split) and companion
figures go to ``<output-dir>/figures/``.

Examples
--------
Toy fast sanity (CPU/GPU)::

    python scripts/run_engine_experiment.py --mode toy --output-dir results/cpu

Real serving benchmark::

    python scripts/run_engine_experiment.py --mode real \\
        --model Qwen/Qwen2.5-0.5B --output-dir results/gpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cli import add_model_args, add_runtime_args, ensure_output_dir, validate_args  # noqa: E402
from make_engine_plots import plot_batch_histogram, plot_metrics_barchart  # noqa: E402

from engine.benchmark import run_serving_experiment  # noqa: E402
from engine.config import (  # noqa: E402
    DEFAULT_PROMPTS,
    EngineConfig,
    SamplingConfig,
    WorkloadConfig,
)
from engine.model import ToyConfig, ToyPagedLM  # noqa: E402
from engine.scheduler import build_kv  # noqa: E402
from engine.utils import save_json, set_seed, setup_logger, sync_device, timestamp  # noqa: E402

logger = setup_logger("run_engine_experiment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure continuous vs static batching on the paged engine: makespan, "
            "throughput, slot utilisation, KV blocks and peak memory"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--mode",
        default="toy",
        choices=["toy", "real"],
        help="toy is a fast, no-download correctness-scale run; real uses Qwen weights",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="model id or local checkpoint dir for --mode real",
    )
    parser.add_argument(
        "--num-requests", type=int, default=12, help="requests served by both policies"
    )
    parser.add_argument("--min-tokens", type=int, default=8, help="min generation budget")
    parser.add_argument("--max-tokens", type=int, default=40, help="max generation budget")
    parser.add_argument("--arrival-window", type=int, default=5, help="arrival spread (steps)")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks", type=int, default=256)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--log-level", default="INFO")
    return parser


# ------------------------------------------------------------------- backends
def _build_paged_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    if args.mode == "real":
        from engine.qwen import build_paged_model

        return build_paged_model(args.model, args.cache_dir, device, dtype)
    return ToyPagedLM(ToyConfig(seed=args.seed), device)


def _workload(args: argparse.Namespace) -> WorkloadConfig:
    return WorkloadConfig(
        prompts=DEFAULT_PROMPTS,
        num_requests=args.num_requests,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        arrival_window=args.arrival_window,
        seed=args.seed,
    )


def _engine_cfg(args: argparse.Namespace, model_id: str) -> EngineConfig:
    return EngineConfig(
        model_id=model_id,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        block_size=args.block_size,
        max_num_blocks=args.max_num_blocks,
        max_batch=args.max_batch,
    )


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.min_tokens < 1 or args.max_tokens < args.min_tokens:
        raise SystemExit("--max-tokens must be >= --min-tokens >= 1")
    logging.getLogger("specdec").setLevel(getattr(logging, args.log_level.upper()))
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("requested %s but CUDA unavailable; falling back to cpu", args.device)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info("building %s paged model on %s", args.mode, device)
    model = _build_paged_model(args, device, dtype)
    engine_cfg = _engine_cfg(args, getattr(model.config, "_name_or_path", None) or "toy")
    kv = build_kv(model, engine_cfg, device)
    sampling = SamplingConfig(temperature=0.0)  # greedy serving baseline

    report: Dict[str, Any] = run_serving_experiment(
        model=model,
        kv=kv,
        engine_cfg=engine_cfg,
        sampling=sampling,
        workload_cfg=_workload(args),
        seed=args.seed,
        measure=True,
    )
    report["mode"] = args.mode
    report["timestamp"] = timestamp()

    name = "engine_serving_{}_{}.json".format(args.mode, timestamp())
    path = os.path.join(args.output_dir, name)
    save_json(report, path)
    logger.info("report written to %s", os.path.abspath(path))

    # ---- figures into <output-dir>/figures
    from make_plots import _apply_paper_style

    _apply_paper_style()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    dev_label = os.path.basename(os.path.normpath(args.output_dir)).upper()
    device_label = dev_label if dev_label in ("CPU", "GPU") else ""
    produced = []
    produced += plot_metrics_barchart(report, fig_dir, device_label)
    produced += plot_batch_histogram(report, fig_dir, device_label)
    for figure in produced:
        logger.info("figure: %s", figure)

    # ---- human summary
    comp = report["comparison"]
    ratio = comp.get("throughput_ratio_cont_over_static")
    logger.info(
        "[result] continuous throughput vs static: %s",
        "{:.3f}x".format(ratio) if ratio is not None else "n/a",
    )
    logger.info(
        "[result] slot utilisation: continuous=%.3f static=%.3f",
        comp.get("slot_utilization_continuous") or 0.0,
        comp.get("slot_utilization_static") or 0.0,
    )
    logger.info(
        "[result] peak KV blocks: continuous=%s static=%s",
        comp.get("kv_peak_blocks_continuous"),
        comp.get("kv_peak_blocks_static"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
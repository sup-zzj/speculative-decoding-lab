"""Serving benchmark: speculative continuous batching vs plain continuous batching.

Phase 3 of the lab. Both policies serve the *same* seeded heterogeneous workload
over the same paged target model and the same physical block pool; the only
difference is the scheduling algorithm (plain ``run_continuous`` vs
``run_spec_continuous`` with a draft/verify round). The headline question is
whether the draft model's extra forwards are repaid by committing more tokens per
target ``decode_block`` -- i.e. whether speculative decoding raises throughput on
a paged, batched serving engine.

Each ``--repeats`` run re-services the identical workload (same seed) so the
spread between repeats is pure timing/measurement variance, and per-metric
mean/std are reported. ``--gamma`` accepts a comma-separated list so a gamma scan
is produced in one invocation (each gamma gets its own JSON file, which
``make_spec_plots.py`` combines).

Two backends mirror the other engine scripts:

* ``--mode toy`` -- fast, no-download, sanity + relative numbers.
* ``--mode real`` -- real Qwen weights (draft ``Qwen/Qwen2.5-0.5B`` ->
  target ``Qwen/Qwen2.5-1.5B``); a CUDA VRAM probe guards the 8 GB case before
  the models are even loaded.

Results are written as ``spec_experiment_*.json`` under ``--output-dir``
(use ``results/cpu`` / ``results/gpu``).

Examples
--------
Toy gamma scan (CPU/GPU)::

    python scripts/run_spec_experiment.py --mode toy --gamma 2,3 --max-batch 4 \\
        --num-requests 6 --repeats 2 --output-dir results/cpu

Real serving benchmark (GPU)::

    python scripts/run_spec_experiment.py --mode real --gamma 2,3 \\
        --max-batch 4 --num-requests 8 --output-dir results/gpu
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cli import add_model_args, add_runtime_args, ensure_output_dir, validate_args  # noqa: E402

from engine.config import (  # noqa: E402
    DEFAULT_PROMPTS,
    EngineConfig,
    SamplingConfig,
    WorkloadConfig,
    WorkloadRequest,
    build_workload,
)
from engine.model import ToyConfig, ToyPagedLM  # noqa: E402
from engine.sampler import build_generator  # noqa: E402
from engine.scheduler import PagedScheduler, SchedulerResult, build_kv  # noqa: E402
from engine.spec_scheduler import SpeculativePagedScheduler  # noqa: E402
from engine.utils import mean_std, save_json, set_seed, setup_logger, timestamp  # noqa: E402

logger = setup_logger("run_spec_experiment")


# ------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure speculative continuous batching vs plain continuous batching: "
            "makespan, throughput, draft/target forward counts, KV peak, round tokens"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(parser)
    add_sampling(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--mode",
        default="toy",
        choices=["toy", "real"],
        help="toy needs no download; real uses Qwen draft/target weights",
    )
    parser.add_argument(
        "--gamma",
        default="3",
        help="comma-separated draft lengths to scan, e.g. '2,3,4'",
    )
    parser.add_argument(
        "--num-requests", type=int, default=8, help="requests served by both policies"
    )
    parser.add_argument("--min-tokens", type=int, default=6, help="min generation budget")
    parser.add_argument("--max-tokens", type=int, default=24, help="max generation budget")
    parser.add_argument("--arrival-window", type=int, default=3, help="arrival spread (steps)")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats per policy")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks", type=int, default=256)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    return parser


def add_sampling(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--temperature", type=float, default=0.0, help="<=0 means greedy"
    )


def parse_gammas(raw: str) -> List[int]:
    """Parse a comma-separated gamma list, validating each entry."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--gamma must contain at least one value")
    gammas = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise SystemExit("invalid gamma value: '{}'".format(part))
        if value < 1 or value > 32:
            raise SystemExit("--gamma must be within [1, 32], got {}".format(value))
        gammas.append(value)
    return gammas


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("requested %s but CUDA unavailable; falling back to cpu", args.device)
        return torch.device("cpu")
    return torch.device(args.device)


# ----------------------------------------------------------- model loading
def _build_models(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
):
    """Return ``(target, draft)`` for the chosen backend."""
    if args.mode == "real":
        from engine.qwen import load_pair

        return load_pair(
            args.draft_model, args.target_model, args.cache_dir, device, dtype
        )
    target = ToyPagedLM(ToyConfig(seed=args.seed, vocab_size=16), device)
    draft = ToyPagedLM(
        ToyConfig(
            seed=7,
            hidden_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            vocab_size=16,
        ),
        device,
    )
    return target, draft


def _probe_vram(args: argparse.Namespace, device: torch.device) -> None:
    """Fail fast if real Qwen draft+target cannot plausibly fit in free VRAM."""
    if args.mode != "real" or device.type != "cuda":
        return
    try:
        props = torch.cuda.get_device_properties(device)
        total = float(props.total_memory) / (1024.0 ** 3)
        free, _ = torch.cuda.mem_get_info(device)
        free_gb = float(free) / (1024.0 ** 3)
    except Exception as exc:  # pragma: no cover
        logger.warning("could not probe VRAM: %s", exc)
        return
    # 0.5B (fp16 ~1 GB) + 1.5B (fp16 ~3 GB) params plus KV/activations.
    estimate = 4.5
    logger.info(
        "VRAM: total=%.1f GB free=%.1f GB (draft+target estimate %.1f GB)",
        total, free_gb, estimate,
    )
    if free_gb < estimate:
        raise SystemExit(
            "only {:.1f} GB VRAM free but real Qwen draft+target estimate ~{:.0f} GB; "
            "lower --max-batch/--num-requests or free memory".format(free_gb, estimate)
        )


# ------------------------------------------------------------ single runs
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


def _run_baseline(
    target: Any,
    engine_cfg: EngineConfig,
    sampling: SamplingConfig,
    requests: List[WorkloadRequest],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """One plain continuous-batching run -> per-repeat metric dict."""
    kv = build_kv(target, engine_cfg, device)
    result = PagedScheduler(target, kv, engine_cfg, sampling).run_continuous(
        requests, seed=seed, measure=True
    )
    r = result.to_dict()
    return {
        "makespan_ms": r["makespan_ms"],
        "throughput_tokens_per_s": r["throughput_tokens_per_s"],
        "target_forward_calls": result.total_forwards,
        "draft_forward_calls": 0,
        "kv_peak_blocks": r["kv_peak_blocks"],
        "draft_kv_peak_blocks": 0,
        "block_util": r["block_util_final"],
        "avg_active_batch": r["avg_active_batch"],
        "mean_tokens_per_round": result.mean_tokens_per_round,
        "mean_acceptance": result.mean_acceptance,
        "peak_memory_mb": result.peak_memory_mb,
    }


def _run_spec(
    target: Any,
    draft: Any,
    engine_cfg: EngineConfig,
    sampling: SamplingConfig,
    requests: List[WorkloadRequest],
    gamma: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """One speculative continuous-batching run -> per-repeat metric dict."""
    kv = build_kv(target, engine_cfg, device)
    dkv = build_kv(draft, engine_cfg, device)
    sched = SpeculativePagedScheduler(
        target, kv, draft, dkv, engine_cfg, sampling, gamma
    )
    result = sched.run_spec_continuous(requests, seed=seed, measure=True)
    r = result.to_dict()
    return {
        "makespan_ms": r["makespan_ms"],
        "throughput_tokens_per_s": r["throughput_tokens_per_s"],
        "target_forward_calls": result.total_forwards,
        "draft_forward_calls": result.draft_forward_calls,
        "kv_peak_blocks": r["kv_peak_blocks"],
        "draft_kv_peak_blocks": result.draft_kv_peak_blocks,
        "block_util": r["block_util_final"],
        "avg_active_batch": r["avg_active_batch"],
        "mean_tokens_per_round": result.mean_tokens_per_round,
        "mean_acceptance": result.mean_acceptance,
        "peak_memory_mb": result.peak_memory_mb,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """mean/std per metric over repeats, dropping None/empty entries."""
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    metrics = rows[0].keys() if rows else []
    for key in metrics:
        values = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = mean_std([float(v) for v in values])  # type: ignore[arg-type]
    return summary


# ------------------------------------------------------------------- main
def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    gammas = parse_gammas(args.gamma)
    args.gamma = gammas[0]  # validate_args reads a single gamma
    validate_args(args)
    args.gamma = ",".join(str(g) for g in gammas)
    if args.min_tokens < 1 or args.max_tokens < args.min_tokens:
        raise SystemExit("--max-tokens must be >= --min-tokens >= 1")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.num_requests < 1:
        raise SystemExit("--num-requests must be >= 1")
    logging.getLogger("specdec").setLevel(getattr(logging, args.log_level.upper()))
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    device = _resolve_device(args)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    _probe_vram(args, device)

    logger.info("building %s model pair on %s", args.mode, device)
    target, draft = _build_models(args, device, dtype)
    engine_cfg = _engine_cfg(
        args, getattr(getattr(target, "config", None), "_name_or_path", None) or "toy"
    )
    sampling = SamplingConfig(temperature=args.temperature)

    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS,
        num_requests=args.num_requests,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        arrival_window=args.arrival_window,
        seed=args.seed,
    )
    requests = build_workload(workload, build_generator(args.seed, device))
    logger.info(
        "workload: %d requests, arrivals in %d steps, budgets in [%d, %d]",
        len(requests), args.arrival_window, args.min_tokens, args.max_tokens,
    )

    config = {
        "engine": engine_cfg.to_dict(),
        "sampling": sampling.to_dict(),
        "workload": workload.to_dict(),
        "mode": args.mode,
        "gamma_list": gammas,
        "repeats": args.repeats,
        "expected_draft_new_tokens": None,
    }

    # Baseline is gamma-independent: run it once per repeat, reuse everywhere.
    base_rows = [
        _run_baseline(target, engine_cfg, sampling, requests, args.seed, device)
        for _ in range(args.repeats)
    ]
    base_summary = _summarize(base_rows)

    for gamma in gammas:
        logger.info("running speculative gamma=%d (%d repeats)", gamma, args.repeats)
        spec_rows = [
            _run_spec(target, draft, engine_cfg, sampling, requests, gamma, args.seed, device)
            for _ in range(args.repeats)
        ]
        spec_summary = _summarize(spec_rows)

        base_thr = base_summary.get("throughput_tokens_per_s", {}).get("mean")
        spec_thr = spec_summary.get("throughput_tokens_per_s", {}).get("mean")
        ratio = (spec_thr / base_thr) if (base_thr and spec_thr) else None

        payload: Dict[str, Any] = {
            "kind": "spec_experiment",
            "mode": args.mode,
            "device": str(torch.cuda.get_device_name(0)) if device.type == "cuda" else "cpu",
            "sampling": sampling.describe(),
            "gamma": gamma,
            "seed": args.seed,
            "config": config,
            "repeats_count": args.repeats,
            "baseline_repeats": base_rows,
            "speculative_repeats": spec_rows,
            "summary": {
                "baseline": base_summary,
                "speculative": spec_summary,
                "spec_vs_baseline_throughput_ratio": ratio,
            },
            "timestamp": timestamp(),
        }
        name = "spec_experiment_{}_g{}.json".format(args.mode, gamma)
        path = os.path.join(args.output_dir, name)
        save_json(payload, path)
        _log_gamma(gamma, base_summary, spec_summary, ratio, path)
    return 0


def _log_gamma(
    gamma: int,
    base_summary: Dict[str, Any],
    spec_summary: Dict[str, Any],
    ratio: Optional[float],
    path: str,
) -> None:
    logger.info("=== gamma=%d ===", gamma)
    logger.info(
        "[result] throughput (tokens/s): speculative=%.3f baseline=%.3f",
        (spec_summary.get("throughput_tokens_per_s") or {}).get("mean") or 0.0,
        (base_summary.get("throughput_tokens_per_s") or {}).get("mean") or 0.0,
    )
    logger.info(
        "[result] spec/baseline throughput ratio: %s",
        "{:.3f}".format(ratio) if ratio is not None else "n/a",
    )
    logger.info(
        "[result] target forward calls: spec=%.1f baseline=%.1f",
        (spec_summary.get("target_forward_calls") or {}).get("mean") or 0.0,
        (base_summary.get("target_forward_calls") or {}).get("mean") or 0.0,
    )
    logger.info(
        "[result] mean tokens/round=%.3f mean acceptance=%.3f",
        (spec_summary.get("mean_tokens_per_round") or {}).get("mean") or 0.0,
        (spec_summary.get("mean_acceptance") or {}).get("mean") or 0.0,
    )
    logger.info("report written to %s", os.path.abspath(path))


if __name__ == "__main__":
    sys.exit(main())
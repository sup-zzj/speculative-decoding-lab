"""Baseline vs speculative decoding: latency, throughput and memory."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cli import add_model_args, add_runtime_args, add_sampling_args, ensure_output_dir, validate_args  # noqa: E402

from specdec.benchmark import run_benchmark  # noqa: E402
from specdec.config import ExperimentConfig, ModelConfig, parse_prompts  # noqa: E402
from specdec.utils import save_json, set_seed, setup_logger, timestamp  # noqa: E402

logger = setup_logger("run_benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure speculative decoding speed-up",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(parser)
    add_sampling_args(parser)
    add_runtime_args(parser)
    parser.add_argument("--gamma", type=int, default=6, help="draft tokens per round")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="run the draft model as a captured CUDA Graph (CUDA only)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
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
        prompts=parse_prompts(args.prompt),
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config.sampling.temperature = args.temperature
    config.sampling.top_k = args.top_k
    config.sampling.top_p = args.top_p

    try:
        payload = run_benchmark(config, use_cuda_graph=args.use_cuda_graph)
    except ValueError as error:
        logger.error("benchmark failed: %s", error)
        return 2

    payload["timestamp"] = timestamp()
    path = os.path.join(args.output_dir, "benchmark_{}.json".format(timestamp()))
    save_json(payload, path)

    aggregate = payload["aggregate"]
    logger.info(
        "mean speed-up %.3fx across %d prompts",
        aggregate["speedup"]["mean"] or float("nan"),
        len(payload["per_prompt"]),
    )
    logger.info("report written to %s", os.path.abspath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
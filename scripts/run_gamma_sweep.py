"""Sweep the draft length gamma and compare measurements against the analytic model.

The point of this script is the *crossover*: the expected number of committed
tokens saturates as ``gamma`` grows while the draft cost keeps accumulating, so
speed-up peaks at a finite ``gamma*``. Reporting that optimum (and the gap
between prediction and measurement) is the analytical core of the project.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cli import add_model_args, add_runtime_args, add_sampling_args, ensure_output_dir, validate_args  # noqa: E402

from specdec.analysis import analyse_sweep  # noqa: E402
from specdec.benchmark import run_gamma_sweep  # noqa: E402
from specdec.config import ExperimentConfig, ModelConfig, parse_prompts  # noqa: E402
from specdec.utils import save_json, set_seed, setup_logger, timestamp  # noqa: E402

logger = setup_logger("run_gamma_sweep")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep gamma and locate the speed-up optimum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(parser)
    add_sampling_args(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--gammas",
        default="1,2,3,4,5,6,8,10,12",
        help="comma separated draft lengths to evaluate",
    )
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


def parse_gammas(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if not 1 <= value <= 32:
            raise SystemExit("gamma out of range [1, 32]: {}".format(value))
        values.append(value)
    if not values:
        raise SystemExit("no valid gamma values supplied")
    return sorted(set(values))


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
        gammas=parse_gammas(args.gammas),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config.sampling.temperature = args.temperature
    config.sampling.top_k = args.top_k
    config.sampling.top_p = args.top_p

    try:
        payload = run_gamma_sweep(config, use_cuda_graph=args.use_cuda_graph)
    except ValueError as error:
        logger.error("sweep failed: %s", error)
        return 2

    payload["analysis"] = analyse_sweep(payload)
    payload["timestamp"] = timestamp()
    path = os.path.join(args.output_dir, "gamma_sweep_{}.json".format(timestamp()))
    save_json(payload, path)

    analysis = payload["analysis"]
    logger.info(
        "best gamma: measured=%s (%.3fx) | predicted=%s (%.3fx) | corr=%.3f",
        analysis.get("best_gamma_measured"),
        analysis.get("best_speedup_measured") or float("nan"),
        analysis.get("best_gamma_predicted"),
        analysis.get("best_speedup_predicted") or float("nan"),
        analysis.get("pearson_measured_vs_predicted") or float("nan"),
    )
    logger.info("report written to %s", os.path.abspath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
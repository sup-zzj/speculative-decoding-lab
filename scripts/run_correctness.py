"""Prove that speculative decoding reproduces the target distribution.

Examples
--------
Toy model (no download, exhaustive joint-distribution check)::

    python scripts/run_correctness.py --mode toy

Real model pair (first-token KL/TV + chi-square)::

    python scripts/run_correctness.py --mode real --num-samples 2000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cli import add_model_args, add_runtime_args, add_sampling_args, ensure_output_dir, sampling_from_args, validate_args  # noqa: E402

from specdec.config import ModelConfig  # noqa: E402
from specdec.correctness import (  # noqa: E402
    EquivalenceReport,
    first_token_test,
    greedy_equivalence,
    sequence_distribution_test,
)
from specdec.models import CausalLM, build_toy_pair  # noqa: E402
from specdec.utils import save_json, set_seed, setup_logger, timestamp  # noqa: E402

logger = setup_logger("run_correctness")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correctness proofs for speculative decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(parser)
    add_sampling_args(parser)
    add_runtime_args(parser)
    parser.add_argument(
        "--mode",
        default="toy",
        choices=["toy", "real", "all"],
        help="toy uses randomly initialised replicas and can enumerate all outcomes",
    )
    parser.add_argument("--gamma", type=int, default=4, help="draft block length under test")
    parser.add_argument("--num-samples", type=int, default=2000, help="Monte-Carlo draws")
    parser.add_argument("--seq-len", type=int, default=2, help="joint-distribution depth (toy)")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="tokens per check")
    parser.add_argument("--toy-vocab", type=int, default=8)
    parser.add_argument(
        "--toy-noise",
        type=float,
        default=0.05,
        help="draft/target weight divergence; controls the acceptance rate",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="verify greedy equivalence with a captured CUDA Graph draft (CUDA only)",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.set_defaults(temperature=1.0)  # distribution tests need real sampling
    return parser


def toy_suite(args: argparse.Namespace) -> List[EquivalenceReport]:
    """Exhaustive checks on a controlled toy pair (no network access needed)."""
    logger.info("building toy pair: vocab=%d noise=%.3f", args.toy_vocab, args.toy_noise)
    target, draft, tokenizer = build_toy_pair(
        vocab_size=args.toy_vocab, noise=args.toy_noise, seed=args.seed
    )
    prompt_ids = [0, 1, 2, 0, 1]
    sampling = sampling_from_args(args)

    reports = [
        greedy_equivalence(
            draft,
            target,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            gamma=args.gamma,
            seed=args.seed,
        )
    ]
    if sampling.is_greedy:
        logger.warning(
            "greedy decoding makes the target distribution degenerate; skipping the "
            "distributional tests (pass --temperature 1.0 to enable them)"
        )
        return reports
    reports.append(
        first_token_test(
            draft,
            target,
            prompt_ids,
            sampling,
            num_samples=args.num_samples,
            gamma=args.gamma,
            seed=args.seed,
            label="toy",
        )
    )
    reports.append(
        sequence_distribution_test(
            draft,
            target,
            prompt_ids,
            sampling,
            seq_len=args.seq_len,
            num_samples=args.num_samples,
            gamma=args.gamma,
            seed=args.seed,
        )
    )
    return reports


def real_suite(args: argparse.Namespace) -> List[EquivalenceReport]:
    """Checks that run against the actual draft/target pair."""
    models = ModelConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
    )
    draft = CausalLM.from_pretrained(models, name="draft")
    target = CausalLM.from_pretrained(models, name="target")
    sampling = sampling_from_args(args)

    reports: List[EquivalenceReport] = []
    for index, prompt in enumerate(_prompts_from_args(args)):
        prompt_ids = target.encode(prompt)
        logger.info("prompt %d: %r -> %d tokens", index + 1, prompt[:48], len(prompt_ids))
        graph_engine = None
        if args.use_cuda_graph:
            from specdec.benchmark import build_graph_engine

            graph_engine = build_graph_engine(
                draft, prompt_ids, args.max_new_tokens, [args.gamma]
            )
            if graph_engine is None:
                logger.warning("--use-cuda-graph ignored: draft is not on CUDA")
        reports.append(
            greedy_equivalence(
                draft,
                target,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                seed=args.seed,
                draft_engine=graph_engine,
            )
        )
        if not sampling.is_greedy:
            reports.append(
                first_token_test(
                    draft,
                    target,
                    prompt_ids,
                    sampling,
                    num_samples=args.num_samples,
                    gamma=args.gamma,
                    seed=args.seed,
                    label="prompt{}".format(index),
                )
            )
    return reports


def _prompts_from_args(args: argparse.Namespace) -> List[str]:
    from specdec.config import parse_prompts

    return parse_prompts(args.prompt)


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logging.getLogger("specdec").setLevel(getattr(logging, args.log_level.upper()))
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    reports: List[EquivalenceReport] = []
    if args.mode in ("toy", "all"):
        reports.extend(toy_suite(args))
    if args.mode in ("real", "all"):
        reports.extend(real_suite(args))

    payload: Dict[str, Any] = {
        "kind": "correctness",
        "timestamp": timestamp(),
        "mode": args.mode,
        "gamma": args.gamma,
        "num_samples": args.num_samples,
        "sampling": sampling_from_args(args).to_dict(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "reports": [report.to_dict() for report in reports],
        "all_passed": all(report.passed for report in reports),
    }
    path = os.path.join(args.output_dir, "correctness_{}.json".format(args.mode))
    save_json(payload, path)

    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        logger.info("[%s] %s (n=%d)", status, report.name, report.num_samples)
    logger.info("report written to %s", os.path.abspath(path))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
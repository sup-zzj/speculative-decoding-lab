"""Quick greedy-equivalence probe for text-level (cross-tokenizer) decoding.

Verifies that ``text_speculative_decode`` with SmolLM2-135M as the draft
reproduces the Qwen2.5-1.5B target's greedy path bit for bit, even though the
two models use completely different tokenizers.

Usage::

    python scripts/_probe_text_level.py [--draft HuggingFaceTB/SmolLM2-135M]
        [--gamma 6] [--max-new-tokens 32] [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specdec.config import SamplingConfig
from specdec.decoding import baseline_decode
from specdec.models import CausalLM, ModelConfig
from specdec.text_speculative import text_speculative_decode
from specdec.utils import setup_logger

logger = setup_logger("probe_text_level")

PROMPTS: List[str] = [
    "The key idea behind speculative decoding is",
    "In a distributed system, exactly-once semantics requires",
    "Photosynthesis converts light energy into chemical energy by",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--cache-dir", default="models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gamma", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    logging.getLogger("specdec").setLevel(getattr(logging, args.log_level.upper()))

    cfg = ModelConfig(
        draft_model=args.draft,
        target_model=args.target,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    draft = CausalLM.from_pretrained(cfg, name="draft")
    target = CausalLM.from_pretrained(cfg, name="target")
    logger.info("draft:  %s", draft.describe())
    logger.info("target: %s", target.describe())
    logger.info(
        "vocab sizes: draft=%d target=%d (cross-tokenizer: %s)",
        draft.vocab_size,
        target.vocab_size,
        draft.vocab_size != target.vocab_size,
    )

    sampling = SamplingConfig(temperature=0.0)
    all_ok = True
    for prompt in PROMPTS:
        prompt_ids = target.encode(prompt)
        reference = baseline_decode(
            target, prompt_ids, args.max_new_tokens, sampling, measure=False
        )
        speculative = text_speculative_decode(
            draft,
            target,
            prompt_ids,
            args.max_new_tokens,
            sampling,
            gamma=args.gamma,
            measure=False,
        )
        matches = reference.token_ids == speculative.token_ids
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(
                    zip(reference.token_ids, speculative.token_ids)
                )
                if a != b
            ),
            None,
        )
        all_ok = all_ok and matches
        logger.info(
            "prompt %r: exact=%s first_diff=%s ref_tokens=%d spec_tokens=%d "
            "acc=%.3f rounds=%d target_calls=%d draft_calls=%d",
            prompt[:40],
            matches,
            first_diff,
            len(reference.token_ids),
            len(speculative.token_ids),
            speculative.stats.acceptance_rate,
            speculative.stats.rounds,
            speculative.stats.target_forward_calls,
            speculative.stats.draft_forward_calls,
        )
        if not matches:
            logger.info("  reference: %r", reference.text[:120])
            logger.info("  speculati: %r", speculative.text[:120])
            all_ok = False

    print("RESULT: {}".format("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

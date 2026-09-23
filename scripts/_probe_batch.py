"""Quick greedy-equivalence probe for same-sequence batch decoding.

Verifies the two correctness claims:
1. For any batch size ``B``, slot 0 of ``speculative_decode_batch`` equals
   ``baseline_decode`` token-for-token (greedy).
2. The batch result is independent of ``B`` (B=1 equals B=2 equals B=4).

Usage::

    python scripts/_probe_batch.py --gamma 4 --max-new-tokens 32 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specdec.batch_speculative import speculative_decode_batch  # noqa: E402
from specdec.config import SamplingConfig  # noqa: E402
from specdec.decoding import baseline_decode  # noqa: E402
from specdec.models import CausalLM, ModelConfig  # noqa: E402
from specdec.utils import setup_logger  # noqa: E402

logger = setup_logger("probe_batch")

PROMPTS: List[str] = [
    "The key idea behind speculative decoding is",
    "In a distributed system, exactly-once semantics requires",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--cache-dir", default="models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batches", default="1,2,4")
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

    sampling = SamplingConfig(temperature=0.0)
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    all_ok = True
    for prompt in PROMPTS:
        prompt_ids = target.encode(prompt)
        reference = baseline_decode(
            target, prompt_ids, args.max_new_tokens, sampling, measure=False
        )
        ref_tokens = reference.token_ids
        per_batch = {}
        for batch_size in batches:
            output = speculative_decode_batch(
                draft,
                target,
                prompt_ids,
                args.max_new_tokens,
                sampling,
                batch_size=batch_size,
                gamma=args.gamma,
                measure=False,
            )
            per_batch[batch_size] = output.token_ids
            matches_reference = ref_tokens == output.token_ids
            all_ok = all_ok and matches_reference
            logger.info(
                "prompt %r | B=%d: ==baseline=%s ref_tokens=%d batch_tokens=%d acc=%.3f rounds=%d",
                prompt[:40],
                batch_size,
                matches_reference,
                len(ref_tokens),
                len(output.token_ids),
                output.stats.acceptance_rate,
                output.stats.rounds,
            )
        # Batch invariance: every B yields the identical stream.
        representative = per_batch[batches[0]]
        for batch_size in batches[1:]:
            invariant = representative == per_batch[batch_size]
            all_ok = all_ok and invariant
            logger.info("  B=1 vs B=%d invariant=%s", batch_size, invariant)

    print("RESULT: {}".format("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
"""Prove the paged engine reproduces a reference decoder.

Two, increasingly strong, checks:

* **toy** -- the authoritative, no-download proof. A heterogeneous request set is
  served by the continuous-batching scheduler; for every sequence the emitted
  greedy token stream must equal an *independent* single-sequence autoregressive
  decode (``engine`` used alone, batch=1). Because the paged attention is
  per-sequence, batching/staggered admission must not change the tokens.

* **real** -- checks the Qwen2.5 adapter against ``transformers``. The same prompt
  is greedy-decoded token-for-token by the paged model and by a native
  ``AutoModelForCausalLM`` decoder, and the two streams must agree on every token.
  This is the strongest claim: the from-scratch paged forward equals PyTorch's
  reference path on real pretrained weights.

Results are written as structured JSON (:ref:`hard constraints
<engine-constraints>`): NaN cleaned to ``null``, one file per run.

Examples
--------
Toy correctness (no network, CPU/GPU)::

    python scripts/run_engine_correctness.py --mode toy

Real-weight equivalence against transformers::

    python scripts/run_engine_correctness.py --mode real \\
        --model Qwen/Qwen2.5-0.5B --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _cli import add_model_args, add_runtime_args, ensure_output_dir, validate_args  # noqa: E402

from engine.config import (  # noqa: E402
    DEFAULT_PROMPTS,
    EngineConfig,
    SamplingConfig,
    WorkloadConfig,
    build_workload,
)
from engine.kv import PagedKVCache  # noqa: E402
from engine.model import ToyConfig, ToyPagedLM  # noqa: E402
from engine.sampler import build_generator  # noqa: E402
from engine.scheduler import PagedScheduler, build_kv  # noqa: E402
from engine.utils import save_json, set_seed, setup_logger, timestamp  # noqa: E402

logger = setup_logger("run_engine_correctness")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correctness proofs for the paged serving engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_runtime_args(parser)
    add_model_args(parser)
    parser.add_argument(
        "--mode",
        default="toy",
        choices=["toy", "real", "all"],
        help="toy needs no download; real validates the Qwen adapter vs transformers",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="model id or local checkpoint dir for the --mode real path",
    )
    parser.add_argument(
        "--num-requests", type=int, default=6, help="requests in the served workload"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=40, help="budget for real-mode prompts"
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks", type=int, default=512)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    return parser


# ---------------------------------------------------------------- toy mode
def _single_sequence_reference(
    model: ToyPagedLM,
    kv: PagedKVCache,
    prompt_ids: List[int],
    max_new_tokens: int,
) -> List[int]:
    """Independent batch=1 greedy decode (the ground truth for the engine)."""
    sampled: List[int] = []
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    logits = model.prefill(ids, kv, [0], [len(prompt_ids)])
    sampled.append(int(logits[0].argmax(dim=-1).item()))
    for _ in range(max_new_tokens - 1):
        ctx = [int(kv.seq_lengths[0])]
        cur = torch.tensor([sampled[-1]], dtype=torch.long, device=model.device)
        logits = model.decode(cur, kv, [0], ctx)
        sampled.append(int(logits[0].argmax(dim=-1).item()))
    return sampled


def toy_suite(args: argparse.Namespace) -> Dict[str, Any]:
    """Serve a workload and check every request against a fresh reference."""
    model = ToyPagedLM(ToyConfig(seed=args.seed), _DEVICE)
    cfg = EngineConfig(
        block_size=args.block_size,
        max_num_blocks=args.max_num_blocks,
        max_batch=args.max_batch,
    )
    kv = build_kv(model, cfg, _DEVICE)
    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS,
        num_requests=args.num_requests,
        min_tokens=4,
        max_tokens=16,
        arrival_window=4,
        seed=args.seed,
    )
    requests = build_workload(workload, build_generator(args.seed, _DEVICE))
    scheduler = PagedScheduler(model, kv, cfg, SamplingConfig(temperature=0.0))
    result = scheduler.run_continuous(requests, seed=args.seed, measure=False)

    rows = []
    failures = 0
    for seq in result.sequences:
        kv.reset()
        kv.add_sequence(0)
        reference = _single_sequence_reference(
            model, kv, seq.prompt_ids, seq.max_new_tokens
        )
        match = reference == seq.generated
        failures += 0 if match else 1
        rows.append(
            {
                "sid": seq.sid,
                "prompt_chars": len(seq.prompt_ids),
                "max_new_tokens": seq.max_new_tokens,
                "generated": seq.generated,
                "matches_reference": match,
            }
        )
    return {
        "name": "toy-continuous-vs-single-reference",
        "num_requests": len(requests),
        "served_tokens": result.generated_tokens,
        "all_match": failures == 0,
        "failures": failures,
        "rows": rows,
    }


# ---------------------------------------------------------------- real mode
def real_suite(args: argparse.Namespace) -> Dict[str, Any]:
    """Greedy-decode the same prompts with paged engine vs ``transformers``."""
    from transformers import AutoTokenizer

    from engine.qwen import build_paged_model

    dtype = torch.float16 if _DEVICE.type == "cuda" else torch.float32
    model = build_paged_model(args.model, args.cache_dir, _DEVICE, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_local(args), cache_dir=args.cache_dir
    )
    prompts = list(DEFAULT_PROMPTS[:2]) + ["The Transformer model was"]

    rows = []
    failures = 0
    for index, prompt in enumerate(prompts):
        paged_tokens = _real_paged_greedy(model, tokenizer, prompt, args.max_new_tokens)
        native_tokens = _real_native_greedy(
            args.model, args.cache_dir, dtype, tokenizer, prompt, args.max_new_tokens
        )
        match = paged_tokens == native_tokens
        failures += 0 if match else 1
        steps = len(paged_tokens)
        first_diff = -1
        for j, (a, b) in enumerate(zip(paged_tokens, native_tokens)):
            if a != b:
                first_diff = j
                break
        rows.append(
            {
                "prompt_index": index,
                "num_tokens": steps,
                "matches_transformers": match,
                "first_divergence_at": first_diff,
                "paged_tokens": paged_tokens,
                "native_tokens": native_tokens,
            }
        )
    return {
        "name": "real-paged-vs-transformers",
        "model": args.model,
        "num_prompts": len(prompts),
        "all_match": failures == 0,
        "failures": failures,
        "rows": rows,
    }


def _real_paged_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> List[int]:
    """Greedy-decode a prompt through the paged engine (batch=1)."""
    from engine.scheduler import build_kv

    prompt_ids = list(tokenizer.encode(prompt))
    cfg = EngineConfig(
        model_id=model.config._name_or_path or "qwen",
        max_num_blocks=512,
        block_size=16,
    )
    kv = build_kv(model, cfg, _DEVICE)
    kv.add_sequence(0)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=_DEVICE)
    logits = model.prefill(ids, kv, [0], [len(prompt_ids)])
    generated = [int(logits[0].argmax(dim=-1).item())]
    for _ in range(max_new_tokens - 1):
        ctx = [int(kv.seq_lengths[0])]
        cur = torch.tensor([generated[-1]], dtype=torch.long, device=_DEVICE)
        logits = model.decode(cur, kv, [0], ctx)
        generated.append(int(logits[0].argmax(dim=-1).item()))
    return generated


def _real_native_greedy(model_id: str, cache_dir: str, dtype, tokenizer, prompt, budget):
    from transformers import AutoModelForCausalLM

    native = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=dtype
    ).to(_DEVICE).eval()
    generated: List[int] = []
    with torch.no_grad():
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(_DEVICE)
        past = ids.clone()
        for _ in range(budget):
            out = native(past).logits[:, -1, :]
            tok = int(out.argmax(dim=-1).item())
            generated.append(tok)
            past = torch.cat((past, torch.tensor([[tok]], device=_DEVICE)), dim=1)
    del native
    return generated


def resolve_local(args: argparse.Namespace) -> str:
    from engine.qwen import resolve_local

    return resolve_local(args.model, args.cache_dir)


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logging.getLogger("specdec").setLevel(getattr(logging, args.log_level.upper()))
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    reports: List[Dict[str, Any]] = []
    if args.mode in ("toy", "all"):
        reports.append(toy_suite(args))
    if args.mode in ("real", "all"):
        reports.append(real_suite(args))

    all_passed = all(report["all_match"] for report in reports)
    payload: Dict[str, Any] = {
        "kind": "engine_correctness",
        "timestamp": timestamp(),
        "mode": args.mode,
        "device": str(torch.cuda.get_device_name(0)) if _DEVICE.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "reports": reports,
        "all_passed": all_passed,
    }
    path = os.path.join(args.output_dir, "engine_correctness_{}.json".format(args.mode))
    save_json(payload, path)
    for report in reports:
        status = "PASS" if report["all_match"] else "FAIL"
        logger.info(
            "[%s] %s (requests/prompts=%d, served tokens=%d)",
            status, report["name"], report.get("num_requests", report.get("num_prompts")),
            report.get("served_tokens", report.get("num_prompts")),
        )
    logger.info("report written to %s", os.path.abspath(path))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
"""Prove speculative continuous batching reproduces the plain engine.

Phase 3 of the lab. The load-bearing claim is that adding a draft/verify round
must NOT change the emitted token stream under greedy decoding: the per-sequence
KV rollback in :class:`~engine.spec_scheduler.SpeculativePagedScheduler` has to be
exact, otherwise the spec engine would deviate from the authoritative Phase-2
``PagedScheduler``. Two suites:

* **toy** -- the no-download proof. A heterogeneous request set (staggered
  arrivals, varying budgets/prompts) is served twice under the same seed (and
  shared target, same vocabulary): once by the plain continuous-batching
  scheduler, once by the speculative scheduler against a *different-capacity*
  draft toy model. Because greedy acceptance is exact, every sequence's generated
  tokens must match token-for-token.

* **real** -- the strongest claim. ``Qwen/Qwen2.5-0.5B`` (draft) proposes into
  ``Qwen/Qwen2.5-1.5B`` (target); each spec sequence is greedy-decoded through the
  paged engine and compared token-for-token against an independent
  ``AutoModelForCausalLM`` reference decoder on the same weights. This requires
  ``transformers`` + a network (or a populated cache dir) and fails loudly rather
  than silently if either is unavailable.

Results are written as structured JSON (NaN/Inf cleaned to ``null``), one file per
``--mode``, mirroring the ``engine_correctness_*.json`` schema.

Examples
--------
Toy equality (no download, CPU/GPU)::

    python scripts/run_spec_correctness.py --mode toy --gamma 3

Real token-equivalence against transformers (GPU recommended)::

    python scripts/run_spec_correctness.py --mode real \\
        --draft-model Qwen/Qwen2.5-0.5B --target-model Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

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
from engine.utils import save_json, set_seed, setup_logger, timestamp  # noqa: E402

logger = setup_logger("run_spec_correctness")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Correctness proofs for the speculative (draft/verify) paged engine: "
            "spec greedy must equal plain continuous greedy"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_runtime_args(parser)
    add_model_args(parser)
    parser.add_argument(
        "--mode",
        default="toy",
        choices=["toy", "real", "all"],
        help="toy needs no download; real validates the Qwen draft/target vs "
        "transformers' autoregressive greedy on real weights",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=3,
        help="draft proposals verified per round",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=6,
        help="requests in the served workload (toy suite)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="generation budget for real-mode prompts",
    )
    parser.add_argument(
        "--block-size", type=int, default=16, help="tokens per physical KV block"
    )
    parser.add_argument(
        "--max-num-blocks", type=int, default=256, help="physical KV block capacity"
    )
    parser.add_argument(
        "--max-batch", type=int, default=16, help="soft cap on sequences per step"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("requested %s but CUDA unavailable; falling back to cpu", args.device)
        return torch.device("cpu")
    return torch.device(args.device)


# ---------------------------------------------------------------- toy mode
def _build_toy_workload(args: argparse.Namespace) -> List[Any]:
    """Heterogeneous request set with staggered arrivals (seeded)."""
    workload = WorkloadConfig(
        prompts=DEFAULT_PROMPTS,
        num_requests=args.num_prompts,
        min_tokens=4,
        max_tokens=16,
        arrival_window=4,
        seed=args.seed,
    )
    return build_workload(workload, build_generator(args.seed, _DEVICE))


def _generation_map(result: SchedulerResult) -> Dict[Tuple, Tuple[int, ...]]:
    """Map a sequence's identity to its emitted greedy tokens (order-independent)."""
    return {
        (
            s.arrive_time,
            s.max_new_tokens,
            tuple(s.prompt_ids),
        ): tuple(s.generated)
        for s in result.sequences
    }


def toy_suite(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Equal-capacity target vs a different-capacity draft over one workload."""
    requests = _build_toy_workload(args)
    cfg = EngineConfig(
        block_size=args.block_size,
        max_num_blocks=args.max_num_blocks,
        max_batch=args.max_batch,
    )
    greedy = SamplingConfig(temperature=0.0)

    target = ToyPagedLM(ToyConfig(seed=args.seed, vocab_size=16), device)
    draft = ToyPagedLM(
        ToyConfig(
            seed=7,
            hidden_size=48,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            vocab_size=16,  # must share the target's vocabulary
        ),
        device,
    )

    # (a) authoritative plain continuous-batching run.
    kv_base = build_kv(target, cfg, device)
    base = PagedScheduler(target, kv_base, cfg, greedy).run_continuous(
        requests, seed=args.seed, measure=False
    )
    base_generated = _generation_map(base)

    # (b) speculative run against the distinct draft (independent caches).
    kv_spec = build_kv(target, cfg, device)
    kv_draft = build_kv(draft, cfg, device)
    sched = SpeculativePagedScheduler(
        target, kv_spec, draft, kv_draft, cfg, greedy, args.gamma
    )
    spec = sched.run_spec_continuous(requests, seed=args.seed, measure=False)

    rows: List[Dict[str, Any]] = []
    failures = 0
    for s in spec.sequences:
        key = (s.arrive_time, s.max_new_tokens, tuple(s.prompt_ids))
        ref = base_generated.get(key)
        if ref is None:
            failures += 1
            rows.append(
                {
                    "sid": s.sid,
                    "prompt_len": len(s.prompt_ids),
                    "max_new_tokens": s.max_new_tokens,
                    "matches": False,
                    "first_divergence_at": 0,
                    "num_tokens": len(s.generated),
                    "reason": "no matching plain-engine sequence",
                }
            )
            continue
        spec_gen = tuple(s.generated)
        match = spec_gen == ref
        failures += 0 if match else 1
        first = -1
        for j, (a, b) in enumerate(zip(spec_gen, ref)):
            if a != b:
                first = j
                break
        rows.append(
            {
                "sid": s.sid,
                "prompt_len": len(s.prompt_ids),
                "max_new_tokens": s.max_new_tokens,
                "matches": match,
                "first_divergence_at": first,
                "num_tokens": len(spec_gen),
                "spec_generated": list(spec_gen),
            }
        )
    return {
        "name": "toy-spec-vs-plain-continuous",
        "gamma": args.gamma,
        "num_requests": len(requests),
        "served_tokens": spec.generated_tokens,
        "all_match": failures == 0,
        "failures": failures,
        "rows": rows,
    }


# ---------------------------------------------------------------- real mode
def _real_workload_warn() -> None:
    logger.info(
        "real suite: greedy spec (draft Qwen/Qwen2.5-0.5B -> target "
        "Qwen/Qwen2.5-1.5B) compared against an autoregressive transformers decode"
    )


def real_suite(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Spec greedy on real weights vs an independent ``transformers`` greedy."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.qwen import build_paged_model, load_pair, resolve_local

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    draft, target = load_pair(
        args.draft_model, args.target_model, args.cache_dir, device, dtype
    )
    tokenizer: AutoTokenizer = target.tokenizer

    _real_workload_warn()
    prompts = list(DEFAULT_PROMPTS[: args.num_prompts])

    cfg = EngineConfig(
        model_id=getattr(target.config, "_name_or_path", None) or "qwen",
        max_num_blocks=args.max_num_blocks,
        block_size=args.block_size,
        max_batch=args.max_batch,
    )
    greedy = SamplingConfig(temperature=0.0)
    kv = build_kv(target, cfg, device)
    dkv = build_kv(draft, cfg, device)
    requests = [
        WorkloadRequest(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            arrive_time=i,
        )
        for i, prompt in enumerate(prompts)
    ]
    sched = SpeculativePagedScheduler(
        target, kv, draft, dkv, cfg, greedy, args.gamma
    )
    spec = sched.run_spec_continuous(requests, seed=args.seed, measure=False)

    rows: List[Dict[str, Any]] = []
    failures = 0
    for s in spec.sequences:
        native = _real_native_greedy_ids(
            args.target_model,
            args.cache_dir,
            dtype,
            device,
            s.prompt_ids,
            s.max_new_tokens,
        )
        spec_gen = tuple(s.generated)
        match = spec_gen == tuple(native)
        failures += 0 if match else 1
        first = -1
        for j, (a, b) in enumerate(zip(spec_gen, native)):
            if a != b:
                first = j
                break
        rows.append(
            {
                "sid": s.sid,
                "num_tokens": len(spec_gen),
                "matches_transformers": match,
                "first_divergence_at": first,
                "spec_tokens": list(spec_gen),
                "native_tokens": native,
            }
        )
    return {
        "name": "real-spec-vs-transformers",
        "gamma": args.gamma,
        "model": args.target_model,
        "num_prompts": len(prompts),
        "all_match": failures == 0,
        "failures": failures,
        "rows": rows,
    }


def _real_native_greedy_ids(
    model_id: str,
    cache_dir: str,
    dtype: torch.dtype,
    device: torch.device,
    prompt_ids: Sequence[int],
    budget: int,
) -> List[int]:
    """Autoregressive greedy reference decode of ``prompt_ids`` (batch=1)."""
    from transformers import AutoModelForCausalLM

    source = _resolve_local_model(model_id, cache_dir)
    native = AutoModelForCausalLM.from_pretrained(
        source, cache_dir=cache_dir, torch_dtype=dtype
    ).to(device).eval()
    generated: List[int] = []
    with torch.no_grad():
        past = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        for _ in range(budget):
            out = native(past).logits[:, -1, :]
            tok = int(out.argmax(dim=-1).item())
            generated.append(tok)
            past = torch.cat(
                (past, torch.tensor([[tok]], dtype=torch.long, device=device)), dim=1
            )
    del native
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return generated


def _resolve_local_model(model_id: str, cache_dir: str) -> str:
    from engine.qwen import resolve_local

    return resolve_local(model_id, cache_dir)


# ------------------------------------------------------------------- main
def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logging.getLogger("specdec").setLevel(getattr(logging, args.log_level.upper()))
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)
    device = _resolve_device(args)

    reports: List[Dict[str, Any]] = []
    if args.mode in ("toy", "all"):
        reports.append(toy_suite(args, device))
    if args.mode in ("real", "all"):
        reports.append(real_suite(args, device))

    all_passed = all(report["all_match"] for report in reports)
    payload: Dict[str, Any] = {
        "kind": "spec_correctness",
        "mode": args.mode,
        "device": str(torch.cuda.get_device_name(0)) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "seed": args.seed,
        "gamma": args.gamma,
        "timestamp": timestamp(),
        "reports": reports,
        "all_passed": all_passed,
    }
    name = "spec_correctness_{}.json".format(args.mode)
    path = os.path.join(args.output_dir, name)
    save_json(payload, path)

    for report in reports:
        status = "PASS" if report["all_match"] else "FAIL"
        logger.info(
            "[%s] %s (gamma=%d, requests/prompts=%d, served tokens=%d)",
            status,
            report["name"],
            report["gamma"],
            report.get("num_requests", report.get("num_prompts")),
            report.get("served_tokens", report.get("num_prompts")),
        )
    logger.info("report written to %s", os.path.abspath(path))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
"""Benchmark entry points: synthesise a workload, run both schedulers, report.

``run_serving_experiment`` is the single entry the CLI drives. It builds a
heterogeneous request set, runs the continuous and static schedulers over the same
paged model, and aggregates the two :class:`SchedulerResult` objects plus the
hardware/metrics payload into a JSON-serialisable report that scripts store under
``results/<device>/``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from .config import EngineConfig, SamplingConfig, WorkloadConfig, WorkloadRequest, build_workload
from .kv import PagedKVCache
from .scheduler import PagedScheduler, SchedulerResult, build_kv
from .utils import setup_logger, sync_device
from .sampler import build_generator

logger = setup_logger(__name__)


def build_engine(
    model_id: str,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Instantiate the real paged model (weights loaded once)."""
    from .qwen import build_paged_model

    return build_paged_model(model_id, cache_dir, device, dtype)


def measure_engine(
    model: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Model/hardware facts recorded in every report."""
    config = getattr(model, "config", None)
    if config is not None:
        name = getattr(config, "_name_or_path", None) or "toy"
        model_type = getattr(config, "model_type", None) or ""
    else:
        name = model.__class__.__name__
        model_type = ""
    return {
        "model": name,
        "model_type": model_type,
        "num_layers": model.num_layers,
        "num_q_heads": model.num_q_heads,
        "num_kv_heads": model.num_kv_heads,
        "head_dim": model.head_dim,
        "dtype": str(model.dtype).replace("torch.", ""),
        "device": str(device),
        "gpu_name": _gpu_name(),
    }


def _gpu_name() -> str:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def run_serving_experiment(
    model: Any,
    kv: PagedKVCache,
    engine_cfg: EngineConfig,
    sampling: SamplingConfig,
    workload_cfg: WorkloadConfig,
    seed: int = 20260920,
    measure: bool = True,
) -> Dict[str, Any]:
    """Serve ``workload_cfg`` with both schedulers and return a full report.

    The same seeded workload and paged cache backend serve both policies, so the
    only difference between the two rows is the scheduling algorithm -- the clean
    isolation the comparison needs.
    """
    device = model.device
    requests: List[WorkloadRequest] = build_workload(workload_cfg, build_generator(seed, device))
    scheduler = PagedScheduler(model, kv, engine_cfg, sampling)

    # Re-seed the KV for each run so continuous/static start from identical state.
    continuous = _fresh_run(scheduler, kv, requests, "continuous", seed, measure)
    static = _fresh_run(scheduler, kv, requests, "static", seed, measure)

    payload: Dict[str, Any] = {
        "engine_config": engine_cfg.to_dict(),
        "sampling": sampling.to_dict(),
        "workload": workload_cfg.to_dict(),
        "seed": seed,
        "hardware": measure_engine(model, device),
        "continuous": continuous.to_dict(),
        "static": static.to_dict(),
        "comparison": _compare_dicts(continuous, static),
    }
    return payload


def _fresh_run(
    scheduler: PagedScheduler,
    kv: PagedKVCache,
    requests: List[WorkloadRequest],
    mode: str,
    seed: int,
    measure: bool,
) -> SchedulerResult:
    # Reset metadata (block tables / lengths) without reallocating the tensors.
    kv.reset()
    if mode == "continuous":
        return scheduler.run_continuous(requests, seed, measure=measure)
    return scheduler.run_static(requests, seed, measure=measure)


def _compare_dicts(continuous: SchedulerResult, static: SchedulerResult) -> Dict[str, Any]:
    cont = continuous.to_dict()
    stat = static.to_dict()
    makespan_ratio = None
    if stat.get("makespan_ms") and cont.get("makespan_ms"):
        makespan_ratio = round(cont["makespan_ms"] / stat["makespan_ms"], 3)
    throughput_ratio = None
    if stat.get("throughput_tokens_per_s") and cont.get("throughput_tokens_per_s"):
        ct = cont["throughput_tokens_per_s"]
        st = stat["throughput_tokens_per_s"]
        if st and ct:
            throughput_ratio = round(ct / st, 3)
    return {
        "makespan_ms_continuous": cont.get("makespan_ms"),
        "makespan_ms_static": stat.get("makespan_ms"),
        "makespan_ratio_cont_over_static": makespan_ratio,
        "throughput_tokens_per_s_continuous": cont.get("throughput_tokens_per_s"),
        "throughput_tokens_per_s_static": stat.get("throughput_tokens_per_s"),
        "throughput_ratio_cont_over_static": throughput_ratio,
        "slot_utilization_continuous": cont.get("slot_utilization"),
        "slot_utilization_static": stat.get("slot_utilization"),
        "kv_peak_blocks_continuous": cont.get("kv_peak_blocks"),
        "kv_peak_blocks_static": stat.get("kv_peak_blocks"),
        "peak_memory_mb_continuous": cont.get("peak_memory_mb"),
        "peak_memory_mb_static": stat.get("peak_memory_mb"),
    }


def compare_schedulers(
    model: Any,
    kv: PagedKVCache,
    engine_cfg: EngineConfig,
    sampling: SamplingConfig,
    requests: List[WorkloadRequest],
    seed: int = 20260920,
    measure: bool = True,
) -> Dict[str, Any]:
    """Finest-grained comparison helper used by tests / notebooks."""
    logger.info("compare_schedulers: %s requests", len(requests))
    # build_workload already consumed the RNG in run_serving_experiment; here we
    # just run both policies on the given requests.
    scheduler = PagedScheduler(model, kv, engine_cfg, sampling)
    cont = _fresh_run(scheduler, kv, requests, "continuous", seed, measure)
    stat = _fresh_run(scheduler, kv, requests, "static", seed, measure)
    return {
        "mode": "compare",
        "continuous": cont.to_dict(),
        "static": stat.to_dict(),
        "comparison": _compare_dicts(cont, stat),
    }
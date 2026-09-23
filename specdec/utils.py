"""Shared utilities: logging, seeding, JSON sanitisation and plotting setup."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logger(name: str = "specdec", level: int = logging.INFO) -> logging.Logger:
    """Create (or fetch) a stdout logger with a timestamped, levelled format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_seed(seed: int = 20260920) -> None:
    """Seed every RNG that can influence a token draw, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with null so the report stays valid JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def save_json(payload: Dict[str, Any], path: str) -> str:
    """Write a JSON report (NaN/Inf sanitised) and return the absolute path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(payload), handle, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


def timestamp() -> str:
    """Compact local timestamp used in artifact file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Stopwatch:
    """Context manager measuring wall-clock latency in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def sync_device(device: Any) -> None:
    """Synchronise CUDA so that wall-clock timings are meaningful."""
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


def configure_chinese_font() -> Optional[str]:
    """Pick an available CJK font so Chinese labels render instead of tofu boxes."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"):
        if candidate in installed:
            matplotlib.rcParams["font.sans-serif"] = [candidate]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return candidate
    return None


def human_bytes(num_bytes: float) -> str:
    """Render a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return "{:.1f}{}".format(num_bytes, unit)
        num_bytes /= 1024.0
    return "{:.1f}TB".format(num_bytes)


def mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    """Return mean/std of a list, tolerating empty input."""
    if not values:
        return {"mean": None, "std": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
    }
"""Re-export the shared lab utilities from ``specdec.utils``.

The Phase-2 engine intentionally reuses Phase-1's infrastructure (seed, JSON,
logging, timing, plotting font pickup) instead of forking it. Scripts are run
from the repository root, which puts ``specdec`` on ``sys.path``; this thin
module gives ``engine``-internal ``from .utils import ...`` a single source of
truth without duplicating any logic.
"""

from __future__ import annotations

from specdec.utils import (  # noqa: F401  (re-exported as the engine's utilities)
    Stopwatch,
    configure_chinese_font,
    human_bytes,
    mean_std,
    sanitize_for_json,
    save_json,
    set_seed,
    setup_logger,
    sync_device,
    timestamp,
)

__all__ = [
    "Stopwatch",
    "configure_chinese_font",
    "human_bytes",
    "mean_std",
    "sanitize_for_json",
    "save_json",
    "set_seed",
    "setup_logger",
    "sync_device",
    "timestamp",
]
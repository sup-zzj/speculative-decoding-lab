"""Publication-style figures for the speculative vs non-speculative experiment.

Reads every ``spec_experiment_*.json`` under ``--dir`` (the reports written by
``run_spec_experiment.py``) and aggregates them by ``gamma``. Because a gamma scan
emits one JSON per gamma, a file covering several gammas -- or several seeds of the
same gamma -- is combined into a single ``(gamma, summary)`` table. Three figures
are produced (PNG + PDF, Agg backend, serif/STIX, inward ticks, open top/right
spines, colour-blind-safe palette):

1. ``spec_throughput_*`` -- grouped bars of mean throughput (speculative vs
   baseline) per gamma with +/- std error bars.
2. ``spec_ratio_vs_gamma_*`` -- the spec/baseline throughput ratio as a function
   of gamma (needs >= 2 distinct gammas).
3. ``spec_adaptation_*`` -- twin-axis line of mean tokens-per-round and mean
   acceptance ratio vs gamma (needs >= 1 gamma where spec ran).

Styling is reused from ``scripts/make_plots.py``.

Examples
--------
::

    python scripts/make_spec_plots.py --dir results/cpu
    python scripts/make_spec_plots.py --dir results/gpu --outdir results/gpu/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from make_plots import _PALETTE, _apply_paper_style, _save, _title  # noqa: E402
from specdec.utils import configure_chinese_font, setup_logger  # noqa: E402

logger = setup_logger("make_spec_plots")


def _load_gamma_table(directory: str) -> List[Dict[str, Any]]:
    """Combine all spec_experiment_*.json into one row per gamma."""
    matches = sorted(glob.glob(os.path.join(directory, "spec_experiment_*.json")))
    if not matches:
        raise SystemExit(
            "no spec_experiment_*.json found under {}".format(directory)
        )
    rows: Dict[int, Dict[str, Any]] = {}
    for path in matches:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        gamma = int(payload["gamma"])
        if gamma not in rows:
            rows[gamma] = {
                "gamma": gamma,
                "mode": payload.get("mode"),
                "device": payload.get("device"),
                "sampling": payload.get("sampling"),
                "seed": payload.get("seed"),
                "summary": payload.get("summary", {}),
                "sources": [os.path.abspath(path)],
            }
        else:
            rows[gamma]["sources"].append(os.path.abspath(path))
            logger.info(
                "multiple seeds for gamma=%d (keeping first summary)", gamma
            )
    return [rows[k] for k in sorted(rows)]


def _spec_mean(row: Dict[str, Any], metric: str) -> Optional[float]:
    return (row["summary"].get("speculative") or {}).get(metric, {}).get("mean")


def _spec_std(row: Dict[str, Any], metric: str) -> float:
    return float((row["summary"].get("speculative") or {}).get(metric, {}).get("std") or 0.0)


def _base_mean(row: Dict[str, Any], metric: str) -> Optional[float]:
    return (row["summary"].get("baseline") or {}).get(metric, {}).get("mean")


def _base_std(row: Dict[str, Any], metric: str) -> float:
    return float((row["summary"].get("baseline") or {}).get(metric, {}).get("std") or 0.0)


def plot_spec_vs_base_throughput(
    rows: List[Dict[str, Any]], outdir: str, device: str
) -> List[str]:
    """Grouped mean throughput bars: speculative vs baseline, with error bars."""
    import matplotlib.pyplot as plt

    gammas = [r["gamma"] for r in rows]
    width = 0.38
    positions = [float(g) for g in gammas]
    spec_mean = [c or 0.0 for c in [_spec_mean(r, "throughput_tokens_per_s") for r in rows]]
    spec_err = [_spec_std(r, "throughput_tokens_per_s") for r in rows]
    base_mean = [c or 0.0 for c in [_base_mean(r, "throughput_tokens_per_s") for r in rows]]
    base_err = [_base_std(r, "throughput_tokens_per_s") for r in rows]

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    axes.bar(
        [p - width / 2 for p in positions], base_mean, width=width,
        yerr=base_err, capsize=3, color=_PALETTE["gray"], alpha=0.85,
        edgecolor=_PALETTE["ink"], linewidth=0.6, label="baseline (plain continuous)",
    )
    axes.bar(
        [p + width / 2 for p in positions], spec_mean, width=width,
        yerr=spec_err, capsize=3, color=_PALETTE["blue"], alpha=0.85,
        edgecolor=_PALETTE["ink"], linewidth=0.6, label="speculative (draft+verify)",
    )
    for p, s in zip(positions, spec_mean):
        axes.annotate("{:.2f}".format(s), (p + width / 2, s),
                      xytext=(0, 3), textcoords="offset points", ha="center",
                      fontsize=8, color=_PALETTE["ink"])
    axes.set_xticks(positions)
    axes.set_xticklabels(["$\\gamma={}$".format(g) for g in gammas])
    axes.set_xlim(min(positions) - 0.6, max(positions) + 0.6)
    axes.set_xlabel("draft length $\\gamma$")
    axes.set_ylabel("throughput (tokens/s)")
    axes.set_title(_title(device, "Speculative vs baseline serving throughput"))
    axes.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    axes.legend(loc="upper left", fontsize=8.5)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "spec_throughput")


def plot_spec_ratio_vs_gamma(rows: List[Dict[str, Any]], outdir: str, device: str) -> List[str]:
    """Line of spec/baseline throughput ratio vs gamma (needs >= 2 points)."""
    import matplotlib.pyplot as plt

    gammas = [r["gamma"] for r in rows]
    ratios = [r["summary"].get("spec_vs_baseline_throughput_ratio") for r in rows]
    if len(gammas) < 2 or not any(v is not None for v in ratios):
        return []

    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.axhline(1.0, color=_PALETTE["accent"], linewidth=1.1, linestyle=":")
    axes.plot(gammas, [v or 0.0 for v in ratios], marker="o", color=_PALETTE["blue"],
              linestyle="-", linewidth=1.9, markersize=5.5)
    for g, v in zip(gammas, ratios):
        if v is not None:
            axes.annotate("{:.2f}".format(v), (g, v), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=8.5,
                          color=_PALETTE["ink"])
    axes.set_xlabel("draft length $\\gamma$")
    axes.set_ylabel("speculative / baseline throughput ratio")
    axes.set_title(_title(device, "Speculative speed-up by draft length"))
    axes.set_xticks(gammas)
    axes.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "spec_ratio_vs_gamma")


def plot_spec_adaptation(rows: List[Dict[str, Any]], outdir: str, device: str) -> List[str]:
    """Twin-axis: mean tokens per round and acceptance ratio vs gamma."""
    import matplotlib.pyplot as plt

    gammas = [r["gamma"] for r in rows]
    tokens = [_spec_mean(r, "mean_tokens_per_round") for r in rows]
    acceptance = [_spec_mean(r, "mean_acceptance") for r in rows]
    if not any(v is not None for v in tokens):
        return []

    figure, left = plt.subplots(figsize=(7.0, 4.1))
    right = left.twinx()
    left.plot(gammas, [v or 0.0 for v in tokens], marker="o", color=_PALETTE["blue"],
              label="tokens / round")
    left.set_ylabel("mean tokens committed per round")
    left.set_xlabel("draft length $\\gamma$")
    left.set_xticks(gammas)
    left.set_title(_title(device, "Speculative adaptation vs draft length"))
    left.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    if any(v is not None for v in acceptance):
        right.plot(gammas, [v or 0.0 for v in acceptance], marker="s", color=_PALETTE["orange"],
                   linestyle="--", markersize=4.5, label="acceptance")
        right.set_ylabel("mean acceptance rate")
        right.set_ylim(0, 1.08)
    handles_l, labels_l = left.get_legend_handles_labels()
    handles_r, labels_r = right.get_legend_handles_labels()
    left.legend(handles_l + handles_r, labels_l + labels_r, loc="upper left", fontsize=8.5)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "spec_adaptation")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description="Speculative-experiment figures from spec_experiment_*.json"
    )
    parser.add_argument(
        "--dir",
        default="results/cpu",
        help="directory holding spec_experiment_*.json; figures go into <dir>/figures",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="figure output directory (default: <dir>/figures)",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dir):
        logger.error("results dir not found: %s", args.dir)
        return 2
    outdir = args.outdir or os.path.join(args.dir, "figures")

    rows = _load_gamma_table(args.dir)
    device_label = os.path.basename(os.path.normpath(args.dir)).upper()
    if device_label not in ("CPU", "GPU"):
        device_label = ""

    configure_chinese_font()
    _apply_paper_style()
    os.makedirs(outdir, exist_ok=True)
    produced: List[str] = []
    produced += plot_spec_vs_base_throughput(rows, outdir, device_label)
    produced += plot_spec_ratio_vs_gamma(rows, outdir, device_label)
    produced += plot_spec_adaptation(rows, outdir, device_label)
    for path in produced:
        logger.info("figure: %s", path)
    logger.info("combined %d gamma row(s)", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
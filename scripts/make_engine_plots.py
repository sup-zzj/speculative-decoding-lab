"""Publication-style figures for the paged serving experiment.

Reads one ``run_serving_experiment`` report (the JSON written by
``run_engine_experiment.py``) and turns it into two figures:

* ``engine_metrics`` -- a grouped bar chart of the headline metrics under the
  two policies: per-step throughput, slot utilisation, and peak KV blocks. The
  three metrics live on very different scales, so each is normalised to the
  static-batch baseline inside the figure (the annotation prints the real value).
* ``engine_batch_hist`` -- the instantaneous batch size per scheduler step for
  continuous (varying, follows arrivals/completions) and static (flat at the
  request count, idle once requests finish). This is the visual that makes
  continuous batching's capacity saving obvious.

Styling reuses ``scripts/make_plots.py`` (Agg, serif/STIX, inward ticks, open
top/right spines, colour-blind-safe palette, PDF with embedded fonts) and is
also callable from the command line against an existing report::

    python scripts/make_engine_plots.py --report results/gpu/engine_serving_real_XXX.json --outdir results/gpu/figures
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from make_plots import _PALETTE, _apply_paper_style, _save, _title  # noqa: E402
from specdec.utils import configure_chinese_font, setup_logger  # noqa: E402

logger = setup_logger("make_engine_plots")


def plot_metrics_barchart(report: Dict[str, Any], outdir: str, device: str) -> List[str]:
    """Grouped bars: throughput, slot utilisation, peak KV blocks (both policies).

    All three subplots are normalised by the static baseline so they share the
    ">1.0 means continuous is better" reading; the annotation shows the absolute
    measured value for reference.
    """
    import matplotlib.pyplot as plt

    cont = report["continuous"]
    stat = report["static"]
    comp = report.get("comparison", {})

    # Normalise relative to static; guard against missing/zero baselines.
    def _rel(value: Any, baseline: Any) -> float:
        if value is None or baseline in (None, 0):
            return float("nan")
        return float(value) / float(baseline)

    labels = ["throughput\n(tokens/s)", "slot\nutilisation", "peak\nKV blocks"]
    values = [
        _rel(cont.get("throughput_tokens_per_s"), stat.get("throughput_tokens_per_s")),
        _rel(cont.get("slot_utilization"), stat.get("slot_utilization")),
        _rel(cont.get("kv_peak_blocks"), stat.get("kv_peak_blocks")),
    ]
    absolutes = [cont.get(k) for k in
                 ("throughput_tokens_per_s", "slot_utilization", "kv_peak_blocks")]

    figure, axes = plt.subplots(figsize=(7.0, 4.1))
    positions = [0, 1, 2]
    colors = [_PALETTE["blue"] if v >= 1.0 else _PALETTE["orange"] for v in values]
    bars = axes.bar(positions, values, width=0.6, color=colors, alpha=0.85,
                    edgecolor=_PALETTE["ink"], linewidth=0.6)
    axes.axhline(1.0, color=_PALETTE["accent"], linewidth=1.3, linestyle=":")
    for bar, value, absv in zip(bars, values, absolutes):
        if value != float("nan"):
            axes.annotate("{:.2f}".format(value), (bar.get_x() + bar.get_width() / 2, value),
                          xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=8.5, color=_PALETTE["ink"])
        if absv is not None:
            axes.annotate("abs {:.2g}".format(absv), (bar.get_x() + bar.get_width() / 2, 0.02),
                          ha="center", fontsize=7.5, color=_PALETTE["ink"])
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylabel("continuous / static (ratio)")
    axes.set_ylim(0, (max(values) if values else 1.0) * 1.35 + 0.1)
    axes.set_title(_title(device, "Continuous batching vs static batching (normalised)"))
    axes.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    note = comp.get("throughput_explanation") or ""
    if note:
        axes.text(0.01, 1.0, note, transform=axes.transAxes, va="bottom",
                  fontsize=7.5, color=_PALETTE["accent"])
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "engine_metrics")


def plot_batch_histogram(report: Dict[str, Any], outdir: str, device: str) -> List[str]:
    """Instantaneous batch size per step: continuous varies, static stays flat."""
    import matplotlib.pyplot as plt

    cont = report.get("continuous", {})
    stat = report.get("static", {})
    cont_hist = cont.get("batch_hist") or []
    stat_hist = stat.get("batch_hist") or []

    steps_c = list(range(len(cont_hist)))
    steps_s = list(range(len(stat_hist)))

    figure, axes = plt.subplots(figsize=(7.0, 4.1))
    if steps_s:
        axes.step(steps_s, stat_hist, where="post", color=_PALETTE["gray"],
                  linestyle="--", linewidth=1.8, label="static (fixed batch)")
    if steps_c:
        axes.step(steps_c, cont_hist, where="post", color=_PALETTE["blue"],
                  linestyle="-", linewidth=2.0, label="continuous (dynamic)")
    axes.set_xlabel("scheduler step")
    axes.set_ylabel("active requests in batch")
    axes.set_title(_title(device, "Batch size over the serving run"))
    axes.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    axes.legend(loc="upper right", fontsize=9)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "engine_batch_hist")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Engine serving figures from a JSON report")
    parser.add_argument("--report", required=True, help="engine_serving_*.json report")
    parser.add_argument("--outdir", default="results/figures", help="figure output directory")
    args = parser.parse_args(argv)

    if not os.path.exists(args.report):
        logger.error("report not found: %s", args.report)
        return 2
    import json

    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    configure_chinese_font()
    _apply_paper_style()
    os.makedirs(args.outdir, exist_ok=True)
    produced = plot_metrics_barchart(report, args.outdir, "")
    produced += plot_batch_histogram(report, args.outdir, "")
    for path in produced:
        logger.info("figure: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
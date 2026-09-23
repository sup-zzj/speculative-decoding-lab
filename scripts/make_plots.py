"""Turn the JSON reports into publication-style figures (PNG + PDF).

The styling follows the conventions used by ML-systems papers: serif fonts
with STIX maths, ticks pointing inward, open top/right spines, a restrained
colour-blind-safe palette, and figures sized for single/double-column
placement.  Every number shown in the figures is also available in the JSON
reports, so the plots never carry information the data does not.

Usage::

    python scripts/make_plots.py --results-dir results/cpu
    python scripts/make_plots.py \\
        --sweep results/gamma_sweep_XXXX.json \\
        --correctness results/correctness_toy.json \\
        --outdir results/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specdec.utils import configure_chinese_font, setup_logger  # noqa: E402

logger = setup_logger("make_plots")

# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------

_PALETTE = {
    "blue": "#00509E",      # primary series (measured / speculative)
    "orange": "#E8922D",    # secondary series (analytic model / theory)
    "gray": "#8C8C8C",      # reference samplers / neutral data
    "ink": "#1A1A1A",       # axis + text
    "grid": "#C8C8C8",      # light gridlines
    "accent": "#57606A",    # annotations
}


def _apply_paper_style() -> None:
    """Global rcParams shared by every figure."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import rcParams

    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.size": 10.0,
            "mathtext.fontset": "stix",
            "mathtext.rm": "STIXGeneral",
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "axes.linewidth": 0.9,
            "axes.edgecolor": _PALETTE["ink"],
            "axes.labelcolor": _PALETTE["ink"],
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "xtick.color": _PALETTE["ink"],
            "ytick.color": _PALETTE["ink"],
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.0,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed TrueType, not bitmaps (papers require this)
            "ps.fonttype": 42,
        }
    )


def _title(device_label: str, text: str) -> str:
    if not device_label:
        return text
    return "{} ({})".format(text, device_label)


def _error_bar(sweep: Dict[str, Any], row: Dict[str, Any], baseline_ms: float) -> float:
    """Propagate the latency mean/std of both decoders into speed-up error bars.

    ``speedup = t_base / t_spec``, so the relative error of the ratio is the
    quadratic sum of the two relative errors.
    """
    spec = row["latency_ms"]
    base = sweep.get("baseline_latency_ms") or {}
    rel_base = (float(base["std"]) / float(base["mean"])) if base.get("mean") else 0.0
    rel_spec = (float(spec["std"]) / float(spec["mean"])) if spec.get("mean") else 0.0
    relative = math.sqrt(rel_base ** 2 + rel_spec ** 2)
    return baseline_ms * relative if baseline_ms > 0 else 0.0


def _latest(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save(figure, outdir: str, name: str) -> List[str]:
    import matplotlib.pyplot as plt

    paths = []
    for extension in ("png", "pdf"):
        target = os.path.join(outdir, "{}.{}".format(name, extension))
        figure.savefig(target, dpi=200)
        paths.append(os.path.abspath(target))
    plt.close(figure)
    return paths


def plot_speedup_vs_gamma(sweep: Dict[str, Any], outdir: str, device: str) -> List[str]:
    """Figure 1 -- speed-up vs draft length, measurement against the analytic model."""
    import matplotlib.pyplot as plt

    rows = sweep["rows"]
    analysis = sweep.get("analysis", {})
    baseline_ms = float(sweep.get("baseline_latency_ms", {}).get("mean") or 0.0)
    gammas = [row["gamma"] for row in rows]
    measured = [row["measured_speedup"] or 0.0 for row in rows]
    predicted = [row["predicted_speedup"] for row in rows]
    errors = [_error_bar(sweep, row, baseline_ms) for row in rows]

    figure, axes = plt.subplots(figsize=(7.0, 4.3))
    axes.errorbar(
        gammas,
        measured,
        yerr=errors,
        capsize=3,
        marker="o",
        color=_PALETTE["blue"],
        linestyle="-",
        label="measured",
    )
    axes.plot(
        gammas,
        predicted,
        marker="s",
        color=_PALETTE["orange"],
        linestyle="--",
        markersize=4.5,
        label="analytic model, $S(\\gamma)=(1-\\alpha^{\\gamma+1})/((1-\\alpha)(1+\\gamma c))$",
    )
    best = analysis.get("best_gamma_measured")
    if best is not None and best in gammas:
        index = gammas.index(best)
        axes.annotate(
            "$\\gamma^* = {}$  ({:.2f}$\\times$)".format(best, measured[index]),
            xy=(best, measured[index]),
            xytext=(best + 0.35, measured[index] - 0.14 * max(measured)),
            arrowprops={"arrowstyle": "->", "color": _PALETTE["accent"], "lw": 1.0},
            fontsize=9.5,
            color=_PALETTE["ink"],
        )
    axes.axhline(1.0, color=_PALETTE["accent"], linewidth=1.0, linestyle=":")
    axes.set_xlabel("draft length $\\gamma$")
    axes.set_ylabel("speed-up vs. autoregressive decoding")
    axes.set_title(_title(device, "Speed-up of speculative decoding"))
    axes.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    axes.legend(loc="lower left", fontsize=8.5)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "fig1_speedup_vs_gamma")


def plot_acceptance_and_tokens(sweep: Dict[str, Any], outdir: str, device: str) -> List[str]:
    """Figure 2 -- acceptance rate and tokens/round on shared gamma axis."""
    import matplotlib.pyplot as plt

    rows = sweep["rows"]
    gammas = [row["gamma"] for row in rows]
    acceptance = [row["acceptance_rate"]["mean"] or 0.0 for row in rows]
    measured = [row["expected_tokens_per_round_measured"]["mean"] or 0.0 for row in rows]
    theory = [row["expected_tokens_per_round_theory"] for row in rows]

    figure, left = plt.subplots(figsize=(7.0, 4.3))
    right = left.twinx()

    left.bar(
        [g - 0.18 for g in gammas],
        acceptance,
        width=0.36,
        color=_PALETTE["gray"],
        alpha=0.7,
        label="acceptance rate $\\alpha$",
        edgecolor=_PALETTE["ink"],
        linewidth=0.5,
    )
    left.set_ylabel("acceptance rate $\\alpha$")
    left.set_ylim(0, 1.08)
    left.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    right.plot(gammas, measured, marker="o", color=_PALETTE["blue"], label="measured")
    right.plot(
        gammas,
        theory,
        marker="s",
        color=_PALETTE["orange"],
        linestyle="--",
        markersize=4.5,
        label="$\\mathbb{E}[\\tau] = \\frac{1-\\alpha^{\\gamma+1}}{1-\\alpha}$",
    )
    right.set_ylabel("tokens committed per round")
    right.set_ylim(0, max(max(theory), max(measured)) * 1.15)

    left.set_xlabel("draft length $\\gamma$")
    left.set_xticks(gammas)
    left.set_title(_title(device, "Acceptance rate and committed tokens per round"))
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        fontsize=8.5,
        ncol=2,
    )
    left.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "fig2_acceptance_tokens")


def plot_tokens_per_round_histogram(
    sweep: Dict[str, Any], outdir: str, device: str
) -> List[str]:
    """Figure 3 -- empirical distribution of committed tokens at the best gamma."""
    import matplotlib.pyplot as plt

    best_gamma = sweep.get("analysis", {}).get("best_gamma_measured")
    row = next(
        (item for item in sweep["rows"] if item["gamma"] == best_gamma),
        sweep["rows"][0],
    )
    counts = row["tokens_per_round_hist"]
    if not counts:
        return []

    observed: Dict[int, int] = {}
    for value in counts:
        observed[value] = observed.get(value, 0) + 1
    keys = sorted(observed)
    values = [observed[key] / len(counts) for key in keys]
    mean_tokens = sum(counts) / len(counts)

    figure, axes = plt.subplots(figsize=(7.0, 3.9))
    axes.bar(
        keys,
        values,
        width=0.75,
        color=_PALETTE["blue"],
        alpha=0.8,
        edgecolor=_PALETTE["ink"],
        linewidth=0.5,
    )
    axes.axvline(
        mean_tokens,
        color=_PALETTE["orange"],
        linestyle="--",
        linewidth=1.6,
        label="mean $= {:.2f}$".format(mean_tokens),
    )
    axes.set_xlabel("tokens committed in one round")
    axes.set_ylabel("empirical frequency")
    axes.set_title(
        _title(
            device,
            "Tokens committed per round at $\\gamma = {}$".format(row["gamma"]),
        )
    )
    axes.set_xticks(keys)
    axes.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    axes.legend(loc="upper right", fontsize=9)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "fig3_tokens_per_round")


def plot_first_token_match(correctness: Dict[str, Any], outdir: str, device: str) -> List[str]:
    """Figure 4 -- exact vs empirical first-token distribution of the target."""
    import matplotlib.pyplot as plt

    report = next(
        (
            item
            for item in correctness.get("reports", [])
            if item["name"].startswith("first_token_test")
        ),
        None,
    )
    if report is None:
        return []
    metrics = report["metrics"]
    target = metrics.get("target_probs_top") or []
    speculative = metrics.get("speculative_probs_top") or []
    if not target:
        return []

    positions = list(range(len(target)))
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.plot(positions, target, marker="o", color=_PALETTE["blue"], label="target $p$ (exact)")
    axes.plot(
        positions,
        speculative,
        marker="s",
        color=_PALETTE["orange"],
        linestyle="--",
        markersize=4.5,
        label="speculative (empirical, $N={}$)".format(metrics.get("monte_carlo_samples")),
    )
    tv = metrics.get("tv_speculative_vs_target")
    kl = metrics.get("kl_speculative_vs_target")
    chi = (metrics.get("chi_square") or {}).get("p_value")
    axes.set_xlabel("token rank under the target distribution")
    axes.set_ylabel("probability")
    subtitle = "TV = {:.4f},  KL = {:.5f}".format(tv, kl)
    if chi is not None:
        subtitle += ",  $\\chi^2$ p = {:.3f}".format(chi)
    axes.set_title(_title(device, "First-token match of the speculative decoder\n" + subtitle))
    axes.set_xticks(positions)
    axes.grid(axis="y", color=_PALETTE["grid"], alpha=0.5, linewidth=0.7)
    axes.legend(loc="upper right", fontsize=9)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "fig4_first_token_match")


def plot_joint_scatter(correctness: Dict[str, Any], outdir: str, device: str) -> List[str]:
    """Figure 5 -- log-log agreement of the toy joint distribution with the exact one."""
    import matplotlib.pyplot as plt

    report = next(
        (
            item
            for item in correctness.get("reports", [])
            if item["name"] == "sequence_distribution_test"
        ),
        None,
    )
    if report is None:
        return []
    metrics = report["metrics"]
    exact = metrics.get("exact_distribution")
    speculative = metrics.get("speculative_distribution")
    baseline = metrics.get("baseline_distribution")
    if not exact or not speculative:
        return []

    floor = 1e-6
    figure, axes = plt.subplots(figsize=(5.0, 4.6))
    axes.scatter(
        [max(v, floor) for v in exact],
        [max(v, floor) for v in baseline],
        s=20,
        alpha=0.6,
        color=_PALETTE["gray"],
        label="autoregressive sampler",
        linewidths=0,
    )
    axes.scatter(
        [max(v, floor) for v in exact],
        [max(v, floor) for v in speculative],
        s=20,
        alpha=0.6,
        color=_PALETTE["blue"],
        label="speculative decoder",
        linewidths=0,
    )
    limits = [floor, max(max(exact), max(speculative)) * 1.6]
    axes.plot(
        limits,
        limits,
        color=_PALETTE["orange"],
        linewidth=1.2,
        linestyle="--",
        label="y = x",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_xlabel("exact joint probability $p(x_1, x_2)$")
    axes.set_ylabel("empirical frequency")
    axes.set_title(
        _title(
            device,
            "Joint distribution over {} outcomes".format(metrics.get("num_outcomes")),
        )
    )
    axes.grid(color=_PALETTE["grid"], alpha=0.4, linewidth=0.6, which="both")
    axes.legend(loc="upper left", fontsize=8.5)
    figure.tight_layout(pad=0.5)
    return _save(figure, outdir, "fig5_joint_distribution")


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build figures from JSON reports")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="directory holding the run JSONs; figures go into <results-dir>/figures",
    )
    parser.add_argument("--sweep", default=None, help="gamma sweep JSON (overrides results-dir)")
    parser.add_argument("--benchmark", default=None, help="benchmark JSON (optional)")
    parser.add_argument("--correctness", default=None, help="correctness JSON (overrides results-dir)")
    parser.add_argument(
        "--outdir",
        default=None,
        help="figure output directory (default: <results-dir>/figures)",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    device_label = os.path.basename(os.path.normpath(results_dir)).upper()
    if device_label not in ("CPU", "GPU"):
        device_label = ""
    outdir = args.outdir or os.path.join(results_dir, "figures")

    configure_chinese_font()
    _apply_paper_style()
    os.makedirs(outdir, exist_ok=True)

    sweep_path = args.sweep or _latest(os.path.join(results_dir, "gamma_sweep_*.json"))
    correctness_path = args.correctness or _latest(
        os.path.join(results_dir, "correctness_*.json")
    )
    sweep = _load(sweep_path)
    correctness = _load(correctness_path)

    produced: List[str] = []
    if sweep:
        logger.info("sweep source: %s", os.path.abspath(sweep_path))
        produced += plot_speedup_vs_gamma(sweep, outdir, device_label)
        produced += plot_acceptance_and_tokens(sweep, outdir, device_label)
        produced += plot_tokens_per_round_histogram(sweep, outdir, device_label)
    if correctness:
        logger.info("correctness source: %s", os.path.abspath(correctness_path))
        produced += plot_first_token_match(correctness, outdir, device_label)
        produced += plot_joint_scatter(correctness, outdir, device_label)
    if not produced:
        logger.error("nothing to plot: run the benchmark or correctness script first")
        return 2

    for path in produced:
        logger.info("figure: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

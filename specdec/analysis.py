"""Analytic model of speculative decoding, used to explain measured numbers.

The classic result (Leviathan et al., 2023) is that with a per-token acceptance
probability ``alpha`` and a draft block length ``gamma``, the expected number of
tokens committed per round is

    E[tokens] = (1 - alpha^(gamma + 1)) / (1 - alpha)

while the round costs ``1 + gamma * c`` target-equivalent steps, where ``c`` is
the draft/target per-step cost ratio. The speed-up therefore saturates: the
numerator converges to ``1 / (1 - alpha)`` as ``gamma`` grows, but the
denominator grows without bound. The optimum sits where the marginal token gain
equals the marginal cost -- that crossover is what this module reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .speculative import expected_tokens_per_round, predicted_speedup


def optimal_gamma(alpha: float, cost_ratio: float, max_gamma: int = 32) -> Dict[str, Any]:
    """Exhaustively locate the ``gamma`` maximising the analytic speed-up."""
    gammas = list(range(1, max_gamma + 1))
    speedups = [predicted_speedup(alpha, g, cost_ratio) for g in gammas]
    best = int(np.argmax(speedups))
    return {
        "gamma_star": gammas[best],
        "speedup_star": float(speedups[best]),
        "speedup_at_gamma_1": float(speedups[0]),
        "plateau_speedup": float(1.0 / (1.0 - min(alpha, 1.0 - 1e-9)) / (1.0 + 1.0 * cost_ratio)),
        "curve": {"gamma": gammas, "predicted_speedup": [float(s) for s in speedups]},
    }


def marginal_table(
    alpha: float, cost_ratio: float, gammas: Sequence[int]
) -> List[Dict[str, Any]]:
    """Marginal benefit of increasing ``gamma`` by one.

    ``marginal_speedup_gain`` is exactly the quantity that must decay to zero at
    the optimum; ``tokens_per_extra_target_step`` expresses the same idea in the
    unit a practitioner reasons about.
    """
    rows: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, float]] = None
    for gamma in gammas:
        tokens = expected_tokens_per_round(alpha, gamma)
        cost = 1.0 + gamma * cost_ratio
        speedup = tokens / cost
        row: Dict[str, Any] = {
            "gamma": gamma,
            "expected_tokens_per_round": tokens,
            "relative_cost": cost,
            "predicted_speedup": speedup,
        }
        if previous is not None:
            delta_tokens = tokens - previous["tokens"]
            delta_cost = cost - previous["cost"]
            row["marginal_tokens"] = delta_tokens
            row["marginal_speedup_gain"] = speedup - previous["speedup"]
            row["tokens_per_extra_target_step"] = (
                delta_tokens / delta_cost if delta_cost > 0 else None
            )
        else:
            row["marginal_tokens"] = None
            row["marginal_speedup_gain"] = None
            row["tokens_per_extra_target_step"] = None
        rows.append(row)
        previous = {"tokens": tokens, "cost": cost, "speedup": speedup}
    return rows


def analyse_sweep(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the measured sweep against the analytic predictions."""
    rows = payload.get("rows", [])
    if not rows:
        return {"error": "no rows in sweep payload"}

    gammas, measured, predicted, alphas = [], [], [], []
    for row in rows:
        if row.get("measured_speedup") is None:
            continue
        gammas.append(int(row["gamma"]))
        measured.append(float(row["measured_speedup"]))
        predicted.append(float(row["predicted_speedup"]))
        alphas.append(float(row["acceptance_rate"]["mean"] or 0.0))

    if not gammas:
        return {"error": "no successful measurements"}

    measured_array = np.asarray(measured)
    predicted_array = np.asarray(predicted)
    best_measured = gammas[int(np.argmax(measured_array))]
    best_predicted = gammas[int(np.argmax(predicted_array))]
    #: Pearson correlation quantifies whether the analytic model ranks gammas correctly
    correlation = (
        float(np.corrcoef(measured_array, predicted_array)[0, 1])
        if len(gammas) > 1
        else None
    )
    return {
        "gammas": gammas,
        "measured_speedup": measured,
        "predicted_speedup": predicted,
        "mean_acceptance_rate": float(np.mean(alphas)),
        "best_gamma_measured": best_measured,
        "best_gamma_predicted": best_predicted,
        "best_speedup_measured": float(measured_array.max()),
        "best_speedup_predicted": float(predicted_array.max()),
        "pearson_measured_vs_predicted": correlation,
        "relative_error_at_best": (
            abs(
                float(measured_array.max())
                - float(predicted_array[int(np.argmax(measured_array))])
            )
            / max(float(measured_array.max()), 1e-9)
        ),
        "cost_ratio": payload.get("cost_ratio", {}).get("cost_ratio"),
        "marginal": marginal_table(
            float(np.mean(alphas)), float(payload.get("cost_ratio", {}).get("cost_ratio") or 0.0), list(gammas)
        ),
        "analytic_optimum": optimal_gamma(
            float(np.mean(alphas)), float(payload.get("cost_ratio", {}).get("cost_ratio") or 0.0)
        ),
    }
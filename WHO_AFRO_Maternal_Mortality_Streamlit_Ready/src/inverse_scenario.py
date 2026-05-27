"""Reverse (target MMR) scenario solver — required indicator values for a goal MMR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import FEATURE_COLUMNS, FEATURE_TO_RISK_MULTIPLIER
from src.predict import _predict_log_mmr_constrained, _row_to_risk_frame, _scale_constrained, predict_mmr

# Levers most actionable in policy dialogue (used when ``priority_only=True``).
PRIORITY_INDICATORS: tuple[str, ...] = (
    "Skilled Birth Attendance",
    "ANC4 Coverage",
    "Facility Delivery Rate",
    "Contraception (modern)",
    "Gov Health Expenditure (% GDP)",
    "Female Literacy",
    "Fertility Rate",
    "Poverty ($2.15/day, %)",
    "Malaria Incidence",
    "Emergency Obstetric Care Coverage",
)


@dataclass
class TargetScenarioResult:
    target_mmr: float
    observed_mmr: float
    achieved_mmr: float
    feasible: bool
    message: str
    solution: dict[str, float]
    baseline: dict[str, float]
    adjustable: list[str]

    def changes_table(self) -> list[dict]:
        rows: list[dict] = []
        for col in self.adjustable:
            b = float(self.baseline[col])
            s = float(self.solution[col])
            if abs(s - b) < 1e-6:
                continue
            rows.append(
                {
                    "Indicator": col,
                    "Status quo": round(b, 2),
                    "Required for target": round(s, 2),
                    "Change": round(s - b, 2),
                }
            )
        rows.sort(key=lambda r: abs(r["Change"]), reverse=True)
        return rows


def _structural_eta(model: dict, features: dict[str, float]) -> float:
    row = _row_to_risk_frame(features)
    xs = _scale_constrained(model, row)
    return float(_predict_log_mmr_constrained(model, xs)[0])


def _eta_target(model: dict, observed_mmr: float, target_mmr: float, eta_baseline: float) -> float:
    gain = float(model.get("scenario_log_gain", 1.0))
    log_obs = float(np.log(max(observed_mmr, 1.0)))
    log_tgt = float(np.log(max(target_mmr, 1.0)))
    return float(eta_baseline + (log_tgt - log_obs) / max(gain, 1e-9))


def solve_target_mmr(
    model: dict,
    baseline: dict[str, float],
    observed_mmr: float,
    target_mmr: float,
    bounds: dict[str, tuple[float, float]],
    *,
    adjustable: list[str] | None = None,
    max_iterations: int = 80,
) -> TargetScenarioResult:
    """Find indicator values that approximate ``target_mmr`` from status-quo baseline.

    Uses the anchored log-linear model: distributes required change in structural
    score η across adjustable indicators with minimum weighted deviation, respecting bounds.
    """
    target_mmr = float(max(target_mmr, 1.0))
    observed_mmr = float(max(observed_mmr, 1.0))
    cols = [c for c in FEATURE_COLUMNS if c in baseline]
    if adjustable is None:
        adjustable = cols
    else:
        adjustable = [c for c in adjustable if c in baseline]

    if not adjustable:
        return TargetScenarioResult(
            target_mmr=target_mmr,
            observed_mmr=observed_mmr,
            achieved_mmr=observed_mmr,
            feasible=False,
            message="No adjustable indicators available.",
            solution=dict(baseline),
            baseline=dict(baseline),
            adjustable=[],
        )

    if abs(target_mmr - observed_mmr) < 0.5:
        return TargetScenarioResult(
            target_mmr=target_mmr,
            observed_mmr=observed_mmr,
            achieved_mmr=observed_mmr,
            feasible=True,
            message="Target equals observed MMR — status quo already meets the goal.",
            solution=dict(baseline),
            baseline=dict(baseline),
            adjustable=adjustable,
        )

    coef = np.array(model["coef"], dtype=float)
    scale = np.array(model["scaler_scale"], dtype=float)
    idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}

    x0 = np.array([float(baseline[c]) for c in FEATURE_COLUMNS], dtype=float)
    lbs = np.array([float(bounds.get(c, (0.0, 100.0))[0]) for c in FEATURE_COLUMNS], dtype=float)
    ubs = np.array([float(bounds.get(c, (0.0, 100.0))[1]) for c in FEATURE_COLUMNS], dtype=float)
    mult = np.array([float(FEATURE_TO_RISK_MULTIPLIER.get(c, 1.0)) for c in FEATURE_COLUMNS], dtype=float)
    # d(eta)/d(x_j) for linear model in raw units
    sensitivity = coef * mult / np.maximum(scale, 1e-9)
    weights = 1.0 / np.maximum(ubs - lbs, 1e-6)

    eta0 = _structural_eta(model, baseline)
    eta_goal = _eta_target(model, observed_mmr, target_mmr, eta0)

    x = x0.copy()
    adj_mask = np.array([c in adjustable for c in FEATURE_COLUMNS], dtype=bool)

    for _ in range(max_iterations):
        feats = {c: float(x[idx[c]]) for c in FEATURE_COLUMNS if c in idx}
        eta_cur = _structural_eta(model, feats)
        remaining = eta_goal - eta_cur
        if abs(remaining) < 1e-4:
            break

        free = adj_mask & (x > lbs + 1e-8) & (x < ubs - 1e-8)
        if target_mmr < observed_mmr and not free.any():
            free = adj_mask.copy()
        if not free.any():
            break

        c = sensitivity[free]
        w = weights[free]
        denom = float(np.sum((c / w) ** 2))
        if denom < 1e-12:
            break
        lam = remaining / denom
        x[free] = x[free] + lam * c / (w**2)
        x = np.clip(x, lbs, ubs)

    solution = {c: float(x[idx[c]]) for c in cols}
    achieved = float(
        predict_mmr(
            model,
            solution,
            anchor=baseline,
            anchor_mmr=observed_mmr,
        )
    )
    rel_err = abs(achieved - target_mmr) / max(target_mmr, 1.0)
    feasible = rel_err <= 0.08

    if target_mmr < observed_mmr and achieved > target_mmr * 1.08:
        message = (
            "Target may be too ambitious within indicator bounds. "
            "Showing best achievable combination — consider widening levers or a less aggressive target."
        )
        feasible = False
    elif target_mmr > observed_mmr and achieved < target_mmr * 0.92:
        message = (
            "Target MMR above current levels requires worsening indicators; "
            "best achievable values shown within bounds."
        )
        feasible = False
    elif feasible:
        message = "Solution found — indicator values approximate the target MMR."
    else:
        message = "Approximate solution — review achieved MMR vs target."

    return TargetScenarioResult(
        target_mmr=target_mmr,
        observed_mmr=observed_mmr,
        achieved_mmr=achieved,
        feasible=feasible,
        message=message,
        solution=solution,
        baseline={c: float(baseline[c]) for c in cols},
        adjustable=adjustable,
    )

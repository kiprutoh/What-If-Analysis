"""Prediction helpers for scenario analysis."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.constants import FEATURE_COLUMNS, FEATURE_TO_RISK_MULTIPLIER, MODEL_PATH, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]


def load_model():
    path = ROOT / MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python scripts/train_model.py"
        )
    return joblib.load(path)


def _row_to_risk_frame(features: dict[str, float]) -> pd.DataFrame:
    row = pd.DataFrame([{col: float(features[col]) for col in FEATURE_COLUMNS}])
    for col in FEATURE_COLUMNS:
        row[col] = row[col].astype(float) * float(FEATURE_TO_RISK_MULTIPLIER.get(col, 1.0))
    return row


def _predict_log_mmr_constrained(model_payload: dict, X: np.ndarray) -> np.ndarray:
    coef = np.array(model_payload["coef"], dtype=float)
    intercept = float(model_payload["intercept"])
    return X @ coef + intercept


def _scale_constrained(model_payload: dict, row: pd.DataFrame) -> np.ndarray:
    mean = np.array(model_payload["scaler_mean"], dtype=float)
    scale = np.array(model_payload["scaler_scale"], dtype=float)
    X = row[FEATURE_COLUMNS].to_numpy(dtype=float)
    return (X - mean) / scale


def predict_mmr(model, features: dict[str, float]) -> float:
    row = _row_to_risk_frame(features)
    if isinstance(model, dict) and model.get("kind") == "constrained_bootstrap_linear_logmmr":
        Xs = _scale_constrained(model, row)
        mu = float(_predict_log_mmr_constrained(model, Xs)[0])
        return float(np.exp(mu))
    return float(np.exp(model.predict(row)[0]))


def predict_mmr_interval(model, features: dict[str, float], z: float = 1.96) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using log-normal approximation from Bayesian predictive std.

    For BayesianRidge, sklearn supports predict(return_std=True) for the target space.
    Here target space is log(MMR), so interval is exp(mu ± z*std).
    """
    row = _row_to_risk_frame(features)
    if isinstance(model, dict) and model.get("kind") == "constrained_bootstrap_linear_logmmr":
        Xs = _scale_constrained(model, row)
        bcoef = np.array(model["bootstrap_coef"], dtype=float)
        bint = np.array(model["bootstrap_intercept"], dtype=float)
        mu_samples = (bcoef @ Xs[0]) + bint
        mmr_samples = np.exp(mu_samples)
        return float(mmr_samples.mean()), float(np.quantile(mmr_samples, 0.025)), float(np.quantile(mmr_samples, 0.975))

    mu, std = model.predict(row, return_std=True)
    mu = float(mu[0])
    std = float(std[0])
    mean = float(np.exp(mu + 0.5 * std * std))
    lo = float(np.exp(mu - z * std))
    hi = float(np.exp(mu + z * std))
    return mean, lo, hi


def scenario_frame(
    baseline: dict[str, float],
    overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    values = {**baseline, **(overrides or {})}
    return pd.DataFrame([{col: values[col] for col in FEATURE_COLUMNS}])

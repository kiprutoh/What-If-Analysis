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


def _scenario_log_gain(model_payload: dict, scenario_log_gain: float | None) -> float:
    if scenario_log_gain is not None:
        return float(scenario_log_gain)
    return float(model_payload.get("scenario_log_gain", 1.0))


def _log_mmr_samples(
    model_payload: dict,
    features: dict[str, float],
) -> np.ndarray:
    row = _row_to_risk_frame(features)
    Xs = _scale_constrained(model_payload, row)
    bcoef = np.array(model_payload["bootstrap_coef"], dtype=float)
    bint = np.array(model_payload["bootstrap_intercept"], dtype=float)
    return (bcoef @ Xs[0]) + bint


def predict_mmr(
    model,
    features: dict[str, float],
    *,
    anchor: dict[str, float] | None = None,
    anchor_mmr: float | None = None,
    scenario_log_gain: float | None = None,
) -> float:
    row = _row_to_risk_frame(features)
    if isinstance(model, dict) and model.get("kind") == "constrained_bootstrap_linear_logmmr":
        Xs = _scale_constrained(model, row)
        log_mu = float(_predict_log_mmr_constrained(model, Xs)[0])
        if anchor is not None and anchor_mmr is not None:
            ref_row = _row_to_risk_frame(anchor)
            log_ref = float(_predict_log_mmr_constrained(model, _scale_constrained(model, ref_row))[0])
            gain = _scenario_log_gain(model, scenario_log_gain)
            log_mu = float(np.log(max(anchor_mmr, 1.0)) + gain * (log_mu - log_ref))
        return float(np.exp(log_mu))
    return float(np.exp(model.predict(row)[0]))


def predict_mmr_interval(
    model,
    features: dict[str, float],
    z: float = 1.96,
    *,
    anchor: dict[str, float] | None = None,
    anchor_mmr: float | None = None,
    scenario_log_gain: float | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) on the MMR scale.

  When ``anchor`` and ``anchor_mmr`` are provided, predictions are shifted so the
  anchor inputs reproduce the observed country MMR; scenario changes apply relative
  log-deltas from the structural model (preserves directionality).
    """
    row = _row_to_risk_frame(features)
    if isinstance(model, dict) and model.get("kind") == "constrained_bootstrap_linear_logmmr":
        log_samples = _log_mmr_samples(model, features)
        if anchor is not None and anchor_mmr is not None:
            log_ref = _log_mmr_samples(model, anchor)
            log_obs = float(np.log(max(anchor_mmr, 1.0)))
            gain = _scenario_log_gain(model, scenario_log_gain)
            log_samples = log_obs + gain * (log_samples - log_ref)
        mmr_samples = np.exp(log_samples)
        return (
            float(mmr_samples.mean()),
            float(np.quantile(mmr_samples, 0.025)),
            float(np.quantile(mmr_samples, 0.975)),
        )

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

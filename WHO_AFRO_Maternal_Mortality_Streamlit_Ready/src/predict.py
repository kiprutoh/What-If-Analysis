"""Prediction helpers for scenario analysis."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.constants import FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]


def load_model():
    path = ROOT / MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python scripts/train_model.py"
        )
    return joblib.load(path)


def predict_mmr(model, features: dict[str, float]) -> float:
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    # Model predicts log(MMR)
    return float(np.exp(model.predict(row)[0]))


def predict_mmr_interval(model, features: dict[str, float], z: float = 1.96) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using log-normal approximation from Bayesian predictive std.

    For BayesianRidge, sklearn supports predict(return_std=True) for the target space.
    Here target space is log(MMR), so interval is exp(mu ± z*std).
    """
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
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

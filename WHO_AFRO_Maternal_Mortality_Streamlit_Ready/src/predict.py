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
    return float(np.exp(model.predict(row)[0]))


def scenario_frame(
    baseline: dict[str, float],
    overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    values = {**baseline, **(overrides or {})}
    return pd.DataFrame([{col: values[col] for col in FEATURE_COLUMNS}])

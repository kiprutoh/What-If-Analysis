#!/usr/bin/env python3
"""Train maternal mortality scenario model on public panel data."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (
    DATA_PATH,
    FEATURE_COLUMNS,
    FEATURE_TO_RISK_MULTIPLIER,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
)


def main() -> None:
    data_path = ROOT / DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python scripts/build_dataset.py"
        )

    df = pd.read_csv(data_path)
    train_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    X = train_df[FEATURE_COLUMNS].copy()
    # Convert to risk-oriented features so that higher => worse (higher expected MMR)
    for col in FEATURE_COLUMNS:
        mult = float(FEATURE_TO_RISK_MULTIPLIER.get(col, 1.0))
        X[col] = X[col].astype(float) * mult
    y = np.log(train_df[TARGET_COLUMN].clip(lower=1))

    # Constrained model in risk-oriented space:
    # coefficients are forced non-negative so that "worse" risk features
    # never reduce predicted MMR.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    base = LinearRegression(positive=True)
    base.fit(Xs, y)

    # Bootstrap uncertainty (lightweight Bayesian-style posterior approximation)
    rng = np.random.default_rng(42)
    B = 120
    coefs = np.zeros((B, Xs.shape[1]), dtype=float)
    intercepts = np.zeros(B, dtype=float)
    n = Xs.shape[0]
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        m = LinearRegression(positive=True)
        m.fit(Xs[idx], y[idx])
        coefs[b] = m.coef_
        intercepts[b] = m.intercept_

    y_pred = np.exp(base.predict(Xs))
    y_true = train_df[TARGET_COLUMN].values
    metrics = {
        "n_training_rows": int(len(train_df)),
        "n_countries": int(train_df["iso3"].nunique()),
        "year_min": int(train_df["year"].min()),
        "year_max": int(train_df["year"].max()),
        "r2_in_sample": round(float(r2_score(y_true, y_pred)), 4),
        "mae_in_sample": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "target_transform": "log(MMR)",
        "model_family": "Constrained linear model (positive coefficients) on risk-oriented features; bootstrap uncertainty",
        "risk_oriented_features": True,
        "feature_to_risk_multiplier": FEATURE_TO_RISK_MULTIPLIER,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "data_source": "World Bank Open Data API + WHO/UN-sourced series (via World Bank) + optional WHO GHO/UNICEF",
        "uncertainty_method": {"type": "bootstrap", "n_models": int(B)},
    }

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "constrained_bootstrap_linear_logmmr",
        "feature_columns": FEATURE_COLUMNS,
        "feature_to_risk_multiplier": FEATURE_TO_RISK_MULTIPLIER,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": base.coef_.tolist(),
        "intercept": float(base.intercept_),
        "bootstrap_coef": coefs.tolist(),
        "bootstrap_intercept": intercepts.tolist(),
    }
    joblib.dump(payload, model_dir / Path(MODEL_PATH).name)
    with open(model_dir / Path(METRICS_PATH).name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved -> {model_dir / Path(MODEL_PATH).name}")


if __name__ == "__main__":
    main()

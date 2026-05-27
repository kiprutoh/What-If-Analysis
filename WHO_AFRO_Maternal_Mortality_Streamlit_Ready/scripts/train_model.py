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
from src.predict import predict_mmr_interval

# Minimum standardized slope so each lever has a visible, directionally correct effect.
MIN_STD_COEF = 0.025


def _fit_univariate_positive_coefs(Xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    coefs = np.zeros(Xs.shape[1], dtype=float)
    y_arr = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)
    for j in range(Xs.shape[1]):
        xj = Xs[:, j : j + 1]
        m = LinearRegression(positive=True)
        m.fit(xj, y_arr)
        coefs[j] = max(float(m.coef_[0]), MIN_STD_COEF)
    return coefs


def _calibrate_intercept(Xs: np.ndarray, y: np.ndarray, coefs: np.ndarray) -> float:
    y_arr = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)
    return float(y_arr.mean() - (Xs @ coefs).mean())


def _shrink_coefs_for_panel_fit(
    Xs: np.ndarray,
    y: np.ndarray,
    coefs: np.ndarray,
    y_true_mmr: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Scale univariate coefficients: prefer stronger lever effects when panel MAE is similar."""
    y_arr = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)
    candidates: list[tuple[float, float]] = []
    for lam in np.linspace(0.2, 1.0, 17):
        scaled = coefs * lam
        intercept = _calibrate_intercept(Xs, y_arr, scaled)
        pred = np.exp(Xs @ scaled + intercept)
        mae = float(mean_absolute_error(y_true_mmr, pred))
        candidates.append((lam, mae))
    best_mae = min(mae for _, mae in candidates)
    threshold = best_mae * 1.12
    chosen_lambda = max(lam for lam, mae in candidates if mae <= threshold)
    return coefs * chosen_lambda, float(chosen_lambda)


def _calibrate_scenario_log_gain(
    train_df: pd.DataFrame,
    payload: dict,
    *,
    target_pct_drop: float = 0.06,
) -> float:
    """Scale anchored scenario deltas so a +15 pp SBA shift yields ~6% MMR reduction (median)."""
    changes: list[float] = []
    for _, row in train_df.groupby("iso3").tail(1).iterrows():
        baseline = {c: float(row[c]) for c in FEATURE_COLUMNS}
        observed = float(row[TARGET_COLUMN])
        if observed <= 0 or baseline.get("Skilled Birth Attendance") is None:
            continue
        sba = baseline["Skilled Birth Attendance"]
        scenario = {
            **baseline,
            "Skilled Birth Attendance": min(100.0, sba + 15.0),
        }
        _, _, m1 = predict_mmr_interval(
            payload,
            scenario,
            anchor=baseline,
            anchor_mmr=observed,
            scenario_log_gain=1.0,
        )
        changes.append((observed - m1) / observed)
    if not changes:
        return 1.0
    median_change = float(np.median(changes))
    if median_change <= 0.005:
        return 2.5
    return float(np.clip(target_pct_drop / median_change, 1.0, 4.0))


def main() -> None:
    data_path = ROOT / DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run: python scripts/build_dataset.py"
        )

    df = pd.read_csv(data_path)
    train_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    X = train_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        mult = float(FEATURE_TO_RISK_MULTIPLIER.get(col, 1.0))
        X[col] = X[col].astype(float) * mult
    y = np.log(train_df[TARGET_COLUMN].clip(lower=1))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    coefs = _fit_univariate_positive_coefs(Xs, y)
    coefs, shrink_lambda = _shrink_coefs_for_panel_fit(
        Xs, y, coefs, train_df[TARGET_COLUMN].values
    )
    intercept = _calibrate_intercept(Xs, y, coefs)

    rng = np.random.default_rng(42)
    B = 120
    boot_coef = np.zeros((B, Xs.shape[1]), dtype=float)
    boot_intercept = np.zeros(B, dtype=float)
    n = Xs.shape[0]
    y_arr = y.to_numpy(dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        raw = _fit_univariate_positive_coefs(Xs[idx], y_arr[idx])
        boot_coef[b] = raw * shrink_lambda
        boot_intercept[b] = _calibrate_intercept(Xs[idx], y_arr[idx], boot_coef[b])

    y_pred = np.exp(Xs @ coefs + intercept)
    y_true = train_df[TARGET_COLUMN].values

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "constrained_bootstrap_linear_logmmr",
        "feature_columns": FEATURE_COLUMNS,
        "feature_to_risk_multiplier": FEATURE_TO_RISK_MULTIPLIER,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": coefs.tolist(),
        "intercept": intercept,
        "bootstrap_coef": boot_coef.tolist(),
        "bootstrap_intercept": boot_intercept.tolist(),
        "scenario_log_gain": 1.0,
    }
    payload["scenario_log_gain"] = _calibrate_scenario_log_gain(train_df, payload)

    metrics = {
        "n_training_rows": int(len(train_df)),
        "n_countries": int(train_df["iso3"].nunique()),
        "year_min": int(train_df["year"].min()),
        "year_max": int(train_df["year"].max()),
        "r2_in_sample": round(float(r2_score(y_true, y_pred)), 4),
        "mae_in_sample": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "target_transform": "log(MMR)",
        "model_family": (
            "Additive monotonic model: one positive coefficient per risk-oriented indicator "
            "(univariate fits avoid multicollinearity zeroing); bootstrap uncertainty; "
            "scenario deltas anchored to observed country MMR in the app"
        ),
        "risk_oriented_features": True,
        "feature_to_risk_multiplier": FEATURE_TO_RISK_MULTIPLIER,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "data_source": "World Bank Open Data API + WHO/UN-sourced series (via World Bank) + optional WHO GHO/UNICEF",
        "uncertainty_method": {
            "type": "bootstrap_additive_monotonic",
            "n_models": int(B),
            "min_std_coef": MIN_STD_COEF,
            "panel_shrink_lambda": shrink_lambda,
            "scenario_log_gain": payload["scenario_log_gain"],
        },
    }

    joblib.dump(payload, model_dir / Path(MODEL_PATH).name)
    with open(model_dir / Path(METRICS_PATH).name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved -> {model_dir / Path(MODEL_PATH).name}")


if __name__ == "__main__":
    main()

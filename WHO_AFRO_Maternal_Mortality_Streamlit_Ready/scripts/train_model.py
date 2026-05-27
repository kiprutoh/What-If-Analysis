#!/usr/bin/env python3
"""Train maternal mortality scenario model on public panel data."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (
    DATA_PATH,
    FEATURE_COLUMNS,
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
    X = train_df[FEATURE_COLUMNS]
    y = np.log(train_df[TARGET_COLUMN].clip(lower=1))

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=2.0, random_state=42)),
        ]
    )
    model.fit(X, y)

    y_pred = np.exp(model.predict(X))
    y_true = train_df[TARGET_COLUMN].values
    metrics = {
        "n_training_rows": int(len(train_df)),
        "n_countries": int(train_df["iso3"].nunique()),
        "year_min": int(train_df["year"].min()),
        "year_max": int(train_df["year"].max()),
        "r2_in_sample": round(float(r2_score(y_true, y_pred)), 4),
        "mae_in_sample": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "cv_r2_mean": round(
            float(
                cross_val_score(
                    model,
                    X,
                    y,
                    cv=5,
                    scoring="r2",
                ).mean()
            ),
            4,
        ),
        "target_transform": "log(MMR)",
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "data_source": "World Bank Open Data API (WHO AFRO countries)",
    }

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / Path(MODEL_PATH).name)
    with open(model_dir / Path(METRICS_PATH).name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved -> {model_dir / Path(MODEL_PATH).name}")


if __name__ == "__main__":
    main()

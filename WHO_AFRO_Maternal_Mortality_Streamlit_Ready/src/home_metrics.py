"""Regional headline metrics for the home page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]


def compute_home_metrics() -> dict:
    latest_path = ROOT / "data" / "afro_country_latest.csv"
    panel = pd.read_csv(ROOT / DATA_PATH)
    if latest_path.exists():
        latest = pd.read_csv(latest_path).dropna(subset=[TARGET_COLUMN])
    else:
        latest_year = int(panel["year"].max())
        latest = panel[panel["year"] == latest_year].dropna(subset=[TARGET_COLUMN])

    latest_year = int(latest["year"].max()) if "year" in latest.columns else int(panel["year"].max())

    mmr_now = float(latest[TARGET_COLUMN].median())
    mmr_2000 = panel[panel["year"] == 2000].dropna(subset=[TARGET_COLUMN])
    mmr_trend = ""
    if len(mmr_2000) >= 5:
        mmr_then = float(mmr_2000[TARGET_COLUMN].median())
        if mmr_then > 0:
            pct = (mmr_then - mmr_now) / mmr_then * 100
            arrow = "↓" if pct >= 0 else "↑"
            mmr_trend = f"{arrow} {abs(pct):.0f}% since 2000"

    sba_col = "Skilled Birth Attendance"
    lit_col = "Female Literacy"
    rural_col = "Rural Population"

    sba_avg = float(latest[sba_col].mean()) if sba_col in latest.columns else float("nan")
    lit_avg = float(latest[lit_col].mean()) if lit_col in latest.columns else float("nan")
    rural_avg = float(latest[rural_col].mean()) if rural_col in latest.columns else float("nan")

    n_countries = int(latest["iso3"].nunique())
    year_min = int(panel["year"].min())
    year_max = int(panel["year"].max())

    return {
        "n_countries": n_countries,
        "year_min": year_min,
        "year_max": year_max,
        "latest_year": latest_year,
        "mmr_median": mmr_now,
        "mmr_trend": mmr_trend or "Regional median (latest year)",
        "sba_avg": sba_avg,
        "literacy_avg": lit_avg,
        "rural_avg": rural_avg,
        "n_feature_columns": len(FEATURE_COLUMNS),
    }

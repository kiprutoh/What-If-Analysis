#!/usr/bin/env python3
"""Download public World Bank indicators and save AFRO panel CSV."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import AFRO_ISO3, DATA_PATH
from src.data_sources import build_afro_panel, latest_country_snapshot
from src.synthetic_data import add_synthetic_predictors
from src.unicef_sdmx import fetch_unicef_bundle
from src.who_gho import fetch_who_gho_bundle


def main() -> None:
    out = ROOT / DATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Fetching World Bank indicators for WHO AFRO countries...")
    panel = build_afro_panel()

    print("Fetching WHO GHO indicators (C-section, workforce densities)...")
    who = fetch_who_gho_bundle(AFRO_ISO3)
    if not who.empty:
        panel = panel.merge(who, on=["iso3", "year"], how="left")
        if "Nursing & Midwifery Density (per 10k)" in panel.columns:
            panel["Midwife Density"] = panel["Nursing & Midwifery Density (per 10k)"]
        if (
            "Doctor Density (per 10k)" in panel.columns
            and "Nursing & Midwifery Density (per 10k)" in panel.columns
        ):
            panel["Health Workforce Density"] = (
                panel["Doctor Density (per 10k)"]
                + panel["Nursing & Midwifery Density (per 10k)"]
            )

    print("Fetching UNICEF SDMX MNCH indicators (facility delivery, C-section)...")
    unicef = fetch_unicef_bundle(AFRO_ISO3)
    if not unicef.empty:
        panel = panel.merge(unicef, on=["iso3", "year"], how="left")
        if "Caesarean Section Rate (UNICEF)" in panel.columns:
            panel["Caesarean Section Rate"] = panel["Caesarean Section Rate"].fillna(
                panel["Caesarean Section Rate (UNICEF)"]
            )

    panel = add_synthetic_predictors(panel)
    panel.to_csv(out, index=False)
    snap_path = out.parent / "afro_country_latest.csv"
    latest_country_snapshot(panel).to_csv(snap_path, index=False)
    print(f"Saved {len(panel)} country-year rows -> {out}")
    print(f"Saved latest snapshot -> {snap_path}")


if __name__ == "__main__":
    main()

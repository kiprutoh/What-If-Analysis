"""UNICEF SDMX data pulls using the `unicefdata` library.

Docs: https://data.unicef.org/sdmx-api-documentation/
Library: https://pypi.org/project/unicefdata/
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _try_import():
    try:
        from unicefdata import unicefData  # type: ignore

        return unicefData
    except Exception:
        return None


def fetch_unicef_bundle(iso3_list: Iterable[str]) -> pd.DataFrame:
    """Fetch a small MNCH bundle. Returns iso3/year columns to merge."""
    unicefData = _try_import()
    if unicefData is None:
        return pd.DataFrame(columns=["iso3", "year"])

    indicators = {
        "MNCH_INSTDEL": "Facility Delivery Rate",
        "MNCH_CSEC": "Caesarean Section Rate (UNICEF)",
    }

    frames = []
    for ind, col in indicators.items():
        try:
            df = unicefData(indicator=ind, countries=list(iso3_list), year="2000:2024", sex="_T")
            # Expected columns vary by package version; normalize.
            df = df.rename(
                columns={
                    "ref_area": "iso3",
                    "time_period": "year",
                    "obs_value": col,
                }
            )
            if "iso3" not in df.columns or "year" not in df.columns or col not in df.columns:
                continue
            df["year"] = df["year"].astype(int)
            frames.append(df[["iso3", "year", col]])
        except Exception:
            continue

    out = None
    for f in frames:
        out = f if out is None else out.merge(f, on=["iso3", "year"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["iso3", "year"])


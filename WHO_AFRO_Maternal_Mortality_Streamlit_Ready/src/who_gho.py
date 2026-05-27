"""WHO Global Health Observatory (GHO) OData client.

Docs: https://www.who.int/data/gho/info/gho-odata-api
Base: https://ghoapi.azureedge.net/api/
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests


BASE = "https://ghoapi.azureedge.net/api"
MAX_RETRIES = 3


def search_indicators(query: str, top: int = 50) -> pd.DataFrame:
    """Search indicator registry by substring match (case-insensitive-ish)."""
    # OData `contains` is case-sensitive on some backends; try both.
    q = query.replace("'", "''")
    url = f"{BASE}/Indicator?$filter=contains(IndicatorName,'{q}') or contains(IndicatorName,'{q.lower()}')&$top={int(top)}"
    j = requests.get(url, timeout=30).json()
    return pd.DataFrame(j.get("value", []))


def fetch_indicator(
    indicator_code: str,
    *,
    iso3_list: Iterable[str],
    year_min: int = 2000,
    year_max: int = 2023,
) -> pd.DataFrame:
    """Fetch country-year numeric values for a given indicator code."""
    iso3 = set(iso3_list)
    url = f"{BASE}/{indicator_code}?$select=SpatialDim,TimeDim,NumericValue,Low,High,Value,SpatialDimType"
    rows: list[dict] = []
    skip = 0
    while True:
        page_url = f"{url}&$top=2000&$skip={skip}"
        payload = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(page_url, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
                break
            except requests.RequestException:
                if attempt == MAX_RETRIES - 1:
                    payload = {"value": []}
                else:
                    time.sleep(1.0 * (attempt + 1))

        vals = (payload or {}).get("value", [])
        if not vals:
            break

        for item in vals:
            if item.get("SpatialDimType") not in (None, "COUNTRY"):
                continue
            c = item.get("SpatialDim")
            y = item.get("TimeDim")
            v = item.get("NumericValue")
            if c not in iso3 or v is None or y is None:
                continue
            y = int(y)
            if y < year_min or y > year_max:
                continue
            rows.append(
                {
                    "iso3": c,
                    "year": y,
                    "value": float(v),
                    "value_low": float(item["Low"]) if item.get("Low") is not None else None,
                    "value_high": float(item["High"]) if item.get("High") is not None else None,
                }
            )

        # Stop condition: if fewer than page size returned, end; else continue.
        if len(vals) < 2000:
            break
        skip += 2000

    return pd.DataFrame(rows)


def fetch_who_gho_bundle(iso3_list: Iterable[str]) -> pd.DataFrame:
    """Fetch a small bundle of key maternal-health predictors from WHO GHO."""
    bundle = [
        # C-section (overall)
        ("WHS4_115", "Caesarean Section Rate"),
        # Workforce densities per 10,000
        ("HWF_0003", "Doctor Density (per 10k)"),
        ("HWF_0006", "Nursing & Midwifery Density (per 10k)"),
    ]

    out = None
    for code, col in bundle:
        df = fetch_indicator(code, iso3_list=iso3_list)
        if df.empty:
            continue
        df = df.rename(columns={"value": col})
        keep = ["iso3", "year", col]
        out = df[keep] if out is None else out.merge(df[keep], on=["iso3", "year"], how="outer")

    return out if out is not None else pd.DataFrame(columns=["iso3", "year"])


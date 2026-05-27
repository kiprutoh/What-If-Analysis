"""Fetch and merge maternal health indicators from the World Bank open API."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from src.constants import AFRO_ISO3, INDICATORS

WORLD_BANK_BASE = "https://api.worldbank.org/v2"
PANEL_START_YEAR = 2000
PANEL_END_YEAR = 2023
MAX_RETRIES = 3
MAX_WORKERS = 12

RENAME_MAP = {
    "mmr": "Maternal Mortality Ratio",
    "sba": "Skilled Birth Attendance",
    "fertility": "Fertility Rate",
    "ado_fertility": "Adolescent Fertility Rate",
    "contraception_modern": "Contraception (modern)",
    "anc4": "ANC4 Coverage",
    "poverty_215": "Poverty ($2.15/day, %)",
    "gov_health_exp": "Gov Health Expenditure (% GDP)",
    "hiv_prev": "HIV Prevalence (15-49)",
    "malaria_inc": "Malaria Incidence",
    "anaemia_wra": "Anaemia (women 15-49)",
    "literacy": "Female Literacy",
    "literacy_total": "Adult Literacy",
    "rural": "Rural Population",
}

VALUE_COLUMNS = list(RENAME_MAP.values())


def _fetch_country_indicator(iso3: str, indicator_code: str) -> list[dict]:
    url = (
        f"{WORLD_BANK_BASE}/country/{iso3}/indicator/{indicator_code}"
        f"?format=json&per_page=500&date={PANEL_START_YEAR}:{PANEL_END_YEAR}"
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2:
                return []
            data = payload[1]
            if not data or (len(data) == 1 and "message" in data[0]):
                return []
            rows = []
            for item in data:
                if item.get("value") is None:
                    continue
                rows.append(
                    {
                        "iso3": item.get("countryiso3code") or iso3,
                        "country": item["country"]["value"],
                        "year": int(item["date"]),
                        "value": float(item["value"]),
                    }
                )
            return rows
        except (requests.RequestException, ValueError, KeyError, TypeError):
            if attempt == MAX_RETRIES - 1:
                return []
            time.sleep(1.0 * (attempt + 1))
    return []


def _fetch_indicator(indicator_code: str, iso3_list: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_country_indicator, iso3, indicator_code): iso3
            for iso3 in iso3_list
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def impute_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Fill sparse indicator gaps using within-country trends and regional medians."""
    filled = panel.sort_values(["iso3", "year"]).copy()
    for col in VALUE_COLUMNS:
        filled[col] = filled.groupby("iso3")[col].transform(
            lambda s: s.ffill().bfill()
        )
        regional = filled.groupby("year")[col].transform("median")
        filled[col] = filled[col].fillna(regional)
    return filled


def build_afro_panel(iso3_list: list[str] | None = None, impute: bool = True) -> pd.DataFrame:
    """Merge indicators into a country-year panel."""
    iso3 = iso3_list or AFRO_ISO3
    frames: dict[str, pd.DataFrame] = {}
    for key, code in INDICATORS.items():
        print(f"  Downloading {RENAME_MAP[key]} ({code})...")
        raw = _fetch_indicator(code, iso3)
        if raw.empty:
            raise RuntimeError(f"No data returned for indicator {code}")
        raw = raw.rename(columns={"value": RENAME_MAP[key]})
        frames[key] = raw[["iso3", "country", "year", RENAME_MAP[key]]]

    panel = frames["mmr"]
    for key in (
        "sba",
        "fertility",
        "ado_fertility",
        "contraception_modern",
        "anc4",
        "poverty_215",
        "gov_health_exp",
        "hiv_prev",
        "malaria_inc",
        "anaemia_wra",
        "literacy",
        "literacy_total",
        "rural",
    ):
        panel = panel.merge(frames[key], on=["iso3", "country", "year"], how="outer")

    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
    if impute:
        panel = impute_panel(panel)
    return panel


def latest_country_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    """Most recent complete observation per country for scenario baselines."""
    # Use only the core columns that are expected to be present after API + synthetic fill.
    # This keeps the app stable even if some optional indicators fail to download.
    required = [c for c in VALUE_COLUMNS if c in panel.columns]
    complete = panel.dropna(subset=required)
    idx = complete.groupby("iso3")["year"].idxmax()
    snap = complete.loc[idx].copy()
    return snap.sort_values("country").reset_index(drop=True)

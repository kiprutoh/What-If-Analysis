"""Realistic synthetic data generation for missing predictors.

Goal: when authoritative APIs don't expose a variable consistently, fill gaps
with plausible values to demonstrate the concept (reproducible).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SynthSpec:
    col: str
    lo: float
    hi: float
    yearly_drift: float  # expected mean change per year
    noise: float  # std dev


SYNTH_SPECS: list[SynthSpec] = [
    # Health system (not reliably in WB)
    SynthSpec("Facility Delivery Rate", 10, 100, 0.9, 6.0),
    SynthSpec("Caesarean Section Rate", 0.5, 35, 0.15, 2.2),
    SynthSpec("Emergency Obstetric Care Coverage", 5, 100, 0.8, 7.0),
    SynthSpec("Health Workforce Density", 0.2, 60, 0.4, 4.0),  # per 10,000
    SynthSpec("Midwife Density", 0.05, 30, 0.25, 2.0),  # per 10,000
    SynthSpec("Blood Availability Index", 0, 1, 0.01, 0.08),
    SynthSpec("Referral Efficiency Index", 0, 1, 0.01, 0.08),
    # Infrastructure
    SynthSpec("Travel Time to Facility (min)", 10, 240, -1.5, 10.0),
    SynthSpec("Road Access Index", 0, 1, 0.01, 0.07),
    SynthSpec("Ambulance Coverage Index", 0, 1, 0.01, 0.08),
    # Governance / data
    SynthSpec("Civil Registration Completeness", 0, 100, 0.8, 8.0),
    SynthSpec("HIS Maturity Index", 0, 1, 0.015, 0.08),
    # Shocks (annualized indices)
    SynthSpec("Epidemic Occurrence (0/1)", 0, 1, 0.0, 0.25),
    SynthSpec("Conflict Intensity Index", 0, 1, 0.0, 0.18),
    SynthSpec("Flood Occurrence (0/1)", 0, 1, 0.0, 0.22),
    SynthSpec("Supply Chain Disruption Index", 0, 1, 0.0, 0.16),
    SynthSpec("Workforce Strike Duration (days)", 0, 120, 0.0, 7.0),
]


def _seed_for(iso3: str, col: str) -> int:
    h = hashlib.sha256(f"{iso3}|{col}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _bounded_walk(
    rng: np.random.Generator,
    years: np.ndarray,
    lo: float,
    hi: float,
    drift: float,
    noise: float,
    start_value: float,
) -> np.ndarray:
    vals = np.zeros(len(years), dtype=float)
    vals[0] = start_value
    for i in range(1, len(years)):
        step = drift + rng.normal(0, noise)
        vals[i] = np.clip(vals[i - 1] + step, lo, hi)
    return vals


def add_synthetic_predictors(panel: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic columns when missing or sparsely populated."""
    out = panel.copy()
    years = np.array(sorted(out["year"].unique()))
    year0 = int(years.min())

    for spec in SYNTH_SPECS:
        if spec.col not in out.columns:
            out[spec.col] = np.nan

        # Fill per-country where missing
        for iso3, g in out.groupby("iso3", sort=False):
            idx = g.index
            s = out.loc[idx, spec.col].astype(float)
            if s.notna().all():
                continue

            rng = np.random.default_rng(_seed_for(str(iso3), spec.col))

            # choose start: if any observed, use first observed; else draw plausible baseline
            if s.notna().any():
                start = float(s.dropna().iloc[0])
            else:
                # draw baseline around mid-range, skewed a bit toward lower-resource settings
                midpoint = (spec.lo + spec.hi) / 2
                start = float(np.clip(rng.normal(midpoint * 0.75, (spec.hi - spec.lo) / 6), spec.lo, spec.hi))

            # align to country-year series ordering
            country_years = out.loc[idx, "year"].to_numpy()
            order = np.argsort(country_years)
            sorted_years = country_years[order]

            # compute relative years for drift scaling (annual)
            rel = sorted_years - sorted_years.min()
            # walk in sorted order, then scatter back
            walk = _bounded_walk(
                rng,
                sorted_years,
                spec.lo,
                spec.hi,
                drift=spec.yearly_drift,
                noise=spec.noise,
                start_value=start,
            )

            # for binary-like indicators, threshold
            if spec.col.endswith("(0/1)"):
                walk = (walk > 0.5).astype(float)

            filled = pd.Series(walk, index=np.array(idx)[order])
            out.loc[filled.index, spec.col] = out.loc[filled.index, spec.col].astype(float).fillna(filled)

    return out


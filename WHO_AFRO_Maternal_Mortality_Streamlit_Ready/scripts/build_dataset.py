#!/usr/bin/env python3
"""Download public World Bank indicators and save AFRO panel CSV."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import DATA_PATH
from src.data_sources import build_afro_panel, latest_country_snapshot


def main() -> None:
    out = ROOT / DATA_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Fetching World Bank indicators for WHO AFRO countries...")
    panel = build_afro_panel()
    panel.to_csv(out, index=False)
    snap_path = out.parent / "afro_country_latest.csv"
    latest_country_snapshot(panel).to_csv(snap_path, index=False)
    print(f"Saved {len(panel)} country-year rows -> {out}")
    print(f"Saved latest snapshot -> {snap_path}")


if __name__ == "__main__":
    main()

"""Consolidate raw parquet files into typed processed datasets.

Outputs:
  data/processed/movements_2025.parquet  - all 2025 movements (DEP+ARR), parsed timestamps
  data/processed/ranking_2026.parquet    - Jan+Jul 2026 movements, parsed timestamps
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
OUT = ROOT / "data" / "processed"

TS_COLS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]


def parse(df: pd.DataFrame) -> pd.DataFrame:
    for c in TS_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    # reporting airport: ADEP for departures, ADES for arrivals
    df["AIRPORT"] = df["ADEP_mvt"].where(df["PHASE_mvt"] == "DEP", df["ADES_mvt"])
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(RAW / "training_*.parquet")))
    assert len(files) == 12, f"expected 12 training files, got {len(files)}"
    tr = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tr = parse(tr)
    tr.to_parquet(OUT / "movements_2025.parquet", index=False)
    print("movements_2025:", tr.shape)

    rk = parse(pd.read_parquet(RAW / "ranking.parquet"))
    rk.to_parquet(OUT / "ranking_2026.parquet", index=False)
    print("ranking_2026:", rk.shape)


if __name__ == "__main__":
    main()

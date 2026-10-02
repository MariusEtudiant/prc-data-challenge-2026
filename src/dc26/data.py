from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# Jul 2026 was regenerated on 2026-09-04: all ten airports now report, and July is
# 55.7% of the test (192,122 of 344,841 departures) where it used to be 29.3%.
JULY_TEST_AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
                      "LFPG", "LIRF", "LSZH", "LTFM"]
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
            "LFPG", "LIRF", "LSZH", "LTFM"]  # LTAI listed on site but absent from data


def load_movements_2025(columns=None) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "movements_2025.parquet", columns=columns)


def load_ranking_2026(columns=None) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "ranking_2026.parquet", columns=columns)

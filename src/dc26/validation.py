"""Mirror validation: reproduce the leaderboard composition on 2025 data.

Leaderboard (verified): Jan 2026 = all 10 airports (152,719 DEP, 71%),
Jul 2026 = EDDF/EGLL/EHAM only (63,157 DEP, 29%).
Mirror holdout: Jan 2025 departures (all airports) + Jul 2025 departures (3 hubs).
Train: the other 10 months (all airports).
"""
import numpy as np
import pandas as pd

from .data import JULY_TEST_AIRPORTS


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def mirror_split(dep: pd.DataFrame):
    """dep: departures of 2025 with parsed MVT_TIME_UTC_mvt and AIRPORT columns.
    Returns (train_df, val_df, val_part) where val_part labels rows 'jan'/'jul'."""
    month = dep["MVT_TIME_UTC_mvt"].dt.month
    is_jan = month == 1
    is_jul_hub = (month == 7) & dep["AIRPORT"].isin(JULY_TEST_AIRPORTS)
    val_mask = is_jan | is_jul_hub
    # July non-hub departures are NOT in val; they stay in train (their info would
    # also be available in real training data).
    train = dep[~val_mask]
    val = dep[val_mask]
    val_part = np.where(is_jan[val_mask], "jan", "jul")
    return train, val, val_part


def rmse_report(y_true, y_pred, val_part) -> dict:
    out = {"overall": rmse(y_true, y_pred)}
    for p in ("jan", "jul"):
        m = val_part == p
        out[p] = rmse(np.asarray(y_true)[m], np.asarray(y_pred)[m])
    return out

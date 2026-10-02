"""Append-only experiment log: experiments/experiments.csv."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import ROOT

LOG = ROOT / "experiments" / "experiments.csv"
COLS = ["ts", "name", "model", "features", "val_rmse", "val_rmse_jan", "val_rmse_jul", "notes", "keep"]


def log_experiment(name, model, features, val_rmse, val_rmse_jan=None, val_rmse_jul=None, notes="", keep=""):
    row = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "name": name, "model": model, "features": features,
        "val_rmse": round(float(val_rmse), 2),
        "val_rmse_jan": round(float(val_rmse_jan), 2) if val_rmse_jan is not None else None,
        "val_rmse_jul": round(float(val_rmse_jul), 2) if val_rmse_jul is not None else None,
        "notes": notes, "keep": keep,
    }
    df = pd.DataFrame([row], columns=COLS)
    if LOG.exists():
        old = pd.read_csv(LOG)
        # replace an existing experiment with the same name (rerun) instead of duplicating
        old = old[old["name"] != name]
        df = pd.concat([old, df], ignore_index=True)
    LOG.parent.mkdir(exist_ok=True)
    df.to_csv(LOG, index=False)
    return row


def load_log() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=COLS)

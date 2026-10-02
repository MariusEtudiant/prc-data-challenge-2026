"""M76: LTFM has a clean target (arrival R2 0.710) but we only reach 0.609 on
departures, and AOBT_3 disagrees with BLOCK_TIME by 419s there - the worst of all
airports. Two candidate causes:

  a) the stand x runway reference time (P10, min 100 flights) has poor coverage,
     because 417 stands x several runways splits the data too thin
  b) AOBT_3 is simply unreliable at LTFM and the model over-relies on it
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
apt = X["AIRPORT"].astype(str).to_numpy()
ok = (y > 0) & (y < 3600)

ref = [c for c in X.columns if "ref" in c.lower() or "unimp" in c.lower() or "p10" in c.lower()]
print("variables de reference detectees:", ref)

print("\n=== a) couverture des variables de reference par aeroport ===")
if ref:
    print("  %-7s " % "aero" + " ".join("%18s" % c[:18] for c in ref))
    for a in sorted(set(apt)):
        m = ok & (apt == a)
        print("  %-7s " % a + " ".join("%17.1f%%" % (100 * X.loc[m, c].isna().mean()) for c in ref))

print("\n=== granularite poste x piste par aeroport ===")
print("  %-7s %8s %8s %10s %14s %16s" % ("aero", "postes", "pistes", "paires", "vols/paire", "paires <100 vols"))
for a in sorted(set(apt)):
    m = ok & (apt == a)
    d = X.loc[m, ["STAND", "RUNWAY"]].astype(str)
    pairs = d.groupby(["STAND", "RUNWAY"], observed=True).size()
    print("  %-7s %8d %8d %10d %14.0f %15.1f%%" % (
        a, d["STAND"].nunique(), d["RUNWAY"].nunique(), len(pairs),
        pairs.mean(), 100 * (pairs < 100).mean()))

print("\n=== b) fiabilite d AOBT_3 : correlation avec la cible, par aeroport ===")
d3 = X["d_aobt3"].to_numpy(dtype="float64")
print("  %-7s %12s %14s %14s" % ("aero", "corr(y,d_aobt3)", "RMSE d_aobt3", "ecart-type y"))
for a in sorted(set(apt)):
    m = ok & (apt == a) & np.isfinite(d3)
    if m.sum() < 500:
        continue
    c = np.corrcoef(d3[m], y[m])[0, 1]
    print("  %-7s %12.3f %14.1f %14.1f" % (
        a, c, np.sqrt(((y[m] - np.clip(d3[m], 60, 86400)) ** 2).mean()), y[m].std()))

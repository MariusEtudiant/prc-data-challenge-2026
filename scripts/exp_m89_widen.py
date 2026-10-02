"""M89: does bagging add to the CURRENT blend?

M46 put ExtraTrees and RandomForest into a blend and reached cleaned 252.0; M42
reached 250.3 three hours later with tuned CatBoost and jointly-fitted weights, and
the forests were dropped. But the two blends are not nested - nobody ever evaluated
M42's five sources PLUS the forests. That matters because M80 measured our sources'
error correlations at 0.93-0.98: the ensemble is effectively one model, and bagging
is the structurally most different family available.

Honest protocol, as in M82:
  fit January ODD days -> measure January EVEN days   (ten airports)
  fit January          -> measure July                (three airports)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse

CAP = 86400
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT", "doy"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
yv = y[va]
apt = X["AIRPORT"].astype(str).to_numpy()[va]
doy = X["doy"].to_numpy()[va]
clean = ~art[va]
vp = meta.loc[va, "val_part"].to_numpy()
jan, jul = (vp == "jan") & clean, (vp == "jul") & clean

t = np.load(EXP / "m42_preds.npz")
n39 = np.load(EXP / "m39_preds.npz")
f46 = np.load(EXP / "m46_preds.npz")
p40 = np.load(EXP / "m40_preds.npz")["P_va"]

base = [np.clip(t["air_lgb_va"], 60, CAP), np.clip(t["air_cb_va"], 60, CAP),
        np.clip(t["glob_lgb_va"], 60, CAP), np.clip(n39["nn_global"], 60, CAP),
        np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]),
                60, CAP)]
forests = [np.clip(f46["et_va"], 60, CAP), np.clip(f46["rf_va"], 60, CAP)]
nets8 = [np.clip(p40[i], 60, CAP) for i in range(p40.shape[0])]

SETS = {
    "A  5 sources (actuel)": base,
    "B  + forets (7)": base + forests,
    "C  + forets + 8 reseaux (15)": base + forests + nets8,
}

print("=== diversite : correlation des erreurs avec la moyenne des arbres ===")
tree = (base[0] + base[1]) / 2
et = yv - tree
print("  %-14s %s" % ("source", "correlation"))
for lab, v in [("air_cb", base[1]), ("glob_lgb", base[2]), ("nn_global", base[3]),
               ("nn_airport", base[4]), ("ExtraTrees", forests[0]), ("RandomForest", forests[1])]:
    m = clean & np.isfinite(v) & np.isfinite(tree)
    print("  %-14s %11.3f" % (lab, np.corrcoef(et[m], (yv - v)[m])[0, 1]))


def fit_apply(V, mf, mt):
    A = np.vstack(V)
    Af = np.where(np.isfinite(A[:, mf].T), A[:, mf].T, 0.0)
    w, _ = nnls(Af, yv[mf])
    w = w / w.sum() if w.sum() > 0 else np.eye(len(V))[0]
    return np.clip(w @ np.nan_to_num(A[:, mt]), 60, CAP), w


for title, (a_, b_, tag) in {
    "ajuste janvier IMPAIR -> mesure janvier PAIR": (jan & (doy % 2 == 1), jan & (doy % 2 == 0), "jan"),
    "ajuste janvier -> mesure JUILLET": (jan, jul, "jul"),
}.items():
    print("\n=== %s ===" % title)
    print("  %-30s %10s" % ("jeu de sources", "RMSE"))
    for lab, V in SETS.items():
        tot, n = 0.0, 0
        for ap in sorted(set(apt)):
            mf, mt = a_ & (apt == ap), b_ & (apt == ap)
            if mf.sum() < 300 or mt.sum() < 300:
                continue
            p, _ = fit_apply(V, mf, mt)
            tot += ((yv[mt] - p) ** 2).sum()
            n += mt.sum()
        print("  %-30s %10.2f" % (lab, np.sqrt(tot / n)))

print("\n=== detail par aeroport, janvier impair -> pair, A contre B ===")
print("  %-8s %10s %10s %9s %s" % ("aero", "A (5)", "B (7)", "gain", "poids des forets"))
for ap in sorted(set(apt)):
    mf, mt = jan & (doy % 2 == 1) & (apt == ap), jan & (doy % 2 == 0) & (apt == ap)
    if mf.sum() < 300 or mt.sum() < 300:
        continue
    pa, _ = fit_apply(SETS["A  5 sources (actuel)"], mf, mt)
    pb, wb = fit_apply(SETS["B  + forets (7)"], mf, mt)
    ra, rb = rmse(yv[mt], pa), rmse(yv[mt], pb)
    print("  %-8s %10.1f %10.1f %+9.1f   ET=%.2f RF=%.2f" % (ap, ra, rb, rb - ra, wb[5], wb[6]))

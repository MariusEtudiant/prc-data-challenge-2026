"""M49: calibration audit. Under RMSE the optimal prediction is the conditional
mean - including the mass of the rare catastrophic flights. Our networks are
trained on a target clipped at 7200s, so by construction they cannot carry that
mass. Question: is the blend systematically below the conditional mean where
extreme risk exists, and how much does that cost?
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
CAP = 86400

X = pd.concat([pd.read_parquet(PROC / f) for f in
               ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet",
                "depregime_train2025.parquet", "aobtq_train2025.parquet",
                "anomreg_train2025.parquet"]], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600

# ---- 1. what rate of catastrophes does the TRAINING distribution carry?
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
n = ok.sum()
print("=== taux dans les donnees d entrainement (n=%d) ===" % n)
for lab, m in [("artefacts jour (y ~ k*86400)", art & ok),
               (">5h hors artefacts", ok & ~art & (y > 18000).to_numpy()),
               (">2h hors artefacts", ok & ~art & (y > 7200).to_numpy())]:
    r = m.sum() / n
    print("  %-28s %6d  taux=%.2e  -> attendu sur 215876 tests = %.1f" % (lab, m.sum(), r, r * 215876))

# ---- 2. what does our score ALLOW to be present in the test?
tot_sq = 215876 * 260.13 ** 2
print("\n=== budget d erreur du test (RMSE 260.13) ===")
print("  erreur quadratique totale = %.3e" % tot_sq)
for lab, val in [("un artefact jour (86400 vs ~1000)", (86400 - 1000) ** 2),
                 ("un vol fantome (30000 vs ~1400)", (30000 - 1400) ** 2)]:
    print("  %-36s %.2e  -> au plus %.1f dans le test" % (lab, val, tot_sq / val))

# ---- 3. reconstruct the submitted blend on the mirror holdout
t = np.load(EXP / "m42_preds.npz")
n39 = np.load(EXP / "m39_preds.npz")
ng = np.clip(n39["nn_global"], 60, CAP)
na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]), 60, CAP)
names = ["air_lgb", "air_cb", "glob_lgb", "nn_global", "nn_airport"]
V = np.vstack([np.clip(t["air_lgb_va"], 60, CAP), np.clip(t["air_cb_va"], 60, CAP),
               np.clip(t["glob_lgb_va"], 60, CAP), ng, na])
W = json.loads((EXP / "m42_weights.json").read_text())
apt_va = X["AIRPORT"].astype(str).to_numpy()[va]
y_va = y[va].to_numpy()
pred = np.full(len(y_va), np.nan)
for a, w in W.items():
    m = apt_va == a
    if not m.any():
        continue
    w = np.array(w, float)
    M = np.where(np.isfinite(V[:, m]), V[:, m], 0.0)
    pred[m] = w @ M
pred = np.clip(pred, 60, CAP)
good = np.isfinite(pred)
np.save(EXP / "m49_blend_va.npy", pred)

# ---- 4. calibration: predicted mean vs realised mean, per bucket
print("\n=== calibration par tranche de prediction (holdout miroir, TOUT inclus) ===")
q = np.quantile(pred[good], np.linspace(0, 1, 11))
q[-1] += 1
print("  %-16s %8s %10s %10s %10s" % ("tranche", "n", "pred moy", "reel moy", "biais"))
for i in range(10):
    m = good & (pred >= q[i]) & (pred < q[i + 1])
    if m.sum() < 50:
        continue
    print("  %5.0f-%-10.0f %8d %10.1f %10.1f %+10.1f" % (
        q[i], q[i + 1], m.sum(), pred[m].mean(), y_va[m].mean(), y_va[m].mean() - pred[m].mean()))
m = good
print("  %-16s %8d %10.1f %10.1f %+10.1f" % ("GLOBAL", m.sum(), pred[m].mean(), y_va[m].mean(),
                                             y_va[m].mean() - pred[m].mean()))

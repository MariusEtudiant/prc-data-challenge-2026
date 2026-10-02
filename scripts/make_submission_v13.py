"""Submission v13 = v12 with gated blend weights.

The five sources are unchanged; only how they are combined changes. The networks
are trained on a target clipped at 7200s, so on heavy-tailed airports their
predictions explode above that cap - and because the per-airport weights were
fitted by NNLS on ungated predictions, a handful of such flights drove the network
weights to zero at LIRF, the airport where the sources disagree most (tree/network
error correlation 0.569 vs ~0.94 elsewhere) and therefore where blending should pay
most.

Gate: outside the ordinary regime (tree consensus above 7200s) the networks are
replaced by that consensus. Validated on four independent January partitions:
LIRF -19 to -29, global -1.6 to -3.4 (M83). The threshold is the networks' own
training cap, not a value tuned on the holdout.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
CAP, GATE = 86400, 7200
NAMES = ["air_lgb", "air_cb", "glob_lgb", "nn_global", "nn_airport"]


def gate(M):
    tree = np.nanmean(M[:2], axis=0)
    W = M.copy()
    for i in (3, 4):
        W[i] = np.where(tree <= GATE, M[i], tree)
    return W


# ---- fit the weights on the mirror holdout's January part, gated
X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64").to_numpy()
va = meta["va"].to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= np.abs(y - k * 86400) < 3600
yv = y[va]
apt_va = X["AIRPORT"].astype(str).to_numpy()[va]
jan = (meta.loc[va, "val_part"].to_numpy() == "jan") & ~art[va]

t = np.load(EXP / "m42_preds.npz")
n39 = np.load(EXP / "m39_preds.npz")
V = gate(np.vstack([
    np.clip(t["air_lgb_va"], 60, CAP), np.clip(t["air_cb_va"], 60, CAP),
    np.clip(t["glob_lgb_va"], 60, CAP), np.clip(n39["nn_global"], 60, CAP),
    np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]),
            60, CAP)]))

W = {}
for a in sorted(set(apt_va)):
    m = jan & (apt_va == a)
    if m.sum() < 300:
        continue
    A = np.where(np.isfinite(V[:, m].T), V[:, m].T, 0.0)
    w, _ = nnls(A, yv[m])
    w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0, 0, 0, 0])
    W[a] = [float(x) for x in w]
    print("  [%s] %s" % (a, " ".join("%s=%.2f" % (n, x) for n, x in zip(NAMES, w) if x > 0.01)),
          flush=True)
(EXP / "v13_weights.json").write_text(json.dumps(W))

# ---- apply to the ranking set, with the same gate
Xr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["AIRPORT"])
apt_r = Xr["AIRPORT"].astype(str).to_numpy()
d10 = np.load(EXP / "v10_rank_preds.npz", allow_pickle=True)
d11 = np.load(EXP / "v11_nn_preds.npz")
cb = np.load(EXP / "v12_cb.npz")["air_cb"]
ng = d11["nn_global"]
na = np.where(np.isfinite(d11["nn_airport"]), d11["nn_airport"], ng)
R = gate(np.vstack([np.clip(d10["air_lgbm"], 60, CAP), np.clip(cb, 60, CAP),
                    np.clip(d10["glob_lgbm"], 60, CAP), np.clip(ng, 60, CAP),
                    np.clip(na, 60, CAP)]))

pred = np.full(len(Xr), np.nan)
for a, w in W.items():
    r = apt_r == a
    if not r.any():
        continue
    w = np.array(w, float)
    M = R[:, r]
    ok = np.all(np.isfinite(M), axis=1)
    if w[~ok].sum() > 0:
        w = np.where(ok, w, 0.0)
        w = w / w.sum() if w.sum() > 0 else None
    if w is None:
        continue
    pred[r] = w @ np.nan_to_num(M)
miss = ~np.isfinite(pred)
if miss.any():
    pred[miss] = np.clip(d10["air_lgbm"], 60, CAP)[miss]
    print("  repli sur %d lignes" % miss.sum(), flush=True)
pred = np.clip(pred, 60, CAP)

ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                on="MVT_ID_mvt", how="left")
assert out["TAXITIME_SEC_mvt"].notna().all() and len(out) == 215876
out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v13.parquet", index=False)
v12 = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v12.parquet")
j = out.merge(v12, on="MVT_ID_mvt", suffixes=("_13", "_12"))
tt = out["TAXITIME_SEC_mvt"]
print("v13 ecrite: mean=%.0f median=%.0f q99=%.0f | ecart moyen vs v12=%.0fs" % (
    tt.mean(), tt.median(), tt.quantile(.99),
    (j["TAXITIME_SEC_mvt_13"] - j["TAXITIME_SEC_mvt_12"]).abs().mean()))

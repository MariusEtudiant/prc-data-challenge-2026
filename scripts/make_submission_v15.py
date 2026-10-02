"""Submission v15 = v14 plus the two networks, blended over four sources.

v14 was deliberately trees-only: it had to put a valid-format file on the new
344,841-row test quickly and establish the scale (it scored 343.31). This restores
the half of the architecture it left out.

Sources, all retrained on the regenerated data:
  lgbm     per-airport LightGBM, 3 seeds, early-stopped on the December window
  cb       per-airport CatBoost, Optuna parameters, GPU
  nn_glob  global entity-embedding MLP, 10 seeds
  nn_apt   per-airport MLP, 3 seeds, falling back to the global one where absent

Weights are fitted per airport by NNLS on ODD days of the mirror and measured on
EVEN days - both January and July, since July is now 55.7% of the test and weights
fitted on January alone were what hid CatBoost's value for weeks.

Measured on that honest split: four sources 281.25 against two sources 282.99.
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
sys.path.insert(0, str(ROOT / "scripts"))
from dc26.validation import rmse

CAP, N_ROWS = 86400, 344841
NAMES = ["lgbm", "cb", "nn_glob", "nn_apt"]


def _mirror_sources():
    n39 = np.load(EXP / "m39_preds.npz")
    ng = np.clip(n39["nn_global"], 60, CAP)
    na = np.clip(np.where(np.isfinite(n39["nn_airport"]), n39["nn_airport"], n39["nn_global"]),
                 60, CAP)
    return np.vstack([np.load(EXP / "m96_reference.npy"),
                      np.load(EXP / "v14_cb_mirror.npy"), ng, na])


def _rank_sources():
    d = np.load(EXP / "v14_rank.npz")
    n = np.load(EXP / "v11_nn_preds.npz")
    ng = np.clip(n["nn_global"], 60, CAP)
    na = np.clip(np.where(np.isfinite(n["nn_airport"]), n["nn_airport"], n["nn_global"]), 60, CAP)
    return np.vstack([d["lgbm"], d["cb"], ng, na])


def main():
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64").to_numpy()
    va = meta["va"].to_numpy()
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= np.abs(y - k * 86400) < 3600
    X = pd.read_parquet(PROC / "X_v2_mirror_t.parquet", columns=["AIRPORT"])
    apt_va = X["AIRPORT"].astype(str).to_numpy()[va]
    yv, clean = y[va], ~art[va]
    mvt = pd.read_parquet(PROC / "movements_2025.parquet",
                          columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"])
    t = mvt[mvt["PHASE_mvt"] == "DEP"]["MVT_TIME_UTC_mvt"].reset_index(drop=True)
    day, mon = t.dt.day.to_numpy()[va], t.dt.month.to_numpy()[va]

    V = _mirror_sources()
    fin = np.all(np.isfinite(V), axis=0)
    W, tot2, tot4, n = {}, 0.0, 0.0, 0
    print("=== poids a quatre sources : ajustes sur les jours IMPAIRS, mesures sur les PAIRS ===")
    for a in sorted(set(apt_va)):
        mo = clean & fin & (apt_va == a) & (day % 2 == 1)
        me = clean & fin & (apt_va == a) & (day % 2 == 0)
        if mo.sum() < 500 or me.sum() < 500:
            continue
        w, _ = nnls(V[:, mo].T, yv[mo])
        w = w / w.sum() if w.sum() > 0 else np.array([1.0, 0, 0, 0])
        w2, _ = nnls(V[:2, mo].T, yv[mo])
        w2 = w2 / w2.sum() if w2.sum() > 0 else np.array([1.0, 0.0])
        W[a] = [float(x) for x in w]
        p4 = np.clip(w @ V[:, me], 60, CAP)
        p2 = np.clip(w2 @ V[:2, me], 60, CAP)
        tot2 += ((yv[me] - p2) ** 2).sum()
        tot4 += ((yv[me] - p4) ** 2).sum()
        n += me.sum()
        print("  %-8s %s | 2src=%.1f -> 4src=%.1f" % (
            a, " ".join("%s=%.2f" % (nm, x) for nm, x in zip(NAMES, w) if x > 0.01),
            rmse(yv[me], p2), rmse(yv[me], p4)))
    print("\n  jours pairs : 2 sources=%.2f -> 4 sources=%.2f (%+.2f)" % (
        np.sqrt(tot2 / n), np.sqrt(tot4 / n), np.sqrt(tot4 / n) - np.sqrt(tot2 / n)))
    (EXP / "v15_weights.json").write_text(json.dumps(W))

    # full-mirror figure for the record, on the same weights
    pm = np.full(len(yv), np.nan)
    for a, w in W.items():
        m = apt_va == a
        if m.any():
            pm[m] = np.clip(np.array(w) @ np.nan_to_num(V[:, m]), 60, CAP)
    g = np.isfinite(pm) & clean
    print("  miroir complet nettoye=%.2f | janvier=%.2f | juillet=%.2f" % (
        rmse(yv[g], pm[g]), rmse(yv[g & (mon == 1)], pm[g & (mon == 1)]),
        rmse(yv[g & (mon == 7)], pm[g & (mon == 7)])))

    R = _rank_sources()
    assert R.shape[1] == N_ROWS, "sources: %d != %d" % (R.shape[1], N_ROWS)
    Xr = pd.read_parquet(PROC / "X_v2_ranking_t.parquet", columns=["AIRPORT"])
    apt_r = Xr["AIRPORT"].astype(str).to_numpy()
    pred = np.full(N_ROWS, np.nan)
    for a, w in W.items():
        r = apt_r == a
        if not r.any():
            continue
        M = R[:, r]
        wa = np.array(w, float)
        okc = np.all(np.isfinite(M), axis=1)
        if wa[~okc].sum() > 0:
            wa = np.where(okc, wa, 0.0)
            wa = wa / wa.sum() if wa.sum() > 0 else np.array([1.0, 0, 0, 0])
        pred[r] = wa @ np.nan_to_num(M)
    miss = ~np.isfinite(pred)
    if miss.any():
        pred[miss] = np.nan_to_num(R[0], nan=900.0)[miss]
        print("  repli sur %d lignes" % miss.sum())
    pred = np.clip(pred, 60, CAP)

    ids = pd.read_parquet(PROC / "rank_ids_t.parquet")["MVT_ID_mvt"]
    tpl = pd.read_parquet(ROOT / "data" / "submitting.parquet").drop(columns=["TAXITIME_SEC_mvt"])
    out = tpl.merge(pd.DataFrame({"MVT_ID_mvt": ids.values, "TAXITIME_SEC_mvt": pred}),
                    on="MVT_ID_mvt", how="left")
    assert out["TAXITIME_SEC_mvt"].notna().all(), "valeurs manquantes"
    assert len(out) == N_ROWS, "%d != %d" % (len(out), N_ROWS)
    assert out["MVT_ID_mvt"].tolist() == tpl["MVT_ID_mvt"].tolist(), "ordre different du gabarit"
    assert out["TAXITIME_SEC_mvt"].between(60, CAP).all()
    out.to_parquet(ROOT / "submissions" / "reliable-hamburger_v15.parquet", index=False)
    v14 = pd.read_parquet(ROOT / "submissions" / "reliable-hamburger_v14.parquet")
    j = out.merge(v14, on="MVT_ID_mvt", suffixes=("_15", "_14"))
    tt = out["TAXITIME_SEC_mvt"]
    print("v15 ecrite: n=%d mean=%.0f median=%.0f q99=%.0f | ecart moyen vs v14=%.0fs" % (
        len(out), tt.mean(), tt.median(), tt.quantile(.99),
        (j["TAXITIME_SEC_mvt_15"] - j["TAXITIME_SEC_mvt_14"]).abs().mean()))


if __name__ == "__main__":
    main()

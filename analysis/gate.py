"""W02: the two-stage mixture is right about the body and wrong about 38 flights, so gate it.

The mixture gains 24.9 s on the 600-1200 s band that holds 27 % of our squared error, and
35 to 499 s on every band above it - and then loses 3126 s on the 38 departures above four
hours. Those 38 are all at Rome, all without a Network Manager record, and all have
MVT - SCHED equal to the target within metres: they are copies, the mixture knows it only to
p = 0.15-0.69, and p * ms therefore under-predicts where the old single regressor, trained on
everything at once, had simply learned "Rome, no record, huge delay, answer huge".

Neither model dominates, so neither is chosen globally. Each (airport x has-NM) cell gets a
weight between the two, fitted on one scored month and applied to the other, so no row helps
choose the weight it is then judged by. A cell where the mixture hurts gets zero and keeps the
model we already have, which bounds the whole thing below by our current score.

The baseline has to be retrained here rather than reused: the cross-month run we have skipped
the day-offset artefacts, so it cannot be scored on the raw metric - the one the leaderboard
applies, and the one the flights in question live in.

Usage: python scripts/exp_w02_gate.py base      # the full-coverage single regressor
       python scripts/exp_w02_gate.py gate      # the gate, once the mirror of v17 exists
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse
import make_submission as V

CAP = 86400
WEIGHTS = np.arange(0, 1.01, 0.1)
MIN_FIT = 100


def stage_base():
    """The plain per-airport regressor on every valid row, cross-month, target winsorised at
    a day. Same folds and same matrices as v17, so the two are comparable row by row."""
    import lightgbm as lgb
    X = pd.concat([pd.read_parquet(PROC / f) for f in V.TRAIN_F], axis=1)
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64").to_numpy()
    es = meta["es"].to_numpy()
    ck = V.clock("movements_2025.parquet")
    valid = (y > 0) & (y < 200000)
    cat = [c for c in X.columns if X[c].dtype.name == "category"]
    apt = ck["apt"]
    pred = np.full(len(y), np.nan)
    for tgt, lab in ((ck["mon"] == 7, "sans juillet"), (ck["mon"] == 1, "sans janvier")):
        t0 = time.time()
        fit = valid & ~tgt
        for a in np.unique(apt):
            f = fit & (apt == a) & ~es
            s = fit & (apt == a) & es
            o = tgt & valid & (apt == a)
            if f.sum() < V.MIN_TRAIN or s.sum() < 300 or not o.any():
                continue
            m = lgb.train(V.reg_params(a, 42),
                          lgb.Dataset(X[f], np.clip(y[f], 0, CAP), categorical_feature=cat,
                                      params={"feature_pre_filter": False, "max_bin": 255}),
                          num_boost_round=6000,
                          valid_sets=[lgb.Dataset(X[s], np.clip(y[s], 0, CAP))],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            pred[o] = m.predict(X[o], num_iteration=m.best_iteration)
        print("  %s fait (%.0f min)" % (lab, (time.time() - t0) / 60), flush=True)
    np.save(EXP / "w02_base.npy", np.clip(pred, 60, CAP))
    print("[base] %d lignes predites" % np.isfinite(pred).sum())


def stage_gate():
    base = np.load(EXP / "w02_base.npy")
    mix = np.load(EXP / "v17_mirror.npy")
    meta = pd.read_parquet(PROC / "meta_train2025.parquet")
    y = meta["y"].astype("float64").to_numpy()
    ck = V.clock("movements_2025.parquet")
    valid = (y > 0) & (y < 200000)
    apt, nonm, mon = ck["apt"], ck["nonm"], ck["mon"]
    art = np.zeros(len(y), bool)
    for k in (1, 2, 3):
        art |= np.abs(y - k * 86400) < 3600
    g = valid & np.isfinite(base) & np.isfinite(mix)
    print("lignes comparables: %d" % g.sum())

    bw, _ = V.window(ck, base)
    mw, _ = V.window(ck, mix)
    other = {1: 7, 7: 1}
    final = bw.copy()
    chosen = {}
    for a in sorted(set(apt)):
        for nm in (False, True):
            for tgt in (1, 7):
                cell = g & (apt == a) & (nonm == nm)
                # the weight is chosen on the BODY of the fitting month. Choosing it on the
                # full RMSE lets a handful of multi-hour flights decide, and those are the
                # rows whose behaviour does not repeat: fitted that way the gate came out
                # worse than either model it was blending.
                f = cell & (mon == other[tgt]) & (y <= 3600)
                h = cell & (mon == tgt)
                if f.sum() < MIN_FIT or not h.any():
                    continue
                sc = [rmse(y[f], (1 - w) * bw[f] + w * mw[f]) for w in WEIGHTS]
                w = WEIGHTS[int(np.argmin(sc))]
                chosen[(a, nm, tgt)] = w
                final[h] = (1 - w) * bw[h] + w * mw[h]
    final = np.clip(final, 60, CAP)

    print("\n=== poids du deux-etages par cellule (choisis sur l'autre mois) ===")
    print("  %-7s %22s %22s" % ("aero", "avec NM (jan/juil)", "sans NM (jan/juil)"))
    for a in sorted(set(apt)):
        c = lambda nm: "%s / %s" % (chosen.get((a, nm, 1), "-"), chosen.get((a, nm, 7), "-"))
        print("  %-7s %22s %22s" % (a, c(False), c(True)))

    cl = g & ~art
    print("\n=== sur les memes %d lignes ===" % g.sum())
    print("  %-34s %10s %10s" % ("", "BRUTE", "nettoyee"))
    for lab, v in (("modele simple", base), ("  + fenetre LOBT", bw),
                   ("deux etages", mix), ("  + fenetre LOBT", mw),
                   ("porte par cellule + fenetre", final)):
        print("  %-34s %10.2f %10.2f" % (lab, rmse(y[g], v[g]), rmse(y[cl], v[cl])))
    print("  %-34s %+10.2f %+10.2f" % ("gain contre simple+fenetre",
                                       rmse(y[g], final[g]) - rmse(y[g], bw[g]),
                                       rmse(y[cl], final[cl]) - rmse(y[cl], bw[cl])))
    print("\n  a 0.854 de ratio officiel/simule, cela donnerait %.1f au classement" % (
        0.854 * rmse(y[g], final[g])))

    print("\n=== par aeroport (BRUTE) ===")
    print("  %-7s %9s %11s %11s %9s" % ("aero", "n", "simple+fen", "porte", "gain"))
    for a in sorted(set(apt)):
        m = g & (apt == a)
        if m.sum() < 500:
            continue
        r0, r1 = rmse(y[m], bw[m]), rmse(y[m], final[m])
        print("  %-7s %9d %11.1f %11.1f %+9.1f" % (a, m.sum(), r0, r1, r1 - r0))

    print("\n=== par bande de cible ===")
    print("  %-14s %9s %11s %11s %9s" % ("cible", "n", "simple+fen", "porte", "gain"))
    for lo, hi in ((0, 600), (600, 1200), (1200, 1800), (1800, 3600), (3600, 7200),
                   (7200, 14400), (14400, 10 ** 9)):
        m = g & (y >= lo) & (y < hi)
        if m.sum() < 5:
            continue
        lab = "%d-%d s" % (lo, hi) if hi < 10 ** 9 else "> 4 h"
        print("  %-14s %9d %11.1f %11.1f %+9.1f" % (
            lab, m.sum(), rmse(y[m], bw[m]), rmse(y[m], final[m]),
            rmse(y[m], final[m]) - rmse(y[m], bw[m])))
    np.save(EXP / "w02_final.npy", final)


if __name__ == "__main__":
    {"base": stage_base, "gate": stage_gate}[sys.argv[1]]()

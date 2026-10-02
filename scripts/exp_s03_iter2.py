"""S03: second iteration of the neighbour feature.

Iteration 1 (S02) built the feature from predictions of a model that did not yet
contain it, and recovered a third of the oracle at LTFM (-11.7 of -33.5), -1.46
overall. Feeding it back should recover more: the source predictions are now better,
so the estimate of the neighbours' taxi is less noisy.

Also sweeps the window K, which was set to 5 arbitrarily.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC, EXP = ROOT / "data" / "processed", ROOT / "experiments"
from dc26.validation import rmse
CAP, MIN_TRAIN = 86400, 20000
SEEDS = [42, 1337]
FOLDS = [(2, 3), (4, 5), (6, 8), (9, 10), (11, 12)]
FILES = ["X_v2_mirror_t.parquet", "flightctx_train2025.parquet", "depregime_train2025.parquet",
         "aobtq_train2025.parquet", "anomreg_train2025.parquet"]
_EPOCH = pd.Timestamp(0, tz="UTC")
X = pd.concat([pd.read_parquet(PROC / f) for f in FILES], axis=1)
meta = pd.read_parquet(PROC / "meta_train2025.parquet")
y = meta["y"].astype("float64")
tr, va, es = meta["tr"].to_numpy(), meta["va"].to_numpy(), meta["es"].to_numpy()
ok = (y > 0).to_numpy() & (y < 200000).to_numpy()
art = np.zeros(len(y), bool)
for k in (1, 2, 3):
    art |= (y - k * 86400).abs().to_numpy() < 3600
ok &= ~art
apt = X["AIRPORT"].astype(str).to_numpy()
apt_va, y_va, clean = apt[va], y[va].to_numpy(), ~art[va]
mv = pd.read_parquet(PROC / "movements_2025.parquet", columns=["PHASE_mvt","RUNWAY_mvt","MVT_TIME_UTC_mvt"])
dep = mv[mv["PHASE_mvt"] == "DEP"].reset_index(drop=True)
t_all = ((dep["MVT_TIME_UTC_mvt"] - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
mon_all = dep["MVT_TIME_UTC_mvt"].dt.month.to_numpy(); mon = mon_all[va]
key = (X["AIRPORT"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)).to_numpy()
NBR1 = None  # built below if absent
def params(a, seed):
    from dc26.mixture import REG_PARAMS
    p = EXP / ("optuna_" + a + ".json")
    b = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=0, bagging_freq=1,
             seed=seed, bagging_seed=seed+1, feature_fraction_seed=seed+2,
             feature_pre_filter=False, max_bin=255)
    if p.exists():
        d = json.loads(p.read_text()); b.update({k:v for k,v in d.items() if not k.startswith("_")})
        for k in ("num_leaves","min_data_in_leaf"): b[k]=int(b[k])
        b["max_bin"]=255
    else: b.update(REG_PARAMS)
    b["seed"]=seed; return b
def fit_predict(XX, fm, sm, om, a):
    P=[lgb.train(params(a,sd), lgb.Dataset(XX[fm], y[fm].clip(upper=CAP),
        categorical_feature=[c for c in XX.columns if XX[c].dtype.name=="category"],
        params={"feature_pre_filter":False,"max_bin":255}), num_boost_round=6000,
        valid_sets=[lgb.Dataset(XX[sm], y[sm].clip(upper=CAP))],
        callbacks=[lgb.early_stopping(150,verbose=False)]) for sd in SEEDS]
    return np.mean([m.predict(XX[om], num_iteration=m.best_iteration) for m in P],0)
def neighbours(src, K):
    nbr=np.full(len(y), np.nan)
    for _, idx in pd.Series(np.arange(len(y))).groupby(key):
        i=idx.to_numpy(); o=i[np.argsort(t_all[i])]; v=src[o]; g=np.isfinite(v)
        run,buf=np.full(len(o),np.nan),[]
        for j in range(len(o)):
            run[j]=np.mean(buf) if len(buf)>=3 else np.nan
            if g[j]:
                buf.append(v[j])
                if len(buf)>K: buf.pop(0)
        nbr[o]=run
    return nbr
t0=time.time()
_f = EXP / "s02_nbr_iter1.npy"
if _f.exists():
    NBR1 = np.load(_f)
else:
    # iteration 1: out-of-fold sources from the plain model, then the neighbour feature
    src1 = np.full(len(y), np.nan); bva = np.full(va.sum(), np.nan)
    for a in np.unique(apt):
        sel=ok&tr&(apt==a); sv=apt_va==a
        if sel.sum()<MIN_TRAIN or not sv.any(): continue
        for months in FOLDS:
            hold=sel&np.isin(mon_all,months); fit=sel&~np.isin(mon_all,months)&~es; stop=sel&~np.isin(mon_all,months)&es
            if hold.sum()<500 or fit.sum()<MIN_TRAIN//2 or stop.sum()<200: continue
            src1[hold]=fit_predict(X, fit, stop, hold, a)
        out=np.zeros(len(y),bool); out[np.where(va)[0][sv]]=True
        bva[sv]=fit_predict(X, sel&~es, sel&es, out, a)
    src1[va]=bva
    NBR1 = neighbours(src1, 5)
    np.save(_f, NBR1)
    print("iteration 1 reconstruite (%.0f min)" % ((time.time()-t0)/60), flush=True)
X1 = pd.concat([X, pd.DataFrame({"nbr_pred_rwy": NBR1})], axis=1)
src2 = np.full(len(y), np.nan); base_va = np.full(va.sum(), np.nan)
for a in np.unique(apt):
    sel=ok&tr&(apt==a); sv=apt_va==a
    if sel.sum()<MIN_TRAIN or not sv.any(): continue
    for months in FOLDS:
        hold=sel&np.isin(mon_all,months); fit=sel&~np.isin(mon_all,months)&~es; stop=sel&~np.isin(mon_all,months)&es
        if hold.sum()<500 or fit.sum()<MIN_TRAIN//2 or stop.sum()<200: continue
        src2[hold]=fit_predict(X1, fit, stop, hold, a)
    out=np.zeros(len(y),bool); out[np.where(va)[0][sv]]=True
    base_va[sv]=fit_predict(X1, sel&~es, sel&es, out, a)
src2[va]=base_va
print("sources iteration 2 pretes (%.0f min)" % ((time.time()-t0)/60), flush=True)
res={}
for tag, feats in (("iter1 (K=5)", {"nbr_pred_rwy": NBR1}),
                   ("iter2 (K=5)", {"nbr_pred_rwy": neighbours(src2,5)}),
                   ("iter2 (K=15)", {"nbr_pred_rwy": neighbours(src2,15)})):
    XX=pd.concat([X, pd.DataFrame(feats)], axis=1)
    pred=np.full(va.sum(), np.nan); per={}
    for a in np.unique(apt):
        sel=ok&tr&(apt==a); sv=apt_va==a
        if sel.sum()<MIN_TRAIN or not sv.any(): continue
        out=np.zeros(len(y),bool); out[np.where(va)[0][sv]]=True
        pred[sv]=fit_predict(XX, sel&~es, sel&es, out, a)
        c=sv&clean; per[a]=rmse(y_va[c], np.clip(pred[c],60,CAP))
    pred=np.clip(pred,60,CAP); g=np.isfinite(pred)&clean
    res[tag]=(rmse(y_va[g],pred[g]), rmse(y_va[g&(mon==7)],pred[g&(mon==7)]), per)
    print("%-14s nettoye=%7.2f | juillet=%7.2f  LTFM=%7.2f  (%.0f min)" % (
        tag, res[tag][0], res[tag][1], per.get("LTFM",np.nan), (time.time()-t0)/60), flush=True)
print()
print("  %-8s %10s %10s %10s" % ("aero","iter1","iter2","iter2 K=15"))
for a in sorted(res["iter1 (K=5)"][2]):
    print("  %-8s %10.1f %10.1f %10.1f" % (a, res["iter1 (K=5)"][2][a], res["iter2 (K=5)"][2][a], res["iter2 (K=15)"][2][a]))

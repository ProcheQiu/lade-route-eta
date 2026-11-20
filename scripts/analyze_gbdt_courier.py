"""How the LightGBM ETA models use courier statistics: gain, mean |SHAP| (20k seen-courier test rows),
hiding one courier statistic at a time (all seen-courier test rows), per-courier rank correlation of training
statistics with the courier's test-period median ETA.  python scripts/analyze_gbdt_courier.py
Needs results/models/gbdt_eta_{pickup,delivery}.txt from eval_hidden_courier_gbdt.py. Output: results/logs_gbdt/courier_analysis.txt"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from lade import config as C
from lade.baselines import gbdt
from lade.snapshots import COURIER_NUM, NODE_NUM, SNAP_NUM
from lade.models.data import unknown_courier_values
from lade.metrics import eta_metrics
print("NODE_NUM", NODE_NUM); print("SNAP_NUM", SNAP_NUM)
for ds in ["pickup", "delivery"]:
    sn = C.load_frame(f"{ds}_snaps"); nd = gbdt.add_context(C.load_frame(f"{ds}_nodes"), sn)
    b = lgb.Booster(model_str=(C.RESULTS/"models"/f"gbdt_eta_{ds}.txt").read_text(encoding="utf-8"))
    F = gbdt.ETA_FEATS
    gain = pd.Series(b.feature_importance("gain"), index=F); gain = (gain/gain.sum()*100).sort_values(ascending=False)
    print(f"\n=== {ds} ETA: gain share % ==="); print(gain.round(1).head(12).to_string())
    seen = nd[(nd.split=="test") & (nd.c_known==1)]
    smp = seen.sample(20000, random_state=0)
    contrib = b.predict(smp[F].values, pred_contrib=True)[:, :-1]
    shap = pd.Series(np.abs(contrib).mean(0), index=F).sort_values(ascending=False)
    print("mean |SHAP| (min):"); print(shap.round(2).head(10).to_string())
    print("courier block |sum SHAP| mean:", np.abs(contrib[:, [F.index(c) for c in COURIER_NUM]].sum(1)).mean().round(2))
    # one-at-a-time hiding on full seen test
    unk = dict(zip(COURIER_NUM, unknown_courier_values(sn)))
    base = eta_metrics(seen.assign(pred_eta=np.clip(b.predict(seen[F].values),0,None)))["mae"]
    print(f"known MAE {base:.2f}; unknown fallback row:", {k: round(v,2) for k,v in unk.items()})
    for c in COURIER_NUM[1:] + ["ALL"]:
        x = seen[F].copy()
        cols = COURIER_NUM if c=="ALL" else [c]
        for cc in cols: x[cc] = unk[cc]
        m = eta_metrics(seen.assign(pred_eta=np.clip(b.predict(x.values),0,None)))["mae"]
        print(f"  hide {c:16s} MAE {m:.2f} ({m-base:+.2f})")
    # is the stat predictive of the courier's own test-period behaviour? per-courier corr
    t = seen.groupby("courier_id").agg(c_median_dur=("c_median_dur","first"), c_median_gap=("c_median_gap","first"),
                                       c_tasks=("c_tasks_per_day","first"), eta_med=("eta","median"), n=("eta","size"))
    t = t[t.n>=200]
    print(f"couriers with >=200 test packages: {len(t)}; spearman with their test median ETA:",
          t[["c_median_dur","c_median_gap","c_tasks"]].corrwith(t.eta_med, method="spearman").round(2).to_dict())
    # dependence: predicted ETA vs c_median_dur decile
    smp = smp.assign(sh=contrib[:, F.index("c_median_dur")])
    print("SHAP of c_median_dur by its decile:", smp.groupby(pd.qcut(smp.c_median_dur, 5, duplicates="drop"), observed=True).sh.mean().round(1).to_dict())

"""Paper mode, step 2. Baselines on paper-mode samples, scored with the official metric definitions
-> results/paper_{method}_{name}.json
  python scripts/run_paper_baselines.py --methods rules --datasets pickup delivery
  python scripts/run_paper_baselines.py --methods gbdt_route --datasets pickup
  python scripts/run_paper_baselines.py --methods gbdt_eta --datasets delivery [--eta-backend lightgbm|xgboost]
                                        [--official-train-cap]
Selection never touches test; --split val evaluates on validation instead."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from lade import config as C
from lade.metrics import route_metrics_paper, eta_metrics_paper
from lade.baselines import paper_baselines as PB

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=["rules"], choices=["rules", "gbdt_route", "gbdt_eta"])
ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
ap.add_argument("--split", default="test")
ap.add_argument("--eta-backend", default="lightgbm", choices=["lightgbm", "xgboost"])
ap.add_argument("--route-backend", default="lightgbm", choices=["lightgbm", "xgboost"])
ap.add_argument("--official-train-cap", action="store_true",
                help="official bug: train on the first <#test windows> training windows only")
ap.add_argument("--max-train-samples", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def route_res(te):
    return {"all": route_metrics_paper(te), "by_courier_known": route_metrics_paper(te, by="c_known"),
            "by_n_open": {"n<=11": route_metrics_paper(te[te.n_open <= 11]),
                          "11<n<=25": route_metrics_paper(te[te.n_open > 11])}}


def eta_res(te):
    return {"all": eta_metrics_paper(te), "by_courier_known": eta_metrics_paper(te, by="c_known")}


def save(obj, fname):
    json.dump(obj, open(C.RESULTS / fname, "w"), indent=1, default=str)
    print(fname, json.dumps(obj.get("all", obj), indent=1, default=str))


for name in args.datasets:
    sn = pd.read_pickle(C.PROC / f"paper_{name}_snaps.pkl")
    nd = pd.read_pickle(C.PROC / f"paper_{name}_nodes.pkl")
    ctx = sn.set_index("sid")[["label_len", "c_known", "n_open", "split"]]
    nd = nd.join(ctx, on="sid")
    te_s = sn[(sn.split == args.split) & (sn.label_len >= 1)]
    te = nd[nd.sid.isin(te_s.sid)].copy()
    tag = f"{args.split}" if args.split != "test" else ""

    if "rules" in args.methods:
        t0 = time.time(); out = {}
        if name == "pickup":
            te["pred_rank"] = PB.time_greedy(te); out["TimeGreedy"] = route_res(te)
        for off in [True, False]:
            te["pred_rank"] = PB.distance_greedy(te, official=off)
            out["DistanceGreedy" + ("" if off else "-fixed")] = route_res(te)
            te["pred_eta"] = PB.speed_eta(te, sn, official=off)
            out["SPEED" + ("" if off else "-fixed")] = eta_res(te)
        out["seconds"] = round(time.time() - t0)
        json.dump(out, open(C.RESULTS / f"paper_rules_{name}{tag}.json", "w"), indent=1, default=str)
        for k, v in out.items():
            if isinstance(v, dict):
                print(name, k, json.dumps(v["all"]))

    tr_s = sn[(sn.split == "train") & (sn.label_len >= 1)]
    if args.max_train_samples and len(tr_s) > args.max_train_samples:
        tr_s = tr_s.sample(args.max_train_samples, random_state=args.seed).sort_values("sid")

    if "gbdt_route" in args.methods:
        t0 = time.time()
        clf, nrows = PB.fit_route_gbdt(tr_s, nd[nd.sid.isin(tr_s.sid)], name, backend=args.route_backend,
                                       seed=args.seed)
        te["pred_rank"] = PB.predict_route_gbdt(clf, te_s, te, name)
        res = route_res(te)
        res.update(backend=args.route_backend, train_samples=int(len(tr_s)), train_rows=int(nrows),
                   test_samples=int(len(te_s)), seconds=round(time.time() - t0))
        suffix = "_xgb" if args.route_backend == "xgboost" else ""
        save(res, f"paper_gbdt_route_{name}{suffix}{tag}.json")

    if "gbdt_eta" in args.methods:
        t0 = time.time()
        tr = tr_s
        if args.official_train_cap:   # official lgb/train.py loops over range(test_dataset.max_iter) windows
            n_win = sn.loc[sn.split == "test", "wid"].nunique()
            keep_w = np.sort(sn.loc[sn.split == "train", "wid"].unique())[:n_win]
            tr = tr[tr.wid.isin(keep_w)]
        m, nrows = PB.fit_eta_gbdt(tr, nd[nd.sid.isin(tr.sid)], name, backend=args.eta_backend, seed=args.seed)
        te["pred_eta"] = PB.predict_eta_gbdt(m, te_s, te, name)
        res = eta_res(te)
        res.update(backend=args.eta_backend, official_train_cap=args.official_train_cap, train_samples=int(nrows),
                   test_samples=int(len(te_s)), seconds=round(time.time() - t0))
        suffix = ("_xgb" if args.eta_backend == "xgboost" else "") + ("_cap" if args.official_train_cap else "")
        save(res, f"paper_gbdt_eta_{name}{suffix}{tag}.json")

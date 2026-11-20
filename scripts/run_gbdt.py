"""Step 3b. Gradient-boosted-tree baselines (LightGBM if installed), full-route setting.
  python scripts/run_gbdt.py --task eta   --datasets delivery pickup                     # all training days
  python scripts/run_gbdt.py --task route --datasets pickup delivery --train-frac 0.25 --max-val-snapshots 30000
--train-frac samples training snapshots exactly like train_joint.py (same seed -> same snapshots as the joint model).
The test split is always used in full (same rows as the joint model) unless --max-test-snapshots is given (preview only).
Writes results/gbdt_{task}_{dataset}[_tag].json"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lade import config as C
from lade.metrics import route_metrics, eta_metrics
from lade.baselines import gbdt

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["eta", "route"], required=True)
ap.add_argument("--datasets", nargs="+", default=None)
ap.add_argument("--train-frac", type=float, default=1.0, help="share of TRAIN snapshots (as train_joint.py --frac)")
ap.add_argument("--max-train-snapshots", type=int, default=None)
ap.add_argument("--max-val-snapshots", type=int, default=None, help="cap on VAL snapshots used for early stopping")
ap.add_argument("--max-test-snapshots", type=int, default=None, help="preview only")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="suffix of the output file name")
args = ap.parse_args()
datasets = args.datasets or (["delivery"] if args.task == "eta" else ["pickup"])
print("backend:", gbdt.BACKEND, flush=True)
for name in datasets:
    t0 = time.time()
    sn = C.load_frame(f"{name}_snaps")
    nd = gbdt.add_context(C.load_frame(f"{name}_nodes"), sn)

    def pick(split, frac=1.0, cap=None, seed=args.seed):
        s = sn[sn.split == split]
        if frac < 1.0:
            s = s.sample(frac=frac, random_state=seed)        # identical to models/data.build
        if cap and len(s) > cap:
            s = s.sample(cap, random_state=seed)
        return nd[nd.sid.isin(s.sid)]
    tr = pick("train", args.train_frac, args.max_train_snapshots)
    va = pick("val", 1.0, args.max_val_snapshots, args.seed + 1)
    te = pick("test", 1.0, args.max_test_snapshots).copy()
    print(f"{name}: train {tr.sid.nunique()} / val {va.sid.nunique()} / test {te.sid.nunique()} snapshots "
          f"({time.time() - t0:.0f}s)", flush=True)
    if args.task == "eta":
        m = gbdt.fit_eta(tr, va, args.seed)
        t1 = time.time()
        te["pred_eta"] = gbdt.predict_eta(m, te)
        res = {"all": eta_metrics(te), "by_courier_known": eta_metrics(te, by="c_known")}
    else:
        m, nrows = gbdt.fit_route(tr, va, None, args.seed)
        print(f"fitted on {nrows} decision rows ({time.time() - t0:.0f}s)", flush=True)
        t1 = time.time()
        te["pred_rank"] = gbdt.predict_route(m, te)
        res = {"all": route_metrics(te), "by_courier_known": route_metrics(te, by="c_known"), "train_rows": nrows}
    res.update(backend=gbdt.BACKEND, best_iteration=getattr(m, "best_iteration_", None),
               seconds=round(time.time() - t0), predict_seconds=round(time.time() - t1),
               train_snapshots=int(tr.sid.nunique()), val_snapshots=int(va.sid.nunique()),
               test_snapshots=int(te.sid.nunique()), args=vars(args),
               preview=bool(args.max_test_snapshots))
    print(name, json.dumps(res, indent=1, default=str), flush=True)
    json.dump(res, open(C.RESULTS / f"gbdt_{args.task}_{name}{'_' + args.tag if args.tag else ''}.json", "w"),
              indent=1, default=str)

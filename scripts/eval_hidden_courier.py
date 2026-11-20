"""Hidden-courier evaluation: how much does KNOWING the courier help the joint model?

Takes the saved joint models (results/models/{run}.pt, no retraining) and scores the SEEN-courier test snapshots
(c_known = 1) in three input modes, on exactly the same rows with the same model:
  known  : normal input (courier id + that courier's training statistics)
  hidden : the courier is shown as an unseen one: id 0, c_known 0, fallback statistics (= a real new courier's input)
  no_id  : diagnostic only: id 0 but the courier's own statistics kept (a combination never seen in training)
known - hidden = the value of courier knowledge, free of the "new couriers get different snapshots" confound.

  python scripts/eval_hidden_courier.py                         # both datasets, seeds 0-4, CPU (~30 min)
  python scripts/eval_hidden_courier.py --datasets pickup --seeds 0 --max-snapshots 5000   # quick check
Writes results/hidden_courier_{dataset}.json
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from lade import config as C
from lade.metrics import route_metrics, eta_metrics
from lade.models.data import fit_encoders, build, to_eval_frame, COURIER_CTX_IDX
from lade.models.joint import JointRouteETA

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--max-snapshots", type=int, default=None, help="quick check only")
ap.add_argument("--batch", type=int, default=1024)
args = ap.parse_args()
KEYS = ["mask", "num", "xy", "tw", "aoi", "aoi_raw", "aoi_type", "eta", "order", "ctx", "courier"]
ROUTE_K, ETA_K = ["hr3", "krc", "lsd", "ed"], ["mae", "rmse", "acc20"]


def subset(arr, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in arr.items()}


def with_mode(arr, mode, enc):
    a = dict(arr)
    if mode in ("hidden", "no_id"):
        a["courier"] = np.zeros_like(arr["courier"])
    if mode == "hidden":
        a["ctx"] = arr["ctx"].copy()
        a["ctx"][:, COURIER_CTX_IDX] = enc["unknown_ctx"]
    return a


@torch.no_grad()
def predict(model, arr):
    orders, etas = [], []
    for i in range(0, len(arr["n"]), args.batch):
        b = {k: torch.as_tensor(arr[k][i:i + args.batch]) for k in KEYS}
        _, order, eta = model(b, teacher=False)
        orders.append(order.numpy()); etas.append(eta.numpy())
    ev = to_eval_frame(arr, np.concatenate(orders), np.concatenate(etas))
    return {**{k: route_metrics(ev)[k] for k in ROUTE_K}, **{k: eta_metrics(ev)[k] for k in ETA_K}}


t0 = time.time()
for ds in args.datasets:
    sn = C.load_frame(f"{ds}_snaps"); nd = C.load_frame(f"{ds}_nodes")
    enc = fit_encoders(nd, sn)
    per_seed = {}
    for seed in args.seeds:
        run = f"{ds}_h64_l3_lam1.0_s{seed}"
        saved = json.load(open(C.RESULTS / f"joint_{run}.json"))
        a = saved["args"]
        te = build(nd, sn, "test", enc, frac=1.0, seed=seed + 2)        # same slot shuffle as in training
        seen = subset(te, te["c_known"] == 1)
        if args.max_snapshots:
            seen = subset(seen, np.arange(len(seen["n"])) < args.max_snapshots)
        model = JointRouteETA(n_num=te["num"].shape[-1], n_ctx=te["ctx"].shape[-1], n_aoi=len(enc["aoi"]),
                              n_aoi_type=len(enc["aoi_type"]), n_courier=len(enc["courier"]), d=a["hidden"],
                              layers=a["layers"], dropout=a["dropout"], use_edges=not a["no_edges"],
                              route_conditioned=not a["no_route_cond"], use_courier=not a["no_courier"],
                              hard_window=a["hard_window"], courier_dropout=0.0)
        model.load_state_dict(torch.load(C.RESULTS / "models" / f"{run}.pt", map_location="cpu"))
        model.eval()
        res = {m: predict(model, with_mode(seen, m, enc)) for m in ["known", "hidden", "no_id"]}
        # sanity: "known" on the full seen set must reproduce the saved test result for seen couriers
        if not args.max_snapshots:
            ref = saved["test_route_by_courier_known"]["1"]["krc"]
            res["check_krc_vs_saved"] = [res["known"]["krc"], ref]
            assert abs(res["known"]["krc"] - ref) < 0.05, (res["known"]["krc"], ref)
        res["snapshots"] = int(len(seen["n"]))
        per_seed[seed] = res
        print(f"{ds} s{seed}: " + "  ".join(f"{m} KRC {res[m]['krc']:.2f} MAE {res[m]['mae']:.2f}"
                                             for m in ["known", "hidden", "no_id"]) + f"  ({time.time() - t0:.0f}s)",
              flush=True)
    summary = {}
    for m in ["known", "hidden", "no_id"]:
        for k in ROUTE_K + ETA_K:
            v = np.array([per_seed[s][m][k] for s in args.seeds])
            summary[f"{m}_{k}"] = [float(v.mean()), float(v.std())]
    for k in ROUTE_K + ETA_K:                     # paired difference per seed: hidden - known
        d = np.array([per_seed[s]["hidden"][k] - per_seed[s]["known"][k] for s in args.seeds])
        summary[f"hidden_minus_known_{k}"] = [float(d.mean()), float(d.std())]
    out = {"dataset": ds, "seeds": args.seeds, "summary": summary, "per_seed": per_seed,
           "preview": bool(args.max_snapshots)}
    json.dump(out, open(C.RESULTS / f"hidden_courier_{ds}{'_preview' if args.max_snapshots else ''}.json", "w"),
              indent=1, default=str)
    print(ds, json.dumps(summary, indent=1), flush=True)

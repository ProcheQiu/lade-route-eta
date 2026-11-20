"""Hidden-courier evaluation for the LightGBM full-route baselines (counterpart of eval_hidden_courier.py).

Retrains the model exactly as run_gbdt.py does (same data, settings and seed), saves it to results/models/, and
scores the SEEN-courier test snapshots twice on the same rows:
  known  : the courier's own training-period statistics (COURIER_NUM)
  hidden : the input a real new courier gets: c_known 0 and the training-courier medians
LightGBM has no courier-ID feature, so the statistics are its only courier information.

  python scripts/eval_hidden_courier_gbdt.py --task eta   --datasets delivery pickup
  python scripts/eval_hidden_courier_gbdt.py --task route --datasets pickup delivery
Writes results/hidden_courier_gbdt_{task}_{dataset}.json
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
from lade import config as C
from lade.metrics import route_metrics, eta_metrics
from lade.baselines import gbdt
from lade.snapshots import COURIER_NUM
from lade.models.data import unknown_courier_values

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["eta", "route"], required=True)
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
# identical to the settings recorded in results/gbdt_{task}_{dataset}.json
TRAIN_FRAC, MAX_VAL = (1.0, None) if args.task == "eta" else (0.25, 30000)
(C.RESULTS / "models").mkdir(exist_ok=True)

for name in args.datasets:
    t0 = time.time()
    sn = C.load_frame(f"{name}_snaps")
    nd = gbdt.add_context(C.load_frame(f"{name}_nodes"), sn)

    def pick(split, frac=1.0, cap=None, seed=args.seed):
        s = sn[sn.split == split]
        if frac < 1.0:
            s = s.sample(frac=frac, random_state=seed)
        if cap and len(s) > cap:
            s = s.sample(cap, random_state=seed)
        return nd[nd.sid.isin(s.sid)]
    tr, va = pick("train", TRAIN_FRAC), pick("val", 1.0, MAX_VAL, args.seed + 1)
    seen = pick("test")
    seen = seen[seen.c_known == 1].copy()
    if args.task == "eta":
        m = gbdt.fit_eta(tr, va, args.seed)
    else:
        m, _ = gbdt.fit_route(tr, va, None, args.seed)
    # LightGBM's own save_model cannot open paths with non-ASCII characters (桌面) on Windows: write from Python
    (C.RESULTS / "models" / f"gbdt_{args.task}_{name}.txt").write_text(m.booster_.model_to_string(), encoding="utf-8")
    print(f"{name}: fitted, best iteration {m.best_iteration_} ({time.time() - t0:.0f}s)", flush=True)

    hidden = seen.copy()
    hidden[COURIER_NUM] = unknown_courier_values(sn)          # one constant row, training days only
    res = {}
    for mode, x in [("known", seen), ("hidden", hidden)]:
        if args.task == "eta":
            x = x.assign(pred_eta=gbdt.predict_eta(m, x)); res[mode] = eta_metrics(x)
        else:
            x = x.assign(pred_rank=gbdt.predict_route(m, x)); res[mode] = route_metrics(x)
        print(f"  {mode}: " + ", ".join(f"{k} {v:.2f}" for k, v in res[mode].items()) + f" ({time.time() - t0:.0f}s)",
              flush=True)
    keys = [k for k in res["known"] if k not in ("snapshots", "packages")]
    res["hidden_minus_known"] = {k: res["hidden"][k] - res["known"][k] for k in keys}
    saved = json.load(open(C.RESULTS / f"gbdt_{args.task}_{name}.json"))["by_courier_known"]["1"]
    k0 = "mae" if args.task == "eta" else "krc"
    res["check_vs_saved"] = {k0: [res["known"][k0], saved[k0]]}        # retrained model should reproduce it
    res.update(best_iteration=m.best_iteration_, seconds=round(time.time() - t0), seed=args.seed,
               train_frac=TRAIN_FRAC, max_val_snapshots=MAX_VAL)
    print(name, json.dumps(res["hidden_minus_known"]), "check", res["check_vs_saved"], flush=True)
    json.dump(res, open(C.RESULTS / f"hidden_courier_gbdt_{args.task}_{name}.json", "w"), indent=1, default=str)

"""Step 3a. TimeGreedy, DistanceGreedy, SPEED on the test split -> results/rule_baselines.json
Usage: python scripts/run_rule_baselines.py [--datasets delivery pickup] [--split test]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from lade import config as C
from lade.metrics import route_metrics, eta_metrics
from lade.baselines.rules import distance_greedy, time_greedy, courier_speed, speed_eta

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS)); ap.add_argument("--split", default="test")
args = ap.parse_args()
res = {}
for name in args.datasets:
    t0 = time.time()
    sn = pd.read_pickle(C.PROC / f"{name}_snaps.pkl"); nd = pd.read_pickle(C.PROC / f"{name}_nodes.pkl")
    split = sn.set_index("sid").split
    nd["split"] = split.reindex(nd.sid).values
    te = nd[nd.split == args.split].copy()
    te["c_known"] = sn.set_index("sid").c_known.reindex(te.sid).values
    r = {}
    for label, fn in [("TimeGreedy", time_greedy), ("DistanceGreedy", distance_greedy)]:
        te["pred_rank"] = fn(te)
        r[label] = {"all": route_metrics(te), "by_courier_known": route_metrics(te, by="c_known")}
    spd = courier_speed(nd[nd.split == "train"], sn[sn.split == "train"])
    te["pred_eta"] = speed_eta(te, sn, spd)
    r["SPEED"] = {"all": eta_metrics(te), "by_courier_known": eta_metrics(te, by="c_known")}
    r["seconds"] = round(time.time() - t0)
    res[name] = r
    print(name, json.dumps({k: v["all"] if isinstance(v, dict) else v for k, v in r.items()}, indent=1))
json.dump(res, open(C.RESULTS / f"rule_baselines_{args.split}.json", "w"), indent=1, default=str)

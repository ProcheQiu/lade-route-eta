"""Step 3c. OR-Tools route baseline -> results/ortools_{dataset}.json
  python scripts/run_ortools.py --dataset pickup --windows [--max-test-snapshots 20000] [--workers 8]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from lade import config as C
from lade.metrics import route_metrics
from lade.baselines.ortools_route import ortools_route

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="pickup"); ap.add_argument("--windows", action="store_true")
ap.add_argument("--max-test-snapshots", type=int, default=None); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--time-limit-ms", type=int, default=200)
args = ap.parse_args()
t0 = time.time()
sn = pd.read_pickle(C.PROC / f"{args.dataset}_snaps.pkl"); nd = pd.read_pickle(C.PROC / f"{args.dataset}_nodes.pkl")
te = nd[nd.sid.isin(sn.sid[sn.split == "test"])].copy()
if args.max_test_snapshots:
    te = te[te.sid.isin(te.sid.drop_duplicates().sample(args.max_test_snapshots, random_state=0))]
chunks = np.array_split(te.sid.unique(), args.workers * 4)
parts = Parallel(n_jobs=args.workers)(delayed(ortools_route)(te[te.sid.isin(c)], args.windows, args.time_limit_ms) for c in chunks)
te["pred_rank"] = pd.concat(parts).reindex(te.index)
te["c_known"] = sn.set_index("sid").c_known.reindex(te.sid).values
res = {"all": route_metrics(te), "by_courier_known": route_metrics(te, by="c_known"), "windows": args.windows,
       "seconds": round(time.time() - t0), "preview": bool(args.max_test_snapshots)}
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(C.RESULTS / f"ortools_{args.dataset}.json", "w"), indent=1, default=str)

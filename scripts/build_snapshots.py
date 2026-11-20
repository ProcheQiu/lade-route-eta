"""Step 2. Build snapshot samples -> data/processed/{name}_snaps.pkl, {name}_nodes.pkl, {name}_courier_stats.pkl
Usage: python scripts/build_snapshots.py [--datasets delivery pickup]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from lade import config as C
from lade.snapshots import build, courier_stats, attach_courier_stats

ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
args = ap.parse_args()
summary = {}
for name in args.datasets:
    t0 = time.time()
    df = pd.read_pickle(C.PROC / f"{name}_clean.pkl")
    snaps, nodes, stats = build(df, name)
    cs = courier_stats(df, name)
    snaps = attach_courier_stats(snaps, cs)
    snaps.to_pickle(C.PROC / f"{name}_snaps.pkl"); nodes.to_pickle(C.PROC / f"{name}_nodes.pkl")
    cs.to_pickle(C.PROC / f"{name}_courier_stats.pkl")
    by = snaps.groupby("split").agg(snapshots=("sid", "size"), nodes=("n_open", "sum"),
                                    new_courier_share=("c_known", lambda s: round(1 - s.mean(), 3)))
    stats["by_split"] = by.to_dict("index"); stats["median_n_open"] = float(snaps.n_open.median())
    stats["seconds"] = round(time.time() - t0)
    summary[name] = stats
    print(name, json.dumps(stats, default=str, indent=1))
json.dump(summary, open(C.RESULTS / "snapshot_summary.json", "w"), indent=1, default=str)

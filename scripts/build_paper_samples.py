"""Paper mode, step 1. Official LaDe preprocessing + sample builder on the RAW Shanghai CSVs
-> data/processed/paper_{name}_snaps.pkl, paper_{name}_nodes.pkl, results/paper_samples_summary.json
Usage: python scripts/build_paper_samples.py [--datasets delivery pickup] [--courier-stats train|all]
                                             [--split official|ours]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from lade import config as C
from lade import paper as PP
from lade.clean import split_of

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
ap.add_argument("--courier-stats", choices=["train", "all"], default="train",
                help="train = leakage-safe (default); all = official (statistics over every day)")
ap.add_argument("--split", choices=["official", "ours"], default="official",
                help="official = dates 60/20/20 with one gap day before val and test; ours = config.SPLIT_CUTS")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
summary = {}
for name in args.datasets:
    t0 = time.time()
    raw = pd.read_csv(C.RAW / name / f"{name}_sh.csv")
    df, log = PP.preprocess(raw, name)
    if args.split == "official":
        split = PP.official_split(df.ds)
    else:
        split = split_of(pd.to_datetime(f"{C.YEAR}" + df.ds.astype(str).str.zfill(4), format="%Y%m%d"), name)
    snaps, nodes, stats = PP.build(df, name, split.values, courier_stats=args.courier_stats, seed=args.seed)
    known = set(snaps.loc[snaps.split == "train", "courier_id"])
    snaps["c_known"] = snaps.courier_id.isin(known).astype(np.int8)
    snaps.to_pickle(C.PROC / f"paper_{name}_snaps.pkl")
    nodes.to_pickle(C.PROC / f"paper_{name}_nodes.pkl")
    by = snaps.groupby("split").agg(samples=("sid", "size"), label_ge1=("label_len", lambda s: int((s >= 1).sum())),
                                    mean_open=("n_open", "mean"), mean_label_len=("label_len", "mean"),
                                    new_courier_share=("c_known", lambda s: round(1 - s.mean(), 3)))
    days = {s: [str(df.ds[split == s].min()), str(df.ds[split == s].max())] for s in ["train", "val", "test"]}
    lab = nodes[nodes.in_label]
    summary[name] = dict(preprocess=log, build=stats, split=args.split, courier_stats=args.courier_stats,
                         split_days=days, by_split=by.to_dict("index"),
                         eta_label_mean=float(lab.eta.mean()), eta_label_median=float(lab.eta.median()),
                         seconds=round(time.time() - t0))
    print(name, json.dumps(summary[name], indent=1, default=str))
json.dump(summary, open(C.RESULTS / "paper_samples_summary.json", "w"), indent=1, default=str)

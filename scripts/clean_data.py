"""Step 1. Clean raw CSVs -> data/processed/{name}_clean.pkl and results/cleaning_log.json
Usage: python scripts/clean_data.py [--datasets delivery pickup]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from lade import config as C
from lade.clean import clean

ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=list(C.DATASETS))
args = ap.parse_args()
C.PROC.mkdir(parents=True, exist_ok=True); C.RESULTS.mkdir(parents=True, exist_ok=True)
logs = {}
for name in args.datasets:
    raw = pd.read_csv(C.RAW / name / f"{name}_sh.csv")
    df, log = clean(raw, name)
    df.to_pickle(C.PROC / f"{name}_clean.pkl")
    logs[name] = log
    print(name, json.dumps(log))
json.dump(logs, open(C.RESULTS / "cleaning_log.json", "w"), indent=1)

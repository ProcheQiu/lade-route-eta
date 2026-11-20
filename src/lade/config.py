"""Paths and fixed settings shared by every step. Change values here, not inside scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"            # put delivery_sh.csv in raw/delivery/, pickup_sh.csv in raw/pickup/
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "figures"

def load_frame(stem):
    """data/processed/<stem>.pkl, or <stem>.parquet when the pickle is absent (Colab bundle,
    written by scripts/export_for_colab.py because pandas-3 pickles may not load on older pandas)."""
    import pandas as pd
    if (PROC / f"{stem}.pkl").exists():
        return pd.read_pickle(PROC / f"{stem}.pkl")
    return pd.read_parquet(PROC / f"{stem}.parquet")


DATASETS = ("delivery", "pickup")
YEAR = 2022              # confirmed: official LaDe README gives 2022-05-01 ~ 2022-10-31. Weekday features are valid.
MAX_N = 25               # paper: snapshots with at most 25 unfinished packages
SHANGHAI_BOX = dict(lng=(120.85, 122.20), lat=(30.65, 31.90))
GPS_MAX_KM = 5.0         # finish GPS farther than this from its stop is treated as a bad fix
CENTER = (121.4737, 31.2304)   # People's Square (approximate), used for distance rings

# Chronological 6:2:2 split BY DAYS (decided 2026-11-14). (last train day, last validation day), MM-DD.
SPLIT_CUTS = {"delivery": ("08-18", "09-24"), "pickup": ("08-22", "09-26")}

# Paper (arXiv 2306.10675 v3) Shanghai targets, used by the reproduction gate.
PAPER = {
    "route_pickup": {  # Table 3
        "TimeGreedy": dict(hr3=59.81, krc=39.93, lsd=5.20, ed=2.24),
        "DistanceGreedy": dict(hr3=61.07, krc=42.84, lsd=5.35, ed=1.94),
        "OR-Tools": dict(hr3=62.50, krc=44.81, lsd=4.69, ed=1.88),
        "LightGBM": dict(hr3=70.63, krc=54.48, lsd=3.27, ed=1.92),
        "FDNET": dict(hr3=69.05, krc=52.72, lsd=4.08, ed=1.86),
        "DeepRoute": dict(hr3=71.66, krc=56.20, lsd=3.26, ed=1.86),
        "CPRoute": dict(hr3=71.73, krc=56.17, lsd=3.39, ed=1.84),
        "Graph2Route": dict(hr3=71.69, krc=56.53, lsd=3.12, ed=1.86),
        "M2G4RTP": dict(hr3=71.73, krc=56.34, lsd=3.16, ed=1.86),
        "DRL4Route": dict(hr3=72.18, krc=57.20, lsd=3.06, ed=1.84),
    },
    "eta_delivery": {  # Table 4
        "SPEED": dict(mae=26.68, rmse=31.31, acc20=52.57),
        "KNN": dict(mae=25.22, rmse=29.57, acc20=65.71),
        "LightGBM": dict(mae=17.24, rmse=20.40, acc20=67.44),
        "FDNET": dict(mae=18.81, rmse=21.15, acc20=64.30),
        "MLP": dict(mae=16.16, rmse=19.31, acc20=72.17),
        "RankETPA": dict(mae=15.76, rmse=19.13, acc20=73.67),
        "M2G4RTP": dict(mae=8.23, rmse=9.59, acc20=91.01),
    },
}

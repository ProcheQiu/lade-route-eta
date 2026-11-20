"""Turn cleaned rows into snapshot samples (one per finish event of a courier-day).

A snapshot at query time q (minutes) right after the courier finishes package j holds every package
with accept_time <= q < finish_time. Only snapshots with 1..MAX_N open packages are kept.

Outputs two tables, joined on `sid`:
  snaps: one row per snapshot  (courier, day, q, courier position, context, split)
  nodes: one row per open package (features known at q + labels `rank` and `eta`)

LEAKAGE CONTRACT (checked by tests/test_leakage.py):
  * a node's features use only: stop coords, AOI, accept time, time window, and the courier position
  * courier position = finish GPS (or stop coords) of the package finished AT q, never later
  * `rank`, `eta`, `finish_min` are labels; they must never be used as model inputs
"""
import numpy as np
import pandas as pd

from . import config as C
from .clean import split_of

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LNG = 111.32 * np.cos(np.radians(31.2))

# model input columns (node level / snapshot level). Labels are deliberately absent.
NODE_NUM = ["dx_km", "dy_km", "dist_km", "min_since_accept", "accept_order", "same_aoi_last",
            "to_window_start", "to_window_end", "has_window"]
NODE_CAT = ["aoi_id", "aoi_type"]
SNAP_NUM = ["hour", "n_open", "n_done"]
LABELS = ["rank", "eta"]


def _minutes(s: pd.Series) -> np.ndarray:
    """Minutes since 1970 as float64 (NaT -> nan)."""
    v = s.values.astype("datetime64[m]").astype("float64")
    v[s.isna().values] = np.nan
    return v


def build(df: pd.DataFrame, name: str, max_n: int = C.MAX_N):
    """df must be the output of clean.clean (sorted by courier, ds, finish_time)."""
    is_p = name == "pickup"
    A, F = _minutes(df.accept_time), _minutes(df.finish_time)
    lng, lat = df.lng.values, df.lat.values
    flng, flat = df.finish_gps_lng.values, df.finish_gps_lat.values
    aoi = df.aoi_id.values
    cid_all, ds_all = df.courier_id.values, df.ds.values

    starts = np.flatnonzero(np.r_[True, (cid_all[1:] != cid_all[:-1]) | (ds_all[1:] != ds_all[:-1])])
    ends = np.r_[starts[1:], len(df)]

    s_rows, n_rows, n_sid = [], [], []
    sid = 0
    stats = {"query_events": 0, "kept": 0, "empty": 0, "too_many": 0}
    for s0, e0 in zip(starts, ends):
        a = A[s0:e0]
        for j in range(e0 - s0 - 1):           # query right after the j-th finish (not after the last one)
            q = F[s0 + j]
            stats["query_events"] += 1
            later = np.arange(s0 + j + 1, e0)  # finish times are unique and sorted -> these finish after q
            sel = later[a[j + 1:] <= q]        # ...and were already accepted at q
            n = len(sel)
            if n == 0:
                stats["empty"] += 1
                continue
            if n > max_n:
                stats["too_many"] += 1
                continue
            r = s0 + j                         # the package just finished at q
            px, py = (flng[r], flat[r]) if not np.isnan(flng[r]) else (lng[r], lat[r])
            s_rows.append((sid, cid_all[r], ds_all[r], q, px, py, aoi[r], j + 1, n, r))
            n_rows.append(sel)
            n_sid.append(np.full(n, sid))
            sid += 1
    stats["kept"] = sid

    snaps = pd.DataFrame(s_rows, columns=["sid", "courier_id", "ds", "q", "pos_lng", "pos_lat",
                                          "last_aoi", "n_done", "n_open", "src_row"])
    rows = np.concatenate(n_rows)
    nodes = pd.DataFrame({"sid": np.concatenate(n_sid), "row": rows})
    sn = snaps.set_index("sid")
    q = sn.q.values[nodes.sid.values]
    nodes["order_id"] = df.order_id.values[rows]
    nodes["aoi_id"] = aoi[rows]
    nodes["aoi_type"] = df.aoi_type.values[rows]
    nodes["lng"], nodes["lat"] = lng[rows], lat[rows]
    nodes["dx_km"] = (lng[rows] - sn.pos_lng.values[nodes.sid.values]) * KM_PER_DEG_LNG
    nodes["dy_km"] = (lat[rows] - sn.pos_lat.values[nodes.sid.values]) * KM_PER_DEG_LAT
    nodes["dist_km"] = np.hypot(nodes.dx_km, nodes.dy_km)
    nodes["accept_min"] = A[rows]
    nodes["min_since_accept"] = q - A[rows]
    nodes["accept_order"] = nodes.groupby("sid").accept_min.rank(method="first").astype(np.int16)
    nodes["same_aoi_last"] = (aoi[rows] == sn.last_aoi.values[nodes.sid.values]).astype(np.int8)
    if is_p:
        nodes["to_window_start"] = _minutes(df.time_window_start)[rows] - q
        nodes["to_window_end"] = _minutes(df.time_window_end)[rows] - q
        nodes["has_window"] = np.int8(1)
    else:
        nodes["to_window_start"] = 0.0
        nodes["to_window_end"] = 0.0
        nodes["has_window"] = np.int8(0)
    # labels
    nodes["finish_min"] = F[rows]
    nodes["eta"] = F[rows] - q
    nodes["rank"] = nodes.groupby("sid").cumcount().astype(np.int16) + 1   # rows are in finish order

    snaps["hour"] = (snaps.q % 1440) / 60.0
    snaps["ds_date"] = pd.to_datetime(f"{C.YEAR}" + snaps.ds.astype(str).str.zfill(4), format="%Y%m%d")
    snaps["split"] = split_of(snaps.ds_date, name).values
    return snaps, nodes, stats


def courier_stats(df_clean: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-courier history from the TRAINING days only (never val/test)."""
    tr = df_clean[split_of(df_clean.ds_date, name) == "train"]
    g = tr.groupby("courier_id")
    out = pd.DataFrame({
        "c_days": g.ds.nunique(),
        "c_tasks_per_day": g.size() / g.ds.nunique(),
        "c_median_dur": g.dur_min.median(),
    })
    # median minutes between consecutive finishes within a courier-day
    gap = tr.groupby(["courier_id", "ds"]).finish_time.diff().dt.total_seconds() / 60
    out["c_median_gap"] = gap.groupby(tr.courier_id).median()
    return out.reset_index()


COURIER_NUM = ["c_known", "c_days", "c_tasks_per_day", "c_median_dur", "c_median_gap"]


def attach_courier_stats(snaps: pd.DataFrame, cstats: pd.DataFrame) -> pd.DataFrame:
    """Unknown couriers get training-wide medians and c_known = 0."""
    s = snaps.merge(cstats, on="courier_id", how="left")
    s["c_known"] = s.c_days.notna().astype(np.int8)
    for c in COURIER_NUM[1:]:
        s[c] = s[c].fillna(cstats[c].median())
    return s

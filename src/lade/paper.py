"""Paper-compatible ("official") sample construction, ported from github.com/wenhaomin/LaDe.

This reproduces the official data pipeline so our numbers can be compared with the paper's tables.
Our own definitions (snapshots.py) stay the default "full-route" setting; see docs/alignment_notes.md
for every difference with file/line references in the official repo.

Official pipeline (route_prediction/data/{preprocess,dataset}.py, time_prediction/data/*_delivery.py):
  preprocess  : minute-of-day clock, sort by (ds, courier, finish), GPS-drift filter (speed > 500 m/min
                on two consecutive rows), drop a row at the same spot < 1 min after the previous one.
  open set    : pickup  -> accept < now < finish        (strict on both sides)
                delivery-> accept < now and finished later in the day's finish order
  windows     : each courier-day is cut into windows of 24 consecutive finishes, stride T = 12;
                at most T = 12 sample steps per window; a step is the moment a package is finished.
  per step    : drop open tasks finishing outside the window (in TEST mode the whole window is dropped),
                drop open tasks whose time label is <= 0 or > 120 min (label_range), require more than
                N_min open tasks (pickup N_min = 2, delivery N_min = 1).
  labels      : ETA label = finish - now for every remaining open task; the ROUTE label (and the set of
                packages that ETA is scored on) is truncated to packages finished before the next new
                package is accepted by the courier within the window (`close_accept_time`).
  split       : chronological by day, 60/20/20 of the dates, with one gap day dropped before val and test.

Leakage notes (paper mode only, all inherited from the official code and kept on purpose for
reproduction): the 120-min label filter and the label truncation use future information to choose which
packages enter a sample. Courier statistics are computed on training days only here (official: all days),
unless courier_stats="all" is passed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T_STEPS = 12          # official --T
N_MAX = 25            # official len_range[1]; window length = N_MAX - 1 = 24
LABEL_RANGE = (0, 120)
N_MIN = {"pickup": 2, "delivery": 1}
FINISH_COL = {"pickup": "pickup_time", "delivery": "delivery_time"}

# WGS84 local-radius distance (meters). Error vs geopy.geodesic < 0.05% at city scale.
_A, _E2 = 6378137.0, 6.69437999014e-3


def geo_m(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = (np.asarray(x, dtype=float) for x in (lat1, lng1, lat2, lng2))
    phi = np.radians((lat1 + lat2) / 2)
    s = 1 - _E2 * np.sin(phi) ** 2
    M = _A * (1 - _E2) / s ** 1.5
    N = _A / np.sqrt(s)
    return np.hypot(M * np.radians(lat2 - lat1), N * np.cos(phi) * np.radians(lng2 - lng1))


def _mod(s: pd.Series):
    """'MM-DD HH:MM:SS' -> (MMDD int, minute of day float). Official time2min()."""
    s = s.astype(str)
    date = s.str.slice(0, 5).str.replace("-", "", regex=False)
    h = s.str.slice(6, 8).astype(float); m = s.str.slice(9, 11).astype(float); sec = s.str.slice(12, 14).astype(float)
    return date, (60 * h + m + sec / 60).values


def official_split(ds: pd.Series, train_ratio=0.6, val_ratio=0.2) -> pd.Series:
    """Official get_data(): dates sorted; one gap day dropped before val and before test ('gap')."""
    days = np.array(sorted(pd.unique(ds)))
    n = len(days)
    a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    lab = {d: "train" for d in days[:a]}
    lab.update({d: "val" for d in days[a + 1:b]})
    lab.update({d: "test" for d in days[int(n * 0.8) + 1:]})
    return ds.map(lab).fillna("gap")


def preprocess(raw: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    """Port of official pre_process()/drop_unnormal() on the raw Shanghai CSV. Returns (df, log)."""
    log = {"raw_rows": len(raw)}
    fcol = FINISH_COL[name]
    df = raw.drop_duplicates().copy()
    log["after_drop_duplicates"] = len(df)
    fdate, df["finish_min"] = _mod(df[fcol])
    df["_fdate"] = fdate.values
    if name == "pickup":
        df = df.sort_values(["ds", "courier_id", "finish_min"], kind="stable")
    else:  # delivery is sorted by the raw time string (date first), clock is still minute-of-day
        df = df.sort_values(["ds", "courier_id", fcol], kind="stable")
    df = df.reset_index(drop=True)
    adate, amin = _mod(df.accept_time)
    df["accept_min"] = np.where(adate.values != df._fdate.values, amin - 1440, amin)
    if name == "pickup":
        _, et = _mod(df.time_window_end)
        df["promise_min"] = np.where(df.time_window_start.isna(), 1440.0, et)

    grp = (df.ds.astype(str) + "_" + df.courier_id.astype(str)).values
    first = np.r_[True, grp[1:] != grp[:-1]]

    def to_last(d):
        pf = np.r_[d.finish_min.values[:1], d.finish_min.values[:-1]]
        plat = np.r_[d.lat.values[:1], d.lat.values[:-1]]; plng = np.r_[d.lng.values[:1], d.lng.values[:-1]]
        f = np.r_[True, d._g.values[1:] != d._g.values[:-1]]
        pf[f] = d.finish_min.values[f]; plat[f] = d.lat.values[f]; plng[f] = d.lng.values[f]
        return d.finish_min.values - pf, np.floor(geo_m(plat, plng, d.lat.values, d.lng.values)).astype(int)

    df["_g"] = grp
    df["time_to_last"], df["dis_to_last"] = to_last(df)

    # GPS drift: drop row i if rows i and i+1 (same courier-day) both have speed > 500 m/min
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.where(df.time_to_last.values != 0, df.dis_to_last.values / df.time_to_last.values, 0)
    flag = spd > 500
    nxt_same = np.r_[grp[1:] == grp[:-1], False]
    nxt_flag = np.r_[flag[1:], False]
    drift = flag & nxt_flag & nxt_same
    log["drop_gps_drift"] = int(drift.sum())
    df = df[~drift].reset_index(drop=True)

    cf = courier_features(df)
    rm = cf[(cf.work_days < 5) & (cf.dis_avg_day < 50) & (cf.order_avg_day < 5) & (cf.time_avg_order < 5)
            & (cf.dis_avg_order < 20)].index
    log["drop_couriers"] = int(len(rm))
    grp = df._g.values
    first = np.r_[True, grp[1:] != grp[:-1]]
    rm_c = df.courier_id.isin(rm).values
    similar = (~first) & (df.dis_to_last.values == 0) & (df.time_to_last.values < 1) & ~rm_c
    log["drop_courier_rows"] = int(rm_c.sum())
    log["drop_same_spot_within_1min"] = int(similar.sum())
    df = df[~(rm_c | similar)].reset_index(drop=True)
    df["time_to_last"], df["dis_to_last"] = to_last(df)   # official recomputes them after filtering
    df["index"] = np.arange(1, len(df) + 1)
    log["rows"] = len(df)
    return df, log


def courier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Official courier_info(): per courier totals (index = courier_id)."""
    g = df.groupby("courier_id")
    out = pd.DataFrame({"order_sum": g.size(), "dis_sum": g.dis_to_last.sum(), "work_days": g.ds.nunique(),
                        "time_avg_order": g.time_to_last.mean(), "dis_avg_order": g.dis_to_last.mean(),
                        "time_sum": g.time_to_last.sum()})
    out["order_avg_day"] = out.order_sum / out.work_days
    out["dis_avg_day"] = out.dis_sum / out.work_days
    out["speed_avg_order"] = np.where(out.time_sum != 0, out.dis_sum / out.time_sum.replace(0, np.nan), 5.0)
    return out


def build(df: pd.DataFrame, name: str, split: pd.Series, courier_stats: str = "train", seed: int = 1):
    """Port of official PickupDataset/DeliveryDataset.get_features (route + time fields only).

    df     : output of preprocess()
    split  : per-row 'train'/'val'/'test'/'gap' (test-mode window dropping applies to 'test' rows)
    Returns (snaps, nodes, stats). One snap row per sample step, one node row per open package.
    nodes columns: slot (shuffled position inside the window, 0 = window's first order), rank (1..n,
    true finish order), in_label (rank <= label_len), eta (minutes), features of the official V tensor,
    d_start (int meters from the start stop) and d_start_off (official E_abs_dis[start_idx] entry,
    which carries the official code's row-overwrite quirk; used only by the *-official rule baselines).
    """
    is_p = name == "pickup"
    n_min = N_MIN[name]
    W = N_MAX - 1
    rng = np.random.default_rng(seed)

    split = np.asarray(split)
    if courier_stats == "train":
        cf = courier_features(df[split == "train"])
    else:
        cf = courier_features(df)
    dad = df.courier_id.map(cf.dis_avg_day).fillna(0.0).values          # unknown courier -> rel dist 0
    spd = df.courier_id.map(cf.speed_avg_order).values
    spd = np.where(np.isnan(spd), np.nanmedian(cf.speed_avg_order.values), spd)

    F, A = df.finish_min.values, df.accept_min.values
    lat, lng = df.lat.values, df.lng.values
    P = df.promise_min.values if is_p else None
    grp = df._g.values
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    ends = np.r_[starts[1:], len(df)]

    S, Nd = [], []      # snap tuples / node arrays
    sid = 0; wid = 0
    stats = dict(windows=0, windows_dropped_test=0, steps=0, steps_label0=0)
    for s0, e0 in zip(starts, ends):
        if split[s0] == "gap":
            continue
        test_mode = split[s0] == "test"
        L = e0 - s0
        a, f = A[s0:e0], F[s0:e0]
        if is_p:
            O = (a[None, :] < f[:, None]) & (f[None, :] > f[:, None])
        else:
            O = (a[None, :] < f[:, None]) & np.triu(np.ones((L, L), bool), 1)
        win_num = (2 + (L - W) // T_STEPS) if L >= W else 1
        for w in range(win_num):
            o = w * T_STEPS
            Lw = (L - o) if w == win_num - 1 else min(W, L - o)
            if Lw < n_min or Lw <= 0:
                continue
            stats["windows"] += 1
            perm = np.r_[0, 1 + rng.permutation(Lw - 1)]       # local index -> slot (first order fixed at 0)
            win_acc = a[o:o + Lw]
            steps = []; drop = False; t = 0
            for st in range(Lw):
                if t == T_STEPS:
                    break
                j = o + st
                todo = np.flatnonzero(O[j])
                if len(todo) == 0:
                    continue
                c = todo - o
                todo = todo[c > st]; c = c[c > st]
                if (c >= W).any():
                    if test_mode:
                        drop = True; break
                    todo = todo[c < W]
                tl = f[todo] - f[j]
                todo = todo[(tl > LABEL_RANGE[0]) & (tl <= LABEL_RANGE[1])]
                if len(todo) <= n_min:
                    continue
                later = win_acc[win_acc > f[j]]
                close = later.min() if len(later) else np.inf
                lab_len = int((f[todo] <= close).sum())       # todo sorted by finish order -> prefix
                steps.append((j, todo, lab_len))
                t += 1
            if drop:
                stats["windows_dropped_test"] += 1
                continue
            if not steps:
                continue
            # official E_abs_dis quirk: row 0 holds, per task, distance from the start of the LAST step
            # whose open set contained that task; rows of starts that were never a valid task are zero.
            last_start = {}
            valid = set()
            for j, todo, _ in steps:
                for k in todo:
                    last_start[k] = j
                valid.update(todo.tolist())
            for j, todo, lab_len in steps:
                g_todo = s0 + todo; gj = s0 + j
                d = np.floor(geo_m(lat[gj], lng[gj], lat[g_todo], lng[g_todo]))
                sslot = perm[j - o]
                if sslot == 0:
                    ls = s0 + np.array([last_start[k] for k in todo])
                    d_off = np.floor(geo_m(lat[ls], lng[ls], lat[g_todo], lng[g_todo]))
                elif j in valid:
                    d_off = d
                else:
                    d_off = np.zeros_like(d)
                n = len(todo)
                S.append((sid, wid, gj, df.courier_id.values[gj], df.ds.values[gj], f[j], n, lab_len, sslot,
                          split[s0], spd[gj]))
                Nd.append(dict(sid=np.full(n, sid), row=g_todo, slot=perm[todo - o], rank=np.arange(1, n + 1),
                               eta=f[todo] - f[j], d_start=d, d_start_off=d_off, rel_dis=np.where(
                                   dad[g_todo] == 0, 0.0, d / np.where(dad[g_todo] == 0, 1, dad[g_todo]) * 100)))
                stats["steps"] += 1; stats["steps_label0"] += lab_len == 0
                sid += 1
            wid += 1
    snaps = pd.DataFrame(S, columns=["sid", "wid", "start_row", "courier_id", "ds", "q", "n_open", "label_len",
                                     "start_slot", "split", "c_speed"])
    nodes = pd.DataFrame({k: np.concatenate([d[k] for d in Nd]) for k in Nd[0]})
    nodes["in_label"] = nodes["rank"].values <= snaps.label_len.values[nodes.sid.values]
    r = nodes.row.values
    q = snaps.q.values[nodes.sid.values]
    nodes["accept_min"] = A[r]; nodes["lng"] = lng[r]; nodes["lat"] = lat[r]
    nodes["aoi_type"] = df.aoi_type.values[r]; nodes["aoi_id"] = df.aoi_id.values[r]
    if is_p:
        nodes["pt_minus_at"] = P[r] - A[r]
        nodes["pt_minus_now"] = P[r] - q
    st = snaps.start_row.values
    snaps["start_accept"] = A[st]; snaps["start_lng"] = lng[st]; snaps["start_lat"] = lat[st]
    if is_p:
        snaps["start_pt"] = P[st]
    snaps["start_ft"] = F[st]
    return snaps, nodes, stats


# official V tensor columns (pickup: 8, delivery: 6) and start_fea
V_COLS = {"pickup": ["accept_min", "lng", "lat", "aoi_type", "rel_dis", "d_start", "pt_minus_at", "pt_minus_now"],
          "delivery": ["accept_min", "lng", "lat", "aoi_type", "rel_dis", "d_start"]}
START_COLS = {"pickup": ["start_accept", "start_lng", "start_lat", "start_pt", "start_ft"],
              "delivery": ["start_accept", "start_lng", "start_lat", "start_ft"]}


def slot_tensor(snaps: pd.DataFrame, nodes: pd.DataFrame, cols) -> np.ndarray:
    """[n_snaps, N_MAX, len(cols)] features at their shuffled slots; empty slots are zero (official mask)."""
    pos = pd.Series(np.arange(len(snaps)), index=snaps.sid.values)
    i = pos.reindex(nodes.sid.values).values
    X = np.zeros((len(snaps), N_MAX, len(cols)), dtype=np.float64)
    X[i, nodes.slot.values, :] = nodes[cols].values.astype(float)
    return X

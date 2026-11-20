"""Training-free baselines from the paper: TimeGreedy, DistanceGreedy (route) and SPEED (ETA).
All are vectorised over snapshots that share the same number of open packages n."""
import numpy as np
import pandas as pd


def by_n(nodes: pd.DataFrame, cols):
    """Yield (n, sids, {col: [B, n] array}) with each row in node (= true-rank) order."""
    d = nodes.sort_values(["sid", "rank"])
    n_of = d.groupby("sid").size()
    for n, sids in n_of.groupby(n_of).groups.items():
        sub = d[d.sid.isin(sids)]
        yield n, sub.index.values, {c: sub[c].values.reshape(-1, n) for c in cols}


def greedy_nearest(X, Y):
    """Nearest-unvisited-first from the courier at (0, 0). Returns order [B, n] of local indices."""
    B, n = X.shape
    px, py = np.zeros(B), np.zeros(B)
    seen = np.zeros((B, n), bool)
    order = np.empty((B, n), int)
    rows = np.arange(B)
    for s in range(n):
        d = np.hypot(X - px[:, None], Y - py[:, None])
        d[seen] = np.inf
        k = d.argmin(1)
        order[:, s] = k
        seen[rows, k] = True
        px, py = X[rows, k], Y[rows, k]
    return order


def ranks_from_order(order):
    B, n = order.shape
    r = np.empty_like(order)
    r[np.arange(B)[:, None], order] = np.arange(1, n + 1)
    return r


def distance_greedy(nodes):
    out = pd.Series(0, index=nodes.index, dtype=np.int64)
    for n, idx, a in by_n(nodes, ["dx_km", "dy_km"]):
        out.loc[idx] = ranks_from_order(greedy_nearest(a["dx_km"], a["dy_km"])).ravel()
    return out


def time_greedy(nodes):
    """Pickup: earliest window end first. Delivery has no windows: earliest accepted first."""
    key = np.where(nodes.has_window.values == 1, nodes.to_window_end.values, nodes.accept_order.values)
    tmp = nodes.assign(_key=key)
    out = pd.Series(0, index=nodes.index, dtype=np.int64)
    for n, idx, a in by_n(tmp, ["_key", "accept_order"]):
        order = np.lexsort((a["accept_order"], a["_key"]), axis=1) if n > 1 else np.zeros((len(a["_key"]), 1), int)
        out.loc[idx] = ranks_from_order(order).ravel()
    return out


def courier_speed(train_nodes: pd.DataFrame, train_snaps: pd.DataFrame) -> pd.Series:
    """km per minute from the first package of each training snapshot (distance / time to finish it)."""
    f = train_nodes[(train_nodes["rank"] == 1) & (train_nodes.eta > 0)]
    f = f.merge(train_snaps[["sid", "courier_id"]], on="sid")
    v = (f.dist_km / f.eta).clip(0.005, 1.0)
    return v.groupby(f.courier_id).median()


def speed_eta(nodes, snaps, speed: pd.Series):
    """SPEED: cumulative distance along the nearest-first route / the courier's historical speed."""
    v_glob = float(speed.median())
    cid = snaps.set_index("sid").courier_id
    out = pd.Series(np.nan, index=nodes.index)
    for n, idx, a in by_n(nodes, ["dx_km", "dy_km", "sid"]):
        X, Y = a["dx_km"], a["dy_km"]
        order = greedy_nearest(X, Y)
        B = X.shape[0]; rows = np.arange(B)[:, None]
        ox, oy = X[rows, order], Y[rows, order]
        step = np.hypot(np.diff(np.c_[np.zeros(B), ox], axis=1), np.diff(np.c_[np.zeros(B), oy], axis=1))
        cum = np.cumsum(step, 1)
        dist_node = np.empty_like(cum); dist_node[rows, order] = cum
        v = speed.reindex(cid.loc[a["sid"][:, 0]].values).fillna(v_glob).values
        out.loc[idx] = (dist_node / v[:, None]).ravel()
    return out

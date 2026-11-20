"""Gradient-boosted-tree baselines (paper: "LightGBM").

ETA  : one regressor, one row per open package.
Route: next-stop classifier applied greedily (the paper's LightGBM / Osquare style). Training rows are
       built with teacher forcing: at step s the courier stands at the true (s-1)-th stop and every
       remaining package is a candidate; label = 1 for the true next stop.

Backend: LightGBM if installed (required for the reproduction gate), otherwise sklearn's
HistGradientBoosting (same algorithm family; numbers are a preview only).
"""
import numpy as np
import pandas as pd

from ..snapshots import NODE_NUM, SNAP_NUM, COURIER_NUM
from .rules import by_n

try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:  # pragma: no cover
    lgb = None
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    BACKEND = "sklearn-hgb"

ETA_FEATS = NODE_NUM + ["aoi_type"] + SNAP_NUM + ["weekday"] + COURIER_NUM


def add_context(nodes: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    s = snaps.set_index("sid")
    cols = SNAP_NUM + COURIER_NUM + ["split", "courier_id"]
    s = s.assign(weekday=s.ds_date.dt.weekday)          # year is 2022 (official LaDe README)
    out = nodes.join(s[cols + ["weekday"]], on="sid")
    return out


def _reg(seed=0):
    if BACKEND == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=50,
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                                         random_state=seed)


def _clf(seed=0):
    if BACKEND == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=50,
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                                          random_state=seed)


def _fit(model, Xtr, ytr, Xva, yva):
    if BACKEND == "lightgbm":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(Xtr, ytr)   # sklearn uses an internal validation split for early stopping
    return model


# ---------------------------------------------------------------- ETA
def fit_eta(train, val, seed=0):
    m = _reg(seed)
    return _fit(m, train[ETA_FEATS].values, train.eta.values, val[ETA_FEATS].values, val.eta.values)


def predict_eta(model, nodes):
    return np.clip(model.predict(nodes[ETA_FEATS].values), 0, None)


# ---------------------------------------------------------------- route
ROUTE_STATIC = ["min_since_accept", "accept_order", "to_window_start", "to_window_end", "has_window", "aoi_type",
                "dist_km"]
ROUTE_CTX = ["hour", "n_open", "c_known", "c_median_gap"]
ROUTE_FEATS = ["step", "n_left", "cur_dist", "cur_dx", "cur_dy", "rank_dist_left", "same_aoi_cur"] + ROUTE_STATIC + ROUTE_CTX


def _step_features(a, cx, cy, caoi, s, left):
    """Feature tensor [B, n, F] for one decision step; `left` = bool mask of remaining candidates."""
    B, n = a["dx_km"].shape
    dx, dy = a["dx_km"] - cx[:, None], a["dy_km"] - cy[:, None]
    d = np.hypot(dx, dy)
    dm = np.where(left, d, np.inf)
    rank_d = dm.argsort(1).argsort(1).astype(float)          # distance rank among remaining
    same = (a["aoi_id"] == caoi[:, None]).astype(float)
    cols = [np.full((B, n), s, float), np.repeat(left.sum(1, keepdims=True), n, 1).astype(float), d, dx, dy,
            rank_d, same] + [a[c].astype(float) for c in ROUTE_STATIC] + [a[c].astype(float) for c in ROUTE_CTX]
    return np.stack(cols, -1)


def _arrays(nodes):
    return list(by_n(nodes, ["dx_km", "dy_km", "aoi_id"] + ROUTE_STATIC + ROUTE_CTX))


def route_training_rows(nodes, max_snapshots=None, seed=0):
    """Teacher-forced decision rows. Returns X [rows, F], y [rows]."""
    if max_snapshots:
        sids = nodes.sid.drop_duplicates().sample(min(max_snapshots, nodes.sid.nunique()), random_state=seed)
        nodes = nodes[nodes.sid.isin(sids)]
    Xs, ys = [], []
    for n, idx, a in _arrays(nodes):
        if n < 2:
            continue
        B = a["dx_km"].shape[0]
        cx, cy, caoi = np.zeros(B), np.zeros(B), np.full(B, -1)
        for s in range(n - 1):                               # the last step has one candidate: no decision
            left = np.zeros((B, n), bool); left[:, s:] = True
            F = _step_features(a, cx, cy, caoi, s, left)[:, s:, :]
            Xs.append(F.reshape(-1, F.shape[-1]).astype(np.float32))   # float32: full-route rows are large
            y = np.zeros((B, n - s)); y[:, 0] = 1               # true next stop is local index s
            ys.append(y.ravel().astype(np.int8))
            cx, cy, caoi = a["dx_km"][:, s], a["dy_km"][:, s], a["aoi_id"][:, s]
    return np.concatenate(Xs), np.concatenate(ys)


def fit_route(train_nodes, val_nodes, max_snapshots=None, seed=0):
    Xtr, ytr = route_training_rows(train_nodes, max_snapshots, seed)
    Xva, yva = route_training_rows(val_nodes, max_snapshots and max(1000, max_snapshots // 4), seed)
    return _fit(_clf(seed), Xtr, ytr, Xva, yva), len(ytr)


def predict_route(model, nodes):
    """Greedy decoding with the next-stop classifier. Returns pred_rank aligned to nodes.index."""
    out = pd.Series(0, index=nodes.index, dtype=np.int64)
    for n, idx, a in _arrays(nodes):
        B = a["dx_km"].shape[0]
        if n == 1:
            out.loc[idx] = 1; continue
        cx, cy, caoi = np.zeros(B), np.zeros(B), np.full(B, -1)
        left = np.ones((B, n), bool)
        rank = np.zeros((B, n), int); rows = np.arange(B)
        for s in range(n):
            F = _step_features(a, cx, cy, caoi, s, left)
            p = model.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(B, n)
            p[~left] = -1
            k = p.argmax(1)
            rank[rows, k] = s + 1
            left[rows, k] = False
            cx, cy, caoi = a["dx_km"][rows, k], a["dy_km"][rows, k], a["aoi_id"][rows, k]
        out.loc[idx] = rank.ravel()
    return out

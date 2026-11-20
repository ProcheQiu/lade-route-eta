"""Baselines on paper-mode samples (lade.paper), ported from github.com/wenhaomin/LaDe.

route_prediction/algorithm/basic/basic_model.py : TimeGreedy, DistanceGreedy (OR-Tools: see ortools_route.py)
route_prediction/algorithm/osqure/train.py      : "LightGBM" route = LGBMClassifier over the 25 shuffled slots,
                                                  teacher-forced on the truncated label, greedy decoding
time_prediction/algorithm/lgb/train.py          : "LightGBM" ETA = MultiOutputRegressor over 25 slots
                                                  (the official code actually uses XGBRegressor)
time_prediction/algorithm/speed/speed.py        : SPEED = straight-line distance start->package / courier speed

Ties are broken by the lowest slot index, as torch.argmin / np.argmax do in the official code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paper as PP
from ..paper import geo_m


def _groups(nodes: pd.DataFrame, cols):
    """Yield (index, {col: [B, n]}) per open-set size n, rows within a sample ordered by slot."""
    d = nodes.sort_values(["sid", "slot"])
    n_of = d.groupby("sid").size()
    for n, sids in n_of.groupby(n_of).groups.items():
        sub = d[d.sid.isin(sids)]
        yield sub.index.values, {c: sub[c].values.reshape(-1, n) for c in cols}


def _ranks(order):
    B, n = order.shape
    r = np.empty_like(order)
    r[np.arange(B)[:, None], order] = np.arange(1, n + 1)
    return r


def time_greedy(nodes):
    """Earliest promised (window-end) time first: argmin of V[..., 7] = promise - now."""
    out = pd.Series(0, index=nodes.index, dtype=np.int64)
    for idx, a in _groups(nodes, ["pt_minus_now"]):
        order = np.argsort(a["pt_minus_now"], axis=1, kind="stable")
        out.loc[idx] = _ranks(order).ravel()
    return out


def distance_greedy(nodes, official=True):
    """Nearest-next from the start stop over int-meter geodesic distances.
    official=True uses the official E_abs_dis row of the start node (with its row-overwrite quirk)."""
    first = "d_start_off" if official else "d_start"
    out = pd.Series(0, index=nodes.index, dtype=np.int64)
    for idx, a in _groups(nodes, ["lat", "lng", first]):
        lat, lng = a["lat"], a["lng"]
        B, n = lat.shape
        D = np.floor(geo_m(lat[:, :, None], lng[:, :, None], lat[:, None, :], lng[:, None, :]))
        seen = np.zeros((B, n), bool); order = np.empty((B, n), int); rows = np.arange(B)
        cur = a[first].astype(float)
        for s in range(n):
            c = np.where(seen, 1e9, cur)
            k = c.argmin(1)
            order[:, s] = k; seen[rows, k] = True
            cur = D[rows, k]
        out.loc[idx] = _ranks(order).ravel()
    return out


def speed_eta(nodes, snaps, official=True):
    """Official SPEED: eta = dist(start, package) / courier speed_avg_order (m/min). Not cumulative."""
    col = "d_start_off" if official else "d_start"
    v = snaps.set_index("sid").c_speed.reindex(nodes.sid.values).values
    return nodes[col].values / v


# ------------------------------------------------------------------------------------------- route GBDT
def _route_rows(X, slot_label, label_len, open_mask):
    """Teacher forcing as in osqure/train.py: one row per label position, chosen slots zeroed."""
    feats, ys = [], []
    mask = open_mask.copy()
    Lmax = int(label_len.max())
    for i in range(Lmax):
        act = label_len > i
        Xi = X[act] * mask[act][:, :, None]
        feats.append(Xi.reshape(Xi.shape[0], -1))
        y = slot_label[act, i]
        ys.append(y)
        mask[np.flatnonzero(act), y] = 0
    return np.concatenate(feats), np.concatenate(ys)


def route_arrays(snaps, nodes, name):
    X = PP.slot_tensor(snaps, nodes, PP.V_COLS[name])
    pos = pd.Series(np.arange(len(snaps)), index=snaps.sid.values)
    i = pos.reindex(nodes.sid.values).values
    open_mask = np.zeros((len(snaps), PP.N_MAX), dtype=np.float64)
    open_mask[i, nodes.slot.values] = 1
    slot_label = np.full((len(snaps), PP.N_MAX), PP.N_MAX - 1)
    slot_label[i, nodes["rank"].values - 1] = nodes.slot.values
    return X, open_mask, slot_label


class _XGBSlotClassifier:
    """XGBClassifier needs labels 0..K-1; slots seen in training (1..23) are encoded and decoded here.
    Exposes classes_ / predict_proba like the LightGBM classifier. zeros (empty slots) are treated as
    missing to mirror the official LGBMClassifier(zero_as_missing=True)."""

    def __init__(self, seed=0, **params):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", missing=0.0,
                                       random_state=seed, n_jobs=-1, **params)

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.model.fit(X, yi)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def fit_route_gbdt(snaps, nodes, name, backend="lightgbm", params=None, seed=0):
    keep = snaps.label_len.values >= 1
    X, M, Y = route_arrays(snaps, nodes, name)
    Xr, yr = _route_rows(X[keep], Y[keep], snaps.label_len.values[keep], M[keep])
    if backend == "xgboost":                          # default XGBoost hyperparameters
        clf = _XGBSlotClassifier(seed=seed, **(params or {}))
    else:
        import lightgbm as lgb
        p = dict(random_state=seed, verbose=-1)      # official: LGBMClassifier(use_missing, zero_as_missing)
        p.update(params or {})
        clf = lgb.LGBMClassifier(use_missing=True, zero_as_missing=True, **p)
    clf.fit(Xr, yr)
    return clf, len(yr)


def predict_route_gbdt(clf, snaps, nodes, name):
    """Greedy decoding over all open slots; returns pred_rank aligned to nodes.index."""
    X, M, _ = route_arrays(snaps, nodes, name)
    B = len(snaps)
    mask = M.copy(); rank_of_slot = np.zeros((B, PP.N_MAX), int)
    cls = clf.classes_
    for s in range(int(M.sum(1).max())):
        act = mask.sum(1) > 0
        if not act.any():
            break
        Xi = (X[act] * mask[act][:, :, None]).reshape(act.sum(), -1)
        pr = np.zeros((act.sum(), PP.N_MAX))
        pr[:, cls] = clf.predict_proba(Xi)
        k = ((pr + 0.01) * mask[act]).argmax(1)
        ia = np.flatnonzero(act)
        rank_of_slot[ia, k] = s + 1
        mask[ia, k] = 0
    pos = pd.Series(np.arange(B), index=snaps.sid.values)
    i = pos.reindex(nodes.sid.values).values
    return pd.Series(rank_of_slot[i, nodes.slot.values], index=nodes.index)


# --------------------------------------------------------------------------------------------- ETA GBDT
def eta_arrays(snaps, nodes, name):
    X = PP.slot_tensor(snaps, nodes, PP.V_COLS[name]).reshape(len(snaps), -1)
    S = snaps[PP.START_COLS[name]].values.astype(float)
    pos = pd.Series(np.arange(len(snaps)), index=snaps.sid.values)
    i = pos.reindex(nodes.sid.values).values
    Y = np.zeros((len(snaps), PP.N_MAX))
    lab = nodes.in_label.values                       # official: label only on truncated-label slots
    Y[i[lab], nodes.slot.values[lab]] = nodes.eta.values[lab]
    return np.concatenate([S, X], 1), Y


def fit_eta_gbdt(snaps, nodes, name, backend="lightgbm", params=None, seed=0):
    from sklearn.multioutput import MultiOutputRegressor
    keep = snaps.label_len.values >= 1
    X, Y = eta_arrays(snaps[keep], nodes[nodes.sid.isin(snaps.sid[keep])], name)
    if backend == "xgboost":
        import xgboost as xgb
        base = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=-1, **(params or {}))
    else:
        import lightgbm as lgb
        base = lgb.LGBMRegressor(random_state=seed, verbose=-1, **(params or {}))
    m = MultiOutputRegressor(base).fit(X, Y)
    return m, len(Y)


def predict_eta_gbdt(model, snaps, nodes, name):
    X, _ = eta_arrays(snaps, nodes, name)
    P = model.predict(X)
    pos = pd.Series(np.arange(len(snaps)), index=snaps.sid.values)
    i = pos.reindex(nodes.sid.values).values
    return P[i, nodes.slot.values]

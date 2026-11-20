"""Padded arrays for the deep models. Pure numpy here; torch tensors are made per batch in train_joint.py.

Each snapshot becomes N = MAX_N slots. Slots are SHUFFLED inside every snapshot so slot position never
encodes the true order; the true visiting order is stored separately in `order` (slot indices, -1 = pad).
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..snapshots import NODE_NUM, SNAP_NUM, COURIER_NUM

CTX_NUM = SNAP_NUM + COURIER_NUM          # + weekday one-hot (7) appended in build()


class Vocab:
    """Maps raw ids to 1..K; 0 = unknown. Fitted on the training split only."""

    def __init__(self, values):
        u = pd.unique(pd.Series(values).dropna())
        self.map = pd.Series(np.arange(1, len(u) + 1), index=u)

    def __call__(self, values):
        return pd.Series(values).map(self.map).fillna(0).astype(np.int64).values

    def __len__(self):
        return len(self.map) + 1


class Scaler:
    """Standardize with training mean/sd. A column that is CONSTANT in training (c_known is always 1 there)
    gets sd = 1: dividing by ~0 turned new couriers' c_known = 0 into -1e6 (bug found 2026-11-18)."""

    def __init__(self, x):
        self.mu = np.nanmean(x, 0)
        sd = np.nanstd(x, 0)
        self.sd = np.where(sd < 1e-6, 1.0, sd + 1e-6)

    def __call__(self, x):
        return np.nan_to_num((x - self.mu) / self.sd)


# columns of `ctx` that describe the courier (positions in CTX_NUM; weekday one-hot comes after CTX_NUM)
COURIER_CTX_IDX = np.arange(len(SNAP_NUM), len(SNAP_NUM) + len(COURIER_NUM))


def unknown_courier_values(snaps):
    """Raw COURIER_NUM values an unseen courier gets from snapshots.attach_courier_stats:
    c_known = 0 and the training-courier medians. They are one constant row derived from training days only,
    so reading it off any unknown-courier snapshot leaks nothing; checked to be identical across those rows."""
    unk = snaps.loc[snaps.c_known == 0, COURIER_NUM]
    if len(unk):
        row = unk.iloc[0].values.astype(float)
        assert np.allclose(unk.values.astype(float), row), "unknown couriers should share one fallback row"
        return row
    tr = snaps[snaps.split == "train"].groupby("courier_id")[COURIER_NUM[1:]].first().median()
    return np.r_[0.0, tr.values.astype(float)]


def fit_encoders(nodes, snaps):
    tr_s = snaps[snaps.split == "train"]
    tr_n = nodes[nodes.sid.isin(tr_s.sid)]
    ctx = Scaler(tr_s[CTX_NUM].values.astype(float))
    raw = unknown_courier_values(snaps)
    unk = ((raw - ctx.mu[COURIER_CTX_IDX]) / ctx.sd[COURIER_CTX_IDX]).astype(np.float32)
    return {"aoi": Vocab(tr_n.aoi_id), "aoi_type": Vocab(tr_n.aoi_type), "courier": Vocab(tr_s.courier_id),
            "node": Scaler(tr_n[NODE_NUM].values.astype(float)), "ctx": ctx, "unknown_ctx": unk}


def simulate_unknown_courier(ctx, courier, enc, p, rng):
    """Courier dropout for training: a share p of samples is made to look exactly like an unseen courier
    (courier id 0, c_known = 0, fallback statistics), so the new-courier input is learned, not extrapolated.
    Works on copies; returns (ctx, courier)."""
    drop = rng.random(len(courier)) < p
    if not drop.any():
        return ctx, courier
    ctx, courier = ctx.copy(), courier.copy()
    courier[drop] = 0
    ctx[np.ix_(np.flatnonzero(drop), COURIER_CTX_IDX)] = enc["unknown_ctx"]
    return ctx, courier


def build(nodes, snaps, split, enc, frac=1.0, seed=0, n_max=C.MAX_N):
    """Return a dict of numpy arrays for one split. `frac` < 1 subsamples snapshots (training only)."""
    s = snaps[snaps.split == split]
    if frac < 1.0:
        s = s.sample(frac=frac, random_state=seed)
    s = s.sort_values("sid")
    d = nodes[nodes.sid.isin(s.sid)].sort_values(["sid", "rank"])
    S, N = len(s), n_max
    rng = np.random.default_rng(seed)
    n = s.n_open.values.astype(int)
    starts = np.r_[0, np.cumsum(n)[:-1]]
    # random slot for every node, padded slots at the end of the permutation
    slot = np.empty(len(d), dtype=np.int64)
    order = np.full((S, N), -1, dtype=np.int64)
    for i in range(S):                                     # simple loop: ~1M iterations, a few seconds
        p = rng.permutation(n[i])
        slot[starts[i]:starts[i] + n[i]] = p
        order[i, :n[i]] = p                                 # true k-th stop sits in slot p[k]
    row = np.repeat(np.arange(S), n)

    def pad(values, fill=0.0, dtype=np.float32):
        out = np.full((S, N) + values.shape[1:], fill, dtype=dtype)
        out[row, slot] = values
        return out

    mask = pad(np.ones(len(d)), 0, bool)
    num = pad(enc["node"](d[NODE_NUM].values.astype(float)).astype(np.float32))
    xy = pad(d[["dx_km", "dy_km"]].values.astype(np.float32))
    tw = pad(d[["to_window_start", "to_window_end"]].values.astype(np.float32) / 60.0)   # hours, raw sign kept
    aoi_raw = pad(d.aoi_id.values, -1, np.int64)
    aoi = pad(enc["aoi"](d.aoi_id.values), 0, np.int64)
    aoi_type = pad(enc["aoi_type"](d.aoi_type.values), 0, np.int64)
    eta = pad(d.eta.values.astype(np.float32))
    wd = np.eye(7, dtype=np.float32)[s.ds_date.dt.weekday.values]          # year 2022 confirmed
    ctx = np.concatenate([enc["ctx"](s[CTX_NUM].values.astype(float)).astype(np.float32), wd], 1)
    return {"sid": s.sid.values, "n": n, "mask": mask, "num": num, "xy": xy, "tw": tw, "aoi": aoi,
            "aoi_raw": aoi_raw, "aoi_type": aoi_type, "eta": eta, "order": order, "ctx": ctx,
            "courier": enc["courier"](s.courier_id.values), "c_known": s.c_known.values.astype(np.int64)}


def to_eval_frame(arr, pred_order, pred_eta):
    """Convert slot-level predictions back to node rows for lade.metrics (sid, rank, pred_rank, eta, pred_eta)."""
    S, N = arr["mask"].shape
    true_rank = np.zeros((S, N), np.int64); pred_rank = np.zeros((S, N), np.int64)
    rows = np.arange(S)[:, None]
    steps = np.arange(1, N + 1)[None, :].repeat(S, 0)
    valid_t = arr["order"] >= 0
    true_rank[np.where(valid_t)[0], arr["order"][valid_t]] = steps[valid_t]
    valid_p = pred_order >= 0
    pred_rank[np.where(valid_p)[0], pred_order[valid_p]] = steps[valid_p]
    m = arr["mask"]
    return pd.DataFrame({"sid": np.repeat(arr["sid"], N).reshape(S, N)[m], "rank": true_rank[m],
                         "pred_rank": pred_rank[m], "eta": arr["eta"][m], "pred_eta": pred_eta[m],
                         "c_known": np.repeat(arr["c_known"], N).reshape(S, N)[m]})

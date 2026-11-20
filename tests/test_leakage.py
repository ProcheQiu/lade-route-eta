"""Leakage and correctness checks on built snapshots. Run: python -m pytest tests -q  (or python tests/test_leakage.py)
Needs data/processed/{name}_clean.pkl, _snaps.pkl, _nodes.pkl from scripts/build_snapshots.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from lade import config as C
from lade.snapshots import NODE_NUM, NODE_CAT, SNAP_NUM, LABELS, COURIER_NUM


def _load(name):
    return (pd.read_pickle(C.PROC / f"{name}_clean.pkl"), pd.read_pickle(C.PROC / f"{name}_snaps.pkl"),
            pd.read_pickle(C.PROC / f"{name}_nodes.pkl"))


def check(name, n_brute=200):
    df, sn, nd = _load(name)
    q = sn.set_index("sid").q.reindex(nd.sid).values
    assert (nd.accept_min.values <= q).all(), "a node was accepted after the query time"
    assert (nd.finish_min.values > q).all(), "a node was already finished at the query time"
    src = sn.src_row.values
    fin = df.finish_time.values.astype("datetime64[m]").astype(float)[src]
    assert np.allclose(fin, sn.q.values), "courier position comes from a package not finished exactly at q"
    assert sn.n_open.between(1, C.MAX_N).all()
    assert (nd.groupby("sid").size().values == sn.n_open.values).all()
    inputs = set(NODE_NUM + NODE_CAT + SNAP_NUM + COURIER_NUM)
    assert not inputs & set(LABELS + ["finish_min"]), "a label is listed as a model input"
    # courier stats must come from training days only: a courier first seen after training is unknown
    first = df.groupby("courier_id").ds_date.min().dt.strftime("%m-%d")
    late = set(first[first > C.SPLIT_CUTS[name][0]].index)
    assert sn[sn.courier_id.isin(late)].c_known.eq(0).all(), "courier stats leak from after the training period"
    # brute-force recount of open sets
    rng = np.random.default_rng(0)
    A = df.accept_time.values.astype("datetime64[m]").astype(float)
    F = df.finish_time.values.astype("datetime64[m]").astype(float)
    key = df.groupby(["courier_id", "ds"]).indices
    for s in rng.choice(sn.sid.values, n_brute, replace=False):
        r = sn.iloc[s]
        idx = key[(r.courier_id, r.ds)]
        truth = set(df.order_id.values[idx][(A[idx] <= r.q) & (F[idx] > r.q)])
        assert truth == set(nd.order_id[nd.sid == s]), f"open set mismatch in snapshot {s}"
    return True


def test_delivery():
    assert check("delivery")


def test_pickup():
    assert check("pickup")


if __name__ == "__main__":
    for n in C.DATASETS:
        print(n, "OK" if check(n) else "FAIL")

"""Checks the padded arrays used by the deep model (numpy only; no torch needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from lade import config as C
from lade.models.data import fit_encoders, build, to_eval_frame
from lade.metrics import route_metrics, eta_metrics


def test_roundtrip_pickup():
    sn = pd.read_pickle(C.PROC / "pickup_snaps.pkl"); nd = pd.read_pickle(C.PROC / "pickup_nodes.pkl")
    enc = fit_encoders(nd, sn)
    arr = build(nd, sn, "val", enc, frac=0.01, seed=0)
    S, N = arr["mask"].shape
    assert N == C.MAX_N and (arr["mask"].sum(1) == arr["n"]).all()
    assert ((arr["order"] >= 0).sum(1) == arr["n"]).all()
    # slots are shuffled: the true first stop is not always slot 0
    assert (arr["order"][:, 0] != 0).mean() > 0.3
    # ETA along the true order must increase
    e = np.take_along_axis(arr["eta"], arr["order"].clip(0), 1)
    v = arr["order"] >= 0
    assert all((np.diff(e[i][v[i]]) > 0).all() for i in range(min(S, 2000)))
    # perfect predictions -> perfect metrics
    ev = to_eval_frame(arr, arr["order"], arr["eta"])
    m = route_metrics(ev)
    assert abs(m["hr3"] - 100) < 1e-9 and abs(m["krc"] - 100) < 1e-9 and m["ed"] == 0
    assert eta_metrics(ev)["mae"] == 0
    assert arr["ctx"].shape[1] == 8 + 7 and np.isfinite(arr["num"]).all()


def test_new_courier_inputs_bounded_and_simulated_exactly():
    """Bug 2026-11-18: c_known is constant (1) in training, so new couriers were scaled to -1e6.
    Their inputs must stay bounded, and the training-time simulation must equal a real new courier's input."""
    from lade.models.data import simulate_unknown_courier, COURIER_CTX_IDX
    for name in ["pickup", "delivery"]:
        sn = pd.read_pickle(C.PROC / f"{name}_snaps.pkl"); nd = pd.read_pickle(C.PROC / f"{name}_nodes.pkl")
        enc = fit_encoders(nd, sn)
        te = build(nd, sn, "test", enc, frac=0.02, seed=0)
        assert np.abs(te["ctx"]).max() < 100, np.abs(te["ctx"]).max(0)
        new = te["c_known"] == 0
        assert new.any() and (te["courier"][new] == 0).all()
        assert np.allclose(te["ctx"][new][:, COURIER_CTX_IDX], enc["unknown_ctx"], atol=1e-5)
        tr = build(nd, sn, "train", enc, frac=0.01, seed=0)
        assert (tr["courier"] > 0).all()
        ctx, cid = simulate_unknown_courier(tr["ctx"], tr["courier"], enc, 1.0, np.random.default_rng(0))
        assert (cid == 0).all() and np.allclose(ctx[:, COURIER_CTX_IDX], enc["unknown_ctx"])
        rest = np.setdiff1d(np.arange(ctx.shape[1]), COURIER_CTX_IDX)
        assert np.array_equal(ctx[:, rest], tr["ctx"][:, rest]) and (tr["courier"] > 0).all()   # input untouched


if __name__ == "__main__":
    test_roundtrip_pickup(); test_new_courier_inputs_bounded_and_simulated_exactly(); print("OK")

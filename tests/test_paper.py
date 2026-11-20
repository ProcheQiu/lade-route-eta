"""Paper mode: vectorised metrics == the official per-sample functions; builder obeys the official rules."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lade.metrics import route_metrics_paper, eta_metrics_paper, _levenshtein_batch  # noqa: E402
from lade import paper as PP  # noqa: E402


# ---- verbatim logic of route_prediction/utils/eval.py (official) ----
def o_hit_rate(pred, label, lab_len, top_n=3):
    e = min(top_n, lab_len)
    return len(set(pred[:e]) & set(label[:e])) / e


def o_krc(pred, label, label_len):
    def conc(i, j):
        return 1 if (lo[i] < lo[j] and po[i] < po[j]) or (lo[i] > lo[j] and po[i] > po[j]) else 0
    if label_len == 1:
        return 1
    label = label[:label_len]
    nil = set(pred) - set(label)
    po = {d: i for i, d in enumerate(pred)}
    lo = {d: i for i, d in enumerate(label)}
    for o in nil:
        lo[o] = len(label)
    n = len(label)
    l1 = [(label[i], label[j]) for i in range(n) for j in range(i + 1, n)]
    l2 = [(i, j) for i in label for j in nil]
    h = sum(conc(i, j) for i, j in l1 + l2)
    return (h - (len(l1) + len(l2) - h)) / (len(l1) + len(l2))


def o_lsd(pred, label, label_len, square=True):
    label = label[:label_len]
    d = [math.fabs(i - pred.index(x)) for i, x in enumerate(label)]
    return sum(x ** 2 if square else x for x in d) / len(label)


def o_ed(pred, label):
    pred = [x for x in pred if x in label]
    a = np.array([label]); b = np.array([pred])
    return int(_levenshtein_batch(a, b)[0])     # Levenshtein, same as edit_distance.SequenceMatcher


def _random_samples(rng, n_samples=400):
    rows = []
    for sid in range(n_samples):
        n = int(rng.integers(3, 15)); L = int(rng.integers(0, n + 1))
        pred = rng.permutation(n) + 1
        for node in range(n):
            rows.append((sid, node + 1, int(np.flatnonzero(pred == node + 1)[0]) + 1, L,
                         float(rng.uniform(1, 120)), float(rng.uniform(1, 120))))
    return pd.DataFrame(rows, columns=["sid", "rank", "pred_rank", "label_len", "eta", "pred_eta"])


def test_route_metrics_match_official():
    d = _random_samples(np.random.default_rng(0))
    got = route_metrics_paper(d)
    hr, krc, lsd, lmd, ed = [], [], [], [], []
    for sid, g in d.groupby("sid"):
        L = g.label_len.iloc[0]
        if L < 1:
            continue
        label = g.sort_values("rank")["rank"].tolist()          # node ids = true rank
        pred = g.sort_values("pred_rank")["rank"].tolist()
        hr.append(o_hit_rate(pred, label, L)); krc.append(o_krc(pred, label, L))
        lsd.append(o_lsd(pred, label, L)); lmd.append(o_lsd(pred, label, L, False)); ed.append(o_ed(pred, label[:L]))
    assert got["samples"] == len(hr)
    np.testing.assert_allclose([got["hr3"], got["krc"], got["lsd"], got["lmd"], got["ed"]],
                               [100 * np.mean(hr), 100 * np.mean(krc), np.mean(lsd), np.mean(lmd), np.mean(ed)])


def test_eta_metrics_match_official():
    d = _random_samples(np.random.default_rng(1))
    d["in_label"] = d["rank"] <= d.label_len
    got = eta_metrics_paper(d)
    maes, rmses, lens, accs = [], [], [], []
    for sid, g in d[d.label_len >= 1].groupby("sid"):
        lab = g[g.in_label]
        e = lab.pred_eta.values - lab.eta.values
        maes.append(np.abs(e).mean()); rmses.append(np.sqrt((e ** 2).mean())); accs.append((np.abs(e) <= 20).mean())
        lens.append(len(lab))
    w = np.array(lens) / np.sum(lens)
    np.testing.assert_allclose([got["mae"], got["rmse"], got["acc20"]],
                               [(np.array(maes) * w).sum(), (np.array(rmses) * w).sum(), 100 * (np.array(accs) * w).sum()])


def _toy_day(name):
    """One courier-day with 30 packages, every package accepted at 08:00, one finish every 9 min."""
    n = 30
    fin = [f"10-01 {(480 + 9 * (i + 1)) // 60:02d}:{(480 + 9 * (i + 1)) % 60:02d}:00" for i in range(n)]
    df = pd.DataFrame({"order_id": range(n), "region_id": 0, "city": "Shanghai", "courier_id": 7,
                       "lng": 121.4 + 0.01 * np.arange(n), "lat": 31.2 + 0.0 * np.arange(n), "aoi_id": np.arange(n),
                       "aoi_type": 1, "accept_time": "10-01 08:00:00", "ds": 1001})
    df[PP.FINISH_COL[name]] = fin
    if name == "pickup":
        df["time_window_start"] = "10-01 09:00:00"; df["time_window_end"] = "10-01 11:00:00"
    return df


def test_builder_follows_official_rules():
    for name in ["pickup", "delivery"]:
        df, _ = PP.preprocess(_toy_day(name), name)
        for mode in ["train", "test"]:
            sn, nd, st = PP.build(df, name, np.full(len(df), mode), courier_stats="all")
            if mode == "test":
                # every window has an open task finishing outside it at its first step -> all dropped
                # except the last window (which runs to the end of the day)
                assert st["windows_dropped_test"] == st["windows"] - 1
            assert (sn.groupby("wid").size() <= PP.T_STEPS).all()
            assert (nd.eta > 0).all() and (nd.eta <= 120).all()          # official label_range (0, 120]
            assert (sn.n_open > PP.N_MIN[name]).all()
            assert (sn.label_len == sn.n_open).all()                     # no new accepts -> no truncation
            assert nd.groupby("sid").slot.apply(lambda s: s.is_unique).all()
            assert (nd.slot > 0).all() and (nd.slot < PP.N_MAX - 1).all()

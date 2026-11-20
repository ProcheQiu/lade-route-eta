import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from lade.metrics import route_metrics, eta_metrics, _levenshtein_batch


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def test_levenshtein_matches_reference():
    rng = np.random.default_rng(1)
    for n in [1, 2, 5, 12, 25]:
        a = np.tile(np.arange(n), (50, 1)); b = np.array([rng.permutation(n) for _ in range(50)])
        assert (_levenshtein_batch(a, b) == [lev(list(x), list(y)) for x, y in zip(a, b)]).all()


def _nodes(pred):
    rows = []
    for sid, p in enumerate(pred):
        for r, pr in enumerate(p, 1):
            rows.append((sid, r, pr))
    return pd.DataFrame(rows, columns=["sid", "rank", "pred_rank"])


def test_perfect_and_reversed():
    m = route_metrics(_nodes([[1, 2, 3, 4], [1, 2]]))
    assert m["hr3"] == 100 and m["krc"] == 100 and m["lsd"] == 0 and m["ed"] == 0
    m = route_metrics(_nodes([[3, 2, 1]]))
    assert m["krc"] == -100 and m["ed"] == 2 and abs(m["hr3"] - 100) < 1e-9


def test_eta():
    d = pd.DataFrame({"eta": [10, 20, 30], "pred_eta": [10, 50, 30]})
    m = eta_metrics(d)
    assert abs(m["mae"] - 10) < 1e-9 and abs(m["acc20"] - 200 / 3) < 1e-6

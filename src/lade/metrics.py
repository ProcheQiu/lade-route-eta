"""Route and ETA metrics, computed per snapshot and averaged over snapshots (route) or packages (ETA).

Predictions live as node-level columns:
  pred_rank : predicted visit position 1..n within the snapshot
  pred_eta  : predicted minutes until finished
True labels are the node columns `rank` and `eta` (see snapshots.py).

!! Definitions to CHECK against the official LaDe code before comparing with the paper:
   - HR@k denominator: here min(k, n) so a 2-package snapshot can still score 100%.
   - LSD: paper prints an absolute-value formula; we report `lsd` (absolute) and `lsd_sq` (squared).
   - ED: plain Levenshtein distance between full true and predicted sequences.
   - KRC: Kendall tau over all node pairs (our label covers every open package, so the paper's
     "nodes not in label" set is empty).
   The paper's ED values (~1.8-2.2) look small for full sequences of median length 10, which hints
   that the official code may evaluate a prefix or normalise. Align before claiming any comparison.
"""
import numpy as np
import pandas as pd


def _levenshtein_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Edit distance between rows of a and b (both [B, n] int), vectorised over B."""
    B, n = a.shape
    prev = np.tile(np.arange(n + 1), (B, 1))
    for i in range(1, n + 1):
        sub = prev[:, :-1] + (a[:, i - 1:i] != b).astype(int)    # substitution / match, j = 1..n
        dele = prev[:, 1:] + 1                                    # deletion
        full = np.concatenate([np.full((B, 1), i), np.minimum(sub, dele)], axis=1)
        # insertion: cur[j] = min(full[j], cur[j-1] + 1)  -> prefix-min trick
        prev = np.minimum.accumulate(full - np.arange(n + 1), axis=1) + np.arange(n + 1)
    return prev[:, n]


def route_metrics(nodes: pd.DataFrame, k: int = 3, by: str | None = None) -> dict:
    """nodes needs sid, rank, pred_rank. Returns means over snapshots (hr/krc in %)."""
    d = nodes[["sid", "rank", "pred_rank"] + ([by] if by else [])].sort_values(["sid", "rank"])
    n_of = d.groupby("sid").size()
    out = []
    for n, sids in n_of.groupby(n_of).groups.items():
        sub = d[d.sid.isin(sids)]
        R = sub["rank"].values.reshape(-1, n)            # rows are 1..n by construction
        P = sub["pred_rank"].values.reshape(-1, n)
        B = R.shape[0]
        kk = min(k, n)
        hr = ((R <= kk) & (P <= kk)).sum(1) / kk
        if n > 1:
            iu = np.triu_indices(n, 1)
            sr = np.sign(R[:, iu[0]] - R[:, iu[1]])
            sp = np.sign(P[:, iu[0]] - P[:, iu[1]])
            krc = (sr == sp).mean(1) * 2 - 1
        else:
            krc = np.ones(B)
        lsd = np.abs(R - P).mean(1)
        lsd_sq = ((R - P) ** 2).mean(1)
        true_seq = np.tile(np.arange(n), (B, 1))
        pred_seq = np.argsort(P, axis=1)                   # node (true position) visited at each step
        ed = _levenshtein_batch(true_seq, pred_seq)
        frame = pd.DataFrame({"sid": np.asarray(sids), "n": n, "hr": hr, "krc": krc, "lsd": lsd,
                              "lsd_sq": lsd_sq, "ed": ed})
        out.append(frame)
    per = pd.concat(out)
    if by:
        per = per.merge(d.drop_duplicates("sid")[["sid", by]], on="sid")
    return _summ_route(per, k) if not by else {g: _summ_route(x, k) for g, x in per.groupby(by)}


def _summ_route(per, k):
    return {f"hr{k}": 100 * per.hr.mean(), "krc": 100 * per.krc.mean(), "lsd": per.lsd.mean(),
            "lsd_sq": per.lsd_sq.mean(), "ed": per.ed.mean(), "snapshots": int(len(per))}


def eta_metrics(nodes: pd.DataFrame, by: str | None = None) -> dict:
    """nodes needs eta, pred_eta (minutes)."""
    def f(x):
        e = x.pred_eta.values - x.eta.values
        return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                "acc20": float(100 * (np.abs(e) < 20).mean()), "acc30": float(100 * (np.abs(e) < 30).mean()),
                "packages": int(len(x))}
    return f(nodes) if not by else {g: f(x) for g, x in nodes.groupby(by)}


# ---------------------------------------------------------------------------------------------------
# Paper-compatible metrics: exact ports of route_prediction/utils/eval.py and time_prediction/utils/eval.py
# in github.com/wenhaomin/LaDe (see docs/alignment_notes.md). Inputs are node rows with
#   sid, rank (true position 1..n among ALL open packages), pred_rank (1..n), label_len (per node, = the
#   snapshot's truncated label length L). The label is the first L packages in true order; the prediction
#   is a full order over all n open packages.
#   HR@k : |pred[:e] & label[:e]| / e, e = min(k, L)
#   KRC  : pairs inside the label + (label, non-label) pairs, non-label items ranked after the label;
#          returns 1 when L == 1 (official shortcut)
#   LSD  : mean over label items of (true pos - pred pos)^2   (official 'square' mode; LMD = absolute)
#   ED   : Levenshtein(pred restricted to label items, label)
#   ACC@k: 1 if the first min(k, L) predicted items equal the label exactly
#   ETA  : errors on label items only; MAE / ACC@m pooled over packages, ACC uses <=; RMSE is the
#          package-weighted mean of per-sample RMSEs (official calc_rmse * label_len / sum label_len)
# Samples with L outside [1, 25] are ignored (official Metric([1, 25])).
# ---------------------------------------------------------------------------------------------------
def route_metrics_paper(nodes: pd.DataFrame, k: int = 3, by: str | None = None, len_range=(1, 25)) -> dict:
    d = nodes[["sid", "rank", "pred_rank", "label_len"] + ([by] if by else [])]
    d = d[(d.label_len >= len_range[0]) & (d.label_len <= len_range[1])].sort_values(["sid", "rank"])
    n_of = d.groupby("sid").size()
    L_of = d.groupby("sid").label_len.first()
    key = pd.DataFrame({"n": n_of, "L": L_of})
    out = []
    for (n, L), grp in key.groupby(["n", "L"]):
        sids = grp.index.values
        sub = d[d.sid.isin(sids)]
        R = sub["rank"].values.reshape(-1, n)
        P = sub["pred_rank"].values.reshape(-1, n)
        B = R.shape[0]
        lab = R <= L
        e = min(k, L)
        hr = ((R <= e) & (P <= e)).sum(1) / e
        acc3 = np.all(np.where(R <= min(3, L), R == P, True), axis=1).astype(float)
        if L == 1:
            krc = np.ones(B)
        else:
            iu = np.triu_indices(n, 1)
            Rl = np.where(lab, R, L + 1)
            both_out = (~lab[:, iu[0]]) & (~lab[:, iu[1]])
            agree = np.sign(Rl[:, iu[0]] - Rl[:, iu[1]]) == np.sign(P[:, iu[0]] - P[:, iu[1]])
            valid = ~both_out
            krc = (2 * (agree & valid).sum(1) - valid.sum(1)) / valid.sum(1)
        diff = np.where(lab, R - P, 0)
        lsd = (diff ** 2).sum(1) / L
        lmd = np.abs(diff).sum(1) / L
        # predicted order restricted to label items, expressed as true positions 1..L
        Pm = np.where(lab, P, n + 10)
        pred_seq = np.take_along_axis(R, np.argsort(Pm, axis=1), axis=1)[:, :L]
        ed = _levenshtein_batch(np.tile(np.arange(1, L + 1), (B, 1)), pred_seq)
        out.append(pd.DataFrame({"sid": sids, "n": n, "L": L, "hr": hr, "krc": krc, "lsd": lsd, "lmd": lmd,
                                 "ed": ed, "acc3": acc3}))
    per = pd.concat(out)

    def summ(x):
        return {f"hr{k}": 100 * x.hr.mean(), "krc": 100 * x.krc.mean(), "lsd": x.lsd.mean(), "lmd": x.lmd.mean(),
                "ed": x.ed.mean(), "acc3": 100 * x.acc3.mean(), "samples": int(len(x))}
    if not by:
        return summ(per)
    per = per.merge(d.drop_duplicates("sid")[["sid", by]], on="sid")
    return {g: summ(x) for g, x in per.groupby(by)}


def eta_metrics_paper(nodes: pd.DataFrame, by: str | None = None, len_range=(1, 25)) -> dict:
    """nodes needs sid, eta, pred_eta, in_label, label_len. Scores label items only (official)."""
    d = nodes[nodes.in_label & (nodes.label_len >= len_range[0]) & (nodes.label_len <= len_range[1])]

    def f(x):
        e = x.pred_eta.values - x.eta.values
        se = pd.Series(e ** 2).groupby(x.sid.values).agg(["mean", "size"])
        rmse = float((np.sqrt(se["mean"]) * se["size"]).sum() / se["size"].sum())
        return {"mae": float(np.abs(e).mean()), "rmse": rmse, "rmse_pooled": float(np.sqrt((e ** 2).mean())),
                "acc20": float(100 * (np.abs(e) <= 20).mean()), "acc30": float(100 * (np.abs(e) <= 30).mean()),
                "packages": int(len(x)), "samples": int(x.sid.nunique())}
    return f(d) if not by else {g: f(x) for g, x in d.groupby(by)}


def order_to_rank(order: np.ndarray) -> np.ndarray:
    """Local visiting order (array of local node indices) -> pred_rank per local node (1-based)."""
    r = np.empty(len(order), dtype=np.int16)
    r[order] = np.arange(1, len(order) + 1)
    return r

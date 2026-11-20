"""Joint route + ETA model (PyTorch, no PyG needed: snapshots have <= 25 packages, so dense attention is cheap).

  encoder : courier node + package nodes, L layers of edge-aware graph attention (GATv2-style)
  route   : pointer decoder with a GRU state; greedy at inference
  ETA     : LSTM along the visiting order -> positive increments -> cumulative sum (monotone ETAs)

Ablation flags (see docs/pipeline_design.md, "Proposed model"):
  use_edges=False        -> plain attention without edge features ("no graph")
  route_conditioned=False-> ETA straight from each node vector
  use_courier=False      -> no courier-ID embedding (courier statistics stay in the context vector)
  hard_window=True       -> a pickup whose window has not opened cannot be chosen while another can
Trained on Colab T4 (2026-11-16, run 1 in docs/experiment_log.md).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_EDGE = 3   # distance (km), same AOI, |window-end gap| (h)


class EdgeGATLayer(nn.Module):
    def __init__(self, d, heads=4, edge_dim=N_EDGE, use_edges=True, dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk, self.use_edges = heads, d // heads, use_edges
        self.wl, self.wr = nn.Linear(d, d), nn.Linear(d, d)
        self.we = nn.Linear(edge_dim, d) if use_edges else None
        self.att = nn.Parameter(torch.randn(heads, self.dk) / self.dk ** 0.5)
        self.out = nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, e, mask):
        # x [B,M,d], e [B,M,M,E], mask [B,M] (True = real node)
        B, M, _ = x.shape
        xl = self.wl(x).view(B, M, 1, self.h, self.dk)          # receiver i
        xr = self.wr(x).view(B, 1, M, self.h, self.dk)          # sender j
        z = xl + xr
        if self.use_edges:
            z = z + self.we(e).view(B, M, M, self.h, self.dk)
        score = (F.leaky_relu(z, 0.2) * self.att).sum(-1)       # [B,M,M,h]
        score = score.masked_fill(~mask[:, None, :, None], float("-inf"))
        a = torch.softmax(score, dim=2).nan_to_num(0.0)
        msg = torch.einsum("bijh,bjhd->bihd", a, self.wr(x).view(B, M, self.h, self.dk)).reshape(B, M, -1)
        x = self.ln1(x + self.drop(self.out(msg)))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x * mask[..., None]


class JointRouteETA(nn.Module):
    def __init__(self, n_num, n_ctx, n_aoi, n_aoi_type, n_courier, d=64, layers=3, heads=4, dropout=0.1,
                 use_edges=True, route_conditioned=True, use_courier=True, hard_window=False, courier_dropout=0.1):
        super().__init__()
        self.use_courier, self.route_conditioned, self.hard_window = use_courier, route_conditioned, hard_window
        self.courier_dropout = courier_dropout
        self.aoi = nn.Embedding(n_aoi, 16)
        self.aoi_type = nn.Embedding(n_aoi_type, 8)
        self.courier = nn.Embedding(n_courier, 16)
        self.node_in = nn.Linear(n_num + 16 + 8, d)
        self.courier_in = nn.Linear(n_ctx + (16 if use_courier else 0), d)
        self.enc = nn.ModuleList([EdgeGATLayer(d, heads, use_edges=use_edges, dropout=dropout) for _ in range(layers)])
        # pointer decoder
        self.gru = nn.GRUCell(d, d)
        self.wq, self.wk = nn.Linear(d, d), nn.Linear(d, d)
        self.v = nn.Linear(d, 1, bias=False)
        self.dyn = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))   # dist from current, window start/end
        # ETA
        self.eta_rnn = nn.LSTM(d + 1, d, batch_first=True)
        self.eta_out = nn.Linear(d, 1)
        nn.init.constant_(self.eta_out.bias, 2.0)        # softplus(2) ~ 2.1 min per step at start
        self.eta_direct = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    # ---------------------------------------------------------------- encoder
    def encode(self, b):
        B, N = b["mask"].shape
        node = torch.cat([b["num"], self.aoi(b["aoi"]), self.aoi_type(b["aoi_type"])], -1)
        cid = b["courier"]
        if self.training and self.courier_dropout > 0:     # teach the "unknown courier" (id 0) fallback
            cid = torch.where(torch.rand_like(cid, dtype=torch.float) < self.courier_dropout, torch.zeros_like(cid), cid)
        ctx = torch.cat([b["ctx"], self.courier(cid)], -1) if self.use_courier else b["ctx"]
        x = torch.cat([self.courier_in(ctx)[:, None], self.node_in(node)], 1)          # [B, N+1, d]
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), b["mask"]], 1)
        xy = torch.cat([torch.zeros(B, 1, 2, device=x.device), b["xy"]], 1)
        dist = torch.cdist(xy, xy)                                                      # km
        aoi = torch.cat([torch.full((B, 1), -2, device=x.device, dtype=torch.long), b["aoi_raw"]], 1)
        same = (aoi[:, :, None] == aoi[:, None, :]).float()
        twe = torch.cat([torch.zeros(B, 1, device=x.device), b["tw"][..., 1]], 1)
        gap = (twe[:, :, None] - twe[:, None, :]).abs()
        e = torch.stack([dist, same, gap], -1)
        for layer in self.enc:
            x = layer(x, e, mask)
        return x[:, 0], x[:, 1:]                                                        # courier, packages

    # ---------------------------------------------------------------- route
    def _scores(self, state, H, cur_xy, b, left):
        q = self.wq(state)[:, None]
        s = self.v(torch.tanh(q + self.wk(H))).squeeze(-1)
        d = (b["xy"] - cur_xy[:, None]).norm(dim=-1)
        s = s + self.dyn(torch.stack([d, b["tw"][..., 0], b["tw"][..., 1]], -1)).squeeze(-1)
        if self.hard_window and not self.training:      # inference-only rule; training sees true choices
            opened = b["tw"][..., 0] <= 0
            any_open = (opened & left).any(1, keepdim=True)
            s = s.masked_fill(any_open & ~opened & left, -1e4)
        return s.masked_fill(~left, float("-inf"))

    def route(self, b, hc, H, teacher=None):
        """Returns (log-probs of the chosen steps [B,N], chosen order [B,N] with -1 pads)."""
        B, N = b["mask"].shape
        left = b["mask"].clone()
        state = hc
        inp = hc
        cur_xy = torch.zeros(B, 2, device=H.device)
        logps, order = [], []
        n = b["mask"].sum(1)
        for s in range(N):
            active = s < n
            state = self.gru(inp, state)
            sc = self._scores(state, H, cur_xy, b, left)
            sc = torch.where(active[:, None], sc, torch.zeros_like(sc))   # keep finished rows finite
            lp = torch.log_softmax(sc, -1)
            k = teacher[:, s].clamp(min=0) if teacher is not None else lp.argmax(-1)
            logps.append(torch.where(active, lp.gather(1, k[:, None]).squeeze(1), torch.zeros_like(n, dtype=lp.dtype)))
            order.append(torch.where(active, k, torch.full_like(k, -1)))
            idx = torch.arange(B, device=H.device)
            left = left.clone(); left[idx[active], k[active]] = False
            inp = torch.where(active[:, None], H[idx, k], inp)
            cur_xy = torch.where(active[:, None], b["xy"][idx, k], cur_xy)
        return torch.stack(logps, 1), torch.stack(order, 1)

    # ---------------------------------------------------------------- ETA
    def eta(self, b, hc, H, order):
        """ETA per slot [B,N] given a visiting order (slot indices, -1 = pad)."""
        B, N = b["mask"].shape
        valid = order >= 0
        o = order.clamp(min=0)
        idx = torch.arange(B, device=H.device)[:, None]
        if not self.route_conditioned:
            return F.softplus(self.eta_direct(torch.cat([H, hc[:, None].expand_as(H)], -1))).squeeze(-1) * 30.0
        seq = H[idx, o]                                                     # [B,N,d] in visiting order
        pts = torch.cat([torch.zeros(B, 1, 2, device=H.device), b["xy"][idx, o]], 1)
        step_km = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1, keepdim=True)
        out, _ = self.eta_rnn(torch.cat([seq, step_km], -1), (hc[None].contiguous(), torch.zeros_like(hc)[None]))
        inc = F.softplus(self.eta_out(out).squeeze(-1)) * 10.0              # minutes per step, > 0
        cum = torch.cumsum(inc * valid, 1)
        eta = torch.zeros(B, N, device=H.device)
        eta[idx.expand(B, N)[valid], o[valid]] = cum[valid]
        return eta

    def forward(self, b, teacher=True):
        hc, H = self.encode(b)
        logp, order = self.route(b, hc, H, b["order"] if teacher else None)
        eta = self.eta(b, hc, H, b["order"] if teacher else order)
        return logp, order, eta


def joint_loss(logp, eta_pred, b, lam=1.0, delta=10.0):
    steps = (b["order"] >= 0).float()
    route = -(logp * steps).sum() / steps.sum()
    m = b["mask"].float()
    eta = (F.huber_loss(eta_pred, b["eta"], delta=delta, reduction="none") * m).sum() / m.sum()
    return route + lam * eta, route.detach(), eta.detach()

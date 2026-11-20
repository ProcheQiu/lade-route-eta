"""Step 4. Train / evaluate the joint route + ETA model.

  python scripts/train_joint.py --dataset pickup --smoke            # 2 minutes on CPU: checks everything runs
  python scripts/train_joint.py --dataset pickup --frac 0.25        # real run (GPU recommended)
  python scripts/train_joint.py --dataset delivery --no-edges       # ablation example

Writes results/joint_{dataset}_{run}.json and results/models/{run}.pt
NOT YET RUN: written in a workspace without PyTorch. Start with --smoke and fix anything that breaks.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch

from lade import config as C
from lade.metrics import route_metrics, eta_metrics
from lade.models.data import fit_encoders, build, to_eval_frame, simulate_unknown_courier
from lade.models.joint import JointRouteETA, joint_loss

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=C.DATASETS, required=True)
ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--lam", type=float, default=1.0, help="weight of the ETA loss")
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--frac", type=float, default=0.25, help="share of TRAIN snapshots used (val/test always full)")
ap.add_argument("--val-frac", type=float, default=1.0, help="share of VAL snapshots used for early stopping")
ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--select", choices=["krc", "mae", "joint"], default="joint")
ap.add_argument("--no-edges", action="store_true"); ap.add_argument("--no-route-cond", action="store_true")
ap.add_argument("--no-courier", action="store_true"); ap.add_argument("--hard-window", action="store_true")
ap.add_argument("--courier-dropout", type=float, default=0.1,
                help="share of training samples shown as an unseen courier (id 0, c_known 0, fallback stats)")
ap.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch, CPU")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--run", default=None)
args = ap.parse_args()
if args.smoke:
    args.frac, args.val_frac, args.epochs, args.device = 0.005, 0.01, 1, "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
run = args.run or f"{args.dataset}_h{args.hidden}_l{args.layers}_lam{args.lam}_s{args.seed}" + \
    "".join(f"_{k}" for k in ["no_edges", "no_route_cond", "no_courier", "hard_window"] if getattr(args, k)) + \
    ("_smoke" if args.smoke else "")
(C.RESULTS / "models").mkdir(parents=True, exist_ok=True)

t0 = time.time()
sn = C.load_frame(f"{args.dataset}_snaps"); nd = C.load_frame(f"{args.dataset}_nodes")
enc = fit_encoders(nd, sn)
tr = build(nd, sn, "train", enc, frac=args.frac, seed=args.seed)
va = build(nd, sn, "val", enc, frac=args.val_frac, seed=args.seed + 1)
te = build(nd, sn, "test", enc, frac=0.01 if args.smoke else 1.0, seed=args.seed + 2)
print(f"data ready in {time.time() - t0:.0f}s: train {len(tr['n'])}, val {len(va['n'])}, test {len(te['n'])} snapshots")

KEYS = ["mask", "num", "xy", "tw", "aoi", "aoi_raw", "aoi_type", "eta", "order", "ctx", "courier"]


drop_rng = np.random.default_rng(args.seed + 100)


def batches(arr, bs, shuffle):
    idx = np.random.permutation(len(arr["n"])) if shuffle else np.arange(len(arr["n"]))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        b = {k: arr[k][j] for k in KEYS}
        if shuffle and args.courier_dropout > 0:     # training only: some samples look like unseen couriers
            b["ctx"], b["courier"] = simulate_unknown_courier(b["ctx"], b["courier"], enc, args.courier_dropout, drop_rng)
        yield {k: torch.as_tensor(v).to(args.device) for k, v in b.items()}


model = JointRouteETA(n_num=tr["num"].shape[-1], n_ctx=tr["ctx"].shape[-1], n_aoi=len(enc["aoi"]),
                      n_aoi_type=len(enc["aoi_type"]), n_courier=len(enc["courier"]), d=args.hidden,
                      layers=args.layers, dropout=args.dropout, use_edges=not args.no_edges,
                      route_conditioned=not args.no_route_cond, use_courier=not args.no_courier,
                      hard_window=args.hard_window,
                      courier_dropout=0.0).to(args.device)   # done in batches() so ctx matches the dropped id
opt = torch.optim.Adam(model.parameters(), lr=args.lr)


@torch.no_grad()
def predict(arr):
    model.eval()
    orders, etas = [], []
    for b in batches(arr, 256, False):
        _, order, eta = model(b, teacher=False)
        orders.append(order.cpu().numpy()); etas.append(eta.cpu().numpy())
    return to_eval_frame(arr, np.concatenate(orders), np.concatenate(etas))


def score(m_route, m_eta):
    if args.select == "krc":
        return m_route["krc"]
    if args.select == "mae":
        return -m_eta["mae"]
    return m_route["krc"] / 100 - m_eta["mae"] / 60


best, best_ep, history = -np.inf, -1, []
for ep in range(args.epochs):
    model.train(); tl = []
    for b in batches(tr, args.batch, True):
        logp, _, eta = model(b, teacher=True)
        loss, lr_, le_ = joint_loss(logp, eta, b, lam=args.lam)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl.append([loss.item(), lr_.item(), le_.item()])
    ev = predict(va); mr, me = route_metrics(ev), eta_metrics(ev)
    sc = score(mr, me)
    history.append({"epoch": ep, "train_loss": np.mean(tl, 0).tolist(), "val_route": mr, "val_eta": me, "score": sc})
    print(f"ep {ep}: loss {np.mean(tl, 0).round(3)}  val HR@3 {mr['hr3']:.2f} KRC {mr['krc']:.2f} "
          f"MAE {me['mae']:.2f}  score {sc:.4f}  ({time.time() - t0:.0f}s)")
    if sc > best:
        best, best_ep = sc, ep
        torch.save(model.state_dict(), C.RESULTS / "models" / f"{run}.pt")
    elif ep - best_ep >= args.patience:
        print("early stop"); break

model.load_state_dict(torch.load(C.RESULTS / "models" / f"{run}.pt", map_location=args.device))
ev = predict(te)
res = {"run": run, "args": vars(args), "best_epoch": best_ep, "test_route": route_metrics(ev),
       "test_eta": eta_metrics(ev), "test_route_by_courier_known": route_metrics(ev, by="c_known"),
       "test_eta_by_courier_known": eta_metrics(ev, by="c_known"), "history": history,
       "seconds": round(time.time() - t0),
       "code_version": (C.ROOT / "CODE_VERSION.txt").read_text().strip() if (C.ROOT / "CODE_VERSION.txt").exists() else "local"}
json.dump(res, open(C.RESULTS / f"joint_{run}.json", "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ["run", "best_epoch", "test_route", "test_eta", "test_route_by_courier_known",
                                      "test_eta_by_courier_known", "seconds", "code_version"]}, indent=1, default=str))

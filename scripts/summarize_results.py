"""Print every results/*.json next to the paper's Shanghai numbers.  Usage: python scripts/summarize_results.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lade import config as C

rows = []
f = C.RESULTS / "rule_baselines_test.json"
if f.exists():
    r = json.load(open(f))
    for ds in r:
        for m in ["TimeGreedy", "DistanceGreedy"]:
            rows.append((m, ds, "route", r[ds][m]["all"]))
        rows.append(("SPEED", ds, "eta", r[ds]["SPEED"]["all"]))
for f in sorted(C.RESULTS.glob("gbdt_*.json")):
    task, ds = f.stem.split("_")[1:3]
    rows.append(("LightGBM" + (" (preview)" if json.load(open(f)).get("preview") else ""), ds, task, json.load(open(f))["all"]))
for f in sorted(C.RESULTS.glob("ortools_*.json")):
    rows.append(("OR-Tools", f.stem.split("_")[1], "route", json.load(open(f))["all"]))
for f in sorted(C.RESULTS.glob("joint_*.json")):
    j = json.load(open(f)); ds = j["args"]["dataset"]
    rows.append((j["run"], ds, "route", j["test_route"])); rows.append((j["run"], ds, "eta", j["test_eta"]))
print(f"{'method':38s} {'data':9s} {'ours':40s} paper")
for m, ds, task, v in rows:
    key = m.split(" ")[0]
    if task == "route":
        ours = f"HR@3 {v['hr3']:.2f} KRC {v['krc']:.2f} LSD {v['lsd']:.2f} ED {v['ed']:.2f}"
        p = C.PAPER["route_pickup"].get(key) if ds == "pickup" else None
        pap = f"HR@3 {p['hr3']} KRC {p['krc']} LSD {p['lsd']} ED {p['ed']}" if p else "-"
    else:
        ours = f"MAE {v['mae']:.2f} RMSE {v['rmse']:.2f} ACC@20 {v['acc20']:.2f}"
        p = C.PAPER["eta_delivery"].get(key) if ds == "delivery" else None
        pap = f"MAE {p['mae']} RMSE {p['rmse']} ACC@20 {p['acc20']}" if p else "-"
    print(f"{m:38s} {ds:9s} {ours:40s} {pap}")

# ---- paper mode (official sample construction + metrics, docs/alignment_notes.md)
prow = []
for f in sorted(C.RESULTS.glob("paper_rules_*.json")):
    r = json.load(open(f)); ds = f.stem.split("_")[2]
    for m, v in r.items():
        if isinstance(v, dict):
            prow.append((m, ds, "eta" if m.startswith("SPEED") else "route", v["all"]))
for f in sorted(C.RESULTS.glob("paper_gbdt_*.json")):
    parts = f.stem.split("_"); task, ds = parts[2], parts[3]
    j = json.load(open(f))
    prow.append(("LightGBM" + f" ({j['backend']}{', train cap' if j.get('official_train_cap') else ''})",
                 ds, task, j["all"]))
if prow:
    print("\nPAPER MODE")
    print(f"{'method':38s} {'data':9s} {'ours':40s} paper")
for m, ds, task, v in prow:
    key = m.split(" ")[0].replace("-fixed", "")
    if task == "route":
        ours = f"HR@3 {v['hr3']:.2f} KRC {v['krc']:.2f} LSD {v['lsd']:.2f} ED {v['ed']:.2f}"
        p = C.PAPER["route_pickup"].get(key) if ds == "pickup" else None
        pap = f"HR@3 {p['hr3']} KRC {p['krc']} LSD {p['lsd']} ED {p['ed']}" if p else "-"
    else:
        ours = f"MAE {v['mae']:.2f} RMSE {v['rmse']:.2f} ACC@20 {v['acc20']:.2f}"
        p = C.PAPER["eta_delivery"].get(key) if ds == "delivery" else None
        pap = f"MAE {p['mae']} RMSE {p['rmse']} ACC@20 {p['acc20']}" if p else "-"
    print(f"{m:38s} {ds:9s} {ours:40s} {pap}")

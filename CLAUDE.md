# CLAUDE.md — guidance for Claude Code in this repo

## What this project is
PhD research project (transportation engineering): joint **route + ETA prediction** for last-mile couriers on the
**LaDe Shanghai** dataset (Cainiao; paper arXiv 2306.10675 v3), compared against every baseline in the paper.
Research angle: predicting well for **couriers never seen in training** (27% of delivery / 24% of pickup test rows).

Full design: `docs/pipeline_design.md` (markdown copy). Live doc with comments: https://claude.ai/artifact/6hiVspCwm7XmA6fjLDCPH3
When a design decision changes, update `docs/pipeline_design.md` and tell the user so the live doc can be updated too.

## Status (handed over 2026-11-14)
| Part | File(s) | State |
| --- | --- | --- |
| Cleaning | `src/lade/clean.py`, `scripts/clean_data.py` | Verified. 1,474,505 delivery / 1,413,208 pickup rows kept |
| Figures | `scripts/make_figures.py` | Verified. Writes `docs/figures/` |
| Snapshot builder | `src/lade/snapshots.py`, `scripts/build_snapshots.py` | Verified. 1,270,492 delivery / 1,231,697 pickup snapshots |
| Leakage + metric tests | `tests/` | Pass (`python -m pytest tests -q`) |
| Metrics | `src/lade/metrics.py` | Verified. Full-route: `route_metrics`/`eta_metrics`; paper mode: `*_paper` (tested against official functions) |
| **Paper mode** (2026-11-15) | `src/lade/paper.py`, `src/lade/baselines/paper_baselines.py`, `scripts/build_paper_samples.py`, `scripts/run_paper_baselines.py`, `docs/alignment_notes.md` | **Reproduction gate passed.** Pickup LightGBM HR@3 69.35 (paper 70.63), delivery ETA MAE 16.72 (paper 17.24); rule baselines match to 2 decimals |
| Rule baselines | `src/lade/baselines/rules.py`, `scripts/run_rule_baselines.py` | Run. Results in `results/rule_baselines_test.json` |
| GBDT (LightGBM) | `src/lade/baselines/gbdt.py`, `scripts/run_gbdt.py` | **Full-route LightGBM done 2026-11-19** (full test = joint-model rows; route trained on the same 25% snapshots). Pickup KRC 57.50, delivery ETA MAE 53.18 / RMSE 89.16. See `docs/experiment_log.md` |
| OR-Tools | `src/lade/baselines/ortools_route.py`, `scripts/run_ortools.py` | Written, **never run** |
| Joint GNN | `src/lade/models/joint.py`, `src/lade/models/data.py`, `scripts/train_joint.py` | data.py tested; `--smoke` runs end to end on CPU. **Run 1 done on Colab (2026-11-16)**: pickup seed 0, test HR@3 74.33 / KRC 58.47 / MAE 64.1 (ties the GBDT preview on route; ETA overfits). Run 2 (5 seeds x 2 datasets, `results_joint/`) found a **bug: new couriers' `c_known` was scaled to -1e6** (constant in training). Fixed 2026-11-18 (Scaler + courier dropout that simulates a real unseen courier). **Run 3 = current valid results** (2026-11-19, in `results/`): pickup HR@3 75.27±0.05, KRC 60.01±0.08, MAE 38.1±0.9; delivery KRC 46.96±0.42, MAE 51.7±0.5. Hidden-courier eval done 2026-11-19: knowing the courier adds ~0 (KRC < 0.15, pickup ETA +0.3 min, delivery ETA -0.8 min), but LightGBM loses 12.1 min pickup ETA MAE when the courier is hidden (34.3 -> 46.4): courier stats carry information the joint model ignores. Direction = modified go (see experiment_log 2026-11-20). Real LightGBM full-route baseline done (joint wins route; ETA mixed: LightGBM lower RMSE). Every run is logged in `docs/experiment_log.md`. Full-route samples only, no paper-mode input yet |
| Deep baselines (FDNET, DeepRoute, Graph2Route, DRL4Route, RankETPA, M2G4RTP, KNN, MLP) | — | Not started: use the official implementations |

## Reproduction (DONE 2026-11-15, details in `docs/alignment_notes.md`)
The mismatch came from the official sample construction, not from our metrics alone: ETA labels limited to
(0, 120] min, route labels truncated at the courier's next accept, 24-row windows with <= 12 steps, whole test
windows dropped when an open task falls outside, >= 3 (pickup) / >= 2 (delivery) open tasks, squared LSD,
ED/KRC on the truncated label. The official "LightGBM" ETA is actually XGBoost. Paper mode reproduces it:
rule baselines match the paper to 2 decimals and LightGBM passes the gate (HR@3 69.35 vs 70.63; MAE 16.72 vs 17.24).
Official repo is cloned at `../official_LaDe/repo` (commit 4d64660), outside this project.
Paper mode keeps the official leakage (future-based sample filters, see alignment notes section 4); our
full-route setting stays the main setting for research claims. Report deep baselines in BOTH settings.

## Environment
`.venv/` (Python 3.13, CPU-only torch 2.15, lightgbm 4.7, xgboost 3.4, ortools 9.15, pandas 3.1, pyarrow).
Run scripts with `.venv/Scripts/python.exe`. No NVIDIA GPU on this machine.
**GPU = Google Colab** (AWS GPU quota is 0, appeal pending, as of 2026-11-16): `python scripts/export_for_colab.py` builds
`colab/lade_colab.zip` (code + model data as Parquet, since pandas-3 pickles may not load on Colab). Upload it to
`MyDrive/lade/`, run `colab/train_joint_colab.ipynb`, then copy `MyDrive/lade/results/` into `results/`.
After any code change: `python scripts/export_for_colab.py --code-only --tag <name>` -> `colab/lade_code_<name>.zip`
(upload to Colab `/content`; the notebook prints `CODE VERSION`, and each results JSON stores `code_version`). `config.load_frame` reads `.pkl`, or `.parquet` when the pickle is absent.

## How to run (in order)
```bash
pip install -r requirements.txt            # torch: pick the build for your CUDA from pytorch.org
# raw data: data/raw/delivery/delivery_sh.csv and data/raw/pickup/pickup_sh.csv
#   (Hugging Face: Cainiao-AI/LaDe-D and Cainiao-AI/LaDe-P, Shanghai files)
python scripts/clean_data.py               # ~1 min  -> data/processed/*_clean.pkl, results/cleaning_log.json
python scripts/build_snapshots.py          # ~1 min  -> *_snaps.pkl, *_nodes.pkl, *_courier_stats.pkl
python -m pytest tests -q                  # leakage, metrics, model arrays
python scripts/make_figures.py             # optional, docs/figures/*.png
python scripts/run_rule_baselines.py       # TimeGreedy, DistanceGreedy, SPEED on the test split
python scripts/build_paper_samples.py      # ~7 min -> paper_*_snaps/nodes.pkl (official sample construction)
python scripts/run_paper_baselines.py --methods rules                          # ~2 min
python scripts/run_paper_baselines.py --methods gbdt_route --datasets pickup   # ~7 min
python scripts/run_paper_baselines.py --methods gbdt_eta --datasets delivery   # ~3 min
python scripts/run_paper_baselines.py --methods gbdt_route --datasets pickup --route-backend xgboost  # ~9 min
python scripts/run_paper_baselines.py --methods gbdt_eta --datasets delivery --eta-backend xgboost    # ~5 min
python scripts/run_gbdt.py --task eta --datasets delivery pickup                 # ~7 min, LightGBM
python scripts/run_gbdt.py --task route --datasets pickup delivery --train-frac 0.25 --max-val-snapshots 30000   # ~40 min
python scripts/run_ortools.py --dataset pickup --windows --workers 8
python scripts/train_joint.py --dataset pickup --smoke      # FIRST: fix whatever breaks
python scripts/train_joint.py --dataset pickup --frac 0.25  # real run, GPU
python scripts/summarize_results.py        # everything vs paper numbers
```
Expected after `run_rule_baselines.py` (our current definitions): pickup TimeGreedy HR@3 65.23, DistanceGreedy 53.94;
delivery SPEED MAE 60.88. If you get different numbers, something changed upstream.

## Fixed decisions (do not change silently)
- **Split:** chronological 6:2:2 **by days** (user's decision). Cuts in `config.SPLIT_CUTS`:
  delivery train <= 08-18, val <= 09-24; pickup train <= 08-22, val <= 09-26. Row-based split = robustness check only.
- **Year is 2022** (official README: 2022-05-01 ~ 2022-10-31) -> weekday features are valid.
- **Max 25 open packages** per snapshot (paper).
- **Scores reported** as in the paper: route on pickup, ETA on delivery; the other combinations are extras.
- Every results table also splits **seen vs new couriers** (`c_known`).

## Rules for any code you add
- **Leakage contract** (`snapshots.py` docstring): features only from information dated <= query time q.
  Courier/AOI statistics and feature scaling come from the **training days only**. Add a test for any new feature.
- **Test set is touched once per final configuration.** All selection (hyperparameters, early stopping, lambda,
  ablation choices) uses validation.
- Labels `rank`, `eta`, `finish_min` must never be model inputs.
- Report mean ± std over 5 seeds for deep models; save every run's JSON in `results/`.
- Keep changes reproducible from the scripts; no notebook-only results.

## Known caveats
- ETA labels are long-tailed (delivery mean 80 min, median 42, p99 520; pickup mean 132). Huber loss is used.
- Pickup snapshots are larger (median 10 open) than delivery (median 4).
- Full snapshot data is ~2.5M snapshots / 19.7M package rows. `train_joint.py --frac` subsamples training only.
- `models/data.py` shuffles node slots per snapshot so slot position never reveals the true order.
- `joint.py` hard-window ablation applies only at inference (a training mask would make true choices -1e4).
- `gbdt.py` route training rows grow ~n²/2 per snapshot; use `--max-train-snapshots` unless RAM is large.
- Licence of the data: CC BY-NC 4.0 (non-commercial research use).

## Research direction (revised 2026-11-20 after the go/no-go)
Hidden-courier evaluation (`scripts/eval_hidden_courier.py`, `scripts/eval_hidden_courier_gbdt.py`): on the same
seen-courier rows, hiding the courier costs the joint model ~0 but LightGBM +12.1 min pickup ETA MAE (+4.5 delivery).
Courier statistics act as a stable per-courier tempo (`scripts/analyze_gbdt_courier.py`); the joint model ignores
them; the unknown-courier fallback (unweighted median over couriers) is biased. So the problem is **estimating a
courier's tempo without history**. Next: package-weighted fallback and courier dropout for LightGBM; q-time-safe
"pace so far today" features; make the joint model use courier info without losing new-courier robustness; then
test a courier-AOI graph profile. Details: `docs/pipeline_design.md` "Courier-knowledge findings", `docs/experiment_log.md`.

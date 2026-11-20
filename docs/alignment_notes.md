# Alignment with the official LaDe code

Official repo: https://github.com/wenhaomin/LaDe, commit `4d64660` (cloned 2026-11-15 into
`../official_LaDe/repo`, outside this project). All line numbers below refer to that commit.

Our pipeline (`clean.py` + `snapshots.py` + `metrics.route_metrics/eta_metrics`) stays the default
**full-route setting**. The paper-compatible setting is a separate port:

| Piece | Ours (full-route) | Paper mode |
| --- | --- | --- |
| Samples | `src/lade/snapshots.py` | `src/lade/paper.py`, `scripts/build_paper_samples.py` |
| Metrics | `metrics.route_metrics`, `metrics.eta_metrics` | `metrics.route_metrics_paper`, `metrics.eta_metrics_paper` |
| Baselines | `baselines/rules.py`, `baselines/gbdt.py` | `baselines/paper_baselines.py`, `scripts/run_paper_baselines.py` |
| Tests | `tests/test_leakage.py`, `tests/test_metrics.py` | `tests/test_paper.py` (metrics checked against verbatim official functions) |

## 1. Sample construction: the differences that move the numbers

R = `route_prediction/data/dataset.py` (pickup), D = `time_prediction/data/dataset_delivery.py` (delivery),
PR / PD = the matching `preprocess*.py`.

| # | Topic | Official | Ours (full-route) | Effect |
| --- | --- | --- | --- | --- |
| 1 | **Time horizon of labels** | Open tasks with time label `<= 0` or `> 120` min are **removed from the sample** (`label_range=(0,120)`, R:147, R:621; D:160, D:540) | Every open package, any horizon | **Main reason for the ETA gap**: delivery labels mean 80 min / p99 520 in ours vs. at most 120 min there. Also shrinks route samples |
| 2 | **Route label truncation** | Label = open tasks finished before the **next package the courier accepts** (`close_accept_time`, R:179-191, D:189-202). Predictions still cover every open task | Full order of all open packages | Labels are short (mean ~2-3), so ED ≈ 2. KRC adds (label, non-label) pairs. ETA is also scored on the truncated set only |
| 3 | **Windows** | Each courier-day is cut into windows of 24 consecutive finishes, stride T=12 (R:537-542); at most T=12 sample steps per window (R:118) | One snapshot after every finish | Overlapping windows can repeat a query time; long days are covered unevenly |
| 4 | **Test-mode window drop** | If any open task of a processed step finishes outside the 24-row window, the **whole window is discarded in test mode** (R:138-143, D:152-156); in train/val those tasks are just removed | Never | Removes busy windows from the test set → the test set is biased toward simpler moments |
| 5 | **Minimum size** | Step kept only if more than `N_min` tasks remain: pickup N_min=2 → ≥3 (R:622, R:154), delivery N_min=1 → ≥2 (D:541, D:167) | ≥1 open | Drops easy 1-2 package snapshots |
| 6 | **Open-set definition** | Pickup: `accept < now < finish` (PR:152-153). Delivery: `accept < now` and later in the finish order (PD:130-131) | `accept <= q < finish` | Small |
| 7 | **Clock** | Minute of day; accept on an earlier day → minus 1440 (PR:214, PD:193). Delivery sorted by the raw time string (PD:274) | Absolute timestamps; R4 drops deliveries finished on another day | Small (R4 rows get odd negative labels there, which rule 1 then removes) |
| 8 | **Cleaning** | GPS-drift filter (speed > 500 m/min on two consecutive rows, PR:61), drop a row at the same spot < 1 min after the previous one (PR:116), drop couriers meeting all five low-activity conditions (PR:94-99) | R1-R8, G1-G2 (`clean.py`) | Small (see counts in `results/paper_samples_summary.json`) |
| 9 | **Split** | Dates sorted, 60/20/20 of the dates; **one gap day dropped** before val and before test (R:583-585) | `config.SPLIT_CUTS` | Delivery: same training days, val starts 08-20 (ours 08-19), test 09-26 (ours 09-25). Pickup: train ≤ 08-21 (ours 08-22), val 08-23..09-25, test 09-27..10-31 (same test days) |
| 10 | **Courier statistics** | Computed on **all days** incl. test (`courier_info`, PR:169-200) and used in features (`dis_avg_day`) and SPEED (`speed_avg_order`) | Training days only | Leakage in the official code. Paper mode uses training days by default; `--courier-stats all` reproduces the official choice |
| 11 | Max open tasks | 25 slots, but window logic caps the open set at 23 | ≤ 25 | — |

## 2. Metrics (`route_prediction/utils/eval.py`, `time_prediction/utils/eval.py`)

| Metric | Official | Ours (full-route) |
| --- | --- | --- |
| HR@k | `|pred[:e] ∩ label[:e]| / e`, `e = min(k, label_len)` (eval.py:10-17); prediction is over **all** open tasks, label is truncated | same formula, label = full order |
| KRC | Pairs inside the label **plus (label, non-label) pairs**, non-label items ranked after the label; returns **1 when label_len = 1** (eval.py:20-53) | Kendall tau over all pairs |
| LSD | **Squared**: mean over label items of (true pos − pred pos)² (`mode='square'`, eval.py:83-107, called at :247). The paper prints an absolute-value formula but the code squares | `lsd` (absolute) and `lsd_sq` |
| ED | Prediction **restricted to label items**, then Levenshtein vs. the truncated label (eval.py:109-124) | Levenshtein over the full sequences |
| Length filter | Samples with `label_len` in [1, 25] (`Metric([1, 25])`); route also reported for [1,5], [1,11], [1,15] | n ≤ 25 |
| MAE / ACC@m | Errors on the **truncated-label packages only**, pooled over packages; ACC uses `<=` (time eval.py:152-161, :480-509) | All open packages; ACC uses `<` |
| RMSE | Package-weighted mean of **per-sample RMSEs** (not a pooled RMSE) | pooled RMSE |
| ACC@20 vs @30 | Both computed (`acc_eta@10..60`); the main text reports @20 | we report both |

## 3. Baselines

| Method | Official implementation | Paper mode port |
| --- | --- | --- |
| TimeGreedy | argmin of `promise − now` (window end) over open tasks; ties → lowest shuffled slot (basic_model.py:229, :62-74) | `paper_baselines.time_greedy` |
| DistanceGreedy | nearest next from the start node using `E_abs_dis` (int geodesic meters), ties → lowest slot (basic_model.py:77-102) | `distance_greedy(official=True)` |
| DistanceGreedy quirk | `E_abs_dis` row 0 is overwritten by every step's start distances (R:313-314), and the row of a start node that was never a valid task in that window is all zeros (only valid tasks get pairwise distances, R:384-403). So the first pick is sometimes random or measured from the wrong point | reproduced in `d_start_off`; `DistanceGreedy-fixed` uses the true start distance |
| "LightGBM" route (OSquare) | `LGBMClassifier(use_missing=True, zero_as_missing=True)` with **default hyperparameters**; input = 25 shuffled slots × 8 node features flattened (accept min, lng, lat, AOI type, rel. dist, abs. dist, promise − accept, promise − now); one row per truncated-label position with chosen slots zeroed; greedy decoding over all open slots (osqure/train.py:90-126, :18-60) | `fit_route_gbdt` / `predict_route_gbdt` |
| "LightGBM" ETA | **Actually `MultiOutputRegressor(XGBRegressor())`** with defaults (lgb/train.py:13, :126); input = start features (accept, lng, lat, finish time) + 25 slots × 6 features; target per slot = ETA on truncated-label slots, 0 elsewhere. **Trains on only the first `len(test)` training windows** (`iter_num = test_dataset.max_iter`, lgb/train.py:95) | `fit_eta_gbdt(backend=lightgbm|xgboost)`; `--official-train-cap` reproduces the training cap |
| SPEED | `dist(start, package) / speed_avg_order` (straight line from the current stop, **not cumulative** along a route), evaluated along the true order (speed.py:66-76, :94) | `speed_eta(official=True)` (uses the quirky `E_abs_dis` row too); `SPEED-fixed` uses the true start distance |
| OR-Tools | open-path TSP from the start point, no time windows in the code (basic_model.py:105-218) | not yet ported to paper mode |

## 4. Leakage in the official setting (kept in paper mode only for comparability)

- Rule 1 (120-min filter) and rule 2 (truncation at the next accept) use future finish/accept times to decide
  which packages enter a sample. A deployed model cannot know which packages will be finished within 2 h.
- Rule 4 (test-mode window drop) uses future finish order to remove test samples.
- Courier statistics over all days (rule 10).

These are acceptable for reproducing the paper's tables but not for our research claims; the full-route
setting stays the main setting for the new-courier study.

## 5. Results in paper mode (2026-11-15, test split, official split, courier stats from training days)

Commands: `python scripts/build_paper_samples.py`, then `python scripts/run_paper_baselines.py --methods rules`,
`--methods gbdt_route --datasets pickup`, `--methods gbdt_eta --datasets delivery`. JSON in `results/paper_*.json`.

Samples (`results/paper_samples_summary.json`): pickup 97,986 test steps, 75,539 with a non-empty label (mean
open 5.8, mean label length 2.85, 11,394 test windows dropped by rule 4); delivery 240,856 scored test steps.
Delivery ETA labels in the scored set: mean 39 min, median 32 (full-route setting: mean 80, median 42).

**Pickup route (paper Table 3)**

| Method | HR@3 | KRC | LSD | ED | Paper HR@3 / KRC / LSD / ED |
| --- | --- | --- | --- | --- | --- |
| TimeGreedy | 59.81 | 40.01 | 5.21 | 2.24 | 59.81 / 39.93 / 5.20 / 2.24 |
| DistanceGreedy (official quirk) | 61.05 | 42.76 | 5.37 | 1.94 | 61.07 / 42.84 / 5.35 / 1.94 |
| DistanceGreedy-fixed | 62.99 | 45.21 | 4.80 | 1.88 | — |
| LightGBM (OSquare, default params) | **69.35** | 52.59 | 3.49 | 1.96 | 70.63 / 54.48 / 3.27 / 1.92 |
| XGBoost (same setup, default `XGBClassifier`, zeros as missing) | 69.24 | 53.06 | 3.48 | 1.97 | — |

**Delivery ETA (paper Table 4)**

| Method | MAE | RMSE | ACC@20 | Paper MAE / RMSE / ACC@20 |
| --- | --- | --- | --- | --- |
| SPEED (official quirk) | 28.20 | 33.28 | 51.59 | 26.68 / 31.31 / 52.57 |
| SPEED-fixed | 26.34 | 31.35 | 55.35 | — |
| LightGBM (multi-output LGBMRegressor, all training samples) | **16.72** | 19.73 | 69.91 | 17.24 / 20.40 / 67.44 |
| XGBoost (multi-output `XGBRegressor`, all training samples) | 17.11 | 20.37 | 68.51 | — |
| XGBoost + official training cap (exact official code path) | 17.79 | 21.14 | 66.48 | — |

XGBoost runs: `--route-backend xgboost` / `--eta-backend xgboost` (results `paper_gbdt_*_xgb.json`). Seen vs new
couriers: route HR@3 70.18 / 67.01, KRC 55.47 / 47.37; ETA MAE 16.52 / 18.57.

**Reproduction gate: PASSED.** LightGBM HR@3 is 1.8% below the paper (gate: within ~5%), ETA MAE 3.0% below
(the exact official XGBoost path is 3.2% above). The rule baselines match the paper to the second decimal, which
confirms that sample construction and metrics are now aligned.

**Seen vs new couriers (go/no-go input)**, LightGBM in paper mode:

| | Seen | New | Gap |
| --- | --- | --- | --- |
| Pickup route HR@3 / KRC (samples) | 70.30 / 55.36 (53,062) | 67.10 / 46.06 (22,477) | −3.2 HR@3, **−9.3 KRC** |
| Delivery ETA MAE / ACC@20 (packages) | 16.47 / 70.03 (882,821) | 17.33 / 69.62 (356,045) | +0.86 min |

The route gap for new couriers is large (KRC −9.3 points, 29% of test samples); the ETA gap is small.
Caveat: the LightGBM inputs have no courier ID; the only courier-specific input is the relative distance
(`dis_avg_day` from training days, set to 0 for new couriers). So the gap mixes "new couriers are harder" with
that one missing statistic, not a missing embedding. The deep models' unknown-courier fallback still has to be measured.

# Experiment log: joint route + ETA model

One entry per training run or configuration change, newest last. Results JSONs are in `results/`.
Setting: **full-route** (ours, `src/lade/snapshots.py`), not paper mode, unless stated.

---

## Run 1: pickup, seed 0, selection = joint (2026-11-16)

**Status:** finished. Superseded as a configuration by run 2 (the selection rule changed), so it is not one of the 5 reported seeds.
**Results file:** `results/joint_pickup_h64_l3_lam1.0_s0_selectjoint.json`. It was rebuilt from the pasted Colab output because
the original JSON and the `.pt` model were lost when Colab disconnected. The seen vs new courier split is **not available**.

### Method
- **Model** (`src/lade/models/joint.py`): one courier node plus up to 25 package nodes, fully connected. Edges carry
  distance, same AOI and window-end gap. 3 layers of edge-aware graph attention (GATv2-style, 4 heads), hidden 64,
  dropout 0.1, with AOI, AOI-type and courier-ID embeddings. The route comes from a pointer decoder with a GRU state
  (teacher forcing in training, greedy at inference). The ETA comes from an LSTM along the visiting order that predicts
  positive increments, summed so ETAs never decrease along the route.
- **Loss:** route cross-entropy + λ · Huber(δ = 10 min) on ETA, with λ = 1.0.
- **Training:** Adam, lr 1e-4, batch 64, grad-clip 1.0. Up to 100 epochs with early stopping, patience 10.
- **Model selection:** after each epoch, score = KRC/100 − MAE/60 on the full validation set. The best epoch's
  weights are reloaded and evaluated on test once.
- **Data:** full-route pickup snapshots with a chronological split by days. 25% of training snapshots (163,538) were
  used; validation (296,771) and test (280,775) were used in full.
- **Hardware:** Google Colab, T4 GPU, about 280 s per epoch including validation. Data came from the Parquet bundle
  (`scripts/export_for_colab.py`).
- **Command:** `python scripts/train_joint.py --dataset pickup --frac 0.25 --seed 0`

### Training log (validation)
| Epoch | Train loss (total / route / ETA) | Val HR@3 | Val KRC | Val MAE | Score |
| --- | --- | --- | --- | --- | --- |
| 0 | 263.8 / 1.251 / 262.5 | 71.48 | 55.72 | 51.79 | −0.3059 |
| **1** | 203.1 / 1.116 / 201.9 | 73.87 | 59.26 | **50.38** | **−0.2471** (best) |
| 2 | 187.0 / 1.059 / 186.0 | 74.14 | 59.58 | 53.33 | −0.2932 |
| 3 | 177.7 / 1.033 / 176.6 | 74.27 | 59.84 | 54.84 | −0.3156 |
| 4 | 171.6 / 1.018 / 170.5 | 74.23 | 59.79 | 54.73 | −0.3142 |
| 5 | 166.8 / 1.010 / 165.8 | 74.27 | 59.90 | 53.52 | −0.2930 |
| 6 | 163.1 / 1.005 / 162.1 | 74.20 | 59.80 | 54.56 | −0.3113 |
| 7 | 160.1 / 1.002 / 159.1 | 74.21 | 59.83 | 54.19 | −0.3049 |
| 8 | 157.2 / 1.000 / 156.2 | 74.19 | 59.78 | 53.16 | −0.2883 |
| 9 | 154.7 / 0.998 / 153.7 | 74.20 | 59.76 | 54.15 | −0.3048 |
| 10 | 152.4 / 0.996 / 151.4 | 74.25 | 59.88 | 54.96 | −0.3171 |
| 11 | 150.3 / 0.994 / 149.3 | 74.26 | **59.96** | 55.35 | −0.3229 |

Early stop after epoch 11. Best epoch was 1. Total time 3,372 s to epoch 11.

### Test results (epoch-1 model)
| Method (pickup, full-route, test) | HR@3 | KRC | LSD | ED | ETA MAE | RMSE | ACC@20 | ACC@30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimeGreedy | 65.23 | 50.00 | 1.65 | 6.61 | – | – | – | – |
| SPEED | – | – | – | – | 111.58 | 173.26 | 25.86 | 33.94 |
| GBDT preview (sklearn stand-in, subsample) | 74.34 | 57.26 | 1.43 | 5.89 | – | – | – | – |
| **Joint model, run 1** | **74.33** | **58.47** | **1.39** | **5.80** | **64.14** | **106.09** | **35.68** | **46.48** |

### Observations
1. **Route converges early.** Validation HR@3 and KRC level off after epoch 3–5, around 74.2 HR@3 and 59.8–60.0 KRC.
2. **The ETA head overfits.** Validation MAE was best at epoch 1 (50.4), then sat around 53–55 while the training
   ETA loss kept falling. Test MAE (64.1) is also well above validation MAE, partly because test days come later in
   the season.
3. **ETA dominates the loss.** The ETA term is about 150–260 and the route term about 1.0. With λ = 1, most of the
   gradient comes from ETA.
4. **The combined score let ETA noise pick the epoch.** The saved model came from epoch 1, before the route part had
   converged (KRC 59.26 vs 59.96 at epoch 11).
5. **Compared with baselines:** clearly better than the rule baselines on route (+9.1 HR@3, +8.5 KRC) and ETA
   (MAE 64 vs 112). Only level with the GBDT preview on route (HR@3 equal, KRC +1.2). A proper LightGBM full-route
   baseline is needed before claiming a gain.

### Decisions for run 2 (made on validation evidence only)
- **Select the saved model by the task each dataset is scored on:** `--select krc` for pickup and `--select mae` for
  delivery. This matches the paper's scoring (route on pickup, ETA on delivery).
- **Patience 10 → 5**, since route converges within about 5 epochs.
- **Log everything:** `train_joint.py` now prints the seen vs new split, the best epoch and run time, and the Colab
  notebook writes `results/log_<dataset>_s<seed>.txt` and downloads the results after every run.
- **Still open:** a λ sweep {0.1, 0.5, 1, 2} on validation (planned in `pipeline_design.md`), and a real LightGBM
  full-route route baseline.

---

## Run 2: pickup and delivery, 5 seeds, per-task selection (2026-11-17/18)

**Status:** finished, but the **new-courier and overall numbers are invalid** because of a bug (below). Seen-courier
numbers are valid. Files are in `results_joint/` (JSON, logs, models), kept out of `results/` on purpose.

**Settings** (checked identical in all 10 JSONs): `--frac 0.25 --select krc` (pickup) / `mae` (delivery), `--patience 5`,
λ 1.0, lr 1e-4, batch 64, hidden 64, 3 layers, dropout 0.1, Colab T4. Each seed uses a different random 25% of the
training snapshots; validation and test are always used in full.
Epochs run (best epoch): pickup 11 (5), 8 (2), 43 (37), 33 (27), 9 (3); delivery 17 (11), 13 (7), 10 (4), 11 (5), 21 (15).
Pickup seed 0 repeated run 1's training exactly (same seed, same T4), as expected.

### Bug found in the results: new couriers get an input of −1,000,000
`c_known` (1 = courier seen in training) is a model input. In training it is always 1, so its training std is 0 and
the scaler (`sd + 1e-6`) turned every new courier's `c_known = 0` into (0 − 1) / 1e-6 = −10⁶. The weight on that
input was never trained, so new-courier predictions depended on its random initial value. That's why pickup
new-courier ETA MAE ranged from 69 to 261 across seeds (163 ± 82, worse than SPEED's 142.6). Seen couriers get 0
there, the same as in training, so their results are unaffected.
A second, related gap: the 10% courier dropout replaced only the courier ID, and kept `c_known = 1` and the real
courier statistics, so the model never saw an input that looks like a real new courier.

**Fix (2026-11-18):** (1) `Scaler` gives columns that are constant in training sd = 1. (2) Courier dropout moved to
`train_joint.batches()`: a dropped sample now gets ID 0, `c_known = 0` and the same fallback statistics a real
unseen courier receives (`--courier-dropout 0.1`). Test `test_new_courier_inputs_bounded_and_simulated_exactly`
checks both. **All 10 runs have to be repeated.**

### Valid part: seen couriers (test, mean ± std over 5 seeds)
| Seen couriers | HR@3 | KRC | ED | ETA MAE | ACC@20 |
| --- | --- | --- | --- | --- | --- |
| Pickup: joint model | 74.98 ± 0.04 | 60.49 ± 0.07 | 5.81 ± 0.01 | 36.66 ± 1.06 | 45.94 ± 1.08 |
| Pickup: TimeGreedy / SPEED | 64.77 | 50.50 | – | 102.76 | 28.42 |
| Pickup: GBDT preview | 74.26 | 58.15 | – | – | – |
| Delivery: joint model | 78.38 ± 0.14 | 46.78 ± 0.19 | 3.47 ± 0.01 | 56.06 ± 0.40 | 49.97 ± 0.44 |
| Delivery: DistanceGreedy / SPEED | 77.27 | 40.99 | – | 64.55 | 43.86 |
| Delivery: GBDT preview | – | – | – | 56.46 | 36.97 |

(For reference only, invalid: overall pickup HR@3 74.94 ± 0.10, KRC 59.55 ± 0.13, ETA MAE 64.58 ± 17.45; overall delivery
HR@3 78.70 ± 0.10, KRC 45.32 ± 0.14, ETA MAE 54.78 ± 1.82.)

### Observations
- **Route is very stable across seeds** (KRC std about 0.1), even when the number of epochs varies a lot (8 to 43).
- **Seen-courier ETA is good on pickup** (MAE 36.7 vs SPEED 102.8). On delivery it is only level with the GBDT preview
  on MAE (56.1 vs 56.5), though better on ACC@20 (50.0 vs 37.0).
- **Delivery new couriers look easier than seen ones** even for SPEED (MAE 49.5 vs 64.6), so their snapshots differ.
  Compare seen vs new against baselines on the same rows, never across groups.

---

## Run 3: rerun of run 2 with the new-courier fix (2026-11-18/19). CURRENT RESULTS

**Status:** finished, valid. All 10 JSONs say `code_version: run3 (built 2026-11-18 12:58)`, and their settings are
identical to run 2 plus `courier_dropout 0.1` (simulated unseen courier). Files: `results_run3/` (original download),
copied to `results/joint_*_s{0..4}.json`, `results/logs_run3/` and `results/models/`.
Epochs run (best epoch): pickup 11 (5), 8 (2), 27 (21), 10 (4), 9 (3); delivery 10 (4), 11 (5), 12 (6), 16 (10), 21 (15).
GPU time: pickup 5.4 h, delivery 5.3 h on a Colab T4 (about 300–350 s per epoch).

### Test results, full-route setting (mean ± std over 5 seeds)
| Pickup | HR@3 | KRC | LSD | ED | ETA MAE | RMSE | ACC@20 | ACC@30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimeGreedy | 65.23 | 50.00 | 1.65 | 6.61 | – | – | – | – |
| SPEED | – | – | – | – | 111.58 | 173.26 | 25.86 | 33.94 |
| GBDT preview* | 74.34 | 57.26 | 1.43 | 5.89 | – | – | – | – |
| **Joint model** | **75.27 ± 0.05** | **60.01 ± 0.08** | **1.33 ± 0.00** | **5.67 ± 0.01** | **38.13 ± 0.91** | **59.85 ± 0.98** | **45.08 ± 0.81** | **57.60 ± 0.90** |

| Delivery | HR@3 | KRC | LSD | ED | ETA MAE | RMSE | ACC@20 | ACC@30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DistanceGreedy | 77.89 | 41.12 | – | 3.45 | – | – | – | – |
| SPEED | – | – | – | – | 60.88 | – | 44.92 | – |
| GBDT preview* | – | – | – | – | 53.90 | 89.97 | 37.05 | – |
| **Joint model** | **78.90 ± 0.11** | **46.96 ± 0.42** | **1.19 ± 0.01** | **3.30 ± 0.02** | **51.66 ± 0.51** | **101.17 ± 0.91** | **51.81 ± 0.35** | **62.47 ± 0.31** |

*GBDT preview = sklearn HistGradientBoosting stand-in, trained on a subsample (route: tested on 20k snapshots only).
It is not a proper LightGBM full-route baseline and not scored on exactly the same rows.

### Seen vs new couriers (test, mean ± std)
| | Seen: KRC | New: KRC | Seen: ETA MAE | New: ETA MAE |
| --- | --- | --- | --- | --- |
| Pickup, joint model | 60.54 ± 0.09 | 58.33 ± 0.18 | 36.86 ± 0.95 | 42.62 ± 0.77 |
| Pickup, TimeGreedy / SPEED | 50.50 | 48.39 | 102.76 | 142.62 |
| Pickup, GBDT preview | 58.15 | 54.45 | – | – |
| Delivery, joint model | 46.63 ± 0.44 | 47.84 ± 0.39 | 55.76 ± 0.60 | 38.85 ± 0.45 |
| Delivery, DistanceGreedy / SPEED | 40.99 | 41.46 | 64.55 | 49.45 |

### Observations
1. **The fix worked.** Seen-courier numbers are unchanged from run 2, as they should be (pickup KRC 60.54 vs 60.49,
   MAE 36.9 vs 36.7). New-courier pickup ETA MAE fell from 162.9 ± 82.1 to 42.6 ± 0.8. Overall pickup ETA std fell
   from 17.45 to 0.91. The "ETA overfitting" seen in runs 1 and 2 was mostly this bug.
2. **Very stable across seeds:** KRC std 0.08 (pickup) and 0.42 (delivery); ETA MAE std under 1 min.
3. **Against baselines:** clearly better than every rule baseline on every metric. Against the GBDT preview: pickup
   route +2.8 KRC and +0.9 HR@3; delivery ETA −2.2 MAE and +14.8 ACC@20, **but RMSE is worse** (101.2 vs 90.0). Huber
   loss under-weights the long tail of very late packages. A real LightGBM full-route baseline on the same rows is
   still needed before claiming a gain.
4. **Seen vs new is confounded by the snapshots.** Pickup: the joint model's KRC gap (−2.2) is about the same as
   TimeGreedy's (−2.1), which uses no courier information. Its ETA gap (+16%) is smaller than SPEED's (+39%). Delivery
   new couriers are easier for every method. So these raw gaps don't measure the value of knowing the courier.
   **Next:** evaluate seen couriers again with their identity hidden (same rows and same model) to isolate it.

## LightGBM full-route baseline (2026-11-19, local CPU)
Real LightGBM 4.7 (not the sklearn preview), seed 0, **full test split = the same rows as the joint model**.
Old preview JSONs were overwritten; logs in `results/logs_gbdt/`.
```
python scripts/run_gbdt.py --task eta   --datasets delivery pickup                       # all training days
python scripts/run_gbdt.py --task route --datasets pickup delivery --train-frac 0.25 --max-val-snapshots 30000
```
- ETA: one regressor per package (L2 loss), trained on all training days, early stopping on the full val split.
  Best iteration: delivery 1039, pickup 103. Fit + predict about 4 min (delivery) and 3 min (pickup).
- Route: greedy next-stop classifier. Trained on the same 25% of training snapshots as joint seed 0 (12.9M decision
  rows for pickup, 3.1M for delivery). Early stopping on 30k val snapshots. Best iteration: pickup 460, delivery 618.
  About 28 min (pickup) and 12 min (delivery).

| Test, full-route | HR@3 | KRC | ED | ETA MAE | RMSE | ACC@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Pickup, LightGBM | 74.37 | 57.50 | 5.88 | **37.60** | **53.77** | 37.41 |
| Pickup, joint (5 seeds) | **75.27 ± 0.05** | **60.01 ± 0.08** | **5.67** | 38.13 ± 0.91 | 59.85 | **45.08** |
| Delivery, LightGBM | 78.01 | 41.77 | 3.44 | 53.18 | **89.16** | 37.08 |
| Delivery, joint (5 seeds) | **78.90 ± 0.11** | **46.96 ± 0.42** | **3.30** | **51.66 ± 0.51** | 101.17 ± 0.91 | **51.81** |

| Seen / new couriers | Seen KRC | New KRC | Seen ETA MAE | New ETA MAE |
| --- | --- | --- | --- | --- |
| Pickup, LightGBM | 58.26 | 55.09 | 34.25 | 49.41 |
| Pickup, joint | 60.54 | 58.33 | 36.86 | 42.62 |
| Delivery, LightGBM | 41.65 | 42.10 | 56.00 | 44.38 |
| Delivery, joint | 46.63 | 47.84 | 55.76 | 38.85 |

Observations:
1. **Route: the joint model wins clearly.** KRC +2.5 on pickup and +5.2 on delivery, larger than 30x the seed std.
2. **ETA is mixed.** The joint model has lower MAE on delivery (−1.5) and much higher ACC@20 (+14.7 on delivery,
   +7.7 on pickup). LightGBM has lower RMSE (89 vs 101 delivery, 54 vs 60 pickup) and ties or wins MAE on pickup
   (37.6 vs 38.1). Loss explains much of it: LightGBM is trained with L2 (mean, good RMSE), the joint model with
   Huber (closer to median, good MAE/ACC@20, weaker on the late tail).
3. **New couriers: the joint model degrades less.** Pickup ETA gap: LightGBM +44% (34.3 -> 49.4), joint +16%
   (36.9 -> 42.6). The joint model is worse than LightGBM on seen couriers' pickup ETA (36.9 vs 34.3) but much better on
   new ones (42.6 vs 49.4). Route KRC gap: LightGBM −3.2, joint −2.2. Courier dropout during training is the likely
   reason; the hidden-courier evaluation should confirm it.
4. Caveat: LightGBM is one seed (trees are near-deterministic; the route subsample would vary slightly with seed).

## Hidden-courier evaluation (2026-11-19, local CPU, run 3 models)
`python scripts/eval_hidden_courier.py` (about 2.4 h on CPU) -> `results/hidden_courier_{pickup,delivery}.json`, log `results/hidden_courier_log.txt`.
Seen-courier test snapshots only (pickup 213,428, delivery 279,424), the 10 saved run 3 models, no retraining.
Each snapshot is scored with the normal input (known) and with the input a real new courier gets (hidden: id 0,
c_known 0, fallback medians). Same rows, same model, so the difference is the value of knowing the courier.
Sanity check passed for every seed: "known" reproduces the saved seen-courier test KRC.

| Seen couriers, mean ± std (5 seeds) | known | hidden | hidden − known (paired) |
| --- | --- | --- | --- |
| Pickup KRC | 60.54 ± 0.08 | 60.53 ± 0.06 | −0.01 ± 0.06 |
| Pickup HR@3 | 74.99 | 74.98 | −0.01 ± 0.06 |
| Pickup ETA MAE | 36.86 ± 0.85 | 37.18 ± 0.77 | **+0.32 ± 0.11** |
| Pickup ETA RMSE | 57.39 | 57.75 | +0.36 ± 0.17 |
| Delivery KRC | 46.63 ± 0.39 | 46.51 ± 0.40 | −0.12 ± 0.07 |
| Delivery HR@3 | 78.34 | 78.27 | −0.07 ± 0.05 |
| Delivery ETA MAE | 55.76 ± 0.53 | 54.98 ± 0.26 | **−0.78 ± 0.39** (hidden is better) |
| Delivery ETA ACC@20 | 50.20 | 51.10 | +0.89 ± 0.23 (hidden is better) |

no_id (id 0, own statistics kept; never seen in training, diagnostic only) is between the two or slightly worse.

Conclusions:
1. **The current joint model gets almost nothing from knowing the courier.** Route: < 0.15 KRC on both datasets.
   ETA: +0.3 min (0.9%) on pickup; on delivery the individual information even hurts by 0.8 min.
2. So the raw seen-vs-new gaps of the joint model (pickup KRC −2.2, ETA +5.8 min) come from the snapshots that new
   couriers handle, not from missing courier knowledge.
3. **Go/no-go for the inductive courier embedding (GraphSAGE): no-go in its current form for the joint model.** Its
   headroom is bounded by known − hidden, which is ~0. Possible reasons it is ~0: courier input is coarse (5 stats +
   16-d id embedding), training-period statistics are stale on the test months (would explain delivery), or the
   model does not exploit it. Richer courier information (courier-AOI habits) could still help, but this result
   gives no evidence for it.
4. Open check: do the same for LightGBM (needs a saved model; the ETA model retrains in about 4 min), to see whether
   a tree model can use the courier statistics where the joint model cannot.

## Hidden-courier evaluation for LightGBM (2026-11-20, local CPU)
`python scripts/eval_hidden_courier_gbdt.py --task {eta,route} --datasets ...` -> `results/hidden_courier_gbdt_{task}_{dataset}.json`,
models saved as `results/models/gbdt_{task}_{dataset}.txt`, logs `results/logs_gbdt/hidden_*.txt`.
Retrained with the same data, settings and seed as the baseline; the check passed exactly (known = saved seen-courier
result). Same seen-courier rows; hidden = c_known 0 + training-courier medians (what a real new courier gets).
LightGBM has no courier id, so its courier information = the courier statistics. Note: LightGBM never saw the
fallback input in training (no courier dropout), so part of its drop is input shift, not only lost information.

| Seen couriers | LightGBM known | LightGBM hidden | diff | Joint known | Joint hidden | diff |
| --- | --- | --- | --- | --- | --- | --- |
| Pickup ETA MAE | **34.25** | 46.38 | **+12.13 (+35%)** | 36.86 | 37.18 | +0.32 |
| Pickup ETA ACC@20 | 41.45 | 23.54 | −17.91 | 45.66 | 45.33 | −0.33 |
| Pickup KRC | 58.26 | 56.85 | −1.41 | 60.54 | 60.53 | −0.01 |
| Delivery ETA MAE | 56.00 | 60.54 | +4.54 | 55.76 | 54.98 | −0.78 |
| Delivery KRC | 41.65 | 41.45 | −0.20 | 46.63 | 46.51 | −0.12 |

Conclusions (revise the "no-go" of the previous section):
1. **Courier statistics DO carry information, especially for pickup ETA.** With them, LightGBM beats the joint model
   on seen couriers' pickup ETA (34.25 vs 36.86); without them it falls to 46.38. Route: a modest 1.4 KRC on pickup.
2. **The joint model ignores this information.** It is robust (hidden ≈ known) but does not exploit it. Courier
   dropout plus the courier entering only through one context node probably taught it to rely on package features.
   This robustness is why it beats LightGBM on real new couriers (pickup ETA 42.6 vs 49.4).
3. So the earlier joint-model result showed that *this model* doesn't use courier knowledge, not that the knowledge
   is worthless. Headroom exists, mainly for pickup ETA:
   - seen couriers: a joint model that actually uses the statistics could gain up to ~2.6 min (toward LightGBM's 34.25);
   - new couriers: the value depends on estimating a new courier's profile without history (the inductive /
     courier-AOI graph idea), bounded by LightGBM's known-vs-hidden gap.
4. Direction becomes a modified go: (a) make the joint model use courier information (stats fed to the ETA head,
   less or scheduled dropout) and check it keeps the new-courier robustness; (b) then test whether profiles inferred
   for new couriers (AOI graph, or their first tasks of the day, q-time safe) recover part of the gap.

## How LightGBM uses courier information (2026-11-20)
`python scripts/analyze_gbdt_courier.py` -> `results/logs_gbdt/courier_analysis.txt` (ETA models, seen-courier test rows).

| | Pickup ETA | Delivery ETA |
| --- | --- | --- |
| Top features (gain) | to_window_start 66%, to_window_end 30% | c_median_dur 16%, n_open 14%, hour 10%, min_since_accept 10% |
| Courier statistics' share of gain | ~2% | ~37% |
| Mean abs SHAP of the courier block | 7.3 min | 20.7 min |
| Hide c_days / c_tasks_per_day / c_median_dur / c_median_gap | +1.2 / +1.3 / +3.8 / +3.3 min MAE | +1.9 / +3.9 / +1.9 / +3.8 min |
| Hide all (sum of single hides) | **+12.1 (9.5)** | +4.5 (11.5) |
| Spearman: training c_median_dur vs courier's test median ETA (couriers >= 200 test packages) | 0.06 (289 couriers) | **0.70** (402 couriers) |

Findings:
1. LightGBM uses the statistics as a **per-courier tempo offset**. Pickup: the time window sets the ETA and
   c_median_dur shifts it (SHAP −15 min for the fastest fifth of couriers, +9 min for the slowest). Delivery: ETA is
   roughly pace × queue position, so c_median_gap / c_tasks_per_day combine with n_open and min_since_accept.
2. Courier tempo is **stable over months** (delivery rank correlation 0.70), so the statistics are not stale. This
   rules out the "stale statistics" explanation for the joint model's delivery result; the joint model simply does
   not use them.
3. **The fallback row is unrepresentative.** It is the median over couriers (unweighted), dominated by small
   couriers: pickup c_tasks_per_day 1.43 and c_median_dur 95.5 (bottom fifth of test packages). For pickup, hiding
   all four costs more than the sum of single hides (12.1 vs 9.5): part of LightGBM's hidden and new-courier loss is
   a biased fallback plus an unseen combination, not lost information.
4. Ideas this suggests (not run yet): a package-weighted fallback and/or courier dropout for LightGBM; a q-time-safe
   "pace so far today" feature that estimates a new courier's tempo online.

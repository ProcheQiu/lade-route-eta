# lade-route-eta

Joint **route and arrival-time (ETA) prediction** for last-mile couriers on the
[LaDe](https://arxiv.org/abs/2306.10675) Shanghai dataset (Cainiao), with a focus on **couriers never seen in
training** (27% of delivery and 24% of pickup test rows).

Given a courier's open packages at a query time, the model predicts the order in which they will be visited and
when each one will be picked up or delivered.

## Model
- **Encoder:** edge-aware graph attention (GATv2-style) over a courier node and up to 25 package nodes
  (distances, time windows, area of interest, courier context).
- **Route decoder:** pointer network with a GRU state, greedy decoding.
- **ETA decoder:** LSTM along the predicted route, predicting positive time increments.
- **Loss:** route cross-entropy + λ · Huber(ETA). Courier dropout during training simulates unseen couriers.

Code: [`src/lade/models/joint.py`](src/lade/models/joint.py), training: [`scripts/train_joint.py`](scripts/train_joint.py).

## Results (test set, full-route setting, mean ± std over 5 seeds)
Chronological split by day (6:2:2). Every method is scored on the same test snapshots.

| Method | Pickup HR@3 | Pickup KRC | Pickup ETA MAE (min) | Delivery KRC | Delivery ETA MAE | Delivery ACC@20 |
| --- | --- | --- | --- | --- | --- | --- |
| TimeGreedy / DistanceGreedy / SPEED | 65.23 | 50.00 | 111.6 | 41.12 | 60.88 | 44.9 |
| LightGBM | 74.37 | 57.50 | **37.60** | 41.77 | 53.18 | 37.1 |
| **Joint model** | **75.27 ± 0.05** | **60.01 ± 0.08** | 38.13 ± 0.91 | **46.96 ± 0.42** | **51.66 ± 0.51** | **51.8** |

- The joint model is clearly better at route prediction. ETA results are mixed: it wins on MAE and ACC@20, while
  LightGBM has lower RMSE (89 vs 101 min on delivery).
- **New couriers:** on pickup ETA the joint model's error is 42.6 min vs 49.4 for LightGBM.
- **Hidden-courier evaluation** ([`scripts/eval_hidden_courier.py`](scripts/eval_hidden_courier.py)): seen
  couriers are scored again with their identity replaced by the new-courier input, on the same rows. LightGBM loses
  12.1 min of pickup ETA MAE; the joint model loses only 0.3. Courier statistics carry a stable per-courier
  "tempo" that the joint model does not yet exploit. Details in [`docs/experiment_log.md`](docs/experiment_log.md).

**Paper mode.** A port of the official LaDe sample construction and metrics reproduces the paper's baselines:
rule baselines match to two decimals, and LightGBM gets pickup HR@3 69.35 (paper 70.63) and delivery ETA MAE 16.72
(paper 17.24). See [`docs/alignment_notes.md`](docs/alignment_notes.md).

## Data
The data are **not included**. LaDe is licensed CC BY-NC 4.0 (non-commercial research use). Download the Shanghai files from
Hugging Face ([Cainiao-AI/LaDe-D](https://huggingface.co/datasets/Cainiao-AI/LaDe-D) and
[Cainiao-AI/LaDe-P](https://huggingface.co/datasets/Cainiao-AI/LaDe-P)), then place them at:
```
data/raw/delivery/delivery_sh.csv
data/raw/pickup/pickup_sh.csv
```

## Quick start
```bash
pip install -r requirements.txt
python scripts/clean_data.py                 # ~1 min
python scripts/build_snapshots.py            # ~1 min
python -m pytest tests -q                    # leakage, metric and model-array tests
python scripts/run_rule_baselines.py
python scripts/run_gbdt.py --task eta --datasets delivery pickup
python scripts/run_gbdt.py --task route --datasets pickup delivery --train-frac 0.25 --max-val-snapshots 30000
python scripts/train_joint.py --dataset pickup --smoke        # CPU check
python scripts/train_joint.py --dataset pickup --select krc --patience 5 --seed 0   # GPU recommended
python scripts/summarize_results.py
```
There is no local GPU, so the joint model was trained on Google Colab: `scripts/export_for_colab.py` builds the
bundle, and `colab/train_joint_colab_nodrive.ipynb` runs it. The full run order is in [`CLAUDE.md`](CLAUDE.md).

## Layout
```
src/lade/            cleaning, snapshots (leakage contract), metrics, paper mode, baselines, models
scripts/             one script per pipeline step
tests/               leakage, metric and model-array tests
docs/                pipeline design, experiment log, alignment with the official code, figures
results/             result JSONs and logs (model weights not included)
results_joint/       run 2 (invalid for new couriers: bug, see its README)
colab/               Colab notebooks
```

## Rules the code follows
- Features use only information dated at or before the query time. Courier and area statistics and feature scaling
  come from training days only.
- The test set is touched once per final configuration. All selection uses validation.
- Every results table reports seen and new couriers separately.

## Reference
Wu, L. et al. *LaDe: The First Comprehensive Last-mile Delivery Dataset from Industry.* arXiv:2306.10675.

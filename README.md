# lade-route-eta

Joint **route and arrival-time (ETA) prediction** for last-mile couriers on the Shanghai part of
**[LaDe](https://github.com/wenhaomin/LaDe)**, the last-mile delivery dataset released by Cainiao Network and its
academic partners (Wu et al., KDD 2024; see [Data and credit](#data-and-credit)). The focus is on **couriers never
seen in training** (27% of delivery and 24% of pickup test rows).

This is an independent research project. It is not affiliated with or endorsed by Cainiao or the LaDe authors.

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

**Paper mode.** A port of the official LaDe sample construction and metrics (from the authors' code at
[wenhaomin/LaDe](https://github.com/wenhaomin/LaDe), commit `4d64660`) reproduces the paper's baselines:
rule baselines match to two decimals, and LightGBM gets pickup HR@3 69.35 (paper 70.63) and delivery ETA MAE 16.72
(paper 17.24). See [`docs/alignment_notes.md`](docs/alignment_notes.md).

## Data and credit
All data come from **LaDe** (Last-mile Delivery dataset), created and released by **Cainiao Network** and the
authors below. It covers 2022-05-01 to 2022-10-31. This project uses the Shanghai files of the delivery
(LaDe-D) and pickup (LaDe-P) subsets. The baseline set, task definitions, metrics and paper-mode sample construction
follow the LaDe paper and its official code.

- Paper: Wu, L., Wen, H., Hu, H., Mao, X., Xia, Y., Shan, E., Zheng, J., Lou, J., Liang, Y., Yang, L., et al. (2024).
  *LaDe: The First Comprehensive Last-mile Express Dataset from Industry.* In Proceedings of the 30th ACM SIGKDD
  Conference on Knowledge Discovery and Data Mining (KDD '24), pp. 5991-6002.
  [doi:10.1145/3637528.3671548](https://dl.acm.org/doi/10.1145/3637528.3671548) · [arXiv:2306.10675](https://arxiv.org/abs/2306.10675)
- Data: [Cainiao-AI/LaDe-D](https://huggingface.co/datasets/Cainiao-AI/LaDe-D) and
  [Cainiao-AI/LaDe-P](https://huggingface.co/datasets/Cainiao-AI/LaDe-P) on Hugging Face
- Official code: [github.com/wenhaomin/LaDe](https://github.com/wenhaomin/LaDe)

**The data are not included in this repository.** LaDe is released under CC BY-NC 4.0, for non-commercial research
use. Read the terms on the dataset pages before downloading. Then put the Shanghai files at:
```
data/raw/delivery/delivery_sh.csv
data/raw/pickup/pickup_sh.csv
```

If you use LaDe, please cite the original paper:
```bibtex
@inproceedings{wu2024lade,
  title={LaDe: The First Comprehensive Last-mile Express Dataset from Industry},
  author={Wu, Lixia and Wen, Haomin and Hu, Haoyuan and Mao, Xiaowei and Xia, Yutong and Shan, Ergang and Zheng, Jianbin and Lou, Junhong and Liang, Yuxuan and Yang, Liuqing and others},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={5991--6002},
  year={2024}
}
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

## Acknowledgments
Thanks to Cainiao Network and the LaDe authors for releasing the dataset, the benchmark and the baseline code.

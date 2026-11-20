Run 2 (2026-11-17/18, Colab T4): joint model, 5 seeds x {pickup, delivery}, --select krc|mae --patience 5, lam 1.0.
**New-courier (c_known = 0) results and the overall results are INVALID**: c_known is constant in training and the
scaler turned it into -1e6 for new couriers (fixed 2026-11-18 in src/lade/models/data.py). Seen-courier (c_known = 1)
results are unaffected. Kept for the record only; do not copy these files into results/. See docs/experiment_log.md.

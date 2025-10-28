# Hyper-parameter Tuning

This summary lists the top-performing configurations uncovered during the hyper-parameter sweeps.
- Next-video tables highlight up to 10 configurations per study ranked by validation accuracy.
- Eligible-only accuracy is shown for comparison next to overall accuracy.
- Opinion regression tables highlight up to 10 configurations per study ranked by MAE relative to the baseline.
- Rows in bold mark the configuration promoted to the final evaluation.

## Next-Video Sweeps

No next-video sweep runs were available when this report was generated.
Run the XGBoost pipeline with `--stage sweeps` or `--stage full` once artifacts are ready.

## Opinion Regression Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.852 | 0.074 | +0.778 | — | 162 | 0.012 | +0.025 | 0.021 | 0.994 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.864 | 0.074 | +0.790 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.864 | 0.074 | +0.790 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.858 | 0.074 | +0.784 | — | 162 | 0.013 | +0.024 | 0.022 | 0.994 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.852 | 0.074 | +0.778 | — | 162 | 0.013 | +0.024 | 0.022 | 0.994 |
| tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.864 | 0.074 | +0.790 | — | 162 | 0.013 | +0.024 | 0.023 | 0.994 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.806 | 0.061 | +0.745 | — | 165 | 0.039 | +0.057 | 0.052 | 0.964 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.800 | 0.061 | +0.739 | — | 165 | 0.039 | +0.057 | 0.053 | 0.963 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.800 | 0.061 | +0.739 | — | 165 | 0.039 | +0.057 | 0.053 | 0.964 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.800 | 0.061 | +0.739 | — | 165 | 0.040 | +0.056 | 0.054 | 0.962 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.806 | 0.061 | +0.745 | — | 165 | 0.040 | +0.056 | 0.054 | 0.962 |
| tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.794 | 0.061 | +0.733 | — | 165 | 0.041 | +0.055 | 0.055 | 0.960 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.743 | 0.058 | +0.685 | — | 257 | 0.049 | +0.035 | 0.068 | 0.933 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.747 | 0.058 | +0.689 | — | 257 | 0.049 | +0.035 | 0.069 | 0.929 |
| tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.774 | 0.058 | +0.716 | — | 257 | 0.049 | +0.035 | 0.069 | 0.930 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.743 | 0.058 | +0.685 | — | 257 | 0.049 | +0.035 | 0.068 | 0.931 |
| tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.747 | 0.058 | +0.689 | — | 257 | 0.049 | +0.035 | 0.068 | 0.931 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.751 | 0.058 | +0.693 | — | 257 | 0.050 | +0.034 | 0.070 | 0.928 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.03 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.036 across 584 participants.
- Weighted baseline MAE 0.074 (+0.039 vs. final).
- Weighted directional accuracy 0.791 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.728 vs. final).
- Weighted RMSE (change) 0.051 across 584 participants.
- Weighted baseline RMSE (change) 0.107 (+0.056 vs. final).
- Weighted calibration ECE 0.010 across 584 participants.
- Weighted KL divergence 0.586 across 584 participants.
- Weighted baseline KL divergence 18.435 (+17.849 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.057).
- Lowest MAE: Study 1 – Gun Control (MTurk) (0.012); Highest MAE: Study 3 – Minimum Wage (YouGov) (0.049).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (0.852).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (0.743).
- Largest directional-accuracy gain: Study 1 – Gun Control (MTurk) (+0.778).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (+0.086).
- Lowest RMSE(change): Study 1 – Gun Control (MTurk) (0.021); Highest: Study 3 – Minimum Wage (YouGov) (0.068).
- Lowest calibration ECE: Study 1 – Gun Control (MTurk) (0.005); Highest: Study 2 – Minimum Wage (MTurk) (0.014).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+20.780).
- Lowest KL divergence: Study 3 – Minimum Wage (YouGov) (0.384); Highest: Study 2 – Minimum Wage (MTurk) (0.823).

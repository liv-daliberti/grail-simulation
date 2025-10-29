# KNN Opinion Shift Study (Next-Video Config)

This section reuses the selected next-video recommendation configuration to estimate post-study opinion change.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: MAE / RMSE / R² / directional accuracy / MAE (change) / RMSE (change) / calibration slope & intercept / calibration ECE / KL divergence, compared against a no-change baseline.

## TF-IDF Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 50 | 0.704 | 0.074 | +0.630 | 0.030 | +0.007 | 0.038 | 0.983 | 0.030 | 0.038 | +0.008 | 0.199 | 0.020 | 0.008 | — | 10.979 | +10.467 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 50 | 0.527 | 0.061 | +0.467 | 0.093 | +0.003 | 0.128 | 0.786 | 0.093 | 0.128 | +0.010 | 1.079 | 0.007 | 0.018 | — | 8.264 | +10.105 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 50 | 0.494 | 0.058 | +0.436 | 0.088 | -0.004 | 0.125 | 0.771 | 0.088 | 0.125 | +0.001 | 0.155 | 0.018 | 0.022 | — | 7.392 | +9.188 | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../tfidf/opinion/)

## Word2Vec Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 50 | 0.704 | 0.074 | +0.630 | 0.030 | +0.007 | 0.038 | 0.983 | 0.030 | 0.038 | +0.008 | 0.156 | 0.022 | 0.008 | — | 11.203 | +10.243 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 25 | 0.545 | 0.061 | +0.485 | 0.089 | +0.007 | 0.124 | 0.798 | 0.089 | 0.124 | +0.014 | 1.140 | 0.007 | 0.027 | — | 7.897 | +10.471 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 50 | 0.463 | 0.058 | +0.405 | 0.089 | -0.005 | 0.127 | 0.764 | 0.089 | 0.127 | -0.001 | -0.190 | 0.026 | 0.032 | — | 9.532 | +7.048 | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../word2vec/opinion/)

### Opinion Change Heatmaps

Plots are refreshed under `reports/knn/<feature-space>/opinion/` including MAE vs. k (`mae_<study>.png`), R² vs. k (`r2_<study>.png`), and change heatmaps (`change_heatmap_<study>.png`).

## Cross-Study Diagnostics

## TF-IDF Feature Space

#### Weighted Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.001 vs. final).
- Weighted directional accuracy 0.562 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.498 vs. final).
- Weighted RMSE (change) 0.102 (+0.005 vs. baseline).
- Weighted calibration ECE 0.017 (— vs. baseline).
- Weighted KL divergence 8.633 (+9.802 vs. baseline).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (TFIDF) (+0.007).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (TFIDF) (+0.010).
- Lowest MAE: Study 1 – Gun Control (MTurk) (TFIDF) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (TFIDF) (0.093).
- Biggest KL divergence reduction: Study 1 – Gun Control (MTurk) (TFIDF) (+10.467).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (TFIDF) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (TFIDF) (0.494).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (TFIDF) (+0.630).

- Unweighted MAE 0.070 (σ 0.028, range 0.030 – 0.093).
- MAE delta mean 0.002 (σ 0.004, range -0.004 – 0.007).
- Unweighted directional accuracy 0.575 (σ 0.092, range 0.494 – 0.704).
- Accuracy delta mean 0.511 (σ 0.085, range 0.436 – 0.630).
- RMSE (change) 0.097 (σ 0.042, range 0.038 – 0.128).
- RMSE (change) delta 0.006 (σ 0.004, range 0.001 – 0.010).
- Calibration ECE 0.016 (σ 0.006, range 0.008 – 0.022).
- KL divergence 8.878 (σ 1.527, range 7.392 – 10.979).
- KL divergence delta 9.920 (σ 0.538, range 9.188 – 10.467).

## Word2Vec Feature Space

#### Weighted Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.002 vs. final).
- Weighted directional accuracy 0.553 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.490 vs. final).
- Weighted RMSE (change) 0.101 (+0.006 vs. baseline).
- Weighted calibration ECE 0.024 (— vs. baseline).
- Weighted KL divergence 9.534 (+8.901 vs. baseline).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (WORD2VEC) (+0.007).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (+0.014).
- Lowest MAE: Study 1 – Gun Control (MTurk) (WORD2VEC) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (0.089).
- Biggest KL divergence reduction: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (+10.471).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (WORD2VEC) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (WORD2VEC) (0.463).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (WORD2VEC) (+0.630).

- Unweighted MAE 0.069 (σ 0.028, range 0.030 – 0.089).
- MAE delta mean 0.003 (σ 0.006, range -0.005 – 0.007).
- Unweighted directional accuracy 0.571 (σ 0.100, range 0.463 – 0.704).
- Accuracy delta mean 0.506 (σ 0.093, range 0.405 – 0.630).
- RMSE (change) 0.096 (σ 0.041, range 0.038 – 0.127).
- RMSE (change) delta 0.007 (σ 0.006, range -0.001 – 0.014).
- Calibration ECE 0.023 (σ 0.010, range 0.008 – 0.032).
- KL divergence 9.544 (σ 1.349, range 7.897 – 11.203).
- KL divergence delta 9.254 (σ 1.563, range 7.048 – 10.471).

## Takeaways

- Study 1 – Gun Control (MTurk): best R² 0.983 with WORD2VEC (k=50); largest MAE reduction +0.007 via WORD2VEC.
- Study 2 – Minimum Wage (MTurk): best R² 0.798 with WORD2VEC (k=25); largest MAE reduction +0.007 via WORD2VEC.
- Study 3 – Minimum Wage (YouGov): best R² 0.771 with TFIDF (k=50); largest MAE reduction -0.004 via TFIDF.

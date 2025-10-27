# KNN Opinion Shift Study

This study evaluates a second KNN baseline that predicts each participant's post-study opinion index.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: MAE / RMSE / R² / directional accuracy / MAE (change) / RMSE (change) / calibration slope & intercept / calibration ECE / KL divergence, compared against a no-change baseline.

## TF-IDF Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 25 | 0.704 | 0.074 | +0.630 | 0.031 | -0.006 | 0.038 | 0.982 | 0.031 | — | — | — | — | — | — | — | — | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 25 | 0.533 | 0.061 | +0.473 | 0.093 | -0.003 | 0.128 | 0.786 | 0.093 | — | — | — | — | — | — | — | — | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 25 | 0.525 | 0.058 | +0.467 | 0.088 | +0.004 | 0.126 | 0.766 | 0.088 | — | — | — | — | — | — | — | — | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../tfidf/opinion/)

### Opinion Change Heatmaps

Plots are refreshed under `reports/knn/<feature-space>/opinion/` including MAE vs. k (`mae_<study>.png`), R² vs. k (`r2_<study>.png`), and change heatmaps (`change_heatmap_<study>.png`).

## Cross-Study Diagnostics

## TF-IDF Feature Space

#### Weighted Summary

- Weighted MAE 0.074 across 584 participants.
- Weighted baseline MAE 0.074 (+0.001 vs. final).
- Weighted directional accuracy 0.577 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.514 vs. final).
- Largest MAE reduction: Study 3 – Minimum Wage (YouGov) (TFIDF) (+0.004).
- Lowest MAE: Study 1 – Gun Control (MTurk) (TFIDF) (0.031); Highest MAE: Study 2 – Minimum Wage (MTurk) (TFIDF) (0.093).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (TFIDF) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (TFIDF) (0.525).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (TFIDF) (+0.630).

- Unweighted MAE 0.071 (σ 0.028, range 0.031 – 0.093).
- MAE delta mean -0.002 (σ 0.004, range -0.006 – 0.004).
- Unweighted directional accuracy 0.587 (σ 0.082, range 0.525 – 0.704).
- Accuracy delta mean 0.523 (σ 0.075, range 0.467 – 0.630).

## Takeaways

- Study 1 – Gun Control (MTurk): best R² 0.982 with TFIDF (k=25); largest MAE reduction -0.006 via TFIDF.
- Study 2 – Minimum Wage (MTurk): best R² 0.786 with TFIDF (k=25); largest MAE reduction -0.003 via TFIDF.
- Study 3 – Minimum Wage (YouGov): best R² 0.766 with TFIDF (k=25); largest MAE reduction +0.004 via TFIDF.

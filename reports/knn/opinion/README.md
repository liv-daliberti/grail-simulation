# KNN Opinion Shift Study

This study evaluates a second KNN baseline that predicts each participant's post-study opinion index.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: MAE / RMSE / R² / directional accuracy on the predicted post index, compared against a no-change baseline.

## TF-IDF Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 25 | 0.704 | 0.074 | +0.630 | 0.031 | -0.006 | 0.038 | 0.982 | 0.031 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 150 | 0.533 | 0.061 | +0.473 | 0.093 | -0.003 | 0.127 | 0.788 | 0.093 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 150 | 0.502 | 0.058 | +0.444 | 0.087 | +0.002 | 0.124 | 0.772 | 0.087 | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../tfidf/opinion/)

### Opinion Change Heatmaps

Plots are refreshed under `reports/knn/<feature-space>/opinion/` including MAE vs. k (`mae_<study>.png`), R² vs. k (`r2_<study>.png`), and change heatmaps (`change_heatmap_<study>.png`).

## Cross-Study Diagnostics

## TF-IDF Feature Space

#### Weighted Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.002 vs. final).
- Weighted directional accuracy 0.567 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.503 vs. final).
- Largest MAE reduction: Study 3 – Minimum Wage (YouGov) (TFIDF) (+0.002).
- Lowest MAE: Study 1 – Gun Control (MTurk) (TFIDF) (0.031); Highest MAE: Study 2 – Minimum Wage (MTurk) (TFIDF) (0.093).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (TFIDF) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (TFIDF) (0.502).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (TFIDF) (+0.630).

- Unweighted MAE 0.070 (σ 0.028, range 0.031 – 0.093).
- MAE delta mean -0.002 (σ 0.004, range -0.006 – 0.002).
- Unweighted directional accuracy 0.580 (σ 0.089, range 0.502 – 0.704).
- Accuracy delta mean 0.515 (σ 0.082, range 0.444 – 0.630).

## Takeaways

- Study 1 – Gun Control (MTurk): best R² 0.982 with TFIDF (k=25); largest MAE reduction -0.006 via TFIDF.
- Study 2 – Minimum Wage (MTurk): best R² 0.788 with TFIDF (k=150); largest MAE reduction -0.003 via TFIDF.
- Study 3 – Minimum Wage (YouGov): best R² 0.772 with TFIDF (k=150); largest MAE reduction +0.002 via TFIDF.

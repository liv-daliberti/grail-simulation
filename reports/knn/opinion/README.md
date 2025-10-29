# KNN Opinion Shift Study

This study evaluates a second KNN baseline that predicts each participant's post-study opinion index.

- Dataset: `data/cleaned_grail`
- Split: validation
- Metrics: MAE / RMSE / R² / directional accuracy / MAE (change) / RMSE (change) / calibration slope & intercept / calibration ECE / KL divergence, compared against a no-change baseline.

## TF-IDF Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 50 | 0.704 | 0.074 | +0.630 | 0.030 | +0.007 | 0.038 | 0.983 | 0.030 | 0.038 | +0.008 | 0.238 | 0.019 | 0.009 | — | 10.978 | +10.467 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 50 | 0.527 | 0.061 | +0.467 | 0.093 | +0.003 | 0.128 | 0.786 | 0.093 | 0.128 | +0.010 | 1.079 | 0.007 | 0.018 | — | 8.264 | +10.105 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 50 | 0.490 | 0.058 | +0.432 | 0.088 | -0.004 | 0.125 | 0.771 | 0.088 | 0.125 | +0.001 | 0.163 | 0.018 | 0.027 | — | 7.386 | +9.194 | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../tfidf/opinion/)

## Word2Vec Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 50 | 0.704 | 0.074 | +0.630 | 0.030 | +0.007 | 0.038 | 0.983 | 0.030 | 0.038 | +0.008 | 0.076 | 0.024 | 0.008 | — | 11.074 | +10.371 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 50 | 0.576 | 0.061 | +0.515 | 0.091 | +0.005 | 0.127 | 0.790 | 0.091 | 0.127 | +0.011 | 1.130 | 0.007 | 0.020 | — | 8.250 | +10.119 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 50 | 0.494 | 0.058 | +0.436 | 0.088 | -0.004 | 0.126 | 0.767 | 0.088 | 0.126 | -0.001 | -0.031 | 0.022 | 0.027 | — | 7.342 | +9.238 | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../word2vec/opinion/)

## Sentence-Transformer Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 50 | 0.704 | 0.074 | +0.630 | 0.030 | +0.007 | 0.038 | 0.983 | 0.030 | 0.038 | +0.008 | 0.030 | 0.025 | 0.009 | — | 14.511 | +6.935 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 25 | 0.552 | 0.061 | +0.491 | 0.089 | +0.007 | 0.126 | 0.791 | 0.089 | 0.126 | +0.012 | 1.073 | 0.007 | 0.032 | — | 8.074 | +10.294 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 50 | 0.521 | 0.058 | +0.463 | 0.089 | -0.004 | 0.125 | 0.768 | 0.089 | 0.125 | -0.000 | 0.083 | 0.019 | 0.020 | — | 9.523 | +7.057 | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../sentence_transformer/opinion/)

### Opinion Change Heatmaps

Plots are refreshed under `reports/knn/<feature-space>/opinion/` including MAE vs. k (`mae_<study>.png`), R² vs. k (`r2_<study>.png`), and change heatmaps (`change_heatmap_<study>.png`).

## Cross-Study Diagnostics

## TF-IDF Feature Space

#### Weighted Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.001 vs. final).
- Weighted directional accuracy 0.560 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.497 vs. final).
- Weighted RMSE (change) 0.102 (+0.005 vs. baseline).
- Weighted calibration ECE 0.019 (— vs. baseline).
- Weighted KL divergence 8.631 (+9.804 vs. baseline).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (TFIDF) (+0.007).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (TFIDF) (+0.010).
- Lowest MAE: Study 1 – Gun Control (MTurk) (TFIDF) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (TFIDF) (0.093).
- Biggest KL divergence reduction: Study 1 – Gun Control (MTurk) (TFIDF) (+10.467).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (TFIDF) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (TFIDF) (0.490).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (TFIDF) (+0.630).

- Unweighted MAE 0.070 (σ 0.028, range 0.030 – 0.093).
- MAE delta mean 0.002 (σ 0.004, range -0.004 – 0.007).
- Unweighted directional accuracy 0.574 (σ 0.093, range 0.490 – 0.704).
- Accuracy delta mean 0.509 (σ 0.086, range 0.432 – 0.630).
- RMSE (change) 0.097 (σ 0.042, range 0.038 – 0.128).
- RMSE (change) delta 0.006 (σ 0.004, range 0.001 – 0.010).
- Calibration ECE 0.018 (σ 0.007, range 0.009 – 0.027).
- KL divergence 8.876 (σ 1.529, range 7.386 – 10.978).
- KL divergence delta 9.922 (σ 0.536, range 9.194 – 10.467).

## Word2Vec Feature Space

#### Weighted Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.002 vs. final).
- Weighted directional accuracy 0.575 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.512 vs. final).
- Weighted RMSE (change) 0.102 (+0.005 vs. baseline).
- Weighted calibration ECE 0.020 (— vs. baseline).
- Weighted KL divergence 8.634 (+9.801 vs. baseline).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (WORD2VEC) (+0.007).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (+0.011).
- Lowest MAE: Study 1 – Gun Control (MTurk) (WORD2VEC) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (0.091).
- Biggest KL divergence reduction: Study 1 – Gun Control (MTurk) (WORD2VEC) (+10.371).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (WORD2VEC) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (WORD2VEC) (0.494).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (WORD2VEC) (+0.630).

- Unweighted MAE 0.070 (σ 0.028, range 0.030 – 0.091).
- MAE delta mean 0.003 (σ 0.005, range -0.004 – 0.007).
- Unweighted directional accuracy 0.591 (σ 0.086, range 0.494 – 0.704).
- Accuracy delta mean 0.527 (σ 0.080, range 0.436 – 0.630).
- RMSE (change) 0.097 (σ 0.042, range 0.038 – 0.127).
- RMSE (change) delta 0.006 (σ 0.005, range -0.001 – 0.011).
- Calibration ECE 0.018 (σ 0.008, range 0.008 – 0.027).
- KL divergence 8.889 (σ 1.589, range 7.342 – 11.074).
- KL divergence delta 9.909 (σ 0.486, range 9.238 – 10.371).

## Sentence-Transformer Feature Space

#### Weighted Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.002 vs. final).
- Weighted directional accuracy 0.580 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.517 vs. final).
- Weighted RMSE (change) 0.101 (+0.005 vs. baseline).
- Weighted calibration ECE 0.021 (— vs. baseline).
- Weighted KL divergence 10.497 (+7.938 vs. baseline).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (+0.007).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (+0.012).
- Lowest MAE: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (0.089).
- Biggest KL divergence reduction: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (+10.294).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (SENTENCE_TRANSFORMER) (0.521).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (+0.630).

- Unweighted MAE 0.069 (σ 0.028, range 0.030 – 0.089).
- MAE delta mean 0.003 (σ 0.005, range -0.004 – 0.007).
- Unweighted directional accuracy 0.592 (σ 0.080, range 0.521 – 0.704).
- Accuracy delta mean 0.528 (σ 0.073, range 0.463 – 0.630).
- RMSE (change) 0.097 (σ 0.042, range 0.038 – 0.126).
- RMSE (change) delta 0.007 (σ 0.005, range -0.000 – 0.012).
- Calibration ECE 0.021 (σ 0.009, range 0.009 – 0.032).
- KL divergence 10.703 (σ 2.757, range 8.074 – 14.511).
- KL divergence delta 8.095 (σ 1.556, range 6.935 – 10.294).

## Takeaways

- Study 1 – Gun Control (MTurk): best R² 0.983 with SENTENCE_TRANSFORMER (k=50); largest MAE reduction +0.007 via WORD2VEC.
- Study 2 – Minimum Wage (MTurk): best R² 0.791 with SENTENCE_TRANSFORMER (k=25); largest MAE reduction +0.007 via SENTENCE_TRANSFORMER.
- Study 3 – Minimum Wage (YouGov): best R² 0.771 with TFIDF (k=50); largest MAE reduction -0.004 via TFIDF.

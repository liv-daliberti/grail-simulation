# KNN Opinion Shift Study (Next-Video Config)

This section reuses the selected next-video recommendation configuration to estimate post-study opinion change.

- Dataset: `data/cleaned_grail`
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
| Study 1 – Gun Control (MTurk) | 162 | 50 | 0.704 | 0.074 | +0.630 | 0.030 | +0.007 | 0.038 | 0.983 | 0.030 | 0.038 | +0.008 | 0.039 | 0.025 | 0.006 | — | 11.146 | +10.300 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 50 | 0.545 | 0.061 | +0.485 | 0.090 | +0.006 | 0.125 | 0.795 | 0.090 | 0.125 | +0.013 | 1.334 | 0.000 | 0.031 | — | 9.678 | +8.690 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 50 | 0.486 | 0.058 | +0.428 | 0.088 | -0.004 | 0.126 | 0.767 | 0.088 | 0.126 | -0.001 | -0.048 | 0.022 | 0.024 | — | 7.363 | +9.217 | 0.084 |
*Assets:* [MAE / R² curves and heatmaps](../word2vec/opinion/)

## Sentence-Transformer Feature Space

| Study | Participants | Best k | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 50 | 0.704 | 0.074 | +0.630 | 0.030 | +0.007 | 0.038 | 0.983 | 0.030 | 0.038 | +0.008 | 0.030 | 0.025 | 0.009 | — | 14.511 | +6.935 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 25 | 0.552 | 0.061 | +0.491 | 0.089 | +0.007 | 0.126 | 0.791 | 0.089 | 0.126 | +0.012 | 1.073 | 0.007 | 0.032 | — | 8.074 | +10.294 | 0.096 |
*Assets:* [MAE / R² curves and heatmaps](../sentence_transformer/opinion/)

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
- Weighted directional accuracy 0.563 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.500 vs. final).
- Weighted RMSE (change) 0.101 (+0.006 vs. baseline).
- Weighted calibration ECE 0.021 (— vs. baseline).
- Weighted KL divergence 9.066 (+9.369 vs. baseline).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (WORD2VEC) (+0.007).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (+0.013).
- Lowest MAE: Study 1 – Gun Control (MTurk) (WORD2VEC) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (0.090).
- Biggest KL divergence reduction: Study 1 – Gun Control (MTurk) (WORD2VEC) (+10.300).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (WORD2VEC) (0.704).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (WORD2VEC) (0.486).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (WORD2VEC) (+0.630).

- Unweighted MAE 0.070 (σ 0.028, range 0.030 – 0.090).
- MAE delta mean 0.003 (σ 0.005, range -0.004 – 0.007).
- Unweighted directional accuracy 0.579 (σ 0.092, range 0.486 – 0.704).
- Accuracy delta mean 0.514 (σ 0.085, range 0.428 – 0.630).
- RMSE (change) 0.096 (σ 0.041, range 0.038 – 0.126).
- RMSE (change) delta 0.007 (σ 0.006, range -0.001 – 0.013).
- Calibration ECE 0.020 (σ 0.010, range 0.006 – 0.031).
- KL divergence 9.396 (σ 1.557, range 7.363 – 11.146).
- KL divergence delta 9.402 (σ 0.670, range 8.690 – 10.300).

## Sentence-Transformer Feature Space

#### Weighted Summary

- Weighted MAE 0.060 across 327 participants.
- Weighted baseline MAE 0.067 (+0.007 vs. final).
- Weighted directional accuracy 0.627 across 327 participants.
- Weighted baseline accuracy 0.067 (+0.560 vs. final).
- Weighted RMSE (change) 0.082 (+0.010 vs. baseline).
- Weighted calibration ECE 0.021 (— vs. baseline).
- Weighted KL divergence 11.263 (+8.630 vs. baseline).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (+0.007).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (+0.012).
- Lowest MAE: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (0.089).
- Biggest KL divergence reduction: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (+10.294).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (0.704).
- Lowest directional accuracy: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (0.552).
- Largest accuracy gain vs. baseline: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (+0.630).

- Unweighted MAE 0.060 (σ 0.029, range 0.030 – 0.089).
- MAE delta mean 0.007 (σ 0.000, range 0.007 – 0.007).
- Unweighted directional accuracy 0.628 (σ 0.076, range 0.552 – 0.704).
- Accuracy delta mean 0.560 (σ 0.069, range 0.491 – 0.630).
- RMSE (change) 0.082 (σ 0.044, range 0.038 – 0.126).
- RMSE (change) delta 0.010 (σ 0.002, range 0.008 – 0.012).
- Calibration ECE 0.021 (σ 0.012, range 0.009 – 0.032).
- KL divergence 11.293 (σ 3.218, range 8.074 – 14.511).
- KL divergence delta 8.614 (σ 1.680, range 6.935 – 10.294).

## Takeaways

- Study 1 – Gun Control (MTurk): best R² 0.983 with SENTENCE_TRANSFORMER (k=50); largest MAE reduction +0.007 via WORD2VEC.
- Study 2 – Minimum Wage (MTurk): best R² 0.795 with WORD2VEC (k=50); largest MAE reduction +0.007 via SENTENCE_TRANSFORMER.
- Study 3 – Minimum Wage (YouGov): best R² 0.771 with TFIDF (k=50); largest MAE reduction -0.004 via TFIDF.

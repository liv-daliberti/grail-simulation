# XGBoost Opinion Regression

This summary captures the opinion-regression baselines trained with XGBoost for the selected participant studies.

- Dataset: `unknown`
- Split: validation
- Metrics track MAE, RMSE, R², directional accuracy, MAE(change), RMSE(change), calibration slope/intercept, calibration ECE, and KL divergence versus the no-change baseline.
- Δ columns capture improvements relative to that baseline when available.

| Study | Participants | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| study1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### TFIDF Opinion Plots

![Change Heatmap Study1](tfidf/opinion/change_heatmap_study1.png)

![Change Heatmap Study2](tfidf/opinion/change_heatmap_study2.png)

![Change Heatmap Study3](tfidf/opinion/change_heatmap_study3.png)

![Error Histogram Study1](tfidf/opinion/error_histogram_study1.png)

![Error Histogram Study2](tfidf/opinion/error_histogram_study2.png)

![Error Histogram Study3](tfidf/opinion/error_histogram_study3.png)

![Post Heatmap Study1](tfidf/opinion/post_heatmap_study1.png)

![Post Heatmap Study2](tfidf/opinion/post_heatmap_study2.png)

![Post Heatmap Study3](tfidf/opinion/post_heatmap_study3.png)

### WORD2VEC Opinion Plots

![Change Heatmap Study1](word2vec/opinion/change_heatmap_study1.png)

![Change Heatmap Study2](word2vec/opinion/change_heatmap_study2.png)

![Error Histogram Study1](word2vec/opinion/error_histogram_study1.png)

![Error Histogram Study2](word2vec/opinion/error_histogram_study2.png)

![Post Heatmap Study1](word2vec/opinion/post_heatmap_study1.png)

![Post Heatmap Study2](word2vec/opinion/post_heatmap_study2.png)

### SENTENCE_TRANSFORMER Opinion Plots

![Change Heatmap Study3](sentence_transformer/opinion/change_heatmap_study3.png)

![Error Histogram Study3](sentence_transformer/opinion/error_histogram_study3.png)

![Post Heatmap Study3](sentence_transformer/opinion/post_heatmap_study3.png)

## Cross-Study Diagnostics


## Observations

- study1: MAE — (Δ vs. baseline —), RMSE(change) —, ECE —, KL —, R² —.

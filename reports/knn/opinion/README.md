# KNN Opinion Shift Study

This study evaluates a second KNN baseline that predicts each participant's post-study opinion index.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: MAE / RMSE / R² on the predicted post index, compared against a no-change baseline.

## TF-IDF Feature Space

| Study | Participants | Best k | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 150 | 0.030 | -0.007 | 0.037 | 0.983 | 0.030 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 20 | 0.092 | -0.004 | 0.127 | 0.790 | 0.092 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 150 | 0.087 | +0.003 | 0.125 | 0.772 | 0.087 | 0.084 |

## Word2Vec Feature Space

| Study | Participants | Best k | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 150 | 0.030 | -0.007 | 0.037 | 0.983 | 0.030 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 20 | 0.089 | -0.007 | 0.122 | 0.805 | 0.089 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 100 | 0.088 | +0.004 | 0.126 | 0.768 | 0.088 | 0.084 |

## Sentence-Transformer Feature Space

| Study | Participants | Best k | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 100 | 0.030 | -0.007 | 0.037 | 0.983 | 0.030 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 125 | 0.088 | -0.008 | 0.124 | 0.801 | 0.088 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 75 | 0.086 | +0.002 | 0.124 | 0.773 | 0.086 | 0.084 |

### Opinion Change Heatmaps

Plots are refreshed under `reports/knn/opinion/<feature-space>/` for MAE, R², and change heatmaps.

## Cross-Study Diagnostics

## TF-IDF Feature Space

#### Weighted Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.002 vs. final).
- Largest MAE reduction: Study 3 – Minimum Wage (YouGov) (TFIDF) (+0.003).
- Lowest MAE: Study 1 – Gun Control (MTurk) (TFIDF) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (TFIDF) (0.092).

- Unweighted MAE 0.070 (σ 0.028, range 0.030 – 0.092).
- MAE delta mean -0.003 (σ 0.004, range -0.007 – 0.003).

## Word2Vec Feature Space

#### Weighted Summary

- Weighted MAE 0.072 across 584 participants.
- Weighted baseline MAE 0.074 (+0.002 vs. final).
- Largest MAE reduction: Study 3 – Minimum Wage (YouGov) (WORD2VEC) (+0.004).
- Lowest MAE: Study 1 – Gun Control (MTurk) (WORD2VEC) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (WORD2VEC) (0.089).

- Unweighted MAE 0.069 (σ 0.028, range 0.030 – 0.089).
- MAE delta mean -0.003 (σ 0.005, range -0.007 – 0.004).

## Sentence-Transformer Feature Space

#### Weighted Summary

- Weighted MAE 0.071 across 584 participants.
- Weighted baseline MAE 0.074 (+0.003 vs. final).
- Largest MAE reduction: Study 3 – Minimum Wage (YouGov) (SENTENCE_TRANSFORMER) (+0.002).
- Lowest MAE: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (SENTENCE_TRANSFORMER) (0.088).

- Unweighted MAE 0.068 (σ 0.027, range 0.030 – 0.088).
- MAE delta mean -0.004 (σ 0.005, range -0.008 – 0.002).

## Takeaways

- Study 1 – Gun Control (MTurk): best R² 0.983 with SENTENCE_TRANSFORMER (k=100); largest MAE reduction -0.007 via WORD2VEC.
- Study 2 – Minimum Wage (MTurk): best R² 0.805 with WORD2VEC (k=20); largest MAE reduction -0.004 via TFIDF.
- Study 3 – Minimum Wage (YouGov): best R² 0.773 with SENTENCE_TRANSFORMER (k=75); largest MAE reduction +0.004 via WORD2VEC.

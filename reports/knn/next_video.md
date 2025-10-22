# KNN Next-Video Baseline

This report summarises the slate-ranking KNN model that predicts the next video a viewer will click.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metric: accuracy on eligible slates (gold index present)
- Uncertainty: participant_bootstrap (n_bootstrap=500, n_groups=162, n_rows=548, seed=2024)

## TF-IDF Feature Space

| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.889 | [0.859, 0.922] | +0.349 | 0.540 | 0.326 | 2 | 548 | 548 |
| Study 2 – Minimum Wage (MTurk) | 0.338 | [0.300, 0.372] | -0.030 | 0.368 | 0.255 | 3 | 671 | 671 |

## Observations

- TFIDF: Study 1 – Gun Control (MTurk): 0.889 (baseline 0.540, Δ +0.349, k=2, eligible 548); Study 2 – Minimum Wage (MTurk): 0.338 (baseline 0.368, Δ -0.030, k=3, eligible 671); averages: mean Δ +0.159, mean random 0.291.

# KNN Next-Video Baseline

This report summarises the slate-ranking KNN model that predicts the next video a viewer will click.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metric: accuracy on eligible slates (gold index present)
- Baseline column: accuracy from always recommending the most-frequent gold index for the study.
- Δ column: improvement over that baseline accuracy.
- Random column: expected accuracy from uniformly sampling one candidate per slate.
- Uncertainty: participant_bootstrap (n_bootstrap=500, n_groups=162, n_rows=548, seed=2024)

## TF-IDF Feature Space

| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.889 | [0.859, 0.922] | +0.349 | 0.540 | 0.326 | 2 | 548 | 548 |
| Study 2 – Minimum Wage (MTurk) | 0.338 | [0.300, 0.372] | -0.030 | 0.368 | 0.255 | 3 | 671 | 671 |
| Study 3 – Minimum Wage (YouGov) | 0.292 | [0.270, 0.317] | -0.187 | 0.479 | 0.255 | 2 | 1,200 | 1,200 |

## Word2Vec Feature Space

| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.859 | [0.824, 0.896] | +0.319 | 0.540 | 0.326 | 2 | 548 | 548 |

## Observations

- TFIDF: Study 1 – Gun Control (MTurk): 0.889 (baseline 0.540, Δ +0.349, k=2, eligible 548); Study 2 – Minimum Wage (MTurk): 0.338 (baseline 0.368, Δ -0.030, k=3, eligible 671); Study 3 – Minimum Wage (YouGov): 0.292 (baseline 0.479, Δ -0.187, k=2, eligible 1,200); averages: mean Δ +0.044, mean random 0.279.
- WORD2VEC: Study 1 – Gun Control (MTurk): 0.859 (baseline 0.540, Δ +0.319, k=2, eligible 548); averages: mean Δ +0.319, mean random 0.326.
- Random values correspond to the expected accuracy from a uniform guess across the slate options.

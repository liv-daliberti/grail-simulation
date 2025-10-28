# KNN Next-Video Baseline

This report summarises the slate-ranking KNN model that predicts the next clicked video.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metric: accuracy on eligible slates (gold index present).
- Baseline column: accuracy from recommending the most frequent gold index.
- Δ column: improvement over that baseline accuracy.
- Random column: expected accuracy from uniformly sampling one candidate per slate.
- Uncertainty: participant_bootstrap (n_bootstrap=500, n_groups=162, n_rows=548, seed=2024)

## Portfolio Summary

| Feature space | Weighted accuracy ↑ | Δ vs baseline ↑ | Random ↑ | Eligible | Studies |
| --- | ---: | ---: | ---: | ---: | ---: |
| TFIDF | 0.577 | +0.036 | 0.326 | 548 | 1 |

Best-performing feature space: **TFIDF** with weighted accuracy 0.577 across 548 eligible slates (1 studies).

## TF-IDF Feature Space

| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.577 | [0.530, 0.618] | +0.036 | 0.540 | 0.326 | 10 | 548 | 548 |

## Accuracy Curves

### Study 1 – Gun Control (MTurk) (TFIDF)

![Accuracy curve](curves/tfidf/study1.png)

## Observations

- TFIDF: Study 1 – Gun Control (MTurk): 0.577 (baseline 0.540, Δ +0.036, k=10, eligible 548); averages: mean Δ +0.036, mean random 0.326.
- Random values approximate the accuracy from uniformly guessing across the slate.

Leave-one-study-out metrics were unavailable when this report was generated.


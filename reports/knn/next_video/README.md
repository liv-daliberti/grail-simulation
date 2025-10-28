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
| TFIDF | 0.520 | +0.075 | 0.287 | 1,219 | 2 |

Best-performing feature space: **TFIDF** with weighted accuracy 0.520 across 1,219 eligible slates (2 studies).

## TF-IDF Feature Space

| Study | Accuracy ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.717 | [0.674, 0.757] | +0.177 | 0.540 | 0.326 | 3 | 548 | 548 |
| Study 2 – Minimum Wage (MTurk) | 0.359 | [0.323, 0.395] | -0.009 | 0.368 | 0.255 | 3 | 671 | 671 |

## Accuracy Curves

### Study 1 – Gun Control (MTurk) (TFIDF)

![Accuracy curve](curves/tfidf/study1.png)

### Study 2 – Minimum Wage (MTurk) (TFIDF)

![Accuracy curve](curves/tfidf/study2.png)

## Observations

- TFIDF: Study 1 – Gun Control (MTurk): 0.717 (baseline 0.540, Δ +0.177, k=3, eligible 548); Study 2 – Minimum Wage (MTurk): 0.359 (baseline 0.368, Δ -0.009, k=3, eligible 671); averages: mean Δ +0.084, mean random 0.291.
- Random values approximate the accuracy from uniformly guessing across the slate.

Leave-one-study-out metrics were unavailable when this report was generated.


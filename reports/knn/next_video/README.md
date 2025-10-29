# KNN Next-Video Baseline

This report summarises the slate-ranking KNN model that predicts the next clicked video.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metric: eligible-only accuracy (gold index present).
- Note: an all-rows accuracy (including ineligible slates) is also recorded in the per-study metrics as `accuracy_overall_all_rows` to ease comparison with XGB's overall accuracy.
- Baseline column: accuracy from recommending the most frequent gold index.
- Δ column: improvement over that baseline accuracy.
- Random column: expected accuracy from uniformly sampling one candidate per slate.
- Uncertainty: participant_bootstrap (n_bootstrap=500, n_groups=162, n_rows=548, seed=2024)

## Portfolio Summary

| Feature space | Weighted accuracy ↑ | Δ vs baseline ↑ | Random ↑ | Eligible | Studies |
| --- | ---: | ---: | ---: | ---: | ---: |
| TFIDF | 0.763 | +0.223 | 0.326 | 548 | 1 |

Best-performing feature space: **TFIDF** with weighted accuracy 0.763 across 548 eligible slates (1 studies).

## TF-IDF Feature Space

| Study | Accuracy ↑ | Accuracy (all rows) ↑ | 95% CI | Δ vs baseline ↑ | Baseline ↑ | Random ↑ | Best k | Eligible | Total |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.763 | 0.763 | [0.727, 0.799] | +0.223 | 0.540 | 0.326 | 2 | 548 | 548 |

## Accuracy Curves

### Study 1 – Gun Control (MTurk) (TFIDF)

![Accuracy curve](curves/tfidf/study1.png)

## Observations

- TFIDF: Study 1 – Gun Control (MTurk): 0.763 (baseline 0.540, Δ +0.223, k=2, eligible 548); averages: mean Δ +0.223, mean random 0.326.
- Random values approximate the accuracy from uniformly guessing across the slate.

## KNN vs XGB (Matched Studies)

This section compares the eligible-only accuracy for KNN and XGB, and also shows an all-rows accuracy for KNN alongside XGB's overall accuracy.

| Study | KNN (feature) eligible-only ↑ | XGB eligible-only ↑ | KNN all-rows ↑ | XGB overall ↑ |
| --- | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.763 (TFIDF) | — | 0.763 | 0.000 |

Leave-one-study-out metrics were unavailable when this report was generated.


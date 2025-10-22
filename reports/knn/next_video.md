# KNN Next-Video Baseline

This report summarises the slate-ranking KNN model that predicts the next video a viewer will click.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metric: accuracy on eligible slates (gold index present)

## TF-IDF Feature Space

| Study | Accuracy ↑ | Best k | Most-frequent baseline ↑ |
| --- | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 0.889 | 2 | 0.540 |

## Observations

- TFIDF: Study 1 – Gun Control (MTurk) accuracy 0.889 (baseline 0.540).

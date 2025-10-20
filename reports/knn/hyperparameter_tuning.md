# KNN Hyperparameter Tuning Notes

This document consolidates the `k` sweeps run for both KNN baselines (next-video classification and post-study opinion regression).

## Next-Video Prediction

| Feature space | Issue | Accuracy @ best k | Best k | Trend |
| --- | --- | ---: | ---: | --- |
| TF-IDF | Gun control | 0.894 | 2 | Sharp drop after k = 5 |
| TF-IDF | Minimum wage | 0.306 | 3 | Accuracy decreases for larger k, below baseline |
| Word2Vec | Gun control | 0.870 | 2 | Declines steadily beyond k = 3 |
| Word2Vec | Minimum wage | 0.288 | 3 | Accuracy degrades quickly with higher k |

Key takeaways:

- Low `k` (2–3) consistently works best; higher neighbourhood sizes dilute the signal.
- Minimum-wage slates remain difficult: even the best configuration trails the most-frequent baseline.

## Post-Study Opinion Regression

| Feature space | Study | R² @ best k | Best k | Trend |
| --- | --- | ---: | ---: | --- |
| TF-IDF | Study 1 (gun control) | 0.184 | 10 | R² plateaus between k = 10 and 25 |
| TF-IDF | Study 2 (MTurk wage) | 0.374 | 25 | Gradual gains up to k ≈ 25 |
| TF-IDF | Study 3 (YouGov wage) | 0.181 | 50 | Requires wide neighbourhoods; still below baseline MAE |
| Word2Vec | Study 1 (gun control) | 0.214 | 5 | Peaks around k = 5–10 |
| Word2Vec | Study 2 (MTurk wage) | 0.440 | 10 | Clear improvement up to k = 10 |
| Word2Vec | Study 3 (YouGov wage) | 0.251 | 50 | Benefits from large k (30–50) |

Key takeaways:

- Opinion regression needs larger neighbourhoods than slate prediction, especially on the YouGov study.
- Word2Vec embeddings consistently reach higher R² than TF-IDF, but both feature spaces are still outperformed by the trivial “no change” baseline on MAE.
- No evidence of overfitting at large k for regression—the curves flatten rather than spike—suggesting further gains may require richer viewer features instead of additional neighbours.

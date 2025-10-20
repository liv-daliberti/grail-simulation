# KNN Hyperparameter Tuning Notes

This document consolidates the `k` sweeps run for both KNN baselines (next-video classification and post-study opinion regression).

## Next-Video Prediction

The latest sweeps cover both TF-IDF and Word2Vec feature spaces with:

- `k ∈ {1,2,3,4,5,10,15,20,25,50,75,100}`
- Distance metrics: cosine and L2
- Text-field augmentations: none, `viewer_profile,state_text`
- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count = 1, epochs = 10  
  (training uses up to 40 worker threads – see `training/training-knn.sh`)

All artefacts (metrics JSON, per-`k` predictions, error-based elbow plots) live under
`models/knn/sweeps/{tfidf,word2vec}/`.

### Best configurations (validation split)

| Feature space | Metric | Text fields | Vec size | Window | Min count | Issue | Accuracy | Best k |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| TF-IDF | cosine | none | — | — | — | Gun control | **0.920** | 2 |
| TF-IDF | cosine | none | — | — | — | Minimum wage | **0.285** | 2 |
| Word2Vec | cosine | none | 128 | 10 | 1 | Gun control | **0.880** | 2 |
| Word2Vec | L2 | viewer_profile,state_text | 128 | 5 | 1 | Minimum wage | **0.305** | 3 |

Observations:

- `k = 2` remains optimal for every gun-control configuration. Minimum-wage improves slightly
  with `k = 3` when using Word2Vec + L2.
- Cosine distance dominates for gun control; L2 is competitive for minimum wage once embeddings
  are used.
- Adding viewer profile/state text helps Word2Vec on minimum wage but hurts performance on gun control.
- TF-IDF still outperforms Word2Vec on gun control, while Word2Vec closes (and slightly beats) the gap on minimum wage.

The elbow plots are now labelled with the data split used (“validation split”), clarifying that the
curves reflect held-out evaluation error rather than the training data.

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

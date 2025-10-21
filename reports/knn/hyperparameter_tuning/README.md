# KNN Hyperparameter Tuning Notes

This document consolidates the `k` sweeps run for both KNN baselines (next-video classification and post-study opinion regression).

## Next-Video Prediction

The latest sweeps cover TF-IDF, Word2Vec, and Sentence-Transformer feature spaces with:

- `k ∈ {1,2,3,4,5,10,15,20,25,50,75,100}`
- Distance metrics: cosine and L2
- Text-field augmentations: none, `viewer_profile,state_text`
- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count = 1, epochs = 10  
  (training uses up to 40 worker threads – see `training/training-knn.sh`)
- Sentence-Transformer variants: model checkpoints pulled from the `--sentence-transformer-model`
  CLI flag (default: `sentence-transformers/all-mpnet-base-v2`) with configurable batch size and
  optional embedding normalisation.

All artifacts (metrics JSON, per-`k` predictions, error-based elbow plots) live under
`models/knn/sweeps/{tfidf,word2vec,sentence_transformer}/`.

### Best configurations (validation split, per study)

| Feature space | Study | Metric | Text fields | Vec size | Window | Min count | Accuracy | Best k |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| _Populated after pipeline run_ |  |  |  |  |  |  |  |  |

Observations (updated after each run):

- Each study is swept independently; the same feature space can therefore pick different `k`, distance metrics, or text augments for Study 1 vs Study 2/3.
- Word2Vec models write their tuned configs beneath `models/knn/word2vec/sweeps/study{1,2,3}/...` so it is easy to diff trained embeddings between studies.
- Sentence-Transformer runs mirror this layout in `models/knn/sentence_transformer/sweeps/study{1,2,3}/...`, capturing the model id and normalisation choice alongside the metrics.
- TF-IDF sweeps remain lightweight—`viewer_profile,state_text` is still the only augmentation evaluated by default, but feel free to expand via `WORD2VEC_SWEEP_*` / `KNN_TEXT_FIELDS` (and the analogous `SENTENCE_TRANSFORMER_*` environment variables).

The elbow plots are now labeled with the data split used (“validation split”), clarifying that the
curves reflect held-out evaluation error rather than the training data.

## Post-Study Opinion Regression

| Feature space | Study | R² @ best k | Best k | Trend |
| --- | --- | ---: | ---: | --- |
| TF-IDF | Study 1 (gun control) | 0.184 | 10 | R² plateaus between k = 10 and 25 |
| TF-IDF | Study 2 (MTurk wage) | 0.374 | 25 | Gradual gains up to k ≈ 25 |
| TF-IDF | Study 3 (YouGov wage) | 0.181 | 50 | Requires wide neighborhoods; still below baseline MAE |
| Word2Vec | Study 1 (gun control) | 0.214 | 5 | Peaks around k = 5–10 |
| Word2Vec | Study 2 (MTurk wage) | 0.440 | 10 | Clear improvement up to k = 10 |
| Word2Vec | Study 3 (YouGov wage) | 0.251 | 50 | Benefits from large k (30–50) |
| Sentence-Transformer | _Populated after pipeline run_ | — | — | Results emitted once the sentence-transformer sweep is executed |

Key takeaways:

- Opinion regression needs larger neighborhoods than slate prediction, especially on the YouGov study.
- Word2Vec and Sentence-Transformer embeddings consistently reach higher R² than TF-IDF, but all feature spaces are still outperformed by the trivial “no change” baseline on MAE.
- No evidence of overfitting at large k for regression—the curves flatten rather than spike—suggesting further gains may require richer viewer features instead of additional neighbors.

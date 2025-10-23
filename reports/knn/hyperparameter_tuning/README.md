# KNN Hyperparameter Tuning Notes

This document consolidates the selected grid searches for the KNN baselines.

## Next-Video Prediction

The latest sweeps cover the TFIDF, WORD2VEC, SENTENCE-TRANSFORMER feature spaces with:
- `k ∈ {1,2,3,4,5,10,15,20,25,50,75,100,125,150}`
- Distance metrics: cosine and L2
- Text-field augmentations: none, `viewer_profile,state_text`
- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count ∈ {1}
- Sentence-transformer model: `sentence-transformers/all-mpnet-base-v2`

| Feature space | Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TFIDF | Study 1 – Gun Control (MTurk) | cosine | none | tfidf | — | — | — | 0.889 | 0.540 | +0.349 | 2 | 548 |
| TFIDF | Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile,state_text | tfidf | — | — | — | 0.338 | 0.368 | -0.030 | 3 | 671 |
| TFIDF | Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | tfidf | — | — | — | 0.292 | 0.479 | -0.187 | 2 | 1,200 |
| WORD2VEC | Study 1 – Gun Control (MTurk) | cosine | none | word2vec | 128 | 5 | 1 | 0.861 | 0.540 | +0.321 | 2 | 548 |
| WORD2VEC | Study 2 – Minimum Wage (MTurk) | cosine | none | word2vec | 256 | 5 | 1 | 0.334 | 0.368 | -0.034 | 10 | 671 |
| WORD2VEC | Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | word2vec | 256 | 5 | 1 | 0.288 | 0.479 | -0.191 | 10 | 1,200 |
| SENTENCE_TRANSFORMER | Study 1 – Gun Control (MTurk) | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.801 | 0.540 | +0.261 | 2 | 548 |
| SENTENCE_TRANSFORMER | Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.308 | 0.368 | -0.060 | 3 | 671 |
| SENTENCE_TRANSFORMER | Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.322 | 0.479 | -0.158 | 2 | 1,200 |

### Configuration Leaderboards

## TF-IDF Feature Space

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-none | 0.889 | 0.000 | 2 | 548 |
| 2 | metric-cosine_text-viewerprofile_statetext | 0.881 | 0.007 | 2 | 548 |
| 3 | metric-l2_text-none | 0.318 | 0.571 | 2 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-viewerprofile_statetext | 0.338 | 0.000 | 3 | 671 |
| 2 | metric-l2_text-viewerprofile_statetext | 0.323 | 0.015 | 3 | 671 |
| 3 | metric-cosine_text-none | 0.311 | 0.027 | 4 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-viewerprofile_statetext | 0.292 | 0.000 | 2 | 1200 |
| 2 | metric-l2_text-viewerprofile_statetext | 0.288 | 0.004 | 2 | 1200 |
| 3 | metric-cosine_text-none | 0.284 | 0.008 | 2 | 1200 |


## Word2Vec Feature Space

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-none_sz128_win5_min1 | 0.861 | 0.000 | 2 | 548 |
| 2 | metric-cosine_text-none_sz128_win10_min1 | 0.861 | 0.000 | 2 | 548 |
| 3 | metric-cosine_text-none_sz256_win5_min1 | 0.861 | 0.000 | 2 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-none_sz256_win5_min1 | 0.334 | 0.000 | 10 | 671 |
| 2 | metric-cosine_text-none_sz256_win10_min1 | 0.334 | 0.000 | 10 | 671 |
| 3 | metric-cosine_text-viewerprofile_statetext_sz256_win10_min1 | 0.334 | 0.000 | 10 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-viewerprofile_statetext_sz256_win5_min1 | 0.288 | 0.000 | 10 | 1200 |
| 2 | metric-cosine_text-none_sz256_win10_min1 | 0.286 | 0.003 | 2 | 1200 |
| 3 | metric-l2_text-none_sz256_win10_min1 | 0.280 | 0.008 | 2 | 1200 |


## Sentence-Transformer Feature Space

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.801 | 0.000 | 2 | 548 |
| 2 | metric-cosine_text-none_model-allmpnetbasev2 | 0.792 | 0.009 | 2 | 548 |
| 3 | metric-l2_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.279 | 0.522 | 2 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.308 | 0.000 | 3 | 671 |
| 2 | metric-l2_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.300 | 0.009 | 3 | 671 |
| 3 | metric-cosine_text-none_model-allmpnetbasev2 | 0.289 | 0.019 | 2 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | metric-cosine_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.322 | 0.000 | 2 | 1200 |
| 2 | metric-l2_text-none_model-allmpnetbasev2 | 0.321 | 0.001 | 3 | 1200 |
| 3 | metric-l2_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.315 | 0.007 | 2 | 1200 |


### Observations

- TFIDF: Study 1 – Gun Control (MTurk): accuracy 0.889 (baseline 0.540, Δ +0.349, k=2) using cosine distance with base prompt only; Study 2 – Minimum Wage (MTurk): accuracy 0.338 (baseline 0.368, Δ -0.030, k=3) using cosine distance with Viewer Profile, State Text; Study 3 – Minimum Wage (YouGov): accuracy 0.292 (baseline 0.479, Δ -0.187, k=2) using cosine distance with Viewer Profile, State Text.
- WORD2VEC: Study 1 – Gun Control (MTurk): accuracy 0.861 (baseline 0.540, Δ +0.321, k=2) using word2vec (128d, window 5, min_count 1) with base prompt only; Study 2 – Minimum Wage (MTurk): accuracy 0.334 (baseline 0.368, Δ -0.034, k=10) using word2vec (256d, window 5, min_count 1) with base prompt only; Study 3 – Minimum Wage (YouGov): accuracy 0.288 (baseline 0.479, Δ -0.191, k=10) using word2vec (256d, window 5, min_count 1) with Viewer Profile, State Text.
- SENTENCE_TRANSFORMER: Study 1 – Gun Control (MTurk): accuracy 0.801 (baseline 0.540, Δ +0.261, k=2) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with Viewer Profile, State Text; Study 2 – Minimum Wage (MTurk): accuracy 0.308 (baseline 0.368, Δ -0.060, k=3) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with Viewer Profile, State Text; Study 3 – Minimum Wage (YouGov): accuracy 0.322 (baseline 0.479, Δ -0.158, k=2) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with Viewer Profile, State Text.


## Post-Study Opinion Regression

Opinion runs reuse the per-study slate configurations gathered above.
See `reports/knn/opinion/README.md` for detailed metrics and plots.

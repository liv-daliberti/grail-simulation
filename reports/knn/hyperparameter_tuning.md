# KNN Hyperparameter Tuning Notes

This document consolidates the selected grid searches for the KNN baselines.

## Next-Video Prediction

The latest sweeps cover the TFIDF, WORD2VEC, SENTENCE-TRANSFORMER feature spaces with:
- `k ∈ {1,2,3,4,5,10,15,20,25,50,75,100,125,150}`
- Distance metrics: cosine and L2
- Text-field augmentations: none, `viewer_profile,state_text`
- Word2Vec variants: vector size ∈ {128, 256}, window ∈ {5, 10}, min_count ∈ {1}
- Sentence-transformer model: `sentence-transformers/all-mpnet-base-v2`

| Feature space | Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy | Best k |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| TFIDF | Study 1 – Gun Control (MTurk) | cosine | none | tfidf | — | — | — | 0.889 | 2 |
| TFIDF | Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile,state_text | tfidf | — | — | — | 0.338 | 3 |
| TFIDF | Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | tfidf | — | — | — | 0.292 | 2 |
| WORD2VEC | Study 1 – Gun Control (MTurk) | cosine | none | word2vec | 128 | 5 | 1 | 0.861 | 2 |
| WORD2VEC | Study 2 – Minimum Wage (MTurk) | cosine | none | word2vec | 256 | 5 | 1 | 0.334 | 10 |
| WORD2VEC | Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | word2vec | 256 | 5 | 1 | 0.288 | 10 |
| SENTENCE_TRANSFORMER | Study 1 – Gun Control (MTurk) | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.801 | 2 |
| SENTENCE_TRANSFORMER | Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.308 | 3 |
| SENTENCE_TRANSFORMER | Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.322 | 2 |

### Configuration Leaderboards

## TF-IDF Feature Space

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-none** | 0.889 | 0.000 | 2 | 548 |
| 2 | metric-cosine_text-viewerprofile_statetext | 0.881 | 0.007 | 2 | 548 |
| 3 | metric-l2_text-none | 0.318 | 0.571 | 2 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-viewerprofile_statetext** | 0.338 | 0.000 | 3 | 671 |
| 2 | metric-l2_text-viewerprofile_statetext | 0.323 | 0.015 | 3 | 671 |
| 3 | metric-cosine_text-none | 0.311 | 0.027 | 4 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-viewerprofile_statetext** | 0.292 | 0.000 | 2 | 1200 |
| 2 | metric-l2_text-viewerprofile_statetext | 0.288 | 0.004 | 2 | 1200 |
| 3 | metric-cosine_text-none | 0.284 | 0.008 | 2 | 1200 |


## Word2Vec Feature Space

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-none_sz128_win5_min1** | 0.861 | 0.000 | 2 | 548 |
| 2 | metric-cosine_text-none_sz128_win10_min1 | 0.861 | 0.000 | 2 | 548 |
| 3 | metric-cosine_text-none_sz256_win5_min1 | 0.861 | 0.000 | 2 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-none_sz256_win5_min1** | 0.334 | 0.000 | 10 | 671 |
| 2 | metric-cosine_text-none_sz256_win10_min1 | 0.334 | 0.000 | 10 | 671 |
| 3 | metric-cosine_text-viewerprofile_statetext_sz256_win10_min1 | 0.334 | 0.000 | 10 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-viewerprofile_statetext_sz256_win5_min1** | 0.288 | 0.000 | 10 | 1200 |
| 2 | metric-cosine_text-none_sz256_win10_min1 | 0.286 | 0.003 | 2 | 1200 |
| 3 | metric-l2_text-none_sz256_win10_min1 | 0.280 | 0.008 | 2 | 1200 |


## Sentence-Transformer Feature Space

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-viewerprofile_statetext_model-allmpnetbasev2** | 0.801 | 0.000 | 2 | 548 |
| 2 | metric-cosine_text-none_model-allmpnetbasev2 | 0.792 | 0.009 | 2 | 548 |
| 3 | metric-l2_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.279 | 0.522 | 2 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-viewerprofile_statetext_model-allmpnetbasev2** | 0.308 | 0.000 | 3 | 671 |
| 2 | metric-l2_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.300 | 0.009 | 3 | 671 |
| 3 | metric-cosine_text-none_model-allmpnetbasev2 | 0.289 | 0.019 | 2 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Best k | Eligible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | **metric-cosine_text-viewerprofile_statetext_model-allmpnetbasev2** | 0.322 | 0.000 | 2 | 1200 |
| 2 | metric-l2_text-none_model-allmpnetbasev2 | 0.321 | 0.001 | 3 | 1200 |
| 3 | metric-l2_text-viewerprofile_statetext_model-allmpnetbasev2 | 0.315 | 0.007 | 2 | 1200 |


### Observations

- TFIDF: Study 1 – Gun Control (MTurk): accuracy 0.889 (k=2) using cosine distance with no extra fields; Study 2 – Minimum Wage (MTurk): accuracy 0.338 (k=3) using cosine distance with extra fields `viewer_profile,state_text`; Study 3 – Minimum Wage (YouGov): accuracy 0.292 (k=2) using cosine distance with extra fields `viewer_profile,state_text`.
- WORD2VEC: Study 1 – Gun Control (MTurk): accuracy 0.861 (k=2) using cosine distance, no extra fields, size=128, window=5, min_count=1; Study 2 – Minimum Wage (MTurk): accuracy 0.334 (k=10) using cosine distance, no extra fields, size=256, window=5, min_count=1; Study 3 – Minimum Wage (YouGov): accuracy 0.288 (k=10) using cosine distance, extra fields `viewer_profile,state_text`, size=256, window=5, min_count=1.
- SENTENCE_TRANSFORMER: Study 1 – Gun Control (MTurk): accuracy 0.801 (k=2) using cosine distance, extra fields `viewer_profile,state_text`, model=sentence-transformers/all-mpnet-base-v2; Study 2 – Minimum Wage (MTurk): accuracy 0.308 (k=3) using cosine distance, extra fields `viewer_profile,state_text`, model=sentence-transformers/all-mpnet-base-v2; Study 3 – Minimum Wage (YouGov): accuracy 0.322 (k=2) using cosine distance, extra fields `viewer_profile,state_text`, model=sentence-transformers/all-mpnet-base-v2.


## Post-Study Opinion Regression

Opinion runs reuse the per-study slate configurations gathered above.
See `reports/knn/opinion/README.md` for detailed metrics and plots.

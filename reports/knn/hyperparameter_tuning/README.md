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
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
- WORD2VEC: Study 1 – Gun Control (MTurk): accuracy 0.861 (baseline 0.540, Δ +0.321, k=2) using word2vec (128d, window 5, min_count 1) with base prompt only; Study 2 – Minimum Wage (MTurk): accuracy 0.334 (baseline 0.368, Δ -0.034, k=10) using word2vec (256d, window 5, min_count 1) with base prompt only; Study 3 – Minimum Wage (YouGov): accuracy 0.288 (baseline 0.479, Δ -0.191, k=10) using word2vec (256d, window 5, min_count 1) with Viewer Profile, State Text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 128 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 10 --knn-k-sweep 10 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 10 --knn-k-sweep 10 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
- SENTENCE_TRANSFORMER: Study 1 – Gun Control (MTurk): accuracy 0.801 (baseline 0.540, Δ +0.261, k=2) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with Viewer Profile, State Text; Study 2 – Minimum Wage (MTurk): accuracy 0.308 (baseline 0.368, Δ -0.060, k=3) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with Viewer Profile, State Text; Study 3 – Minimum Wage (YouGov): accuracy 0.322 (baseline 0.479, Δ -0.158, k=2) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with Viewer Profile, State Text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`


## Post-Study Opinion Regression

Configurations are ranked by validation MAE (lower is better). Bold rows indicate the selections promoted to the finalize stage.

## TF-IDF Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | l2 | none | tfidf | — | — | — | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile,state_text | tfidf | — | — | — | 0.704 | 0.074 | +0.630 | 125 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | none | tfidf | — | — | — | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile,state_text | tfidf | — | — | — | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile,state_text | tfidf | — | — | — | 0.552 | 0.061 | +0.491 | 20 | 165 | 0.092 | -0.004 | 0.127 | 0.790 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile,state_text | tfidf | — | — | — | 0.558 | 0.061 | +0.497 | 20 | 165 | 0.092 | -0.004 | 0.127 | 0.790 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | none | tfidf | — | — | — | 0.545 | 0.061 | +0.485 | 10 | 165 | 0.094 | -0.003 | 0.127 | 0.791 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | none | tfidf | — | — | — | 0.545 | 0.061 | +0.485 | 10 | 165 | 0.094 | -0.003 | 0.127 | 0.791 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | l2 | none | tfidf | — | — | — | 0.502 | 0.058 | +0.444 | 150 | 257 | 0.086 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | none | tfidf | — | — | — | 0.502 | 0.058 | +0.444 | 150 | 257 | 0.086 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile,state_text | tfidf | — | — | — | 0.498 | 0.058 | +0.440 | 150 | 257 | 0.087 | +0.003 | 0.124 | 0.772 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | tfidf | — | — | — | 0.502 | 0.058 | +0.444 | 150 | 257 | 0.087 | +0.003 | 0.125 | 0.772 | 257 |

- **TFIDF selections**: Study 1 – Gun Control (MTurk): MAE 0.030 (Δ -0.007, k=150); Study 2 – Minimum Wage (MTurk): MAE 0.092 (Δ -0.004, k=20); Study 3 – Minimum Wage (YouGov): MAE 0.086 (Δ +0.002, k=150).

  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric l2 --knn-k 150 --knn-k-sweep 150 --out-dir '<run_dir>' --knn-text-fields`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 20 --knn-k-sweep 20 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric l2 --knn-k 150 --knn-k-sweep 150 --out-dir '<run_dir>' --knn-text-fields`

## Word2Vec Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | cosine | none | word2vec | 256 | 10 | 1 | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | none | word2vec | 256 | 5 | 1 | 0.704 | 0.074 | +0.630 | 125 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile,state_text | word2vec | 256 | 10 | 1 | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile,state_text | word2vec | 128 | 10 | 1 | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile,state_text | word2vec | 256 | 10 | 1 | 0.704 | 0.074 | +0.630 | 125 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
*Showing top 5 of 16 configurations for Study 1 – Gun Control (MTurk).*
| **Study 2 – Minimum Wage (MTurk)** | cosine | none | word2vec | 256 | 5 | 1 | 0.564 | 0.061 | +0.503 | 25 | 165 | 0.088 | -0.008 | 0.123 | 0.803 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | none | word2vec | 128 | 5 | 1 | 0.570 | 0.061 | +0.509 | 25 | 165 | 0.089 | -0.007 | 0.123 | 0.801 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | none | word2vec | 128 | 5 | 1 | 0.533 | 0.061 | +0.473 | 50 | 165 | 0.089 | -0.007 | 0.124 | 0.798 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile,state_text | word2vec | 256 | 10 | 1 | 0.552 | 0.061 | +0.491 | 50 | 165 | 0.089 | -0.007 | 0.125 | 0.797 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | none | word2vec | 256 | 5 | 1 | 0.545 | 0.061 | +0.485 | 25 | 165 | 0.089 | -0.007 | 0.123 | 0.801 | 165 |
*Showing top 5 of 16 configurations for Study 2 – Minimum Wage (MTurk).*
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile,state_text | word2vec | 256 | 10 | 1 | 0.471 | 0.058 | +0.412 | 75 | 257 | 0.088 | +0.004 | 0.125 | 0.769 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile,state_text | word2vec | 128 | 10 | 1 | 0.463 | 0.058 | +0.405 | 75 | 257 | 0.088 | +0.004 | 0.125 | 0.769 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile,state_text | word2vec | 256 | 10 | 1 | 0.494 | 0.058 | +0.436 | 150 | 257 | 0.088 | +0.004 | 0.126 | 0.768 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | none | word2vec | 256 | 10 | 1 | 0.490 | 0.058 | +0.432 | 150 | 257 | 0.088 | +0.004 | 0.126 | 0.767 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile,state_text | word2vec | 128 | 5 | 1 | 0.479 | 0.058 | +0.420 | 100 | 257 | 0.088 | +0.004 | 0.125 | 0.770 | 257 |
*Showing top 5 of 16 configurations for Study 3 – Minimum Wage (YouGov).*

- **WORD2VEC selections**: Study 1 – Gun Control (MTurk): MAE 0.030 (Δ -0.007, k=50); Study 2 – Minimum Wage (MTurk): MAE 0.088 (Δ -0.008, k=25); Study 3 – Minimum Wage (YouGov): MAE 0.088 (Δ +0.004, k=75).

  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 256 --word2vec-window 10 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 75 --knn-k-sweep 75 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --word2vec-size 256 --word2vec-window 10 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`

## Sentence-Transformer Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | l2 | none | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.704 | 0.074 | +0.630 | 75 | 162 | 0.030 | -0.008 | 0.037 | 0.984 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.704 | 0.074 | +0.630 | 100 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.704 | 0.074 | +0.630 | 100 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.539 | 0.061 | +0.479 | 125 | 165 | 0.088 | -0.008 | 0.124 | 0.801 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.539 | 0.061 | +0.479 | 125 | 165 | 0.088 | -0.008 | 0.124 | 0.800 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | none | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.545 | 0.061 | +0.485 | 150 | 165 | 0.089 | -0.008 | 0.124 | 0.800 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | none | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.545 | 0.061 | +0.485 | 150 | 165 | 0.089 | -0.008 | 0.124 | 0.800 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.537 | 0.058 | +0.479 | 75 | 257 | 0.086 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile,state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.541 | 0.058 | +0.482 | 75 | 257 | 0.087 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | none | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.533 | 0.058 | +0.475 | 50 | 257 | 0.088 | +0.004 | 0.124 | 0.772 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | none | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.502 | 0.058 | +0.444 | 20 | 257 | 0.089 | +0.005 | 0.124 | 0.772 | 257 |

- **SENTENCE_TRANSFORMER selections**: Study 1 – Gun Control (MTurk): MAE 0.030 (Δ -0.008, k=75); Study 2 – Minimum Wage (MTurk): MAE 0.088 (Δ -0.008, k=125); Study 3 – Minimum Wage (YouGov): MAE 0.086 (Δ +0.002, k=75).

  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues gun_control --participant-studies study1 --knn-metric l2 --knn-k 75 --knn-k-sweep 75 --out-dir '<run_dir>' --knn-text-fields --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 125 --knn-k-sweep 125 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 75 --knn-k-sweep 75 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`

### Portfolio Summary

- Weighted MAE 0.072 across 1,752 participants.
- Weighted baseline MAE 0.074 (+0.003 vs. final).
- Largest MAE reduction: Study 3 – Minimum Wage (YouGov) (WORD2VEC) (+0.004).
- Lowest MAE: Study 1 – Gun Control (MTurk) (SENTENCE_TRANSFORMER) (0.030); Highest MAE: Study 2 – Minimum Wage (MTurk) (TFIDF) (0.092).

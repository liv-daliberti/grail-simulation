# Hyper-Parameter Sweep Results

This catalog aggregates the grid-search results used to select the production KNN configurations. Each table lists the top configurations per study, ranked by validation accuracy (for the slate-ranking task) or validation MAE (for the opinion task).

Key settings:
- Studies: Study 1 – Gun Control (MTurk), Study 2 – Minimum Wage (MTurk), Study 3 – Minimum Wage (YouGov) (study1, study2, study3)
- k sweep: 1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150
- Feature spaces: TFIDF, WORD2VEC, SENTENCE_TRANSFORMER
- Sentence-transformer baseline: `sentence-transformers/all-mpnet-base-v2`

Tables bold the configurations promoted to the finalize stage. Commands beneath each table reproduce the selected configuration.


## Slate-Ranking Sweep Leaders

### Configuration Leaderboards

## TF-IDF Feature Space

| Study | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | Command |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Study 1 – Gun Control (MTurk)** | cosine | — | 0.889 | 0.540 | +0.349 | 2 | 548 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields` |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | 0.338 | 0.368 | -0.030 | 3 | 671 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | 0.292 | 0.479 | -0.187 | 2 | 1,200 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |

## Word2Vec Feature Space

| Study | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | Command |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Study 1 – Gun Control (MTurk)** | cosine | — | 0.861 | 0.540 | +0.321 | 2 | 548 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 128 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40` |
| **Study 2 – Minimum Wage (MTurk)** | cosine | — | 0.334 | 0.368 | -0.034 | 10 | 671 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 10 --knn-k-sweep 10 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40` |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | 0.288 | 0.479 | -0.191 | 10 | 1,200 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 10 --knn-k-sweep 10 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40` |

## Sentence-Transformer Feature Space

| Study | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | Command |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Study 1 – Gun Control (MTurk)** | cosine | viewer_profile, state_text | 0.801 | 0.540 | +0.261 | 2 | 548 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize` |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | 0.308 | 0.368 | -0.060 | 3 | 671 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize` |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | 0.322 | 0.479 | -0.158 | 2 | 1,200 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize` |


### Observations

- TFIDF: Study 1 – Gun Control (MTurk): accuracy 0.889 (baseline 0.540, Δ +0.349, k=2) using cosine distance; Study 2 – Minimum Wage (MTurk): accuracy 0.338 (baseline 0.368, Δ -0.030, k=3) using cosine distance with viewer_profile, state_text; Study 3 – Minimum Wage (YouGov): accuracy 0.292 (baseline 0.479, Δ -0.187, k=2) using cosine distance with viewer_profile, state_text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
- WORD2VEC: Study 1 – Gun Control (MTurk): accuracy 0.861 (baseline 0.540, Δ +0.321, k=2) using word2vec (128d, window 5, min_count 1); Study 2 – Minimum Wage (MTurk): accuracy 0.334 (baseline 0.368, Δ -0.034, k=10) using word2vec (256d, window 5, min_count 1); Study 3 – Minimum Wage (YouGov): accuracy 0.288 (baseline 0.479, Δ -0.191, k=10) using word2vec (256d, window 5, min_count 1) with viewer_profile, state_text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 128 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 10 --knn-k-sweep 10 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 10 --knn-k-sweep 10 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
- SENTENCE_TRANSFORMER: Study 1 – Gun Control (MTurk): accuracy 0.801 (baseline 0.540, Δ +0.261, k=2) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with viewer_profile, state_text; Study 2 – Minimum Wage (MTurk): accuracy 0.308 (baseline 0.368, Δ -0.060, k=3) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with viewer_profile, state_text; Study 3 – Minimum Wage (YouGov): accuracy 0.322 (baseline 0.479, Δ -0.158, k=2) using sentence-transformer `sentence-transformers/all-mpnet-base-v2` with viewer_profile, state_text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`


### Configuration Leaderboards

| Order | Study | Feature space | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | — | 0.889 | 0.540 | +0.349 | 2 | 548 |
| 1 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | — | 0.311 | 0.368 | -0.057 | 4 | 671 |
| 2 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | — | 0.284 | 0.479 | -0.195 | 2 | 1,200 |
| 3 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.881 | 0.540 | +0.341 | 2 | 548 |
| 4 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.338 | 0.368 | -0.030 | 3 | 671 |
| 5 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text | 0.292 | 0.479 | -0.187 | 2 | 1,200 |
| 6 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | — | 0.318 | 0.540 | -0.223 | 2 | 548 |
| 7 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | — | 0.295 | 0.368 | -0.073 | 4 | 671 |
| 8 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | — | 0.280 | 0.479 | -0.199 | 2 | 1,200 |
| 9 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.305 | 0.540 | -0.235 | 2 | 548 |
| 10 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.323 | 0.368 | -0.045 | 3 | 671 |
| 11 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text | 0.288 | 0.479 | -0.191 | 2 | 1,200 |
| 12 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | — | 0.861 | 0.540 | +0.321 | 2 | 548 |
| 13 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | — | 0.325 | 0.368 | -0.043 | 10 | 671 |
| 14 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | — | 0.277 | 0.479 | -0.203 | 2 | 1,200 |
| 15 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | — | 0.861 | 0.540 | +0.321 | 2 | 548 |
| 16 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | — | 0.267 | 0.368 | -0.101 | 2 | 671 |
| 17 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | — | 0.279 | 0.479 | -0.200 | 2 | 1,200 |
| 18 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | — | 0.861 | 0.540 | +0.321 | 2 | 548 |
| 19 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | — | 0.334 | 0.368 | -0.034 | 10 | 671 |
| 20 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | — | 0.273 | 0.479 | -0.206 | 2 | 1,200 |
| 21 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | — | 0.859 | 0.540 | +0.319 | 2 | 548 |
| 22 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | — | 0.334 | 0.368 | -0.034 | 10 | 671 |
| 23 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | — | 0.286 | 0.479 | -0.193 | 2 | 1,200 |
| 24 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.847 | 0.540 | +0.307 | 2 | 548 |
| 25 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.294 | 0.368 | -0.075 | 2 | 671 |
| 26 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | viewer_profile, state_text | 0.268 | 0.479 | -0.212 | 2 | 1,200 |
| 27 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.845 | 0.540 | +0.305 | 2 | 548 |
| 28 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.313 | 0.368 | -0.055 | 4 | 671 |
| 29 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | viewer_profile, state_text | 0.273 | 0.479 | -0.206 | 2 | 1,200 |
| 30 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.849 | 0.540 | +0.308 | 2 | 548 |
| 31 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.329 | 0.368 | -0.039 | 10 | 671 |
| 32 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | viewer_profile, state_text | 0.288 | 0.479 | -0.191 | 10 | 1,200 |
| 33 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.845 | 0.540 | +0.305 | 2 | 548 |
| 34 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.334 | 0.368 | -0.034 | 10 | 671 |
| 35 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | cosine | viewer_profile, state_text | 0.274 | 0.479 | -0.205 | 2 | 1,200 |
| 36 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | — | 0.303 | 0.540 | -0.237 | 2 | 548 |
| 37 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | — | 0.289 | 0.368 | -0.079 | 5 | 671 |
| 38 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | — | 0.273 | 0.479 | -0.206 | 2 | 1,200 |
| 39 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | — | 0.294 | 0.540 | -0.246 | 2 | 548 |
| 40 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | — | 0.262 | 0.368 | -0.106 | 2 | 671 |
| 41 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | — | 0.278 | 0.479 | -0.202 | 2 | 1,200 |
| 42 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | — | 0.305 | 0.540 | -0.235 | 2 | 548 |
| 43 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | — | 0.298 | 0.368 | -0.070 | 5 | 671 |
| 44 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | — | 0.277 | 0.479 | -0.203 | 2 | 1,200 |
| 45 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | — | 0.296 | 0.540 | -0.245 | 2 | 548 |
| 46 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | — | 0.303 | 0.368 | -0.066 | 10 | 671 |
| 47 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | — | 0.280 | 0.479 | -0.199 | 2 | 1,200 |
| 48 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.290 | 0.540 | -0.250 | 2 | 548 |
| 49 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.291 | 0.368 | -0.077 | 2 | 671 |
| 50 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | viewer_profile, state_text | 0.277 | 0.479 | -0.203 | 5 | 1,200 |
| 51 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.290 | 0.540 | -0.250 | 2 | 548 |
| 52 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.279 | 0.368 | -0.089 | 2 | 671 |
| 53 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | viewer_profile, state_text | 0.277 | 0.479 | -0.203 | 4 | 1,200 |
| 54 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.292 | 0.540 | -0.248 | 2 | 548 |
| 55 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.288 | 0.368 | -0.080 | 2 | 671 |
| 56 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | viewer_profile, state_text | 0.278 | 0.479 | -0.202 | 4 | 1,200 |
| 57 | Study 1 – Gun Control (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.292 | 0.540 | -0.248 | 2 | 548 |
| 58 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | l2 | viewer_profile, state_text | 0.300 | 0.368 | -0.069 | 4 | 671 |
| 59 | Study 3 – Minimum Wage (YouGov) | WORD2VEC | l2 | viewer_profile, state_text | 0.271 | 0.479 | -0.208 | 2 | 1,200 |
| 60 | Study 1 – Gun Control (MTurk) | SENTENCE_TRANSFORMER | cosine | — | 0.792 | 0.540 | +0.252 | 2 | 548 |
| 61 | Study 2 – Minimum Wage (MTurk) | SENTENCE_TRANSFORMER | cosine | — | 0.289 | 0.368 | -0.079 | 2 | 671 |
| 62 | Study 3 – Minimum Wage (YouGov) | SENTENCE_TRANSFORMER | cosine | — | 0.314 | 0.479 | -0.165 | 2 | 1,200 |
| 63 | Study 1 – Gun Control (MTurk) | SENTENCE_TRANSFORMER | cosine | viewer_profile, state_text | 0.801 | 0.540 | +0.261 | 2 | 548 |
| 64 | Study 2 – Minimum Wage (MTurk) | SENTENCE_TRANSFORMER | cosine | viewer_profile, state_text | 0.308 | 0.368 | -0.060 | 3 | 671 |
| 65 | Study 3 – Minimum Wage (YouGov) | SENTENCE_TRANSFORMER | cosine | viewer_profile, state_text | 0.322 | 0.479 | -0.158 | 2 | 1,200 |
| 66 | Study 1 – Gun Control (MTurk) | SENTENCE_TRANSFORMER | l2 | — | 0.266 | 0.540 | -0.274 | 2 | 548 |
| 67 | Study 2 – Minimum Wage (MTurk) | SENTENCE_TRANSFORMER | l2 | — | 0.279 | 0.368 | -0.089 | 2 | 671 |
| 68 | Study 3 – Minimum Wage (YouGov) | SENTENCE_TRANSFORMER | l2 | — | 0.321 | 0.479 | -0.158 | 3 | 1,200 |
| 69 | Study 1 – Gun Control (MTurk) | SENTENCE_TRANSFORMER | l2 | viewer_profile, state_text | 0.279 | 0.540 | -0.261 | 2 | 548 |
| 70 | Study 2 – Minimum Wage (MTurk) | SENTENCE_TRANSFORMER | l2 | viewer_profile, state_text | 0.300 | 0.368 | -0.069 | 3 | 671 |
| 71 | Study 3 – Minimum Wage (YouGov) | SENTENCE_TRANSFORMER | l2 | viewer_profile, state_text | 0.315 | 0.479 | -0.164 | 2 | 1,200 |


## Post-Study Opinion Regression

Configurations are ranked by validation MAE (lower is better). Bold rows indicate the selections promoted to the finalize stage.

## TF-IDF Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | l2 | — | — | — | — | — | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text | — | — | — | — | 0.704 | 0.074 | +0.630 | 125 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | — | — | — | — | — | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text | — | — | — | — | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | — | — | — | — | 0.552 | 0.061 | +0.491 | 20 | 165 | 0.092 | -0.004 | 0.127 | 0.790 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile, state_text | — | — | — | — | 0.558 | 0.061 | +0.497 | 20 | 165 | 0.092 | -0.004 | 0.127 | 0.790 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | — | — | — | — | — | 0.545 | 0.061 | +0.485 | 10 | 165 | 0.094 | -0.003 | 0.127 | 0.791 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | — | — | — | — | — | 0.545 | 0.061 | +0.485 | 10 | 165 | 0.094 | -0.003 | 0.127 | 0.791 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | l2 | — | — | — | — | — | 0.502 | 0.058 | +0.444 | 150 | 257 | 0.086 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | — | — | — | — | — | 0.502 | 0.058 | +0.444 | 150 | 257 | 0.086 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text | — | — | — | — | 0.498 | 0.058 | +0.440 | 150 | 257 | 0.087 | +0.003 | 0.124 | 0.772 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile, state_text | — | — | — | — | 0.502 | 0.058 | +0.444 | 150 | 257 | 0.087 | +0.003 | 0.125 | 0.772 | 257 |

## Word2Vec Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | cosine | — | — | 256 | 10 | 1 | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | — | — | 256 | 5 | 1 | 0.704 | 0.074 | +0.630 | 125 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text | — | 256 | 10 | 1 | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text | — | 128 | 10 | 1 | 0.704 | 0.074 | +0.630 | 150 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text | — | 256 | 10 | 1 | 0.704 | 0.074 | +0.630 | 125 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | — | — | 256 | 5 | 1 | 0.564 | 0.061 | +0.503 | 25 | 165 | 0.088 | -0.008 | 0.123 | 0.803 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | — | — | 128 | 5 | 1 | 0.570 | 0.061 | +0.509 | 25 | 165 | 0.089 | -0.007 | 0.123 | 0.801 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | — | — | 128 | 5 | 1 | 0.533 | 0.061 | +0.473 | 50 | 165 | 0.089 | -0.007 | 0.124 | 0.798 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile, state_text | — | 256 | 10 | 1 | 0.552 | 0.061 | +0.491 | 50 | 165 | 0.089 | -0.007 | 0.125 | 0.797 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | — | — | 256 | 5 | 1 | 0.545 | 0.061 | +0.485 | 25 | 165 | 0.089 | -0.007 | 0.123 | 0.801 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | — | 256 | 10 | 1 | 0.471 | 0.058 | +0.412 | 75 | 257 | 0.088 | +0.004 | 0.125 | 0.769 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile, state_text | — | 128 | 10 | 1 | 0.463 | 0.058 | +0.405 | 75 | 257 | 0.088 | +0.004 | 0.125 | 0.769 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text | — | 256 | 10 | 1 | 0.494 | 0.058 | +0.436 | 150 | 257 | 0.088 | +0.004 | 0.126 | 0.768 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | — | — | 256 | 10 | 1 | 0.490 | 0.058 | +0.432 | 150 | 257 | 0.088 | +0.004 | 0.126 | 0.767 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text | — | 128 | 5 | 1 | 0.479 | 0.058 | +0.420 | 100 | 257 | 0.088 | +0.004 | 0.125 | 0.770 | 257 |

## Sentence-Transformer Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | l2 | — | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.704 | 0.074 | +0.630 | 75 | 162 | 0.030 | -0.008 | 0.037 | 0.984 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.704 | 0.074 | +0.630 | 100 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.704 | 0.074 | +0.630 | 100 | 162 | 0.030 | -0.007 | 0.037 | 0.983 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.539 | 0.061 | +0.479 | 125 | 165 | 0.088 | -0.008 | 0.124 | 0.801 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile, state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.539 | 0.061 | +0.479 | 125 | 165 | 0.088 | -0.008 | 0.124 | 0.800 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | — | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.545 | 0.061 | +0.485 | 150 | 165 | 0.089 | -0.008 | 0.124 | 0.800 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | — | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.545 | 0.061 | +0.485 | 150 | 165 | 0.089 | -0.008 | 0.124 | 0.800 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.537 | 0.058 | +0.479 | 75 | 257 | 0.086 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.541 | 0.058 | +0.482 | 75 | 257 | 0.087 | +0.002 | 0.124 | 0.773 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | — | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.533 | 0.058 | +0.475 | 50 | 257 | 0.088 | +0.004 | 0.124 | 0.772 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | — | sentence-transformers/all-mpnet-base-v2 | — | — | — | 0.502 | 0.058 | +0.444 | 20 | 257 | 0.089 | +0.005 | 0.124 | 0.772 | 257 |

### Portfolio Summary

- Weighted MAE 0.073 across 7,430 participants.
- Weighted baseline MAE 0.075 (+0.002 vs. final).
- Largest MAE reduction: SENTENCE_TRANSFORMER – Study 3 – Minimum Wage (YouGov) (+0.005).
- Lowest MAE: SENTENCE_TRANSFORMER – Study 1 – Gun Control (MTurk) (0.030); Highest MAE: TFIDF – Study 2 – Minimum Wage (MTurk) (0.094).

### Opinion Reproduction Commands

- TFIDF:
  - Study 1 – Gun Control (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric l2 --knn-k 150 --knn-k-sweep 150 --out-dir '<run_dir>' --knn-text-fields`
  - Study 2 – Minimum Wage (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 20 --knn-k-sweep 20 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 3 – Minimum Wage (YouGov): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric l2 --knn-k 150 --knn-k-sweep 150 --out-dir '<run_dir>' --knn-text-fields`

- WORD2VEC:
  - Study 1 – Gun Control (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 256 --word2vec-window 10 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  - Study 2 – Minimum Wage (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  - Study 3 – Minimum Wage (YouGov): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 75 --knn-k-sweep 75 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --word2vec-size 256 --word2vec-window 10 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`

- SENTENCE_TRANSFORMER:
  - Study 2 – Minimum Wage (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 125 --knn-k-sweep 125 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  - Study 3 – Minimum Wage (YouGov): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 75 --knn-k-sweep 75 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`
  - Study 1 – Gun Control (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space sentence_transformer --issues gun_control --participant-studies study1 --knn-metric l2 --knn-k 75 --knn-k-sweep 75 --out-dir '<run_dir>' --knn-text-fields --sentence-transformer-model sentence-transformers/all-mpnet-base-v2 --sentence-transformer-batch-size 32 --sentence-transformer-normalize`


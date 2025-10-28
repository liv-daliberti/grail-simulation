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
| **Study 1 – Gun Control (MTurk)** | cosine | viewer_profile, state_text | 0.717 | 0.540 | +0.177 | 3 | 548 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text, freq_youtube | 0.359 | 0.368 | -0.009 | 3 | 671 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,freq_youtube` |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text, newsint | 0.305 | 0.479 | -0.174 | 3 | 1,200 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,newsint` |

## Word2Vec Feature Space

| Study | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | Command |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Study 1 – Gun Control (MTurk)** | cosine | viewer_profile, state_text, ideo2 | 0.276 | 0.540 | -0.265 | 3 | 548 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,ideo2 --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40` |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | 0.308 | 0.368 | -0.060 | 5 | 671 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 5 --knn-k-sweep 5 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40` |


### Observations

- TFIDF: Study 1 – Gun Control (MTurk): accuracy 0.717 (baseline 0.540, Δ +0.177, k=3) using cosine distance with viewer_profile, state_text; Study 2 – Minimum Wage (MTurk): accuracy 0.359 (baseline 0.368, Δ -0.009, k=3) using cosine distance with viewer_profile, state_text, freq_youtube; Study 3 – Minimum Wage (YouGov): accuracy 0.305 (baseline 0.479, Δ -0.174, k=3) using cosine distance with viewer_profile, state_text, newsint.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,freq_youtube`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,newsint`
- WORD2VEC: Study 1 – Gun Control (MTurk): accuracy 0.276 (baseline 0.540, Δ -0.265, k=3) using word2vec (256d, window 5, min_count 1) with viewer_profile, state_text, ideo2; Study 2 – Minimum Wage (MTurk): accuracy 0.308 (baseline 0.368, Δ -0.060, k=5) using word2vec (256d, window 5, min_count 1) with viewer_profile, state_text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,ideo2 --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space word2vec --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 5 --knn-k-sweep 5 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text --word2vec-size 256 --word2vec-window 5 --word2vec-min-count 1 --word2vec-epochs 10 --word2vec-workers 40`


### Configuration Leaderboards

| Order | Study | Feature space | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 1 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 2 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text | 0.299 | 0.479 | -0.180 | 3 | 1,200 |
| 3 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.661 | 0.540 | +0.120 | 3 | 548 |
| 4 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.349 | 0.368 | -0.019 | 3 | 671 |
| 5 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.299 | 0.479 | -0.180 | 3 | 1,200 |
| 6 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.681 | 0.540 | +0.141 | 3 | 548 |
| 7 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.349 | 0.368 | -0.019 | 3 | 671 |
| 8 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.299 | 0.479 | -0.180 | 3 | 1,200 |
| 9 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.695 | 0.540 | +0.155 | 3 | 548 |
| 10 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 11 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.297 | 0.479 | -0.182 | 3 | 1,200 |
| 12 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 13 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 14 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.301 | 0.479 | -0.178 | 5 | 1,200 |
| 15 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, freq_youtube | 0.692 | 0.540 | +0.151 | 3 | 548 |
| 16 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, freq_youtube | 0.359 | 0.368 | -0.009 | 3 | 671 |
| 17 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, freq_youtube | 0.295 | 0.479 | -0.184 | 3 | 1,200 |
| 18 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, youtube_time | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 19 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, youtube_time | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 20 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, youtube_time | 0.299 | 0.479 | -0.180 | 3 | 1,200 |
| 21 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, newsint | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 22 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, newsint | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 23 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, newsint | 0.305 | 0.479 | -0.174 | 3 | 1,200 |
| 24 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, slate_source | 0.697 | 0.540 | +0.157 | 3 | 548 |
| 25 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, slate_source | 0.347 | 0.368 | -0.021 | 3 | 671 |
| 26 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, slate_source | 0.302 | 0.479 | -0.177 | 5 | 1,200 |
| 27 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, educ | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 28 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, educ | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 29 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, educ | 0.298 | 0.479 | -0.181 | 3 | 1,200 |
| 30 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, employ | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 31 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, employ | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 32 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, employ | 0.302 | 0.479 | -0.177 | 5 | 1,200 |
| 33 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, child18 | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 34 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, child18 | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 35 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, child18 | 0.302 | 0.479 | -0.177 | 5 | 1,200 |
| 36 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, inputstate | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 37 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, inputstate | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 38 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, inputstate | 0.302 | 0.479 | -0.177 | 5 | 1,200 |
| 39 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 40 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 41 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text | 0.294 | 0.479 | -0.185 | 3 | 1,200 |
| 42 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.277 | 0.540 | -0.263 | 3 | 548 |
| 43 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.341 | 0.368 | -0.027 | 3 | 671 |
| 44 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.294 | 0.479 | -0.185 | 3 | 1,200 |
| 45 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 46 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.346 | 0.368 | -0.022 | 3 | 671 |
| 47 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.294 | 0.479 | -0.185 | 3 | 1,200 |
| 48 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 49 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.341 | 0.368 | -0.027 | 3 | 671 |
| 50 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.292 | 0.479 | -0.187 | 3 | 1,200 |
| 51 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 52 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 53 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.297 | 0.479 | -0.182 | 5 | 1,200 |
| 54 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, freq_youtube | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 55 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, freq_youtube | 0.347 | 0.368 | -0.021 | 3 | 671 |
| 56 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, freq_youtube | 0.296 | 0.479 | -0.183 | 3 | 1,200 |
| 57 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, youtube_time | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 58 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, youtube_time | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 59 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, youtube_time | 0.294 | 0.479 | -0.185 | 3 | 1,200 |
| 60 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, newsint | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 61 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, newsint | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 62 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, newsint | 0.301 | 0.479 | -0.178 | 3 | 1,200 |
| 63 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, slate_source | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 64 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, slate_source | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 65 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, slate_source | 0.295 | 0.479 | -0.184 | 5 | 1,200 |
| 66 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, educ | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 67 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, educ | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 68 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, educ | 0.293 | 0.479 | -0.186 | 3 | 1,200 |
| 69 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, employ | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 70 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, employ | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 71 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, employ | 0.299 | 0.479 | -0.180 | 5 | 1,200 |
| 72 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, child18 | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 73 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, child18 | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 74 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, child18 | 0.294 | 0.479 | -0.185 | 5 | 1,200 |
| 75 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, inputstate | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 76 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, inputstate | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 77 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, inputstate | 0.297 | 0.479 | -0.182 | 3 | 1,200 |
| 81 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.274 | 0.540 | -0.266 | 3 | 548 |
| 82 | Study 2 – Minimum Wage (MTurk) | WORD2VEC | cosine | viewer_profile, state_text | 0.308 | 0.368 | -0.060 | 5 | 671 |
| 87 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text, ideo1 | 0.272 | 0.540 | -0.268 | 3 | 548 |
| 93 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text, ideo2 | 0.276 | 0.540 | -0.265 | 3 | 548 |
| 99 | Study 1 – Gun Control (MTurk) | WORD2VEC | cosine | viewer_profile, state_text, pol_interest | 0.274 | 0.540 | -0.266 | 3 | 548 |


## Post-Study Opinion Regression

Configurations are ranked by validation MAE (lower is better). Bold rows indicate the selections promoted to the finalize stage.

## TF-IDF Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | cosine | viewer_profile, state_text | — | — | — | — | 0.704 | 0.074 | +0.630 | 25 | 162 | 0.031 | -0.006 | 0.038 | 0.982 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text, religpew | — | — | — | — | 0.704 | 0.074 | +0.630 | 25 | 162 | 0.031 | -0.006 | 0.038 | 0.982 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text, youtube_time | — | — | — | — | 0.704 | 0.074 | +0.630 | 25 | 162 | 0.031 | -0.006 | 0.038 | 0.982 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text, newsint | — | — | — | — | 0.704 | 0.074 | +0.630 | 25 | 162 | 0.031 | -0.006 | 0.038 | 0.982 | 162 |
| Study 1 – Gun Control (MTurk) | cosine | viewer_profile, state_text, educ | — | — | — | — | 0.704 | 0.074 | +0.630 | 25 | 162 | 0.031 | -0.006 | 0.038 | 0.982 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text, freq_youtube | — | — | — | — | 0.558 | 0.061 | +0.497 | 25 | 165 | 0.092 | -0.004 | 0.127 | 0.788 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile, state_text, ideo1 | — | — | — | — | 0.545 | 0.061 | +0.485 | 25 | 165 | 0.093 | -0.004 | 0.128 | 0.787 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile, state_text, freq_youtube | — | — | — | — | 0.564 | 0.061 | +0.503 | 25 | 165 | 0.093 | -0.004 | 0.127 | 0.788 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile, state_text, ideo1 | — | — | — | — | 0.552 | 0.061 | +0.491 | 25 | 165 | 0.093 | -0.004 | 0.128 | 0.787 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile, state_text, pol_interest | — | — | — | — | 0.539 | 0.061 | +0.479 | 25 | 165 | 0.093 | -0.004 | 0.128 | 0.787 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | l2 | viewer_profile, state_text, employ | — | — | — | — | 0.518 | 0.058 | +0.459 | 25 | 257 | 0.088 | +0.004 | 0.126 | 0.767 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text, child18 | — | — | — | — | 0.529 | 0.058 | +0.471 | 25 | 257 | 0.088 | +0.004 | 0.126 | 0.767 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile, state_text, employ | — | — | — | — | 0.514 | 0.058 | +0.455 | 25 | 257 | 0.088 | +0.004 | 0.126 | 0.767 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile, state_text, child18 | — | — | — | — | 0.521 | 0.058 | +0.463 | 25 | 257 | 0.088 | +0.004 | 0.126 | 0.767 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text, pol_interest | — | — | — | — | 0.502 | 0.058 | +0.444 | 25 | 257 | 0.088 | +0.004 | 0.126 | 0.767 | 257 |

### Portfolio Summary

- Weighted MAE 0.073 across 2,920 participants.
- Weighted baseline MAE 0.074 (+0.001 vs. final).
- Weighted directional accuracy 0.578 across 2,920 participants.
- Weighted baseline accuracy 0.063 (+0.515 vs. final).
- Weighted RMSE (change) 0.102 (+0.005 vs. baseline).
- Weighted calibration ECE 0.022 (— vs. baseline).
- Weighted KL divergence 7.524 (+10.911 vs. baseline).
- Largest MAE reduction: TFIDF – Study 3 – Minimum Wage (YouGov) (+0.004).
- Largest RMSE(change) reduction: TFIDF – Study 2 – Minimum Wage (MTurk) (+0.011).
- Lowest MAE: TFIDF – Study 1 – Gun Control (MTurk) (0.031); Highest MAE: TFIDF – Study 2 – Minimum Wage (MTurk) (0.093).
- Biggest KL divergence reduction: TFIDF – Study 2 – Minimum Wage (MTurk) (+11.953).
- Highest directional accuracy: TFIDF – Study 1 – Gun Control (MTurk) (0.704).
- Lowest directional accuracy: TFIDF – Study 3 – Minimum Wage (YouGov) (0.502).
- Largest accuracy gain vs. baseline: TFIDF – Study 1 – Gun Control (MTurk) (+0.630).

### Opinion Reproduction Commands

- TFIDF:
  - Study 1 – Gun Control (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 2 – Minimum Wage (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,freq_youtube`
  - Study 3 – Minimum Wage (YouGov): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric l2 --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,employ`


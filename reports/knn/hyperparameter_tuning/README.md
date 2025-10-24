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
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | 0.352 | 0.368 | -0.016 | 3 | 671 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | 0.299 | 0.479 | -0.180 | 3 | 1,200 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |


### Observations

- TFIDF: Study 1 – Gun Control (MTurk): accuracy 0.717 (baseline 0.540, Δ +0.177, k=3) using cosine distance with viewer_profile, state_text; Study 2 – Minimum Wage (MTurk): accuracy 0.352 (baseline 0.368, Δ -0.016, k=3) using cosine distance with viewer_profile, state_text; Study 3 – Minimum Wage (YouGov): accuracy 0.299 (baseline 0.479, Δ -0.180, k=3) using cosine distance with viewer_profile, state_text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`


### Configuration Leaderboards

| Order | Study | Feature space | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 1 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 2 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text | 0.299 | 0.479 | -0.180 | 3 | 1,200 |
| 3 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.279 | 0.540 | -0.261 | 3 | 548 |
| 4 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 5 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text | 0.294 | 0.479 | -0.185 | 3 | 1,200 |


## Post-Study Opinion Regression

Configurations are ranked by validation MAE (lower is better). Bold rows indicate the selections promoted to the finalize stage.

## TF-IDF Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | cosine | viewer_profile, state_text | — | — | — | — | 0.704 | 0.074 | +0.630 | 25 | 162 | 0.031 | -0.006 | 0.038 | 0.982 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text | — | — | — | — | 0.704 | 0.074 | +0.630 | 25 | 162 | 0.031 | -0.006 | 0.038 | 0.982 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | — | — | — | — | 0.533 | 0.061 | +0.473 | 25 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| Study 2 – Minimum Wage (MTurk) | l2 | viewer_profile, state_text | — | — | — | — | 0.533 | 0.061 | +0.473 | 25 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | l2 | viewer_profile, state_text | — | — | — | — | 0.525 | 0.058 | +0.467 | 25 | 257 | 0.088 | +0.004 | 0.126 | 0.766 | 257 |
| Study 3 – Minimum Wage (YouGov) | cosine | viewer_profile, state_text | — | — | — | — | 0.518 | 0.058 | +0.459 | 25 | 257 | 0.088 | +0.004 | 0.126 | 0.766 | 257 |

### Portfolio Summary

- Weighted MAE 0.074 across 1,168 participants.
- Weighted baseline MAE 0.074 (+0.001 vs. final).
- Weighted directional accuracy 0.575 across 1,168 participants.
- Weighted baseline accuracy 0.063 (+0.512 vs. final).
- Largest MAE reduction: TFIDF – Study 3 – Minimum Wage (YouGov) (+0.004).
- Lowest MAE: TFIDF – Study 1 – Gun Control (MTurk) (0.031); Highest MAE: TFIDF – Study 2 – Minimum Wage (MTurk) (0.093).
- Highest directional accuracy: TFIDF – Study 1 – Gun Control (MTurk) (0.704).
- Lowest directional accuracy: TFIDF – Study 3 – Minimum Wage (YouGov) (0.518).
- Largest accuracy gain vs. baseline: TFIDF – Study 1 – Gun Control (MTurk) (+0.630).

### Opinion Reproduction Commands

- TFIDF:
  - Study 1 – Gun Control (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 2 – Minimum Wage (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 3 – Minimum Wage (YouGov): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric l2 --knn-k 25 --knn-k-sweep 25 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`


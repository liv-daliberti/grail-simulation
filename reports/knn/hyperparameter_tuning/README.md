# Hyper-Parameter Sweep Results

This catalog aggregates the grid-search results used to select the production KNN configurations. Each table lists the top configurations per study, ranked by validation accuracy (for the slate-ranking task) or validation MAE (for the opinion task).

Key settings:
- Studies: Study 1 – Gun Control (MTurk), Study 2 – Minimum Wage (MTurk), Study 3 – Minimum Wage (YouGov) (study1, study2, study3)
- k sweep: 1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150
- Feature spaces: TFIDF, WORD2VEC, SENTENCE_TRANSFORMER
- Sentence-transformer baseline: `sentence-transformers/all-mpnet-base-v2`

Tables bold the configurations promoted to the finalize stage. Commands beneath each table reproduce the selected configuration.
Accuracy values reflect eligible-only accuracy on the validation split at the selected best k (per the configured k-selection method).


## Slate-Ranking Sweep Leaders

### Configuration Leaderboards

## TF-IDF Feature Space

| Study | Metric | Text fields | Acc (best k) ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | Command |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | 0.324 | 0.479 | -0.155 | 50 | 1,200 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |


### Observations

- TFIDF: Study 3 – Minimum Wage (YouGov): accuracy 0.324 (baseline 0.479, Δ -0.155, k=50) using cosine distance with viewer_profile, state_text.
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`


### Configuration Leaderboards

| Order | Study | Feature space | Metric | Text fields | Acc (best k) ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text | 0.324 | 0.479 | -0.155 | 50 | 1,200 |


## Post-Study Opinion Regression

Configurations are ranked by validation MAE (lower is better). Bold rows indicate the selections promoted to the finalize stage.

## TF-IDF Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | cosine | viewer_profile, state_text | — | — | — | — | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | — | — | — | — | 0.527 | 0.061 | +0.467 | 50 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | — | — | — | — | 0.494 | 0.058 | +0.436 | 50 | 257 | 0.088 | +0.004 | 0.125 | 0.771 | 257 |

### Portfolio Summary

- Weighted MAE 0.073 across 584 participants.
- Weighted baseline MAE 0.074 (+0.001 vs. final).
- Weighted directional accuracy 0.562 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.498 vs. final).
- Weighted RMSE (change) 0.102 (+0.005 vs. baseline).
- Weighted calibration ECE 0.017 (— vs. baseline).
- Weighted KL divergence 8.633 (+9.802 vs. baseline).
- Largest MAE reduction: TFIDF – Study 3 – Minimum Wage (YouGov) (+0.004).
- Largest RMSE(change) reduction: TFIDF – Study 2 – Minimum Wage (MTurk) (+0.010).
- Lowest MAE: TFIDF – Study 1 – Gun Control (MTurk) (0.030); Highest MAE: TFIDF – Study 2 – Minimum Wage (MTurk) (0.093).
- Biggest KL divergence reduction: TFIDF – Study 1 – Gun Control (MTurk) (+10.467).
- Highest directional accuracy: TFIDF – Study 1 – Gun Control (MTurk) (0.704).
- Lowest directional accuracy: TFIDF – Study 3 – Minimum Wage (YouGov) (0.494).
- Largest accuracy gain vs. baseline: TFIDF – Study 1 – Gun Control (MTurk) (+0.630).

### Opinion Reproduction Commands

- TFIDF:
  - Study 1 – Gun Control (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 2 – Minimum Wage (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 3 – Minimum Wage (YouGov): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`


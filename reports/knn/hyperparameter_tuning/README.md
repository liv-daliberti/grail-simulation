# Hyper-Parameter Sweep Results

This catalog aggregates the grid-search results used to select the production KNN configurations. Each table lists the top configurations per study, ranked by validation accuracy (for the slate-ranking task) or validation MAE (for the opinion task).

Key settings:
- Studies: Study 1 – Gun Control (MTurk), Study 2 – Minimum Wage (MTurk), Study 3 – Minimum Wage (YouGov) (study1, study2, study3)
- k sweep: 1, 2, 3, 4, 5, 10, 25, 50
- Feature spaces: TFIDF, WORD2VEC, SENTENCE_TRANSFORMER
- Sentence-transformer baseline: `sentence-transformers/all-mpnet-base-v2`

Tables bold the configurations promoted to the finalize stage. Commands beneath each table reproduce the selected configuration.
Accuracy values reflect eligible-only accuracy on the validation split at the selected best k (per the configured k-selection method).


## Slate-Ranking Sweep Leaders

### Configuration Leaderboards

## TF-IDF Feature Space

| Study | Metric | Text fields | Acc (best k) ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | Command |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Study 1 – Gun Control (MTurk)** | cosine | viewer_profile, state_text | 0.763 | 0.540 | +0.223 | 2 | 548 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text, ideo1 | 0.355 | 0.368 | -0.013 | 2 | 671 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,ideo1` |
| **Study 3 – Minimum Wage (YouGov)** | cosine | viewer_profile, state_text | 0.309 | 0.479 | -0.170 | 2 | 1,200 | `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text` |


### Observations

- TFIDF: Study 1 – Gun Control (MTurk): accuracy 0.763 (baseline 0.540, Δ +0.223, k=2) using cosine distance with viewer_profile, state_text; Study 2 – Minimum Wage (MTurk): accuracy 0.355 (baseline 0.368, Δ -0.013, k=2) using cosine distance with viewer_profile, state_text, ideo1; Study 3 – Minimum Wage (YouGov): accuracy 0.309 (baseline 0.479, Δ -0.170, k=2) using cosine distance with viewer_profile, state_text.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,ideo1`
  Command (Study 3 – Minimum Wage (YouGov)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric cosine --knn-k 2 --knn-k-sweep 2 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`


### Configuration Leaderboards

| Order | Study | Feature space | Metric | Text fields | Acc (best k) ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.763 | 0.540 | +0.223 | 2 | 548 |
| 1 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.350 | 0.368 | -0.018 | 2 | 671 |
| 2 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text | 0.309 | 0.479 | -0.170 | 2 | 1,200 |
| 3 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.699 | 0.540 | +0.159 | 2 | 548 |
| 4 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.355 | 0.368 | -0.013 | 2 | 671 |
| 5 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.309 | 0.479 | -0.170 | 2 | 1,200 |
| 6 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.726 | 0.540 | +0.186 | 2 | 548 |
| 7 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.344 | 0.368 | -0.024 | 2 | 671 |
| 8 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.309 | 0.479 | -0.170 | 2 | 1,200 |
| 9 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.735 | 0.540 | +0.195 | 2 | 548 |
| 10 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.346 | 0.368 | -0.022 | 2 | 671 |
| 11 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.306 | 0.479 | -0.173 | 2 | 1,200 |
| 12 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.763 | 0.540 | +0.223 | 2 | 548 |
| 13 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.350 | 0.368 | -0.018 | 2 | 671 |
| 14 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.309 | 0.479 | -0.170 | 2 | 1,200 |
| 45 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.719 | 0.540 | +0.179 | 2 | 548 |
| 46 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.349 | 0.368 | -0.019 | 2 | 671 |
| 47 | Study 3 – Minimum Wage (YouGov) | TFIDF | cosine | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.303 | 0.479 | -0.176 | 2 | 1,200 |
| 48 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.308 | 0.540 | -0.232 | 2 | 548 |
| 49 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 50 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text | 0.305 | 0.479 | -0.174 | 2 | 1,200 |
| 51 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.307 | 0.540 | -0.234 | 2 | 548 |
| 52 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.332 | 0.368 | -0.036 | 2 | 671 |
| 53 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.305 | 0.479 | -0.174 | 2 | 1,200 |
| 54 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.308 | 0.540 | -0.232 | 2 | 548 |
| 55 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.346 | 0.368 | -0.022 | 3 | 671 |
| 56 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.305 | 0.479 | -0.174 | 2 | 1,200 |
| 57 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.308 | 0.540 | -0.232 | 2 | 548 |
| 58 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.341 | 0.368 | -0.027 | 3 | 671 |
| 59 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.300 | 0.479 | -0.179 | 2 | 1,200 |
| 60 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.308 | 0.540 | -0.232 | 2 | 548 |
| 61 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 62 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.302 | 0.479 | -0.177 | 2 | 1,200 |
| 93 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.301 | 0.540 | -0.239 | 2 | 548 |
| 94 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.332 | 0.368 | -0.036 | 2 | 671 |
| 95 | Study 3 – Minimum Wage (YouGov) | TFIDF | l2 | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.300 | 0.479 | -0.179 | 2 | 1,200 |


## Post-Study Opinion Regression

Configurations are ranked by validation MAE (lower is better). Bold rows indicate the selections promoted to the finalize stage.

## TF-IDF Feature Space

| Study | Metric | Text fields | Model | Vec size | Window | Min count | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Study 1 – Gun Control (MTurk)** | l2 | viewer_profile, state_text | — | — | — | — | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text, ideo1 | — | — | — | — | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text, ideo2 | — | — | — | — | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text, pol_interest | — | — | — | — | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| Study 1 – Gun Control (MTurk) | l2 | viewer_profile, state_text, religpew | — | — | — | — | 0.704 | 0.074 | +0.630 | 50 | 162 | 0.030 | -0.007 | 0.038 | 0.983 | 162 |
| **Study 2 – Minimum Wage (MTurk)** | cosine | viewer_profile, state_text | — | — | — | — | 0.527 | 0.061 | +0.467 | 50 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile, state_text, ideo1 | — | — | — | — | 0.527 | 0.061 | +0.467 | 50 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile, state_text, ideo2 | — | — | — | — | 0.527 | 0.061 | +0.467 | 50 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile, state_text, pol_interest | — | — | — | — | 0.527 | 0.061 | +0.467 | 50 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| Study 2 – Minimum Wage (MTurk) | cosine | viewer_profile, state_text, religpew | — | — | — | — | 0.527 | 0.061 | +0.467 | 50 | 165 | 0.093 | -0.003 | 0.128 | 0.786 | 165 |
| **Study 3 – Minimum Wage (YouGov)** | l2 | viewer_profile, state_text | — | — | — | — | 0.490 | 0.058 | +0.432 | 50 | 257 | 0.088 | +0.004 | 0.125 | 0.771 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text, ideo1 | — | — | — | — | 0.490 | 0.058 | +0.432 | 50 | 257 | 0.088 | +0.004 | 0.125 | 0.771 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text, ideo2 | — | — | — | — | 0.490 | 0.058 | +0.432 | 50 | 257 | 0.088 | +0.004 | 0.125 | 0.771 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text, pol_interest | — | — | — | — | 0.490 | 0.058 | +0.432 | 50 | 257 | 0.088 | +0.004 | 0.125 | 0.771 | 257 |
| Study 3 – Minimum Wage (YouGov) | l2 | viewer_profile, state_text, religpew | — | — | — | — | 0.490 | 0.058 | +0.432 | 50 | 257 | 0.088 | +0.004 | 0.125 | 0.771 | 257 |

### Portfolio Summary

- Weighted MAE 0.073 across 2,920 participants.
- Weighted baseline MAE 0.074 (+0.001 vs. final).
- Weighted directional accuracy 0.560 across 2,920 participants.
- Weighted baseline accuracy 0.063 (+0.497 vs. final).
- Weighted RMSE (change) 0.102 (+0.005 vs. baseline).
- Weighted calibration ECE 0.019 (— vs. baseline).
- Weighted KL divergence 8.631 (+9.804 vs. baseline).
- Largest MAE reduction: TFIDF – Study 3 – Minimum Wage (YouGov) (+0.004).
- Largest RMSE(change) reduction: TFIDF – Study 2 – Minimum Wage (MTurk) (+0.010).
- Lowest MAE: TFIDF – Study 1 – Gun Control (MTurk) (0.030); Highest MAE: TFIDF – Study 2 – Minimum Wage (MTurk) (0.093).
- Biggest KL divergence reduction: TFIDF – Study 1 – Gun Control (MTurk) (+10.467).
- Highest directional accuracy: TFIDF – Study 1 – Gun Control (MTurk) (0.704).
- Lowest directional accuracy: TFIDF – Study 3 – Minimum Wage (YouGov) (0.490).
- Largest accuracy gain vs. baseline: TFIDF – Study 1 – Gun Control (MTurk) (+0.630).

### Opinion Reproduction Commands

- TFIDF:
  - Study 1 – Gun Control (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric l2 --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 2 – Minimum Wage (MTurk): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  - Study 3 – Minimum Wage (YouGov): `python -m knn.cli --task opinion --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study3 --knn-metric l2 --knn-k 50 --knn-k-sweep 50 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`


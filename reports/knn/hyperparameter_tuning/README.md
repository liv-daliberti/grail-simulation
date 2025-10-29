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
| 51 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.307 | 0.540 | -0.234 | 2 | 548 |
| 52 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1 | 0.332 | 0.368 | -0.036 | 2 | 671 |
| 54 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.308 | 0.540 | -0.232 | 2 | 548 |
| 55 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo2 | 0.346 | 0.368 | -0.022 | 3 | 671 |
| 57 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.308 | 0.540 | -0.232 | 2 | 548 |
| 58 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, pol_interest | 0.341 | 0.368 | -0.027 | 3 | 671 |
| 60 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.308 | 0.540 | -0.232 | 2 | 548 |
| 61 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, religpew | 0.343 | 0.368 | -0.025 | 3 | 671 |
| 93 | Study 1 – Gun Control (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.301 | 0.540 | -0.239 | 2 | 548 |
| 94 | Study 2 – Minimum Wage (MTurk) | TFIDF | l2 | viewer_profile, state_text, ideo1, ideo2, pol_interest, religpew, freq_youtube, youtube_time, newsint, slate_source, educ, employ, child18, inputstate, income, participant_study | 0.332 | 0.368 | -0.036 | 2 | 671 |


## Post-Study Opinion Regression

No opinion sweeps were available when this report was generated.
Run the KNN pipeline with `--stage sweeps` or `--stage full` once artifacts are ready.


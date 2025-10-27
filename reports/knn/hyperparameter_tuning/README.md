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


### Observations

- TFIDF: Study 1 – Gun Control (MTurk): accuracy 0.717 (baseline 0.540, Δ +0.177, k=3) using cosine distance with viewer_profile, state_text; Study 2 – Minimum Wage (MTurk): accuracy 0.359 (baseline 0.368, Δ -0.009, k=3) using cosine distance with viewer_profile, state_text, freq_youtube.
  Command (Study 1 – Gun Control (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues gun_control --participant-studies study1 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text`
  Command (Study 2 – Minimum Wage (MTurk)): `python -m knn.cli --task slate --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --feature-space tfidf --issues minimum_wage --participant-studies study2 --knn-metric cosine --knn-k 3 --knn-k-sweep 3 --out-dir '<run_dir>' --knn-text-fields viewer_profile,state_text,freq_youtube`


### Configuration Leaderboards

| Order | Study | Feature space | Metric | Text fields | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 1 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 3 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.661 | 0.540 | +0.120 | 3 | 548 |
| 4 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo1 | 0.349 | 0.368 | -0.019 | 3 | 671 |
| 6 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.681 | 0.540 | +0.141 | 3 | 548 |
| 7 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, ideo2 | 0.349 | 0.368 | -0.019 | 3 | 671 |
| 9 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.695 | 0.540 | +0.155 | 3 | 548 |
| 10 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, pol_interest | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 12 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 13 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, religpew | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 15 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, freq_youtube | 0.692 | 0.540 | +0.151 | 3 | 548 |
| 16 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, freq_youtube | 0.359 | 0.368 | -0.009 | 3 | 671 |
| 18 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, youtube_time | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 19 | Study 2 – Minimum Wage (MTurk) | TFIDF | cosine | viewer_profile, state_text, youtube_time | 0.352 | 0.368 | -0.016 | 3 | 671 |
| 21 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, newsint | 0.717 | 0.540 | +0.177 | 3 | 548 |
| 24 | Study 1 – Gun Control (MTurk) | TFIDF | cosine | viewer_profile, state_text, slate_source | 0.697 | 0.540 | +0.157 | 3 | 548 |


## Post-Study Opinion Regression

No opinion sweeps were available when this report was generated.
Run the KNN pipeline with `--stage sweeps` or `--stage full` once artifacts are ready.


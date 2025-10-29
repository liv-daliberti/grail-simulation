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


### Observations



### Configuration Leaderboards

No sweep outcomes were recorded for this run.


## Post-Study Opinion Regression

No opinion sweeps were available when this report was generated.
Run the KNN pipeline with `--stage sweeps` or `--stage full` once artifacts are ready.


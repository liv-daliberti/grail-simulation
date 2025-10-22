# KNN Next-Video Baseline

This report summarizes the slate-ranking KNN models that predict the next video a viewer will click.  
All results are now organized by the three study cohorts that appear in the cleaned GRAIL dataset:

- **Study 1 – Gun Control (MTurk)**
- **Study 2 – Minimum Wage (MTurk)**
- **Study 3 – Minimum Wage (YouGov)**

Each pipeline run refreshes the tables below with validation accuracy, the elbow-selected `k`, and baseline comparisons for TF-IDF, Word2Vec, and Sentence-Transformer feature spaces.

- Dataset: `data/cleaned_grail`
- Split: validation
- Metric: accuracy on eligible slates (gold index present)

## TF-IDF Feature Space

| Study | Accuracy ↑ | Best k | Most-frequent baseline ↑ |
| --- | ---: | ---: | ---: |
| _Pipeline run will populate these rows_ |  |  |  |

- Plots: `tfidf/elbow_study1.png`, `tfidf/elbow_study2.png`, `tfidf/elbow_study3.png`

## Word2Vec Feature Space

| Study | Accuracy ↑ | Best k | Most-frequent baseline ↑ |
| --- | ---: | ---: | ---: |
| _Pipeline run will populate these rows_ |  |  |  |

- Plots: `word2vec/elbow_study1.png`, `word2vec/elbow_study2.png`, `word2vec/elbow_study3.png`

## Sentence-Transformer Feature Space

| Study | Accuracy ↑ | Best k | Most-frequent baseline ↑ |
| --- | ---: | ---: | ---: |
| _Pipeline run will populate these rows_ |  |  |  |

- Plots: `sentence_transformer/elbow_study1.png`, `sentence_transformer/elbow_study2.png`, `sentence_transformer/elbow_study3.png`

## Observations

- Slate metrics are scoped per study, allowing each feature space to tune `k` and hyperparameters per cohort.
- Study-specific directories live under `models/knn/<feature-space>/study{1,2,3}` with per-`k` predictions and metrics JSON.
- Regenerate this README via `reports/build-reports.sh` (or `python -m knn.pipeline --stage reports`) after any new training run.

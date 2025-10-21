# KNN Report Catalog

This directory centralises the artefacts and write-ups for the KNN baselines we maintain:

- `next_video/` – slate-ranking accuracy results for the next-video baseline, with feature-specific assets in `tfidf/` and `word2vec/`.
- `opinion/` – post-study opinion regression analysis, including metric tables and heatmaps grouped by feature space.
- `hyperparameter_tuning/` – consolidated notes from the k-sweeps that produced the current configurations.

Generated plots live alongside the README for each task so it is easy to browse results without hunting through sibling folders. Model outputs (predictions, metrics JSONL) remain in `models/knn/...` as referenced inside the individual reports.

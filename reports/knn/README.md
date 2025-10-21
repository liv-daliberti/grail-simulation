# KNN Report Catalog

This directory centralises the artefacts and write-ups for the KNN baselines we maintain:

- `next_video/` – slate-ranking accuracy results for the next-video baseline, with feature-specific assets in `tfidf/` and `word2vec/`.
- `opinion/` – post-study opinion regression analysis, including metric tables and heatmaps grouped by feature space.
- `hyperparameter_tuning/` – consolidated notes from the k-sweeps that produced the current configurations.

Generated plots live alongside the README for each task so it is easy to browse results without hunting through sibling folders. Model outputs (predictions, metrics JSONL) remain in `models/knn/...` as referenced inside the individual reports.

## Training Walkthrough

**Sample prompt**
```text
VIEWER 31-year-old, Black or African-American (non-Hispanic) woman; democrat liberal; $70,000-$79,999; college-educated; watches YouTube weekly.
Initial Viewpoint: Opposes stricter gun laws

CURRENTLY WATCHING How to Create a Gun-Free America in 5 Easy Steps (from ReasonTV)

RECENTLY WATCHED (NEWEST LAST)
1. Do We Need Stricter Gun Control? - The People Speak (watched 259s of 259s (100% complete), from VICE News)

SURVEY HIGHLIGHTS
party identification is Democrat, party lean is Not very strong Democrat, ideology is Liberal, political interest is not very interested in politics, watches YouTube weekly, gun policy importance is not at all important, gun regulation support score is 25%, and does not identify as enthusiastic about guns.

OPTIONS
1. Democrats Need to Listen to Gun Owners (The Atlantic) — Engagement: views 29,001, likes 379
2. Piers Morgan Argues With Pro-Gun Campaigner About Orlando Shooting | Good Morning Britain — Engagement: views 2,433,982, likes 18,834
```

**TF-IDF snapshot**
| Token | Weight |
| --- | ---: |
| gun | 0.559 |
| control | 0.226 |
| steps | 0.140 |
| free | 0.140 |
| easy | 0.140 |
| create | 0.140 |
| britain | 0.137 |
| america | 0.137 |

**Word2Vec neighbours (vector size = 256)**
| Token | Similarity |
| --- | ---: |
| control | 0.550 |
| inner | 0.490 |
| e9nrtzpxcdo | 0.489 |
| 2. | 0.486 |
| liberal | 0.475 |

**Train & evaluate**
```bash
PYTHONPATH=src python -m knn.pipeline \
  --dataset data/cleaned_grail \
  --issues gun_control,minimum_wage \
  --out-dir models/knn \
  --reports-dir reports/knn \
  --feature-space tfidf \
  --fit-index \
  --knn-k-sweep 1,2,3,4,5,10,15,20,25,50,100 \
  --eval-max 0 \
  --train-curve-max 2000
```

The command above prepares prompt documents, fits the requested feature space (TF-IDF by default, switchable to Word2Vec via `--feature-space word2vec`), trains the FAISS-backed index, and writes per-`k` metrics. Evaluation runs through the validation split and refreshes Markdown assets under `reports/knn/`.

**Validation snapshot**
| Issue | Feature space | Best k | Accuracy | Majority baseline |
| --- | --- | ---: | ---: | ---: |
| Gun control | TF-IDF | 2 | 0.894 | 0.540 |
| Gun control | Word2Vec | 2 | 0.870 | 0.540 |
| Minimum wage | TF-IDF | 3 | 0.306 | 0.439 |
| Minimum wage | Word2Vec | 4 | 0.292 | 0.439 |

- Accuracy values come from `knn_eval_*_validation_metrics.json` under `models/knn/(tfidf|word2vec)/(issue)/` (for example, `models/knn/tfidf/gun_control/knn_eval_gun_control_validation_metrics.json`).
- The sweep over `k ∈ {1,2,3,4,5,10,15,20,25,50,100}` ensures we surface the most competitive neighbourhood size for each issue/feature pairing. For minimum-wage cohorts the slate imbalance keeps majority-choice accuracy (0.439) above any KNN configuration, signalling that richer features or additional filtering are required to close the gap.
- For gun-control slates, both TF-IDF and Word2Vec comfortably beat the 0.540 most-frequent baseline, and TF-IDF with `k=2` delivers the strongest validation accuracy while preserving high coverage.

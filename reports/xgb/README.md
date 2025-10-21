# XGBoost Report Catalog

This directory mirrors the assets emitted by `python -m xgb.pipeline` and
`training/training-xgb.sh`:

- `next_video/` – selected slate-ranking evaluation summaries.
- `opinion/` – post-study opinion regression metrics derived from the same hyper-parameters.
- `hyperparameter_tuning/` – notes from the learning-rate/depth sweeps that choose the current configuration.

The pipeline regenerates these Markdown files on every run so that dashboards in
the repository stay in sync with the latest experiment sweep.

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

**Train & evaluate**
```bash
PYTHONPATH=src python -m xgb.pipeline \
  --dataset data/cleaned_grail \
  --issues gun_control,minimum_wage \
  --out-dir models/xgb \
  --reports-dir reports/xgb \
  --max-train 200000 \
  --max-features 200000 \
  --learning-rate-grid 0.05,0.1 \
  --max-depth-grid 4,6 \
  --n-estimators-grid 200,300 \
  --subsample-grid 0.7 \
  --colsample-grid 0.7 \
  --reg-lambda-grid 1.0 \
  --reg-alpha-grid 0.0,0.5
```

The pipeline mirrors the KNN document builder, fits a TF-IDF vectorizer over the training corpus, encodes candidate video ids with a label encoder, and trains an `XGBClassifier` for each issue/configuration in the grid. Validation metrics and the Markdown summaries under `reports/xgb/` are rewritten once the sweep completes.

**Validation snapshot**
| Issue | Accuracy | Coverage | Selected config |
| --- | ---: | ---: | --- |
| Gun control | 0.960 | 0.963 | lr = 0.05, depth = 4, estimators = 200, subsample = 0.7, colsample = 0.7, λ = 1.0, α = 0.5 |
| Minimum wage | 0.375 | 0.375 | lr = 0.05, depth = 4, estimators = 200, subsample = 0.7, colsample = 0.7, λ = 1.0, α = 0.0 |

- Metrics originate from the highest-accuracy `metrics.json` files under `models/xgb/sweeps/<issue>/<config>/` (for example, `models/xgb/sweeps/gun_control/lr0p05_depth4_estim200_sub0p7_col0p7_l21_l10p5/gun_control/metrics.json`).
- The gun-control model substantially outperforms the 0.540 most-frequent baseline drawn from the validation split while also covering 96% of slates with known candidates.
- Minimum-wage slates remain challenging: after enumerating the specified depth, learning-rate, and regularization grids the best configuration tops out at 0.375 accuracy, still below the 0.439 majority baseline captured in the KNN diagnostics. This indicates that, within the current feature construction and hyper-parameter bounds, the model has effectively saturated and further gains will require either richer text fields or alternative objectives.

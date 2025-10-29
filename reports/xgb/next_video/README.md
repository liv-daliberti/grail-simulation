# XGBoost Next-Video Baseline

Slate-ranking accuracy for the selected XGBoost configuration.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics include overall accuracy, eligible-only accuracy (gold present in slate), coverage of known candidates, and availability of known neighbors.
- In the summary table below, the Accuracy column reports eligible-only accuracy to match KNN reports.
- `Known hits / total` counts successes among slates that contained a known candidate; `Known availability` is the share of evaluations with any known candidate present.
- `Avg prob` reports the mean predicted probability assigned to known candidate hits.

## Portfolio Summary

- Weighted eligible-only accuracy 0.468 across 2,419 eligible slates.
- Weighted known-candidate coverage 0.467 over 2,417 eligible slates.
- Known-candidate availability 0.999 relative to all evaluated slates.
- Mean predicted probability on known candidates 0.468 (across 3 studies with recorded probabilities).
- Highest study accuracy: Study 1 – Gun Control (MTurk) (0.874).
- Lowest study accuracy: Study 2 – Minimum Wage (MTurk) (0.329).

| Study | Issue | Acc (eligible) ↑ | Baseline ↑ | Random ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.874 | 0.540 | 0.326 | 479/548 | 0.874 | 477/546 | 0.996 | 0.610 |
| Study 2 – Minimum Wage (MTurk) | Minimum Wage | 0.329 | 0.368 | 0.255 | 221/671 | 0.329 | 221/671 | 1.000 | 0.360 |
| Study 3 – Minimum Wage (YouGov) | Minimum Wage | 0.359 | 0.479 | 0.255 | 431/1,200 | 0.359 | 431/1,200 | 1.000 | 0.435 |

## Accuracy Curves

![Slate accuracy overview](curves/accuracy_overview.png)

## Cross-Study Holdouts

- Highest holdout accuracy: Study 1 – Gun Control (MTurk) (0.874).
- Lowest holdout accuracy: Study 2 – Minimum Wage (MTurk) (0.329).
- Average holdout accuracy 0.521.
- Highest holdout eligible-only accuracy: Study 1 – Gun Control (MTurk) (0.874).
- Lowest holdout eligible-only accuracy: Study 2 – Minimum Wage (MTurk) (0.329).
- Average holdout eligible-only accuracy 0.521.

| Holdout study | Issue | Accuracy ↑ | Acc (eligible) ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.874 | 0.874 | 479/548 | 0.874 | 477/546 | 0.996 | 0.610 |
| Study 2 – Minimum Wage (MTurk) | Minimum Wage | 0.329 | 0.329 | 221/671 | 0.329 | 221/671 | 1.000 | 0.360 |
| Study 3 – Minimum Wage (YouGov) | Minimum Wage | 0.359 | 0.359 | 431/1,200 | 0.359 | 431/1,200 | 1.000 | 0.435 |

## Observations

- Study 1 – Gun Control (MTurk): accuracy 0.874, eligible accuracy 0.874, coverage 0.874, known availability 0.996.
- Study 2 – Minimum Wage (MTurk): accuracy 0.329, eligible accuracy 0.329, coverage 0.329, known availability 1.000.
- Study 3 – Minimum Wage (YouGov): accuracy 0.359, eligible accuracy 0.359, coverage 0.359, known availability 1.000.
- Average accuracy 0.521.
- Average eligible-only accuracy 0.521.
- Known coverage averages 0.521.
- Known candidate availability averages 0.999.

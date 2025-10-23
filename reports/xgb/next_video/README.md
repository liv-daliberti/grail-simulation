# XGBoost Next-Video Baseline

Slate-ranking accuracy for the selected XGBoost configuration.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: accuracy, coverage of known candidates, and availability of known neighbours.

## Portfolio Summary

- Weighted accuracy 0.546 across 2,419 evaluated slates.
- Weighted known-candidate coverage 0.546 over 2,417 eligible slates.
- Known-candidate availability 0.999 relative to all evaluated slates.
- Mean predicted probability on known candidates 0.600 (across 3 studies with recorded probabilities).
- Highest study accuracy: Study 1 – Gun Control (MTurk) (0.973).
- Lowest study accuracy: Study 2 – Minimum Wage (MTurk) (0.382).

| Study | Issue | Accuracy ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.973 | 533/548 | 0.976 | 533/546 | 0.996 | 0.924 |
| Study 2 – Minimum Wage (MTurk) | Minimum Wage | 0.382 | 256/671 | 0.382 | 256/671 | 1.000 | 0.435 |
| Study 3 – Minimum Wage (YouGov) | Minimum Wage | 0.443 | 531/1,200 | 0.443 | 531/1,200 | 1.000 | 0.442 |

## Accuracy Curves

![Study 2 – Minimum Wage (MTurk)](curves/study_2_–_minimum_wage_(mturk).png)

![Study 3 – Minimum Wage (YouGov)](curves/study_3_–_minimum_wage_(yougov).png)

## Observations

- Study 1 – Gun Control (MTurk): accuracy 0.973, coverage 0.976, known availability 0.996, avg probability 0.924.
- Study 2 – Minimum Wage (MTurk): accuracy 0.382, coverage 0.382, known availability 1.000, avg probability 0.435.
- Study 3 – Minimum Wage (YouGov): accuracy 0.443, coverage 0.443, known availability 1.000, avg probability 0.442.
- Portfolio mean accuracy 0.599 across 3 studies.
- Mean coverage 0.600.
- Known candidate availability averages 0.999.

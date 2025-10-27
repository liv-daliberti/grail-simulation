# XGBoost Next-Video Baseline

Slate-ranking accuracy for the selected XGBoost configuration.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: accuracy, coverage of known candidates, and availability of known neighbors.

## Portfolio Summary

- Weighted accuracy 0.991 across 1,219 evaluated slates.
- Weighted known-candidate coverage 0.993 over 1,217 eligible slates.
- Known-candidate availability 0.998 relative to all evaluated slates.
- Mean predicted probability on known candidates 0.949 (across 2 studies with recorded probabilities).
- Highest study accuracy: Study 2 – Minimum Wage (MTurk) (0.996).
- Lowest study accuracy: Study 1 – Gun Control (MTurk) (0.985).

| Study | Issue | Accuracy ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.985 | 540/548 | 0.989 | 540/546 | 0.996 | 0.935 |
| Study 2 – Minimum Wage (MTurk) | Minimum Wage | 0.996 | 668/671 | 0.996 | 668/671 | 1.000 | 0.963 |

## Accuracy Curves

![Slate accuracy overview](curves/accuracy_overview.png)

## Observations

- Study 1 – Gun Control (MTurk): accuracy 0.985, coverage 0.989, known availability 0.996.
- Study 2 – Minimum Wage (MTurk): accuracy 0.996, coverage 0.996, known availability 1.000.
- Average accuracy 0.990.
- Known coverage averages 0.992.
- Known candidate availability averages 0.998.

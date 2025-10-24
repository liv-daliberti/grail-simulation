# XGBoost Next-Video Baseline

Slate-ranking accuracy for the selected XGBoost configuration.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: accuracy, coverage of known candidates, and availability of known neighbors.

## Portfolio Summary

- Weighted accuracy 0.985 across 548 evaluated slates.
- Weighted known-candidate coverage 0.989 over 546 eligible slates.
- Known-candidate availability 0.996 relative to all evaluated slates.
- Mean predicted probability on known candidates 0.935 (across 1 study with recorded probabilities).
- Highest study accuracy: Study 1 – Gun Control (MTurk) (0.985).

| Study | Issue | Accuracy ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.985 | 540/548 | 0.989 | 540/546 | 0.996 | 0.935 |

## Accuracy Curves

![Slate accuracy overview](curves/accuracy_overview.png)

## Observations

- Study 1 – Gun Control (MTurk): accuracy 0.985, coverage 0.989, known availability 0.996.
- Average accuracy 0.985.
- Known coverage averages 0.989.
- Known candidate availability averages 0.996.

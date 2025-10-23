# XGBoost Next-Video Baseline

Slate-ranking accuracy for the selected XGBoost configuration.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: accuracy, coverage of known candidates, and availability of known neighbours.

| Study | Issue | Accuracy ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.973 | 533/548 | 0.976 | 533/546 | 0.996 | 0.924 |

## Observations

- Study 1 – Gun Control (MTurk): accuracy 0.973, coverage 0.976, known availability 0.996, avg probability 0.924.
- Portfolio mean accuracy 0.973 across 1 studies.
- Mean coverage 0.976.
- Known candidate availability averages 0.996.


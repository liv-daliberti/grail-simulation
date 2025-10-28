# XGBoost Next-Video Baseline

Slate-ranking accuracy for the selected XGBoost configuration.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: accuracy, coverage of known candidates, and availability of known neighbors.
- Table columns capture validation accuracy, counts of correct predictions, known-candidate recall, and probability calibration for the selected slates.
- `Known hits / total` counts successes among slates that contained a known candidate; `Known availability` is the share of evaluations with any known candidate present.
- `Avg prob` reports the mean predicted probability assigned to known candidate hits.

## Portfolio Summary

- Weighted accuracy 0.987 across 548 evaluated slates.
- Weighted known-candidate coverage 0.991 over 546 eligible slates.
- Known-candidate availability 0.996 relative to all evaluated slates.
- Mean predicted probability on known candidates 0.936 (across 1 study with recorded probabilities).
- Highest study accuracy: Study 1 – Gun Control (MTurk) (0.987).

| Study | Issue | Accuracy ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.987 | 541/548 | 0.991 | 541/546 | 0.996 | 0.936 |

## Accuracy Curves

![Slate accuracy overview](curves/accuracy_overview.png)

## Cross-Study Holdouts

Leave-one-study-out metrics were unavailable when this report was generated.

## Observations

- Study 1 – Gun Control (MTurk): accuracy 0.987, coverage 0.991, known availability 0.996.
- Average accuracy 0.987.
- Known coverage averages 0.991.
- Known candidate availability averages 0.996.

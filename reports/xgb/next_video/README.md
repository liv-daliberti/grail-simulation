# XGBoost Next-Video Baseline

Slate-ranking accuracy for the selected XGBoost configuration.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: overall accuracy, eligible-only accuracy (gold present in slate), coverage of known candidates, and availability of known neighbors.
- Table columns capture validation accuracy, counts of correct predictions, known-candidate recall, and probability calibration for the selected slates.
- `Known hits / total` counts successes among slates that contained a known candidate; `Known availability` is the share of evaluations with any known candidate present.
- `Avg prob` reports the mean predicted probability assigned to known candidate hits.

## Portfolio Summary

- Weighted accuracy 0.990 across 1,219 evaluated slates.
- Weighted known-candidate coverage 0.992 over 1,217 eligible slates.
- Known-candidate availability 0.998 relative to all evaluated slates.
- Mean predicted probability on known candidates 0.953 (across 2 studies with recorded probabilities).
- Highest study accuracy: Study 2 – Minimum Wage (MTurk) (0.993).
- Lowest study accuracy: Study 1 – Gun Control (MTurk) (0.987).

| Study | Issue | Accuracy ↑ | Baseline ↑ | Random ↑ | Correct / evaluated | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Study 1 – Gun Control (MTurk) | Gun Control | 0.987 | 0.540 | 0.326 | 541/548 | 0.991 | 541/546 | 0.996 | 0.940 |
| Study 2 – Minimum Wage (MTurk) | Minimum Wage | 0.993 | 0.368 | 0.255 | 666/671 | 0.993 | 666/671 | 1.000 | 0.966 |

## Accuracy Curves

![Slate accuracy overview](curves/accuracy_overview.png)

## Cross-Study Holdouts

Leave-one-study-out metrics were unavailable when this report was generated.

## Observations

- Study 1 – Gun Control (MTurk): accuracy 0.987, eligible accuracy —, coverage 0.991, known availability 0.996.
- Study 2 – Minimum Wage (MTurk): accuracy 0.993, eligible accuracy —, coverage 0.993, known availability 1.000.
- Average accuracy 0.990.
- Known coverage averages 0.992.
- Known candidate availability averages 0.998.

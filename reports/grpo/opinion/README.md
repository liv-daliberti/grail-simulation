# GRPO Opinion Regression

Opinion-shift evaluation across the canonical participant studies. Baseline metrics treat the pre-study opinion index as the prediction.

## Combined Metrics

| Metric | Value |
| --- | ---: |
| Eligible | 522 |
| MAE (post-study) | 1.895 |
| MAE (change) | 1.895 |
| Direction accuracy | 0.563 |
| RMSE (post-study) | 2.061 |
| RMSE (change) | 2.061 |
| Calibration ECE | 1.895 |

## Per-Study Breakdown

| Study | Participants | Eligible | MAE ↓ | Baseline MAE ↓ | Direction ↑ | Baseline Direction ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 100 | 100 | 2.740 | 0.228 | 0.730 | 0.080 |
| Study 2 – Minimum Wage (MTurk) | 165 | 165 | 1.689 | 0.240 | 0.545 | 0.061 |
| Study 3 – Minimum Wage (YouGov) | 257 | 257 | 1.697 | 0.216 | 0.510 | 0.058 |

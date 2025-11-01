# GRAIL Opinion Regression

Opinion-shift evaluation across the canonical participant studies. Baseline metrics treat the pre-study opinion index as the prediction.

## Combined Metrics

| Metric | Value |
| --- | ---: |
| Eligible | 522 |
| MAE (post-study) | 1.249 |
| MAE (change) | 1.249 |
| Direction accuracy | 0.563 |
| RMSE (post-study) | 1.861 |
| RMSE (change) | 1.861 |
| Calibration ECE | 1.249 |

## Per-Study Breakdown

| Study | Participants | Eligible | MAE ↓ | Baseline MAE ↓ | Direction ↑ | Baseline Direction ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 100 | 100 | 2.740 | 0.228 | 0.730 | 0.080 |
| Study 2 – Minimum Wage (MTurk) | 165 | 165 | 0.902 | 0.240 | 0.545 | 0.061 |
| Study 3 – Minimum Wage (YouGov) | 257 | 257 | 0.892 | 0.216 | 0.510 | 0.058 |

# GRAIL Opinion Regression

Opinion-shift evaluation across the canonical participant studies. Baseline metrics treat the pre-study opinion index as the prediction.

## Combined Metrics

| Metric | Value |
| --- | ---: |
| Eligible | 472 |
| MAE (post-study) | 1.103 |
| MAE (change) | 1.103 |
| Direction accuracy | 0.551 |
| RMSE (post-study) | 1.694 |
| RMSE (change) | 1.694 |
| Calibration ECE | 1.103 |

## Per-Study Breakdown

| Study | Participants | Eligible | MAE ↓ | Baseline MAE ↓ | Direction ↑ | Baseline Direction ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 50 | 50 | 2.853 | 0.226 | 0.780 | 0.040 |
| Study 2 – Minimum Wage (MTurk) | 165 | 165 | 0.902 | 0.240 | 0.545 | 0.061 |
| Study 3 – Minimum Wage (YouGov) | 257 | 257 | 0.892 | 0.216 | 0.510 | 0.058 |

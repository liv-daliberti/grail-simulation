# GRPO Opinion Regression

Opinion-shift evaluation across the canonical participant studies. Baseline metrics treat the pre-study opinion index as the prediction.

## Combined Metrics

| Metric | Value |
| --- | ---: |
| Eligible | 584 |
| MAE (post-study) | 0.712 |
| MAE (change) | 0.712 |
| Direction accuracy | 0.574 |
| RMSE (post-study) | 0.981 |
| RMSE (change) | 0.981 |
| Calibration ECE | 0.712 |

## Per-Study Breakdown

| Study | Participants | Eligible | MAE ↓ | Baseline MAE ↓ | Direction ↑ | Baseline Direction ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 162 | 0.554 | — | 0.704 | 0.074 |
| Study 2 – Minimum Wage (MTurk) | 165 | 165 | 0.853 | — | 0.545 | 0.061 |
| Study 3 – Minimum Wage (YouGov) | 257 | 257 | 0.722 | — | 0.510 | 0.058 |

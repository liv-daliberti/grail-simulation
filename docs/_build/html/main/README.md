# Portfolio Comparison (Main)

This report compares grail, grpo, gpt4o, knn, and xgb across the next-video and opinion tasks. KNN and XGB opinion metrics come from their `opinion_from_next` runs.

## Next-Video Eligible Accuracy (↑)

| Study | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 0.261 | 0.960 | 0.977 | 0.763 | 0.874 |
| Study 2 – Minimum Wage (MTurk) | 0.286 | 0.446 | 0.506 | 0.355 | 0.329 |
| Study 3 – Minimum Wage (YouGov) | 0.251 | 0.432 | 0.479 | 0.320 | 0.359 |

## Opinion Directional Accuracy (↑)

| Study | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 0.704 | 0.704 | 0.704 | 0.704 | 0.759 |
| Study 2 – Minimum Wage (MTurk) | 0.545 | 0.545 | 0.545 | 0.564 | 0.558 |
| Study 3 – Minimum Wage (YouGov) | 0.510 | 0.510 | 0.510 | 0.521 | 0.549 |

## Opinion MAE (↓)

| Study | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 0.535 | 2.831 | 2.831 | 0.030 | 0.026 |
| Study 2 – Minimum Wage (MTurk) | 0.671 | 0.853 | 0.902 | 0.091 | 0.090 |
| Study 3 – Minimum Wage (YouGov) | 0.697 | 0.722 | 0.892 | 0.089 | 0.083 |

Notes

- KNN/XGB opinion metrics reflect training on next-video representations (`opinion_from_next`).
- GPT-4o next-video accuracies and per-study opinion CSVs are sourced from their report artefacts.
- GRPO/GRAIL metrics are read from `models/<family>/` caches when available.

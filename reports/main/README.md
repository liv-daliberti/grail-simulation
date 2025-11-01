# Portfolio Comparison (Main)

This report compares grail, grpo, gpt4o, knn, and xgb across the next-video and opinion tasks. KNN and XGB opinion metrics come from their `opinion_from_next` runs.

## Next-Video Eligible Accuracy (↑)

| Study | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 0.261 | 0.963 | 0.980 | 0.763 | — |
| Study 2 – Minimum Wage (MTurk) | 0.286 | 0.478 | 0.477 | 0.355 | — |
| Study 3 – Minimum Wage (YouGov) | 0.251 | 0.405 | 0.504 | 0.320 | — |

## Opinion Directional Accuracy (↑)

| Study | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 0.704 | 0.704 | 0.704 | 0.704 | — |
| Study 2 – Minimum Wage (MTurk) | 0.545 | 0.545 | 0.545 | 0.564 | — |
| Study 3 – Minimum Wage (YouGov) | 0.510 | 0.510 | 0.510 | 0.521 | — |

## Opinion MAE (↓)

| Study | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 0.535 | 0.554 | nan | 0.030 | — |
| Study 2 – Minimum Wage (MTurk) | 0.671 | 0.853 | 0.902 | 0.091 | — |
| Study 3 – Minimum Wage (YouGov) | 0.697 | 0.722 | 0.892 | 0.089 | — |

Notes

- KNN/XGB opinion metrics reflect training on next-video representations (`opinion_from_next`).
- GPT-4o next-video accuracies and per-study opinion CSVs are sourced from their report artefacts.
- GRPO/GRAIL metrics are read from `models/<family>/` caches when available.

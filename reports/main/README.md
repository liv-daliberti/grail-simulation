# Portfolio Comparison (Main)

This report compares grail, grpo, gpt4o, knn, and xgb across the next-video and opinion tasks. KNN and XGB opinion metrics come from their `opinion_from_next` runs.

## Next-Video Eligible Accuracy (↑)

| Study | N | Random | Most-Freq. | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 548 | 0.326 | 0.540 | 0.261 | 0.962 | 0.964 | 0.763 | 0.874 |
| Study 2 – Minimum Wage (MTurk) | 671 | 0.256 | 0.450 | 0.286 | 0.394 | 0.419 | 0.355 | 0.329 |
| Study 3 – Minimum Wage (YouGov) | 1,200 | 0.256 | 0.450 | 0.251 | 0.447 | 0.494 | 0.320 | 0.359 |

## Opinion Directional Accuracy (↑)

| Study | N | Random (1/3) | No-change | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 162 | 0.333 | 0.074 | 0.704 | 0.704 | 0.704 | 0.704 | 0.759 |
| Study 2 – Minimum Wage (MTurk) | 165 | 0.333 | 0.061 | 0.545 | 0.545 | 0.545 | 0.564 | 0.558 |
| Study 3 – Minimum Wage (YouGov) | 257 | 0.333 | 0.058 | 0.510 | 0.510 | 0.510 | 0.521 | 0.549 |

## Opinion MAE (↓)

| Study | N | Global Mean | Using Before | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 162 | 0.259 | 0.037 | 0.535 | 2.831 | 2.831 | 0.030 | 0.026 |
| Study 2 – Minimum Wage (MTurk) | 165 | 0.240 | 0.096 | 0.671 | 0.853 | 0.902 | 0.091 | 0.090 |
| Study 3 – Minimum Wage (YouGov) | 257 | 0.216 | 0.084 | 0.697 | 0.722 | 0.892 | 0.089 | 0.083 |

Notes

- KNN/XGB opinion metrics reflect training on next-video representations (`opinion_from_next`).
- GPT-4o next-video accuracies and per-study opinion CSVs are sourced from their report artefacts.
- GRPO/GRAIL metrics are read from `models/<family>/` caches when available.
- Baselines: For next-video, 'Random' is the expected accuracy of uniformly picking among the slate; 'Most-Freq.' always chooses the most common gold index in the split. For opinion direction, 'Random (1/3)' assumes equal probability of up/none/down; 'No-change' predicts the pre-study opinion. For opinion MAE, 'Global Mean' predicts the dataset mean of the post-study index; 'Using Before' predicts the pre-study index (when available).
- Column 'N' reports the number of eligible evaluation examples per study.

# Portfolio Comparison (Main)

This report compares grail, grpo, gpt4o, knn, and xgb across the next-video and opinion tasks. KNN and XGB opinion metrics come from their `opinion_from_next` runs.

## Next-Video Eligible Accuracy (↑)

| Study | N | Random | Most-Freq. | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 548 | 0.326 [0.288, 0.366] | 0.540 [0.498, 0.581] | 0.261 [0.226, 0.299] | **0.962 [0.942, 0.975]** | **0.964 [0.944, 0.976]** | **0.763 [0.725, 0.796]** | **0.874 [0.844, 0.899]** |
| Study 2 – Minimum Wage (MTurk) | 671 | 0.256 [0.224, 0.290] | 0.450 [0.413, 0.488] | 0.286 [0.253, 0.321] | 0.357 [0.322, 0.394] | 0.388 [0.352, 0.426] | 0.355 [0.319, 0.392] | 0.329 [0.295, 0.366] |
| Study 3 – Minimum Wage (YouGov) | 1,200 | 0.256 [0.232, 0.281] | 0.450 [0.422, 0.478] | 0.251 [0.227, 0.276] | 0.440 [0.412, 0.468] | 0.485 [0.457, 0.513] | 0.320 [0.294, 0.347] | 0.359 [0.333, 0.387] |

## Opinion Directional Accuracy (↑)

| Study | N | Random (1/3) | No-change | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 162 | 0.333 [0.265, 0.409] | 0.074 [0.043, 0.125] | **0.704 [0.629, 0.769]** | **0.562 [0.485, 0.636]** | **0.562 [0.485, 0.636]** | **0.704 [0.629, 0.769]** | **0.759 [0.688, 0.819]** |
| Study 2 – Minimum Wage (MTurk) | 165 | 0.333 [0.266, 0.408] | 0.061 [0.033, 0.108] | **0.545 [0.469, 0.620]** | **0.545 [0.469, 0.620]** | **0.545 [0.469, 0.620]** | **0.564 [0.487, 0.637]** | **0.558 [0.481, 0.631]** |
| Study 3 – Minimum Wage (YouGov) | 257 | 0.333 [0.279, 0.393] | 0.058 [0.036, 0.094] | **0.510 [0.449, 0.570]** | **0.510 [0.449, 0.570]** | **0.510 [0.449, 0.570]** | **0.521 [0.460, 0.582]** | **0.549 [0.488, 0.608]** |

## Opinion MAE (↓)

| Study | N | Global Mean | Using Before | GPT-4o | GRPO | GRAIL | KNN | XGB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Study 1 – Gun Control (MTurk) | 162 | 0.259 | 0.037 | 0.535 | 0.290 | 0.290 | 0.030 | 0.026 |
| Study 2 – Minimum Wage (MTurk) | 165 | 0.240 | 0.096 | 0.671 | 0.853 | 0.902 | 0.091 | 0.090 |
| Study 3 – Minimum Wage (YouGov) | 257 | 0.216 | 0.084 | 0.697 | 0.722 | 0.892 | 0.089 | 0.083 |

Notes

- KNN/XGB opinion metrics reflect training on next-video representations (`opinion_from_next`).
- GPT-4o next-video accuracies and per-study opinion CSVs are sourced from their report artefacts.
- GRPO/GRAIL metrics are read from `models/<family>/` caches when available.
- Baselines: For next-video, 'Random' is the expected accuracy of uniformly picking among the slate; 'Most-Freq.' always chooses the most common gold index in the split. For opinion direction, 'Random (1/3)' assumes equal probability of up/none/down; 'No-change' predicts the pre-study opinion. For opinion MAE, 'Global Mean' predicts the dataset mean of the post-study index; 'Using Before' predicts the pre-study index (when available).
- Column 'N' reports the number of eligible evaluation examples per study.
- Accuracy entries include 95% Wilson CIs in brackets. Bold indicates the model's lower CI exceeds the baseline's upper CI (Most‑Freq. for next‑video; No‑change for opinion).

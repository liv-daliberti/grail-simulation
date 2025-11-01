# GPT-4o Opinion Shift

- **Selected configuration:** `temp0_tok500_tp1` (temperature=0.00, top_p=1.00, max_tokens=500)
- **Participants evaluated:** 584
- **Overall metrics:** MAE=0.645, RMSE=0.704, Direction accuracy=0.574

| Study | Issue | Participants | Eligible | MAE (after) | RMSE (after) | Direction ↑ | No-change ↑ | Δ Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | gun control | 162 | 162 | 0.535 | 0.607 | 0.704 | 0.074 | 0.630 |
| Study 2 – Minimum Wage (MTurk) | minimum wage | 165 | 165 | 0.671 | 0.726 | 0.545 | 0.061 | 0.485 |
| Study 3 – Minimum Wage (YouGov) | minimum wage | 257 | 257 | 0.697 | 0.744 | 0.510 | 0.058 | 0.451 |

`opinion_metrics.csv` summarises per-study metrics.

## Artefacts

- `study1` metrics: `models/gpt-4o/opinion/temp0_tok500_tp1/study1/metrics.json` (predictions: `models/gpt-4o/opinion/temp0_tok500_tp1/study1/predictions.jsonl`, QA log: `logs/gpt/opinion/temp0_tok500_tp1/study1/qa.log`)
- `study2` metrics: `models/gpt-4o/opinion/temp0_tok500_tp1/study2/metrics.json` (predictions: `models/gpt-4o/opinion/temp0_tok500_tp1/study2/predictions.jsonl`, QA log: `logs/gpt/opinion/temp0_tok500_tp1/study2/qa.log`)
- `study3` metrics: `models/gpt-4o/opinion/temp0_tok500_tp1/study3/metrics.json` (predictions: `models/gpt-4o/opinion/temp0_tok500_tp1/study3/predictions.jsonl`, QA log: `logs/gpt/opinion/temp0_tok500_tp1/study3/qa.log`)

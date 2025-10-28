# Additional Text Features


This report tracks the supplementary text columns appended to the prompt builder output during training and evaluation.

## Next-Video Pipeline

### Sweep Configurations

| Study | Configuration | Extra text fields |
| --- | --- | --- |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text` |

### Final Evaluations

| Study | Issue | Extra text fields |
| --- | --- | --- |
| Study 2 – Minimum Wage (MTurk) | minimum_wage | `viewer_profile`, `state_text` |

## Opinion Regression

### Sweep Configurations

No opinion sweep metrics were provided.

### Final Evaluations

Opinion final metrics were not generated.

## Summary

- Default extra text fields: `viewer_profile`, `state_text`
- Additional fields observed: none (defaults only).

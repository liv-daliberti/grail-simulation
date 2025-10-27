# Additional Text Features


This report tracks the supplementary text columns appended to the prompt builder output during training and evaluation.

## Next-Video Pipeline

### Sweep Configurations

| Study | Configuration | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 3 – Minimum Wage (YouGov) | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 1 – Gun Control (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | `viewer_profile`, `state_text` |
| Study 3 – Minimum Wage (YouGov) | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | `viewer_profile`, `state_text` |
| Study 1 – Gun Control (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 3 – Minimum Wage (YouGov) | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 1 – Gun Control (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text` |
| Study 1 – Gun Control (MTurk) | tfidf_lr0p03_depth3_estim300_sub0p75_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p03_depth3_estim300_sub0p75_col0p8_l20p5_l10 | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p03_depth3_estim300_sub0p75_col0p8_l21_l10 | `viewer_profile`, `state_text` |

### Final Evaluations

| Study | Issue | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control (MTurk) | gun_control | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | minimum_wage | `viewer_profile`, `state_text` |
| Study 3 – Minimum Wage (YouGov) | minimum_wage | `viewer_profile`, `state_text` |

## Opinion Regression

### Sweep Configurations

No opinion sweep metrics were provided.

### Final Evaluations

Opinion final metrics were not generated.

## Summary

- Default extra text fields: `viewer_profile`, `state_text`
- Additional fields observed: none (defaults only).

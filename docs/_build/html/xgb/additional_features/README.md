# Additional Text Features


This report tracks the supplementary text columns appended to the prompt builder output during training and evaluation.

## Next-Video Pipeline

### Sweep Configurations

| Study | Configuration | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control | tfidf_lr0p1_depth4_estim100_sub0p8_col0p8_l21_l10 | `viewer_profile`, `state_text` |

### Final Evaluations

| Study | Issue | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control | gun_control | `viewer_profile`, `state_text` |

## Opinion Regression

### Sweep Configurations

| Study | Configuration | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control | tfidf_lr0p1_depth4_estim100_sub0p8_col0p8_l21_l10 | `viewer_profile`, `state_text` |

### Final Evaluations

| Study | Extra text fields |
| --- | --- |
| Study 1 – Gun Control | `viewer_profile`, `state_text` |

## Summary

- Default extra text fields: `viewer_profile`, `state_text`
- Additional fields observed: none (defaults only).

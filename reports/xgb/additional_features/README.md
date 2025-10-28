# Additional Text Features


This report tracks the supplementary text columns appended to the prompt builder output during training and evaluation.

## Next-Video Pipeline

### Sweep Configurations

| Study | Configuration | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control (MTurk) | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text`, `pid1`, `pid2`, `ideo1`, `ideo2`, `pol_interest`, `religpew`, `educ`, `employ`, `child18`, `inputstate`, `freq_youtube`, `youtube_time`, `newsint`, `q31`, `participant_study`, `slate_source`, `minwage_text_w2`, `minwage_text_w1`, `mw_support_w2`, `mw_support_w1`, `minwage15_w2`, `minwage15_w1`, `mw_index_w2`, `mw_index_w1`, `gun_importance`, `gun_index`, `gun_enthusiasm`, `gun_identity` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text`, `pid1`, `pid2`, `ideo1`, `ideo2`, `pol_interest`, `religpew`, `educ`, `employ`, `child18`, `inputstate`, `freq_youtube`, `youtube_time`, `newsint`, `q31`, `participant_study`, `slate_source`, `minwage_text_w2`, `minwage_text_w1`, `mw_support_w2`, `mw_support_w1`, `minwage15_w2`, `minwage15_w1`, `mw_index_w2`, `mw_index_w1`, `gun_importance`, `gun_index`, `gun_enthusiasm`, `gun_identity` |
| Study 3 – Minimum Wage (YouGov) | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text`, `pid1`, `pid2`, `ideo1`, `ideo2`, `pol_interest`, `religpew`, `educ`, `employ`, `child18`, `inputstate`, `freq_youtube`, `youtube_time`, `newsint`, `q31`, `participant_study`, `slate_source`, `minwage_text_w2`, `minwage_text_w1`, `mw_support_w2`, `mw_support_w1`, `minwage15_w2`, `minwage15_w1`, `mw_index_w2`, `mw_index_w1`, `gun_importance`, `gun_index`, `gun_enthusiasm`, `gun_identity` |

### Final Evaluations

| Study | Issue | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control (MTurk) | gun_control | `viewer_profile`, `state_text`, `pid1`, `pid2`, `ideo1`, `ideo2`, `pol_interest`, `religpew`, `educ`, `employ`, `child18`, `inputstate`, `freq_youtube`, `youtube_time`, `newsint`, `q31`, `participant_study`, `slate_source`, `minwage_text_w2`, `minwage_text_w1`, `mw_support_w2`, `mw_support_w1`, `minwage15_w2`, `minwage15_w1`, `mw_index_w2`, `mw_index_w1`, `gun_importance`, `gun_index`, `gun_enthusiasm`, `gun_identity` |
| Study 2 – Minimum Wage (MTurk) | minimum_wage | `viewer_profile`, `state_text`, `pid1`, `pid2`, `ideo1`, `ideo2`, `pol_interest`, `religpew`, `educ`, `employ`, `child18`, `inputstate`, `freq_youtube`, `youtube_time`, `newsint`, `q31`, `participant_study`, `slate_source`, `minwage_text_w2`, `minwage_text_w1`, `mw_support_w2`, `mw_support_w1`, `minwage15_w2`, `minwage15_w1`, `mw_index_w2`, `mw_index_w1`, `gun_importance`, `gun_index`, `gun_enthusiasm`, `gun_identity` |
| Study 3 – Minimum Wage (YouGov) | minimum_wage | `viewer_profile`, `state_text`, `pid1`, `pid2`, `ideo1`, `ideo2`, `pol_interest`, `religpew`, `educ`, `employ`, `child18`, `inputstate`, `freq_youtube`, `youtube_time`, `newsint`, `q31`, `participant_study`, `slate_source`, `minwage_text_w2`, `minwage_text_w1`, `mw_support_w2`, `mw_support_w1`, `minwage15_w2`, `minwage15_w1`, `mw_index_w2`, `mw_index_w1`, `gun_importance`, `gun_index`, `gun_enthusiasm`, `gun_identity` |

## Opinion Regression

### Sweep Configurations

| Study | Configuration | Extra text fields |
| --- | --- | --- |
| Study 1 – Gun Control (MTurk) | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text` |
| Study 3 – Minimum Wage (YouGov) | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | `viewer_profile`, `state_text` |

### Final Evaluations

| Study | Extra text fields |
| --- | --- |
| Study 1 – Gun Control (MTurk) | `viewer_profile`, `state_text` |
| Study 2 – Minimum Wage (MTurk) | `viewer_profile`, `state_text` |
| Study 3 – Minimum Wage (YouGov) | `viewer_profile`, `state_text` |

## Summary

- Default extra text fields: `viewer_profile`, `state_text`
- Additional fields observed: `child18`, `educ`, `employ`, `freq_youtube`, `gun_enthusiasm`, `gun_identity`, `gun_importance`, `gun_index`, `ideo1`, `ideo2`, `inputstate`, `minwage15_w1`, `minwage15_w2`, `minwage_text_w1`, `minwage_text_w2`, `mw_index_w1`, `mw_index_w2`, `mw_support_w1`, `mw_support_w2`, `newsint`, `participant_study`, `pid1`, `pid2`, `pol_interest`, `q31`, `religpew`, `slate_source`, `youtube_time`

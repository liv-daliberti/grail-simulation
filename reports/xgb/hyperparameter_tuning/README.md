# Hyper-parameter Tuning

This summary lists the top-performing configurations uncovered during the hyper-parameter sweeps.
- Next-video tables highlight up to 10 configurations per study ranked by validation accuracy.
- Eligible-only accuracy is shown for comparison next to overall accuracy.
- Opinion regression tables highlight up to 10 configurations per study ranked by MAE relative to the baseline.
- Rows in bold mark the configuration promoted to the final evaluation.

## Next-Video Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.786 | 0.786 | 0.789 | 431/546 | 0.996 | 0.495 | 548 |

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.219 | 0.219 | 0.219 | 147/671 | 1.000 | 0.027 | 671 |

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.786 | 0.000 | 0.789 | 0.000 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.219 | 0.000 | 0.219 | 0.000 | 671 |

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.786 (coverage 0.789) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.219 (coverage 0.219) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×2 |
| Learning rate | 0.05 ×2 |
| Max depth | 4 ×2 |
| Estimators | 300 ×2 |
| Subsample | 0.9 ×2 |
| Column subsample | 0.8 ×2 |
| L2 regularisation | 1 ×2 |
| L1 regularisation | 0 ×2 |

## Opinion Regression Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.772 | 0.074 | +0.698 | — | 162 | 0.035 | +0.002 | 0.061 | 0.955 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.861 | 0.061 | +0.800 | — | 165 | 0.033 | +0.063 | 0.045 | 0.974 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.782 | 0.058 | +0.724 | — | 257 | 0.048 | +0.036 | 0.070 | 0.927 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.040 across 584 participants.
- Weighted baseline MAE 0.074 (+0.035 vs. final).
- Weighted directional accuracy 0.801 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.738 vs. final).
- Weighted RMSE (change) 0.061 across 584 participants.
- Weighted baseline RMSE (change) 0.107 (+0.046 vs. final).
- Weighted calibration ECE 0.016 across 584 participants.
- Weighted KL divergence 0.187 across 584 participants.
- Weighted baseline KL divergence 18.435 (+18.249 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.063).
- Lowest MAE: Study 2 – Minimum Wage (MTurk) (0.033); Highest MAE: Study 3 – Minimum Wage (YouGov) (0.048).
- Highest directional accuracy: Study 2 – Minimum Wage (MTurk) (0.861).
- Lowest directional accuracy: Study 1 – Gun Control (MTurk) (0.772).
- Largest directional-accuracy gain: Study 2 – Minimum Wage (MTurk) (+0.800).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (+0.093).
- Lowest RMSE(change): Study 2 – Minimum Wage (MTurk) (0.045); Highest: Study 3 – Minimum Wage (YouGov) (0.070).
- Lowest calibration ECE: Study 2 – Minimum Wage (MTurk) (0.007); Highest: Study 1 – Gun Control (MTurk) (0.025).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+21.271).
- Lowest KL divergence: Study 2 – Minimum Wage (MTurk) (0.058); Highest: Study 3 – Minimum Wage (YouGov) (0.277).

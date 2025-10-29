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
| **tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10** | 0.803 | 0.803 | 0.802 | 438/546 | 0.996 | 0.513 | 548 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.803 | 0.803 | 0.802 | 438/546 | 0.996 | 0.513 | 548 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.792 | 0.792 | 0.791 | 432/546 | 0.996 | 0.495 | 548 |
| tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.790 | 0.790 | 0.789 | 431/546 | 0.996 | 0.495 | 548 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.785 | 0.785 | 0.784 | 428/546 | 0.996 | 0.492 | 548 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.783 | 0.783 | 0.782 | 427/546 | 0.996 | 0.492 | 548 |

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10** | 0.240 | 0.240 | 0.240 | 161/671 | 1.000 | 0.034 | 671 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.240 | 0.240 | 0.240 | 161/671 | 1.000 | 0.034 | 671 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.219 | 0.219 | 0.219 | 147/671 | 1.000 | 0.027 | 671 |
| tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.219 | 0.219 | 0.219 | 147/671 | 1.000 | 0.027 | 671 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.215 | 0.215 | 0.215 | 144/671 | 1.000 | 0.027 | 671 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.215 | 0.215 | 0.215 | 144/671 | 1.000 | 0.027 | 671 |

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10** | 0.250 | 0.250 | 0.250 | 300/1,200 | 1.000 | 0.022 | 1,200 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.250 | 0.250 | 0.250 | 300/1,200 | 1.000 | 0.022 | 1,200 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.229 | 0.229 | 0.229 | 275/1,200 | 1.000 | 0.020 | 1,200 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.229 | 0.229 | 0.229 | 275/1,200 | 1.000 | 0.020 | 1,200 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.220 | 0.220 | 0.220 | 264/1,200 | 1.000 | 0.020 | 1,200 |
| tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.220 | 0.220 | 0.220 | 264/1,200 | 1.000 | 0.020 | 1,200 |

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10** | 0.803 | 0.000 | 0.802 | 0.000 | 548 |
| 2 | tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.803 | 0.000 | 0.802 | 0.000 | 548 |
| 3 | tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.792 | 0.011 | 0.791 | 0.011 | 548 |
| 4 | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.790 | 0.013 | 0.789 | 0.013 | 548 |
| 5 | tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.785 | 0.018 | 0.784 | 0.018 | 548 |
*Showing top 5 of 6 configurations.*

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10** | 0.240 | 0.000 | 0.240 | 0.000 | 671 |
| 2 | tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.240 | 0.000 | 0.240 | 0.000 | 671 |
| 3 | tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.219 | 0.021 | 0.219 | 0.021 | 671 |
| 4 | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.219 | 0.021 | 0.219 | 0.021 | 671 |
| 5 | tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.215 | 0.025 | 0.215 | 0.025 | 671 |
*Showing top 5 of 6 configurations.*

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10** | 0.250 | 0.000 | 0.250 | 0.000 | 1,200 |
| 2 | tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.250 | 0.000 | 0.250 | 0.000 | 1,200 |
| 3 | tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.229 | 0.021 | 0.229 | 0.021 | 1,200 |
| 4 | tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10 | 0.229 | 0.021 | 0.229 | 0.021 | 1,200 |
| 5 | tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.220 | 0.030 | 0.220 | 0.030 | 1,200 |
*Showing top 5 of 6 configurations.*

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.803 (coverage 0.802) using vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer tfidf --xgb_learning_rate 0.1 --xgb_max_depth 3 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.240 (coverage 0.240) using vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer tfidf --xgb_learning_rate 0.1 --xgb_max_depth 3 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 3 – Minimum Wage (YouGov) (issue Minimum Wage)**: accuracy 0.250 (coverage 0.250) using vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study3 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer tfidf --xgb_learning_rate 0.1 --xgb_max_depth 3 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×3 |
| Learning rate | 0.1 ×3 |
| Max depth | 3 ×3 |
| Estimators | 300 ×3 |
| Subsample | 0.9 ×3 |
| Column subsample | 0.8 ×3 |
| L2 regularisation | 1 ×3 |
| L1 regularisation | 0 ×3 |

## Opinion Regression Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.852 | 0.074 | +0.778 | — | 162 | 0.012 | +0.025 | 0.021 | 0.994 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.864 | 0.074 | +0.790 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.864 | 0.074 | +0.790 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.858 | 0.074 | +0.784 | — | 162 | 0.013 | +0.024 | 0.022 | 0.994 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.852 | 0.074 | +0.778 | — | 162 | 0.013 | +0.024 | 0.022 | 0.994 |
| tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.864 | 0.074 | +0.790 | — | 162 | 0.013 | +0.024 | 0.023 | 0.994 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.806 | 0.061 | +0.745 | — | 165 | 0.039 | +0.057 | 0.052 | 0.964 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.800 | 0.061 | +0.739 | — | 165 | 0.039 | +0.057 | 0.053 | 0.963 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.800 | 0.061 | +0.739 | — | 165 | 0.039 | +0.057 | 0.053 | 0.964 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.800 | 0.061 | +0.739 | — | 165 | 0.040 | +0.056 | 0.054 | 0.962 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.806 | 0.061 | +0.745 | — | 165 | 0.040 | +0.056 | 0.054 | 0.962 |
| tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.794 | 0.061 | +0.733 | — | 165 | 0.041 | +0.055 | 0.055 | 0.960 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.743 | 0.058 | +0.685 | — | 257 | 0.049 | +0.035 | 0.068 | 0.933 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.747 | 0.058 | +0.689 | — | 257 | 0.049 | +0.035 | 0.069 | 0.929 |
| tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.774 | 0.058 | +0.716 | — | 257 | 0.049 | +0.035 | 0.069 | 0.930 |
| tfidf_lr0p1_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.743 | 0.058 | +0.685 | — | 257 | 0.049 | +0.035 | 0.068 | 0.931 |
| tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.747 | 0.058 | +0.689 | — | 257 | 0.049 | +0.035 | 0.068 | 0.931 |
| tfidf_lr0p05_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.751 | 0.058 | +0.693 | — | 257 | 0.050 | +0.034 | 0.070 | 0.928 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.03 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.036 across 584 participants.
- Weighted baseline MAE 0.074 (+0.039 vs. final).
- Weighted directional accuracy 0.791 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.728 vs. final).
- Weighted RMSE (change) 0.051 across 584 participants.
- Weighted baseline RMSE (change) 0.107 (+0.056 vs. final).
- Weighted calibration ECE 0.010 across 584 participants.
- Weighted KL divergence 0.586 across 584 participants.
- Weighted baseline KL divergence 18.435 (+17.849 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.057).
- Lowest MAE: Study 1 – Gun Control (MTurk) (0.012); Highest MAE: Study 3 – Minimum Wage (YouGov) (0.049).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (0.852).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (0.743).
- Largest directional-accuracy gain: Study 1 – Gun Control (MTurk) (+0.778).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (+0.086).
- Lowest RMSE(change): Study 1 – Gun Control (MTurk) (0.021); Highest: Study 3 – Minimum Wage (YouGov) (0.068).
- Lowest calibration ECE: Study 1 – Gun Control (MTurk) (0.005); Highest: Study 2 – Minimum Wage (MTurk) (0.014).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+20.780).
- Lowest KL divergence: Study 3 – Minimum Wage (YouGov) (0.384); Highest: Study 2 – Minimum Wage (MTurk) (0.823).

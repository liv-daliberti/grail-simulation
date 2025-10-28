# Hyper-parameter Tuning

This summary lists the top-performing configurations uncovered during the hyper-parameter sweeps.
- Next-video tables highlight up to 10 configurations per study ranked by validation accuracy.
- Opinion regression tables highlight up to 10 configurations per study ranked by MAE relative to the baseline.
- Rows in bold mark the configuration promoted to the final evaluation.

## Next-Video Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10** | 0.987 | 0.991 | 541/546 | 0.996 | 0.936 | 548 |
| tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.930 | 548 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l20p5_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.940 | 548 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.940 | 548 |
| tfidf_lr0p03_depth3_estim400_sub0p9_col0p8_l20p5_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.940 | 548 |
| tfidf_lr0p03_depth3_estim400_sub0p9_col0p8_l21_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.940 | 548 |
| tfidf_lr0p03_depth4_estim200_sub0p9_col0p8_l20p5_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.936 | 548 |
| tfidf_lr0p03_depth4_estim200_sub0p9_col0p8_l21_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.930 | 548 |
| tfidf_lr0p03_depth4_estim300_sub0p75_col0p8_l20p5_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.933 | 548 |
| tfidf_lr0p03_depth4_estim300_sub0p75_col0p8_l21_l10 | 0.987 | 0.991 | 541/546 | 0.996 | 0.932 | 548 |
*Showing top 10 of 72 configurations.*

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.994 | 0.994 | 667/671 | 1.000 | 0.963 | 671 |
| tfidf_lr0p05_depth3_estim400_sub0p75_col0p8_l21_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.963 | 671 |
| tfidf_lr0p05_depth4_estim200_sub0p9_col0p8_l20p5_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.967 | 671 |
| tfidf_lr0p05_depth4_estim300_sub0p75_col0p8_l20p5_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.963 | 671 |
| tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l20p5_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.967 | 671 |
| tfidf_lr0p05_depth4_estim400_sub0p75_col0p8_l21_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.963 | 671 |
| tfidf_lr0p05_depth4_estim400_sub0p9_col0p8_l20p5_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.967 | 671 |
| tfidf_lr0p1_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.966 | 671 |
| tfidf_lr0p1_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.966 | 671 |
| tfidf_lr0p1_depth3_estim400_sub0p9_col0p8_l21_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.966 | 671 |
*Showing top 10 of 72 configurations.*

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.985 | 1,200 |
| tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.982 | 1,200 |
| tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.987 | 1,200 |
| tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.985 | 1,200 |
| tfidf_lr0p03_depth3_estim300_sub0p75_col0p8_l20p5_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.988 | 1,200 |
| tfidf_lr0p03_depth3_estim300_sub0p75_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.988 | 1,200 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l20p5_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.991 | 1,200 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.990 | 1,200 |
| tfidf_lr0p03_depth3_estim400_sub0p75_col0p8_l20p5_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.988 | 1,200 |
| tfidf_lr0p03_depth3_estim400_sub0p75_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.988 | 1,200 |
*Showing top 10 of 72 configurations.*

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10** | 0.987 | 0.000 | 0.991 | 0.000 | 548 |
| 2 | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.987 | 0.000 | 0.991 | 0.000 | 548 |
| 3 | tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l20p5_l10 | 0.987 | 0.000 | 0.991 | 0.000 | 548 |
| 4 | tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10 | 0.987 | 0.000 | 0.991 | 0.000 | 548 |
| 5 | tfidf_lr0p03_depth3_estim400_sub0p9_col0p8_l20p5_l10 | 0.987 | 0.000 | 0.991 | 0.000 | 548 |
*Showing top 5 of 72 configurations.*

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.994 | 0.000 | 0.994 | 0.000 | 671 |
| 2 | tfidf_lr0p05_depth3_estim400_sub0p75_col0p8_l21_l10 | 0.994 | 0.000 | 0.994 | 0.000 | 671 |
| 3 | tfidf_lr0p05_depth4_estim200_sub0p9_col0p8_l20p5_l10 | 0.994 | 0.000 | 0.994 | 0.000 | 671 |
| 4 | tfidf_lr0p05_depth4_estim300_sub0p75_col0p8_l20p5_l10 | 0.994 | 0.000 | 0.994 | 0.000 | 671 |
| 5 | tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l20p5_l10 | 0.994 | 0.000 | 0.994 | 0.000 | 671 |
*Showing top 5 of 72 configurations.*

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 2 | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 3 | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 4 | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 5 | tfidf_lr0p03_depth3_estim300_sub0p75_col0p8_l20p5_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
*Showing top 5 of 72 configurations.*

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.987 (coverage 0.991) using vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.9, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 200 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.994 (coverage 0.994) using vectorizer=tfidf, lr=0.05, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 3 --xgb_n_estimators 200 --xgb_subsample 0.75 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 3 – Minimum Wage (YouGov) (issue Minimum Wage)**: accuracy 0.998 (coverage 0.998) using vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study3 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 200 --xgb_subsample 0.75 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×3 |
| Learning rate | 0.03 ×2, 0.05 ×1 |
| Max depth | 3 ×3 |
| Estimators | 200 ×3 |
| Subsample | 0.75 ×2, 0.9 ×1 |
| Column subsample | 0.8 ×3 |
| L2 regularisation | 0.5 ×3 |
| L1 regularisation | 0 ×3 |

## Opinion Regression Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l20p5_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.648 | 0.074 | +0.574 | — | 162 | 0.082 | -0.045 | 0.114 | 0.843 |
| tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.648 | 0.074 | +0.574 | — | 162 | 0.082 | -0.045 | 0.114 | 0.843 |
| tfidf_lr0p05_depth4_estim300_sub0p75_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.654 | 0.074 | +0.580 | — | 162 | 0.082 | -0.045 | 0.114 | 0.841 |
| tfidf_lr0p05_depth4_estim200_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=200, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.648 | 0.074 | +0.574 | — | 162 | 0.083 | -0.046 | 0.114 | 0.843 |
| tfidf_lr0p03_depth4_estim400_sub0p9_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=400, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.654 | 0.074 | +0.580 | — | 162 | 0.083 | -0.046 | 0.114 | 0.843 |
| tfidf_lr0p03_depth4_estim400_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=400, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.654 | 0.074 | +0.580 | — | 162 | 0.083 | -0.046 | 0.114 | 0.841 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.642 | 0.074 | +0.568 | — | 162 | 0.083 | -0.046 | 0.113 | 0.844 |
| tfidf_lr0p03_depth4_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.648 | 0.074 | +0.574 | — | 162 | 0.083 | -0.046 | 0.114 | 0.842 |
| tfidf_lr0p05_depth4_estim200_sub0p75_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.654 | 0.074 | +0.580 | — | 162 | 0.083 | -0.046 | 0.115 | 0.840 |
| tfidf_lr0p05_depth4_estim200_sub0p9_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=200, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.642 | 0.074 | +0.568 | — | 162 | 0.083 | -0.046 | 0.114 | 0.842 |
*Showing top 10 of 44 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 0.5 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim200_sub0p75_col0p8_l20p5_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.788 | 0.061 | +0.727 | — | 165 | 0.042 | +0.054 | 0.053 | 0.963 |
| tfidf_lr0p05_depth4_estim300_sub0p75_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.794 | 0.061 | +0.733 | — | 165 | 0.043 | +0.054 | 0.054 | 0.963 |
| tfidf_lr0p05_depth4_estim200_sub0p75_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=200, subsample=0.75, colsample=0.8, λ=1, α=0 | 0.764 | 0.061 | +0.703 | — | 165 | 0.043 | +0.054 | 0.054 | 0.963 |
| tfidf_lr0p05_depth4_estim300_sub0p75_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.75, colsample=0.8, λ=1, α=0 | 0.776 | 0.061 | +0.715 | — | 165 | 0.043 | +0.053 | 0.054 | 0.962 |
| tfidf_lr0p03_depth3_estim400_sub0p75_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=400, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.776 | 0.061 | +0.715 | — | 165 | 0.043 | +0.053 | 0.054 | 0.962 |
| tfidf_lr0p03_depth4_estim300_sub0p75_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=300, subsample=0.75, colsample=0.8, λ=1, α=0 | 0.752 | 0.061 | +0.691 | — | 165 | 0.043 | +0.053 | 0.054 | 0.962 |
| tfidf_lr0p05_depth3_estim200_sub0p75_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=1, α=0 | 0.770 | 0.061 | +0.709 | — | 165 | 0.043 | +0.053 | 0.054 | 0.962 |
| tfidf_lr0p05_depth3_estim200_sub0p75_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.758 | 0.061 | +0.697 | — | 165 | 0.043 | +0.053 | 0.054 | 0.962 |
| tfidf_lr0p03_depth4_estim400_sub0p75_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=400, subsample=0.75, colsample=0.8, λ=1, α=0 | 0.764 | 0.061 | +0.703 | — | 165 | 0.043 | +0.053 | 0.054 | 0.962 |
| tfidf_lr0p03_depth4_estim400_sub0p75_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=400, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.764 | 0.061 | +0.703 | — | 165 | 0.043 | +0.053 | 0.054 | 0.962 |
*Showing top 10 of 43 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 200 --subsample-grid 0.75 --colsample-grid 0.8 --reg-lambda-grid 0.5 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10**<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.700 | 0.058 | +0.642 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.712 | 0.058 | +0.654 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.696 | 0.058 | +0.638 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p05_depth3_estim200_sub0p9_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=200, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.712 | 0.058 | +0.654 | — | 257 | 0.058 | +0.026 | 0.080 | 0.907 |
| tfidf_lr0p05_depth3_estim200_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=200, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.700 | 0.058 | +0.642 | — | 257 | 0.058 | +0.026 | 0.079 | 0.907 |
| tfidf_lr0p03_depth3_estim300_sub0p9_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.708 | 0.058 | +0.650 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0 | 0.693 | 0.058 | +0.634 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=1, α=0 | 0.696 | 0.058 | +0.638 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p03_depth3_estim300_sub0p75_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=300, subsample=0.75, colsample=0.8, λ=1, α=0 | 0.704 | 0.058 | +0.646 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p03_depth3_estim400_sub0p9_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=400, subsample=0.9, colsample=0.8, λ=0.5, α=0 | 0.720 | 0.058 | +0.661 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
*Showing top 10 of 43 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.03 --max-depth-grid 3 --n-estimators-grid 200 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 0.5 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.060 across 584 participants.
- Weighted baseline MAE 0.074 (+0.014 vs. final).
- Weighted directional accuracy 0.711 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.647 vs. final).
- Weighted RMSE (change) 0.082 across 584 participants.
- Weighted baseline RMSE (change) 0.107 (+0.025 vs. final).
- Weighted calibration ECE 0.037 across 584 participants.
- Weighted KL divergence 0.324 across 584 participants.
- Weighted baseline KL divergence 18.435 (+18.111 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.054).
- Lowest MAE: Study 2 – Minimum Wage (MTurk) (0.042); Highest MAE: Study 1 – Gun Control (MTurk) (0.082).
- Highest directional accuracy: Study 2 – Minimum Wage (MTurk) (0.788).
- Lowest directional accuracy: Study 1 – Gun Control (MTurk) (0.648).
- Largest directional-accuracy gain: Study 2 – Minimum Wage (MTurk) (+0.727).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (+0.085).
- Lowest RMSE(change): Study 2 – Minimum Wage (MTurk) (0.053); Highest: Study 1 – Gun Control (MTurk) (0.114).
- Lowest calibration ECE: Study 2 – Minimum Wage (MTurk) (0.018); Highest: Study 1 – Gun Control (MTurk) (0.078).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+20.980).
- Lowest KL divergence: Study 2 – Minimum Wage (MTurk) (0.209); Highest: Study 1 – Gun Control (MTurk) (0.466).

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
| **tfidf_lr0p03_depth3_estim250_sub0p8_col0p6_l20p5_l10** | 0.973 | 0.976 | 533/546 | 0.996 | 0.921 | 548 |
| tfidf_lr0p03_depth3_estim250_sub0p8_col0p6_l20p5_l10p1 | 0.973 | 0.976 | 533/546 | 0.996 | 0.920 | 548 |
| tfidf_lr0p03_depth3_estim250_sub0p8_col0p8_l20p5_l10 | 0.973 | 0.976 | 533/546 | 0.996 | 0.924 | 548 |
| tfidf_lr0p03_depth3_estim250_sub0p8_col1_l20p5_l10 | 0.973 | 0.976 | 533/546 | 0.996 | 0.924 | 548 |
| tfidf_lr0p03_depth3_estim250_sub0p8_col1_l20p5_l10p1 | 0.973 | 0.976 | 533/546 | 0.996 | 0.924 | 548 |
| tfidf_lr0p03_depth3_estim250_sub0p9_col1_l20p5_l10 | 0.973 | 0.976 | 533/546 | 0.996 | 0.929 | 548 |
| tfidf_lr0p03_depth3_estim250_sub0p9_col1_l20p5_l10p1 | 0.973 | 0.976 | 533/546 | 0.996 | 0.928 | 548 |
| tfidf_lr0p03_depth3_estim350_sub0p8_col0p6_l20p5_l10 | 0.973 | 0.976 | 533/546 | 0.996 | 0.924 | 548 |
| tfidf_lr0p03_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.973 | 0.976 | 533/546 | 0.996 | 0.926 | 548 |
| tfidf_lr0p03_depth3_estim350_sub0p9_col1_l20p5_l10 | 0.973 | 0.976 | 533/546 | 0.996 | 0.931 | 548 |
*Showing top 10 of 1094 configurations.*

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim150_sub0p7_col1_l20p5_l10p1** | 0.382 | 0.382 | 256/671 | 1.000 | 0.435 | 671 |
| tfidf_lr0p03_depth3_estim150_sub0p7_col1_l20p5_l10p5 | 0.382 | 0.382 | 256/671 | 1.000 | 0.429 | 671 |
| tfidf_lr0p03_depth3_estim150_sub1_col0p6_l21_l11 | 0.380 | 0.380 | 255/671 | 1.000 | 0.415 | 671 |
| tfidf_lr0p03_depth3_estim150_sub1_col1_l21p5_l11 | 0.380 | 0.380 | 255/671 | 1.000 | 0.411 | 671 |
| tfidf_lr0p03_depth3_estim150_sub0p7_col0p6_l20p5_l10 | 0.377 | 0.377 | 253/671 | 1.000 | 0.433 | 671 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col0p8_l21p5_l10p1 | 0.377 | 0.377 | 253/671 | 1.000 | 0.422 | 671 |
| tfidf_lr0p03_depth3_estim150_sub1_col0p6_l21p5_l10p1 | 0.377 | 0.377 | 253/671 | 1.000 | 0.420 | 671 |
| tfidf_lr0p03_depth3_estim150_sub0p7_col0p6_l20p5_l10p1 | 0.376 | 0.376 | 252/671 | 1.000 | 0.432 | 671 |
| tfidf_lr0p03_depth3_estim150_sub0p7_col0p6_l21_l11 | 0.376 | 0.376 | 252/671 | 1.000 | 0.409 | 671 |
| tfidf_lr0p03_depth3_estim150_sub0p7_col0p8_l20p5_l10p1 | 0.376 | 0.376 | 252/671 | 1.000 | 0.432 | 671 |
*Showing top 10 of 1104 configurations.*

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim150_sub0p9_col0p6_l20p5_l10p1** | 0.443 | 0.443 | 531/1,200 | 1.000 | 0.442 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col0p8_l20p5_l10 | 0.442 | 0.442 | 530/1,200 | 1.000 | 0.445 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p8_col1_l21_l10p5 | 0.441 | 0.441 | 529/1,200 | 1.000 | 0.437 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col0p6_l21_l10 | 0.441 | 0.441 | 529/1,200 | 1.000 | 0.437 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p8_col0p6_l21_l11 | 0.440 | 0.440 | 528/1,200 | 1.000 | 0.429 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p8_col0p8_l20p5_l11 | 0.440 | 0.440 | 528/1,200 | 1.000 | 0.437 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p8_col0p6_l21_l10 | 0.439 | 0.439 | 527/1,200 | 1.000 | 0.437 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col0p6_l20p5_l10p5 | 0.439 | 0.439 | 527/1,200 | 1.000 | 0.440 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col0p6_l20p5_l11 | 0.439 | 0.439 | 527/1,200 | 1.000 | 0.435 | 1,200 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col0p8_l20p5_l11 | 0.439 | 0.439 | 527/1,200 | 1.000 | 0.436 | 1,200 |
*Showing top 10 of 1030 configurations.*

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim250_sub0p8_col0p6_l20p5_l10** | 0.973 | 0.000 | 0.976 | 0.000 | 548 |
| 2 | tfidf_lr0p03_depth3_estim250_sub0p8_col0p6_l20p5_l10p1 | 0.973 | 0.000 | 0.976 | 0.000 | 548 |
| 3 | tfidf_lr0p03_depth3_estim250_sub0p8_col0p8_l20p5_l10 | 0.973 | 0.000 | 0.976 | 0.000 | 548 |
| 4 | tfidf_lr0p03_depth3_estim250_sub0p8_col1_l20p5_l10 | 0.973 | 0.000 | 0.976 | 0.000 | 548 |
| 5 | tfidf_lr0p03_depth3_estim250_sub0p8_col1_l20p5_l10p1 | 0.973 | 0.000 | 0.976 | 0.000 | 548 |
*Showing top 5 of 1094 configurations.*

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim150_sub0p7_col1_l20p5_l10p1** | 0.382 | 0.000 | 0.382 | 0.000 | 671 |
| 2 | tfidf_lr0p03_depth3_estim150_sub0p7_col1_l20p5_l10p5 | 0.382 | 0.000 | 0.382 | 0.000 | 671 |
| 3 | tfidf_lr0p03_depth3_estim150_sub1_col0p6_l21_l11 | 0.380 | 0.001 | 0.380 | 0.001 | 671 |
| 4 | tfidf_lr0p03_depth3_estim150_sub1_col1_l21p5_l11 | 0.380 | 0.001 | 0.380 | 0.001 | 671 |
| 5 | tfidf_lr0p03_depth3_estim150_sub0p7_col0p6_l20p5_l10 | 0.377 | 0.004 | 0.377 | 0.004 | 671 |
*Showing top 5 of 1104 configurations.*

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim150_sub0p9_col0p6_l20p5_l10p1** | 0.443 | 0.000 | 0.443 | 0.000 | 1,200 |
| 2 | tfidf_lr0p03_depth3_estim150_sub0p9_col0p8_l20p5_l10 | 0.442 | 0.001 | 0.442 | 0.001 | 1,200 |
| 3 | tfidf_lr0p03_depth3_estim150_sub0p8_col1_l21_l10p5 | 0.441 | 0.002 | 0.441 | 0.002 | 1,200 |
| 4 | tfidf_lr0p03_depth3_estim150_sub0p9_col0p6_l21_l10 | 0.441 | 0.002 | 0.441 | 0.002 | 1,200 |
| 5 | tfidf_lr0p03_depth3_estim150_sub0p8_col0p6_l21_l11 | 0.440 | 0.003 | 0.440 | 0.003 | 1,200 |
*Showing top 5 of 1030 configurations.*

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.973 (coverage 0.976) using vectorizer=tfidf, lr=0.03, depth=3, estimators=250, subsample=0.8, colsample=0.6, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 250 --xgb_subsample 0.8 --xgb_colsample_bytree 0.6 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.382 (coverage 0.382) using vectorizer=tfidf, lr=0.03, depth=3, estimators=150, subsample=0.7, colsample=1, λ=0.5, α=0.1. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 150 --xgb_subsample 0.7 --xgb_colsample_bytree 1.0 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.1 --out_dir '<run_dir>'`
- **Study 3 – Minimum Wage (YouGov) (issue Minimum Wage)**: accuracy 0.443 (coverage 0.443) using vectorizer=tfidf, lr=0.03, depth=3, estimators=150, subsample=0.9, colsample=0.6, λ=0.5, α=0.1. Δ accuracy vs. runner-up +0.001; Δ coverage +0.001.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study3 --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 150 --xgb_subsample 0.9 --xgb_colsample_bytree 0.6 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.1 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×3 |
| Learning rate | 0.03 ×3 |
| Max depth | 3 ×3 |
| Estimators | 150 ×2, 250 ×1 |
| Subsample | 0.8 ×1, 0.7 ×1, 0.9 ×1 |
| Column subsample | 0.6 ×2, 1 ×1 |
| L2 regularisation | 0.5 ×3 |
| L1 regularisation | 0.1 ×2, 0 ×1 |

## Opinion Regression Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p03_depth4_estim150_sub0p8_col0p8_l20p5_l10**<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.8, colsample=0.8, λ=0.5, α=0 | 0.082 | -0.045 | 0.111 | 0.851 | 162 |
| tfidf_lr0p03_depth4_estim150_sub0p8_col0p8_l21p5_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.8, colsample=0.8, λ=1.5, α=0 | 0.083 | -0.046 | 0.111 | 0.849 | 162 |
| tfidf_lr0p03_depth4_estim150_sub1_col0p8_l21p5_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=1, colsample=0.8, λ=1.5, α=0.1 | 0.083 | -0.046 | 0.112 | 0.847 | 162 |
| tfidf_lr0p03_depth4_estim150_sub0p8_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.8, colsample=0.8, λ=1, α=0 | 0.083 | -0.046 | 0.111 | 0.849 | 162 |
| tfidf_lr0p03_depth4_estim150_sub1_col0p8_l21_l10p5<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=1, colsample=0.8, λ=1, α=0.5 | 0.083 | -0.046 | 0.112 | 0.847 | 162 |
| tfidf_lr0p03_depth4_estim150_sub1_col0p8_l20p5_l10p5<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=1, colsample=0.8, λ=0.5, α=0.5 | 0.083 | -0.046 | 0.113 | 0.846 | 162 |
| tfidf_lr0p03_depth4_estim150_sub0p8_col1_l20p5_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.8, colsample=1, λ=0.5, α=0 | 0.083 | -0.046 | 0.112 | 0.848 | 162 |
| tfidf_lr0p03_depth4_estim150_sub0p8_col1_l21_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.8, colsample=1, λ=1, α=0 | 0.083 | -0.046 | 0.112 | 0.847 | 162 |
| tfidf_lr0p03_depth4_estim150_sub1_col0p8_l20p5_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=1, colsample=0.8, λ=0.5, α=0.1 | 0.083 | -0.046 | 0.113 | 0.844 | 162 |
| tfidf_lr0p03_depth4_estim150_sub0p9_col0p6_l20p5_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.9, colsample=0.6, λ=0.5, α=0.1 | 0.083 | -0.046 | 0.112 | 0.848 | 162 |
*Showing top 10 of 660 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.03 --max-depth-grid 4 --n-estimators-grid 150 --subsample-grid 0.8 --colsample-grid 0.8 --reg-lambda-grid 0.5 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim450_sub0p8_col0p8_l21_l11**<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=450, subsample=0.8, colsample=0.8, λ=1, α=1 | 0.041 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim450_sub0p8_col0p8_l20p5_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=450, subsample=0.8, colsample=0.8, λ=0.5, α=1 | 0.041 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim450_sub0p7_col0p8_l21_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=450, subsample=0.7, colsample=0.8, λ=1, α=1 | 0.042 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim350_sub0p7_col0p8_l21_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=350, subsample=0.7, colsample=0.8, λ=1, α=1 | 0.042 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim350_sub0p7_col0p8_l20p5_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=350, subsample=0.7, colsample=0.8, λ=0.5, α=1 | 0.042 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim450_sub0p7_col0p8_l20p5_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=450, subsample=0.7, colsample=0.8, λ=0.5, α=1 | 0.042 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim350_sub0p7_col0p8_l21p5_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=350, subsample=0.7, colsample=0.8, λ=1.5, α=1 | 0.042 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim450_sub1_col0p8_l20p5_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=450, subsample=1, colsample=0.8, λ=0.5, α=1 | 0.042 | +0.055 | 0.054 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim350_sub0p8_col0p8_l21_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=1 | 0.042 | +0.055 | 0.053 | 0.963 | 165 |
| tfidf_lr0p03_depth3_estim450_sub0p8_col1_l21p5_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=450, subsample=0.8, colsample=1, λ=1.5, α=1 | 0.042 | +0.055 | 0.053 | 0.963 | 165 |
*Showing top 10 of 660 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.03 --max-depth-grid 3 --n-estimators-grid 450 --subsample-grid 0.8 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 1 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | Participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p03_depth4_estim150_sub0p9_col0p6_l21_l10p1**<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.9, colsample=0.6, λ=1, α=0.1 | 0.057 | +0.027 | 0.077 | 0.912 | 257 |
| tfidf_lr0p03_depth4_estim150_sub0p9_col0p6_l20p5_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.9, colsample=0.6, λ=0.5, α=0.1 | 0.057 | +0.027 | 0.077 | 0.913 | 257 |
| tfidf_lr0p03_depth4_estim150_sub0p9_col0p6_l21p5_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.9, colsample=0.6, λ=1.5, α=0.1 | 0.057 | +0.027 | 0.077 | 0.912 | 257 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col1_l20p5_l10p1<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=150, subsample=0.9, colsample=1, λ=0.5, α=0.1 | 0.057 | +0.027 | 0.079 | 0.909 | 257 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col1_l21p5_l10<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=150, subsample=0.9, colsample=1, λ=1.5, α=0 | 0.057 | +0.027 | 0.079 | 0.909 | 257 |
| tfidf_lr0p03_depth3_estim250_sub0p9_col1_l21_l11<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=250, subsample=0.9, colsample=1, λ=1, α=1 | 0.057 | +0.027 | 0.080 | 0.907 | 257 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col1_l21_l10p1<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=150, subsample=0.9, colsample=1, λ=1, α=0.1 | 0.057 | +0.027 | 0.079 | 0.909 | 257 |
| tfidf_lr0p03_depth4_estim150_sub0p9_col0p6_l21p5_l10<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=150, subsample=0.9, colsample=0.6, λ=1.5, α=0 | 0.057 | +0.027 | 0.077 | 0.912 | 257 |
| tfidf_lr0p03_depth3_estim250_sub0p9_col1_l21_l10p1<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=250, subsample=0.9, colsample=1, λ=1, α=0.1 | 0.057 | +0.027 | 0.078 | 0.910 | 257 |
| tfidf_lr0p03_depth3_estim150_sub0p9_col1_l21p5_l10p1<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=150, subsample=0.9, colsample=1, λ=1.5, α=0.1 | 0.057 | +0.027 | 0.079 | 0.909 | 257 |
*Showing top 10 of 660 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.03 --max-depth-grid 4 --n-estimators-grid 150 --subsample-grid 0.9 --colsample-grid 0.6 --reg-lambda-grid 1 --reg-alpha-grid 0.1 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.060 across 584 participants.
- Weighted baseline MAE 0.074 (+0.015 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.055).
- Lowest MAE: Study 2 – Minimum Wage (MTurk) (0.041); Highest MAE: Study 1 – Gun Control (MTurk) (0.082).

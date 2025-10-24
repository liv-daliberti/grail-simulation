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
| **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10** | 0.985 | 0.989 | 540/546 | 0.996 | 0.935 | 548 |
| tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.985 | 0.989 | 540/546 | 0.996 | 0.934 | 548 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.985 | 0.989 | 540/546 | 0.996 | 0.935 | 548 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.984 | 0.987 | 539/546 | 0.996 | 0.935 | 548 |

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10** | 0.996 | 0.996 | 668/671 | 1.000 | 0.963 | 671 |
| tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.964 | 671 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.963 | 671 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.994 | 0.994 | 667/671 | 1.000 | 0.963 | 671 |

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10** | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.989 | 1,200 |
| tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.989 | 1,200 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.989 | 1,200 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.989 | 1,200 |

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10** | 0.985 | 0.000 | 0.989 | 0.000 | 548 |
| 2 | tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.985 | 0.000 | 0.989 | 0.000 | 548 |
| 3 | tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.985 | 0.000 | 0.989 | 0.000 | 548 |
| 4 | tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.984 | 0.002 | 0.987 | 0.002 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10** | 0.996 | 0.000 | 0.996 | 0.000 | 671 |
| 2 | tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.994 | 0.001 | 0.994 | 0.001 | 671 |
| 3 | tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.994 | 0.001 | 0.994 | 0.001 | 671 |
| 4 | tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.994 | 0.001 | 0.994 | 0.001 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10** | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 2 | tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 3 | tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 4 | tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.985 (coverage 0.989) using vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 3 --xgb_n_estimators 350 --xgb_subsample 0.8 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.996 (coverage 0.996) using vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=0. Δ accuracy vs. runner-up +0.001; Δ coverage +0.001.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 3 --xgb_n_estimators 350 --xgb_subsample 0.8 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 3 – Minimum Wage (YouGov) (issue Minimum Wage)**: accuracy 0.998 (coverage 0.998) using vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study3 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 3 --xgb_n_estimators 350 --xgb_subsample 0.8 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×3 |
| Learning rate | 0.05 ×3 |
| Max depth | 3 ×3 |
| Estimators | 350 ×3 |
| Subsample | 0.8 ×3 |
| Column subsample | 0.8 ×3 |
| L2 regularisation | 0.5 ×2, 1 ×1 |
| L1 regularisation | 0 ×3 |

## Opinion Regression Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=0 | 0.648 | 0.074 | +0.574 | — | 162 | 0.084 | -0.047 | 0.115 | 0.840 |
| tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0 | 0.654 | 0.074 | +0.580 | — | 162 | 0.085 | -0.047 | 0.117 | 0.834 |
| tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=0 | 0.654 | 0.074 | +0.580 | — | 162 | 0.085 | -0.048 | 0.118 | 0.832 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0 | 0.623 | 0.074 | +0.549 | — | 162 | 0.088 | -0.051 | 0.120 | 0.826 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.1 --max-depth-grid 3 --n-estimators-grid 350 --subsample-grid 0.8 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=0 | 0.770 | 0.061 | +0.709 | — | 165 | 0.044 | +0.053 | 0.055 | 0.961 |
| tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0 | 0.776 | 0.061 | +0.715 | — | 165 | 0.044 | +0.053 | 0.055 | 0.960 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0 | 0.776 | 0.061 | +0.715 | — | 165 | 0.045 | +0.051 | 0.057 | 0.958 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=0 | 0.782 | 0.061 | +0.721 | — | 165 | 0.046 | +0.050 | 0.058 | 0.957 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 3 --n-estimators-grid 350 --subsample-grid 0.8 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=0 | 0.696 | 0.058 | +0.638 | — | 257 | 0.058 | +0.026 | 0.080 | 0.906 |
| tfidf_lr0p05_depth3_estim350_sub0p8_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.05, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0 | 0.704 | 0.058 | +0.646 | — | 257 | 0.059 | +0.025 | 0.081 | 0.904 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l20p5_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=0.5, α=0 | 0.712 | 0.058 | +0.654 | — | 257 | 0.060 | +0.024 | 0.082 | 0.901 |
| tfidf_lr0p1_depth3_estim350_sub0p8_col0p8_l21_l10<br>vectorizer=tfidf, lr=0.1, depth=3, estimators=350, subsample=0.8, colsample=0.8, λ=1, α=0 | 0.700 | 0.058 | +0.642 | — | 257 | 0.061 | +0.023 | 0.081 | 0.902 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 3 --n-estimators-grid 350 --subsample-grid 0.8 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.061 across 584 participants.
- Weighted baseline MAE 0.074 (+0.013 vs. final).
- Weighted directional accuracy 0.704 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.640 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.053).
- Lowest MAE: Study 2 – Minimum Wage (MTurk) (0.044); Highest MAE: Study 1 – Gun Control (MTurk) (0.084).
- Highest directional accuracy: Study 2 – Minimum Wage (MTurk) (0.770).
- Lowest directional accuracy: Study 1 – Gun Control (MTurk) (0.648).
- Largest directional-accuracy gain: Study 2 – Minimum Wage (MTurk) (+0.709).

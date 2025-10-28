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
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.987 | 0.991 | 541/546 | 0.996 | 0.940 | 548 |

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.993 | 0.993 | 666/671 | 1.000 | 0.966 | 671 |

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.990 | 1,200 |

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.987 | 0.000 | 0.991 | 0.000 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.993 | 0.000 | 0.993 | 0.000 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.987 (coverage 0.991) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.993 (coverage 0.993) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 3 – Minimum Wage (YouGov) (issue Minimum Wage)**: accuracy 0.998 (coverage 0.998) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study3 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×3 |
| Learning rate | 0.05 ×3 |
| Max depth | 4 ×3 |
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
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.648 | 0.074 | +0.574 | — | 162 | 0.082 | -0.045 | 0.114 | 0.843 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.764 | 0.061 | +0.703 | — | 165 | 0.045 | +0.051 | 0.056 | 0.960 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.693 | 0.058 | +0.634 | — | 257 | 0.059 | +0.025 | 0.080 | 0.907 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.062 across 584 participants.
- Weighted baseline MAE 0.074 (+0.013 vs. final).
- Weighted directional accuracy 0.700 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.637 vs. final).
- Weighted RMSE (change) 0.082 across 584 participants.
- Weighted baseline RMSE (change) 0.107 (+0.025 vs. final).
- Weighted calibration ECE 0.037 across 584 participants.
- Weighted KL divergence 0.464 across 584 participants.
- Weighted baseline KL divergence 18.435 (+17.971 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.051).
- Lowest MAE: Study 2 – Minimum Wage (MTurk) (0.045); Highest MAE: Study 1 – Gun Control (MTurk) (0.082).
- Highest directional accuracy: Study 2 – Minimum Wage (MTurk) (0.764).
- Lowest directional accuracy: Study 1 – Gun Control (MTurk) (0.648).
- Largest directional-accuracy gain: Study 2 – Minimum Wage (MTurk) (+0.703).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (+0.082).
- Lowest RMSE(change): Study 2 – Minimum Wage (MTurk) (0.056); Highest: Study 1 – Gun Control (MTurk) (0.114).
- Lowest calibration ECE: Study 2 – Minimum Wage (MTurk) (0.016); Highest: Study 1 – Gun Control (MTurk) (0.076).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+20.912).
- Lowest KL divergence: Study 2 – Minimum Wage (MTurk) (0.179); Highest: Study 3 – Minimum Wage (YouGov) (0.603).

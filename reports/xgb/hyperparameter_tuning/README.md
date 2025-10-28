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

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.987 | 0.000 | 0.991 | 0.000 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.993 | 0.000 | 0.993 | 0.000 | 671 |

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.987 (coverage 0.991) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.993 (coverage 0.993) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

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
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0 | 0.648 | 0.074 | +0.574 | — | 162 | 0.082 | -0.045 | 0.114 | 0.843 |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.05 --max-depth-grid 4 --n-estimators-grid 300 --subsample-grid 0.9 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.082 across 162 participants.
- Weighted baseline MAE 0.037 (-0.045 vs. final).
- Weighted directional accuracy 0.648 across 162 participants.
- Weighted baseline accuracy 0.074 (+0.574 vs. final).
- Weighted RMSE (change) 0.114 across 162 participants.
- Weighted baseline RMSE (change) 0.046 (-0.068 vs. final).
- Weighted calibration ECE 0.076 across 162 participants.
- Weighted KL divergence 0.533 across 162 participants.
- Weighted baseline KL divergence 21.446 (+20.912 vs. final).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (-0.045).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (0.648).
- Largest directional-accuracy gain: Study 1 – Gun Control (MTurk) (+0.574).
- Largest RMSE(change) reduction: Study 1 – Gun Control (MTurk) (-0.068).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+20.912).

# Hyper-parameter Tuning

This summary lists the top-performing configurations uncovered during the hyper-parameter sweeps.
- Next-video tables highlight up to 10 configurations per study ranked by validation accuracy.
- Eligible-only accuracy is shown for comparison next to overall accuracy.
- Opinion regression tables highlight up to 10 configurations per study ranked by MAE relative to the baseline.
- Rows in bold mark the configuration promoted to the final evaluation.

## Next-Video Sweeps

### Study 1 – Gun Control

*Issue:* Gun Control

| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p1_depth4_estim100_sub0p8_col0p8_l21_l10** | — | — | — | — | — | — | — |

### Configuration Leaderboards

#### Study 1 – Gun Control

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p1_depth4_estim100_sub0p8_col0p8_l21_l10** | 0.000 | 0.000 | 0.000 | 0.000 | 0 |

### Selection Summary

- **Study 1 – Gun Control (issue Gun Control)**: accuracy 0.000 (coverage 0.000) using vectorizer=tfidf, lr=0.1, depth=4, estimators=100, subsample=0.8, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset data/cleaned_grail --issues gun_control --participant_studies study1 --text_vectorizer tfidf --xgb_learning_rate 0.1 --xgb_max_depth 4 --xgb_n_estimators 100 --xgb_subsample 0.8 --xgb_colsample_bytree 0.8 --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --xgb_tree_method hist --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×1 |
| Learning rate | 0.1 ×1 |
| Max depth | 4 ×1 |
| Estimators | 100 ×1 |
| Subsample | 0.8 ×1 |
| Column subsample | 0.8 ×1 |
| L2 regularisation | 1 ×1 |
| L1 regularisation | 0 ×1 |

## Opinion Regression Sweeps

### Study 1 – Gun Control

*Issue:* Gun Control

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p1_depth4_estim100_sub0p8_col0p8_l21_l10**<br>vectorizer=tfidf, lr=0.1, depth=4, estimators=100, subsample=0.8, colsample=0.8, λ=1, α=0 | — | — | — | — | — | — | — | — | — |
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.1 --max-depth-grid 4 --n-estimators-grid 100 --subsample-grid 0.8 --colsample-grid 0.8 --reg-lambda-grid 1 --reg-alpha-grid 0 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset data/cleaned_grail`

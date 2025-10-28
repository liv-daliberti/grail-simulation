# Hyper-parameter Tuning

This summary lists the top-performing configurations uncovered during the hyper-parameter sweeps.
- Next-video tables highlight up to 10 configurations per study ranked by validation accuracy.
- Opinion regression tables highlight up to 10 configurations per study ranked by MAE relative to the baseline.
- Rows in bold mark the configuration promoted to the final evaluation.

## Next-Video Sweeps

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.993 | 0.993 | 666/671 | 1.000 | 0.966 | 671 |

### Configuration Leaderboards

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p05_depth4_estim300_sub0p9_col0p8_l21_l10** | 0.993 | 0.000 | 0.993 | 0.000 | 671 |

### Selection Summary

- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.993 (coverage 0.993) using vectorizer=tfidf, lr=0.05, depth=4, estimators=300, subsample=0.9, colsample=0.8, λ=1, α=0.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.05 --xgb_max_depth 4 --xgb_n_estimators 300 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 1.0 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×1 |
| Learning rate | 0.05 ×1 |
| Max depth | 4 ×1 |
| Estimators | 300 ×1 |
| Subsample | 0.9 ×1 |
| Column subsample | 0.8 ×1 |
| L2 regularisation | 1 ×1 |
| L1 regularisation | 0 ×1 |

## Opinion Regression Sweeps

No opinion sweep runs were available when this report was generated.
Run the XGBoost pipeline with `--stage sweeps` or `--stage full` once artifacts are ready.

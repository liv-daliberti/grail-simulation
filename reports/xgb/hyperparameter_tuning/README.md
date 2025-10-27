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
| tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10 | 0.985 | 0.989 | 540/546 | 0.996 | 0.925 | 548 |
| tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.985 | 0.989 | 540/546 | 0.996 | 0.914 | 548 |

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.993 | 0.993 | 666/671 | 1.000 | 0.960 | 671 |
| tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | 0.993 | 0.993 | 666/671 | 1.000 | 0.965 | 671 |
| tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.993 | 0.993 | 666/671 | 1.000 | 0.962 | 671 |
| tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.991 | 0.991 | 665/671 | 1.000 | 0.955 | 671 |

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.985 | 1,200 |
| tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.982 | 1,200 |
| tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | 0.998 | 0.998 | 1,198/1,200 | 1.000 | 0.987 | 1,200 |

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10** | 0.987 | 0.000 | 0.991 | 0.000 | 548 |
| 2 | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.987 | 0.000 | 0.991 | 0.000 | 548 |
| 3 | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10 | 0.985 | 0.002 | 0.989 | 0.002 | 548 |
| 4 | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.985 | 0.002 | 0.989 | 0.002 | 548 |

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.993 | 0.000 | 0.993 | 0.000 | 671 |
| 2 | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | 0.993 | 0.000 | 0.993 | 0.000 | 671 |
| 3 | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l21_l10 | 0.993 | 0.000 | 0.993 | 0.000 | 671 |
| 4 | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.991 | 0.001 | 0.991 | 0.001 | 671 |

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l20p5_l10** | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 2 | tfidf_lr0p03_depth3_estim200_sub0p75_col0p8_l21_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |
| 3 | tfidf_lr0p03_depth3_estim200_sub0p9_col0p8_l20p5_l10 | 0.998 | 0.000 | 0.998 | 0.000 | 1,200 |

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.987 (coverage 0.991) using vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.9, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 200 --xgb_subsample 0.9 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.993 (coverage 0.993) using vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 200 --xgb_subsample 0.75 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`
- **Study 3 – Minimum Wage (YouGov) (issue Minimum Wage)**: accuracy 0.998 (coverage 0.998) using vectorizer=tfidf, lr=0.03, depth=3, estimators=200, subsample=0.75, colsample=0.8, λ=0.5, α=0. Δ accuracy vs. runner-up +0.000; Δ coverage +0.000.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study3 --extra_text_fields state_text,viewer_profile --text_vectorizer tfidf --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 200 --xgb_subsample 0.75 --xgb_colsample_bytree 0.8 --xgb_tree_method hist --xgb_reg_lambda 0.5 --xgb_reg_alpha 0.0 --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | tfidf ×3 |
| Learning rate | 0.03 ×3 |
| Max depth | 3 ×3 |
| Estimators | 200 ×3 |
| Subsample | 0.75 ×2, 0.9 ×1 |
| Column subsample | 0.8 ×3 |
| L2 regularisation | 0.5 ×3 |
| L1 regularisation | 0 ×3 |

## Opinion Regression Sweeps

No opinion sweep runs were available when this report was generated.
Run the XGBoost pipeline with `--stage sweeps` or `--stage full` once artifacts are ready.

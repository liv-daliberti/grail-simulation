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
| **w2v256_lr0p06_depth4_estim350_sub0p8_col0p7_l20p7_l10p1** | 0.874 | 0.874 | 0.874 | 477/546 | 0.996 | 0.610 | 548 |
| w2v256_lr0p03_depth4_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.867 | 0.867 | 0.866 | 473/546 | 0.996 | 0.582 | 548 |
| w2v256_lr0p06_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.865 | 0.865 | 0.864 | 472/546 | 0.996 | 0.620 | 548 |
| w2v256_lr0p06_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.865 | 0.865 | 0.864 | 472/546 | 0.996 | 0.600 | 548 |
| w2v256_lr0p06_depth4_estim350_sub0p9_col0p7_l20p7_l10p1 | 0.865 | 0.865 | 0.864 | 472/546 | 0.996 | 0.612 | 548 |
| w2v256_lr0p03_depth3_estim350_sub0p9_col0p7_l20p7_l10p1 | 0.861 | 0.861 | 0.861 | 470/546 | 0.996 | 0.590 | 548 |
| w2v256_lr0p06_depth3_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.861 | 0.861 | 0.861 | 470/546 | 0.996 | 0.608 | 548 |
| w2v256_lr0p06_depth3_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.861 | 0.861 | 0.861 | 470/546 | 0.996 | 0.604 | 548 |
| w2v256_lr0p03_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.859 | 0.859 | 0.859 | 469/546 | 0.996 | 0.588 | 548 |
| w2v256_lr0p03_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.859 | 0.859 | 0.859 | 469/546 | 0.996 | 0.547 | 548 |
*Showing top 10 of 47 configurations.*

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **w2v256_lr0p03_depth3_estim250_sub0p8_col0p7_l20p7_l10p1** | 0.329 | 0.329 | 0.329 | 221/671 | 1.000 | 0.360 | 671 |
| w2v256_lr0p06_depth4_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.328 | 0.328 | 0.328 | 220/671 | 1.000 | 0.506 | 671 |
| w2v256_lr0p06_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.322 | 0.322 | 0.322 | 216/671 | 1.000 | 0.480 | 671 |
| w2v256_lr0p06_depth4_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.319 | 0.319 | 0.319 | 214/671 | 1.000 | 0.479 | 671 |
| w2v256_lr0p03_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.316 | 0.316 | 0.316 | 212/671 | 1.000 | 0.403 | 671 |
| st_all-mpnet-base-v2_lr0p06_depth4_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.316 | 0.316 | 0.316 | 212/671 | 1.000 | 0.388 | 671 |
| st_all-mpnet-base-v2_lr0p06_depth4_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.316 | 0.316 | 0.316 | 212/671 | 1.000 | 0.416 | 671 |
| st_all-mpnet-base-v2_lr0p06_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.314 | 0.314 | 0.314 | 211/671 | 1.000 | 0.390 | 671 |
| w2v256_lr0p03_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.313 | 0.313 | 0.313 | 210/671 | 1.000 | 0.391 | 671 |
| w2v256_lr0p03_depth4_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.313 | 0.313 | 0.313 | 210/671 | 1.000 | 0.435 | 671 |
*Showing top 10 of 48 configurations.*

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Acc (eligible) ↑ | Coverage ↑ | Known hits / total | Known availability ↑ | Avg prob ↑ | Evaluated |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **st_all-mpnet-base-v2_lr0p06_depth4_estim350_sub0p9_col0p7_l20p7_l10p1** | 0.359 | 0.359 | 0.359 | 431/1,200 | 1.000 | 0.435 | 1,200 |
| st_all-mpnet-base-v2_lr0p06_depth3_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.358 | 0.358 | 0.358 | 430/1,200 | 1.000 | 0.359 | 1,200 |
| st_all-mpnet-base-v2_lr0p06_depth3_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.355 | 0.355 | 0.355 | 426/1,200 | 1.000 | 0.360 | 1,200 |
| st_all-mpnet-base-v2_lr0p06_depth4_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.355 | 0.355 | 0.355 | 426/1,200 | 1.000 | 0.403 | 1,200 |
| st_all-mpnet-base-v2_lr0p06_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.353 | 0.353 | 0.353 | 424/1,200 | 1.000 | 0.392 | 1,200 |
| st_all-mpnet-base-v2_lr0p03_depth3_estim350_sub0p9_col0p7_l20p7_l10p1 | 0.352 | 0.352 | 0.352 | 422/1,200 | 1.000 | 0.323 | 1,200 |
| st_all-mpnet-base-v2_lr0p06_depth3_estim350_sub0p9_col0p7_l20p7_l10p1 | 0.350 | 0.350 | 0.350 | 420/1,200 | 1.000 | 0.389 | 1,200 |
| st_all-mpnet-base-v2_lr0p03_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.349 | 0.349 | 0.349 | 419/1,200 | 1.000 | 0.322 | 1,200 |
| st_all-mpnet-base-v2_lr0p03_depth4_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.348 | 0.348 | 0.348 | 418/1,200 | 1.000 | 0.358 | 1,200 |
| st_all-mpnet-base-v2_lr0p03_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.344 | 0.344 | 0.344 | 413/1,200 | 1.000 | 0.325 | 1,200 |
*Showing top 10 of 48 configurations.*

### Configuration Leaderboards

#### Study 1 – Gun Control (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **w2v256_lr0p06_depth4_estim350_sub0p8_col0p7_l20p7_l10p1** | 0.874 | 0.000 | 0.874 | 0.000 | 548 |
| 2 | w2v256_lr0p03_depth4_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.867 | 0.007 | 0.866 | 0.007 | 548 |
| 3 | w2v256_lr0p06_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.865 | 0.009 | 0.864 | 0.009 | 548 |
| 4 | w2v256_lr0p06_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.865 | 0.009 | 0.864 | 0.009 | 548 |
| 5 | w2v256_lr0p06_depth4_estim350_sub0p9_col0p7_l20p7_l10p1 | 0.865 | 0.009 | 0.864 | 0.009 | 548 |
*Showing top 5 of 47 configurations.*

#### Study 2 – Minimum Wage (MTurk)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **w2v256_lr0p03_depth3_estim250_sub0p8_col0p7_l20p7_l10p1** | 0.329 | 0.000 | 0.329 | 0.000 | 671 |
| 2 | w2v256_lr0p06_depth4_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.328 | 0.001 | 0.328 | 0.001 | 671 |
| 3 | w2v256_lr0p06_depth4_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.322 | 0.007 | 0.322 | 0.007 | 671 |
| 4 | w2v256_lr0p06_depth4_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.319 | 0.010 | 0.319 | 0.010 | 671 |
| 5 | w2v256_lr0p03_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.316 | 0.013 | 0.316 | 0.013 | 671 |
*Showing top 5 of 48 configurations.*

#### Study 3 – Minimum Wage (YouGov)

| Rank | Config | Accuracy ↑ | Δ accuracy ↓ | Coverage ↑ | Δ coverage ↓ | Evaluated |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **st_all-mpnet-base-v2_lr0p06_depth4_estim350_sub0p9_col0p7_l20p7_l10p1** | 0.359 | 0.000 | 0.359 | 0.000 | 1,200 |
| 2 | st_all-mpnet-base-v2_lr0p06_depth3_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.358 | 0.001 | 0.358 | 0.001 | 1,200 |
| 3 | st_all-mpnet-base-v2_lr0p06_depth3_estim250_sub0p8_col0p7_l20p7_l10p1 | 0.355 | 0.004 | 0.355 | 0.004 | 1,200 |
| 4 | st_all-mpnet-base-v2_lr0p06_depth4_estim250_sub0p9_col0p7_l20p7_l10p1 | 0.355 | 0.004 | 0.355 | 0.004 | 1,200 |
| 5 | st_all-mpnet-base-v2_lr0p06_depth3_estim350_sub0p8_col0p7_l20p7_l10p1 | 0.353 | 0.006 | 0.353 | 0.006 | 1,200 |
*Showing top 5 of 48 configurations.*

### Selection Summary

- **Study 1 – Gun Control (MTurk) (issue Gun Control)**: accuracy 0.874 (coverage 0.874) using vectorizer=word2vec (w2v256), lr=0.06, depth=4, estimators=350, subsample=0.8, colsample=0.7, λ=0.7, α=0.1. Δ accuracy vs. runner-up +0.007; Δ coverage +0.007.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues gun_control --participant_studies study1 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer word2vec --xgb_learning_rate 0.06 --xgb_max_depth 4 --xgb_n_estimators 350 --xgb_subsample 0.8 --xgb_colsample_bytree 0.7 --xgb_tree_method hist --xgb_reg_lambda 0.7 --xgb_reg_alpha 0.1 --word2vec_size 256 --word2vec_window 5 --word2vec_min_count 2 --word2vec_epochs 10 --word2vec_workers 1 --out_dir '<run_dir>'`
- **Study 2 – Minimum Wage (MTurk) (issue Minimum Wage)**: accuracy 0.329 (coverage 0.329) using vectorizer=word2vec (w2v256), lr=0.03, depth=3, estimators=250, subsample=0.8, colsample=0.7, λ=0.7, α=0.1. Δ accuracy vs. runner-up +0.001; Δ coverage +0.001.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study2 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer word2vec --xgb_learning_rate 0.03 --xgb_max_depth 3 --xgb_n_estimators 250 --xgb_subsample 0.8 --xgb_colsample_bytree 0.7 --xgb_tree_method hist --xgb_reg_lambda 0.7 --xgb_reg_alpha 0.1 --word2vec_size 256 --word2vec_window 5 --word2vec_min_count 2 --word2vec_epochs 10 --word2vec_workers 1 --out_dir '<run_dir>'`
- **Study 3 – Minimum Wage (YouGov) (issue Minimum Wage)**: accuracy 0.359 (coverage 0.359) using vectorizer=sentence_transformer (st_all-mpnet-base-v2), lr=0.06, depth=4, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1. Δ accuracy vs. runner-up +0.001; Δ coverage +0.001.
  Command: `python -m xgb.cli --fit_model --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --issues minimum_wage --participant_studies study3 --extra_text_fields child18,educ,employ,freq_youtube,gun_enthusiasm,gun_identity,gun_importance,gun_index,ideo1,ideo2,inputstate,minwage15_w1,minwage15_w2,minwage_text_w1,minwage_text_w2,mw_index_w1,mw_index_w2,mw_support_w1,mw_support_w2,newsint,participant_study,pid1,pid2,pol_interest,q31,religpew,slate_source,state_text,viewer_profile,youtube_time --text_vectorizer sentence_transformer --xgb_learning_rate 0.06 --xgb_max_depth 4 --xgb_n_estimators 350 --xgb_subsample 0.9 --xgb_colsample_bytree 0.7 --xgb_tree_method hist --xgb_reg_lambda 0.7 --xgb_reg_alpha 0.1 --sentence_transformer_model sentence-transformers/all-mpnet-base-v2 --sentence_transformer_batch_size 32 --sentence_transformer_normalize --out_dir '<run_dir>'`

### Parameter Frequency Across Selected Configurations

| Parameter | Preferred values (count) |
| --- | --- |
| Vectorizer | word2vec ×2, sentence_transformer ×1 |
| Learning rate | 0.06 ×2, 0.03 ×1 |
| Max depth | 4 ×2, 3 ×1 |
| Estimators | 350 ×2, 250 ×1 |
| Subsample | 0.8 ×2, 0.9 ×1 |
| Column subsample | 0.7 ×3 |
| L2 regularisation | 0.7 ×3 |
| L1 regularisation | 0.1 ×3 |

## Opinion Regression Sweeps

### Study 1 – Gun Control (MTurk)

*Issue:* Gun Control

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p03_depth4_estim250_sub0p9_col0p7_l20p7_l10p1**<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.864 | 0.074 | +0.790 | — | 162 | 0.012 | +0.025 | 0.021 | 0.994 |
| tfidf_lr0p03_depth4_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.864 | 0.074 | +0.790 | — | 162 | 0.012 | +0.025 | 0.021 | 0.994 |
| tfidf_lr0p03_depth4_estim350_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=350, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.870 | 0.074 | +0.796 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p03_depth4_estim250_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=250, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.877 | 0.074 | +0.802 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p06_depth4_estim250_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.858 | 0.074 | +0.784 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p06_depth4_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.858 | 0.074 | +0.784 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p06_depth4_estim250_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=250, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.852 | 0.074 | +0.778 | — | 162 | 0.012 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p06_depth3_estim250_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.864 | 0.074 | +0.790 | — | 162 | 0.013 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p06_depth4_estim350_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=350, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.852 | 0.074 | +0.778 | — | 162 | 0.013 | +0.025 | 0.022 | 0.994 |
| tfidf_lr0p06_depth3_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.864 | 0.074 | +0.790 | — | 162 | 0.013 | +0.025 | 0.022 | 0.994 |
*Showing top 10 of 48 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues gun_control --studies study1 --tree-method hist --learning-rate-grid 0.03 --max-depth-grid 4 --n-estimators-grid 250 --subsample-grid 0.9 --colsample-grid 0.7 --reg-lambda-grid 0.7 --reg-alpha-grid 0.1 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 2 – Minimum Wage (MTurk)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p06_depth4_estim350_sub0p9_col0p7_l20p7_l10p1**<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.836 | 0.061 | +0.776 | — | 165 | 0.036 | +0.060 | 0.049 | 0.969 |
| tfidf_lr0p06_depth4_estim350_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=350, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.830 | 0.061 | +0.770 | — | 165 | 0.036 | +0.060 | 0.049 | 0.968 |
| tfidf_lr0p06_depth4_estim250_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.824 | 0.061 | +0.764 | — | 165 | 0.037 | +0.059 | 0.049 | 0.968 |
| tfidf_lr0p06_depth4_estim250_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=4, estimators=250, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.812 | 0.061 | +0.752 | — | 165 | 0.037 | +0.059 | 0.050 | 0.967 |
| tfidf_lr0p06_depth3_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.818 | 0.061 | +0.758 | — | 165 | 0.037 | +0.059 | 0.050 | 0.968 |
| tfidf_lr0p06_depth3_estim250_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.818 | 0.061 | +0.758 | — | 165 | 0.037 | +0.059 | 0.050 | 0.967 |
| tfidf_lr0p03_depth4_estim350_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=350, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.818 | 0.061 | +0.758 | — | 165 | 0.037 | +0.059 | 0.050 | 0.967 |
| tfidf_lr0p03_depth4_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.806 | 0.061 | +0.745 | — | 165 | 0.038 | +0.059 | 0.051 | 0.966 |
| tfidf_lr0p03_depth3_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.812 | 0.061 | +0.752 | — | 165 | 0.038 | +0.058 | 0.051 | 0.966 |
| tfidf_lr0p03_depth4_estim250_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=250, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.818 | 0.061 | +0.758 | — | 165 | 0.038 | +0.058 | 0.051 | 0.966 |
*Showing top 10 of 48 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study2 --tree-method hist --learning-rate-grid 0.06 --max-depth-grid 4 --n-estimators-grid 350 --subsample-grid 0.9 --colsample-grid 0.7 --reg-lambda-grid 0.7 --reg-alpha-grid 0.1 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Study 3 – Minimum Wage (YouGov)

*Issue:* Minimum Wage

| Config | Accuracy ↑ | Baseline ↑ | Δ vs baseline ↑ | Best k | Eligible | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **tfidf_lr0p06_depth3_estim250_sub0p9_col0p7_l20p7_l10p1**<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.739 | 0.058 | +0.681 | — | 257 | 0.049 | +0.036 | 0.069 | 0.930 |
| tfidf_lr0p06_depth3_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.747 | 0.058 | +0.689 | — | 257 | 0.049 | +0.035 | 0.069 | 0.931 |
| tfidf_lr0p03_depth3_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.755 | 0.058 | +0.696 | — | 257 | 0.049 | +0.035 | 0.069 | 0.931 |
| tfidf_lr0p06_depth3_estim350_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=350, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.747 | 0.058 | +0.689 | — | 257 | 0.049 | +0.035 | 0.068 | 0.931 |
| tfidf_lr0p06_depth3_estim250_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.06, depth=3, estimators=250, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.735 | 0.058 | +0.677 | — | 257 | 0.049 | +0.035 | 0.068 | 0.931 |
| tfidf_lr0p03_depth4_estim350_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=350, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.759 | 0.058 | +0.700 | — | 257 | 0.049 | +0.035 | 0.068 | 0.933 |
| tfidf_lr0p03_depth3_estim250_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=3, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.743 | 0.058 | +0.685 | — | 257 | 0.049 | +0.035 | 0.069 | 0.930 |
| tfidf_lr0p03_depth4_estim350_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=350, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.743 | 0.058 | +0.685 | — | 257 | 0.049 | +0.035 | 0.067 | 0.933 |
| tfidf_lr0p03_depth4_estim250_sub0p9_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=250, subsample=0.9, colsample=0.7, λ=0.7, α=0.1 | 0.755 | 0.058 | +0.696 | — | 257 | 0.049 | +0.035 | 0.068 | 0.932 |
| tfidf_lr0p03_depth4_estim250_sub0p8_col0p7_l20p7_l10p1<br>vectorizer=tfidf, lr=0.03, depth=4, estimators=250, subsample=0.8, colsample=0.7, λ=0.7, α=0.1 | 0.747 | 0.058 | +0.689 | — | 257 | 0.049 | +0.035 | 0.068 | 0.932 |
*Showing top 10 of 48 configurations.*
  Command: `python -m xgb.pipeline --stage full --tasks opinion --issues minimum_wage --studies study3 --tree-method hist --learning-rate-grid 0.06 --max-depth-grid 3 --n-estimators-grid 250 --subsample-grid 0.9 --colsample-grid 0.7 --reg-lambda-grid 0.7 --reg-alpha-grid 0.1 --text-vectorizer-grid tfidf --out-dir '<models_dir>' --dataset /n/fs/similarity/grail-simulation/data/cleaned_grail --max-features 200000`

### Portfolio Summary

- Weighted MAE 0.035 across 584 participants.
- Weighted baseline MAE 0.074 (+0.040 vs. final).
- Weighted directional accuracy 0.801 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.738 vs. final).
- Weighted RMSE (change) 0.050 across 584 participants.
- Weighted baseline RMSE (change) 0.107 (+0.057 vs. final).
- Weighted calibration ECE 0.008 across 584 participants.
- Weighted KL divergence 0.940 across 584 participants.
- Weighted baseline KL divergence 18.435 (+17.495 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.060).
- Lowest MAE: Study 1 – Gun Control (MTurk) (0.012); Highest MAE: Study 3 – Minimum Wage (YouGov) (0.049).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (0.864).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (0.739).
- Largest directional-accuracy gain: Study 1 – Gun Control (MTurk) (+0.790).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (+0.089).
- Lowest RMSE(change): Study 1 – Gun Control (MTurk) (0.021); Highest: Study 3 – Minimum Wage (YouGov) (0.069).
- Lowest calibration ECE: Study 1 – Gun Control (MTurk) (0.004); Highest: Study 2 – Minimum Wage (MTurk) (0.013).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+19.888).
- Lowest KL divergence: Study 2 – Minimum Wage (MTurk) (0.572); Highest: Study 1 – Gun Control (MTurk) (1.557).

# XGBoost Opinion Regression

MAE / RMSE / R² / directional accuracy / MAE (change) / RMSE (change) / calibration slope & intercept / calibration ECE / KL divergence, all compared against a no-change baseline (pre-study opinion).

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics: MAE, RMSE, R², directional accuracy, MAE(change), RMSE(change), calibration slope & intercept, calibration ECE, and KL divergence.

| Study | Participants | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 0.660 | 0.074 | +0.586 | 0.084 | -0.047 | 0.114 | 0.841 | — | — | — | — | — | — | — | — | — | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 0.752 | 0.061 | +0.691 | 0.043 | +0.053 | 0.054 | 0.962 | — | — | — | — | — | — | — | — | — | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 0.696 | 0.058 | +0.638 | 0.058 | +0.026 | 0.080 | 0.907 | — | — | — | — | — | — | — | — | — | 0.084 |

## Cross-Study Diagnostics

### Weighted Summary Portfolio Summary

- Weighted MAE 0.061 across 584 participants.
- Weighted baseline MAE 0.074 (+0.014 vs. final).
- Weighted directional accuracy 0.702 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.639 vs. final).
- Largest MAE reduction: Study 2 – Minimum Wage (MTurk) (+0.053).
- Lowest MAE: Study 2 – Minimum Wage (MTurk) (0.043); Highest MAE: Study 1 – Gun Control (MTurk) (0.084).
- Highest directional accuracy: Study 2 – Minimum Wage (MTurk) (0.752).
- Lowest directional accuracy: Study 1 – Gun Control (MTurk) (0.660).
- Largest directional-accuracy gain: Study 2 – Minimum Wage (MTurk) (+0.691).

- Unweighted MAE 0.061 (σ 0.017, range 0.043 – 0.084).
- MAE delta mean 0.011 (σ 0.042, range -0.047 – 0.053).
- Directional accuracy mean 0.703 (σ 0.037, range 0.660 – 0.752).
- Accuracy delta mean 0.638 (σ 0.043, range 0.586 – 0.691).

## Observations

- Study 1 – Gun Control (MTurk): MAE 0.084 (Δ vs. baseline -0.047), RMSE(change) —, ECE —, KL —, R² 0.841.
- Study 2 – Minimum Wage (MTurk): MAE 0.043 (Δ vs. baseline +0.053), RMSE(change) —, ECE —, KL —, R² 0.962.
- Study 3 – Minimum Wage (YouGov): MAE 0.058 (Δ vs. baseline +0.026), RMSE(change) —, ECE —, KL —, R² 0.907.
- Average MAE reduction +0.011 across 3 studies.
- Mean R² 0.903.

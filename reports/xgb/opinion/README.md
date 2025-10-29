# XGBoost Opinion Regression

This summary captures the opinion-regression baselines trained with XGBoost for the selected participant studies.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics track MAE, RMSE, R², directional accuracy, MAE(change), RMSE(change), calibration slope/intercept, calibration ECE, and KL divergence versus the no-change baseline.
- Δ columns capture improvements relative to that baseline when available.

| Study | Participants | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 0.864 | 0.074 | +0.790 | 0.012 | +0.025 | 0.021 | 0.994 | 0.012 | 0.021 | +0.024 | 1.084 | -0.003 | 0.004 | — | 1.557 | +19.888 | 0.037 |

## Cross-Study Diagnostics

### Weighted Summary Portfolio Summary

- Weighted MAE 0.012 across 162 participants.
- Weighted baseline MAE 0.037 (+0.025 vs. final).
- Weighted directional accuracy 0.864 across 162 participants.
- Weighted baseline accuracy 0.074 (+0.790 vs. final).
- Weighted RMSE (change) 0.021 across 162 participants.
- Weighted baseline RMSE (change) 0.046 (+0.024 vs. final).
- Weighted calibration ECE 0.004 across 162 participants.
- Weighted KL divergence 1.557 across 162 participants.
- Weighted baseline KL divergence 21.446 (+19.888 vs. final).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (+0.025).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (0.864).
- Largest directional-accuracy gain: Study 1 – Gun Control (MTurk) (+0.790).
- Largest RMSE(change) reduction: Study 1 – Gun Control (MTurk) (+0.024).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+19.888).

- Unweighted MAE 0.012 (σ 0.000, range 0.012 – 0.012).
- MAE delta mean 0.025 (σ 0.000, range 0.025 – 0.025).
- Directional accuracy mean 0.864 (σ 0.000, range 0.864 – 0.864).
- Accuracy delta mean 0.790 (σ 0.000, range 0.790 – 0.790).
- RMSE(change) mean 0.021 (σ 0.000, range 0.021 – 0.021).
- RMSE(change) delta mean 0.024 (σ 0.000, range 0.024 – 0.024).
- Calibration ECE mean 0.004 (σ 0.000, range 0.004 – 0.004).
- KL divergence mean 1.557 (σ 0.000, range 1.557 – 1.557).
- KL divergence delta mean 19.888 (σ 0.000, range 19.888 – 19.888).

## Observations

- Study 1 – Gun Control (MTurk): MAE 0.012 (Δ vs. baseline +0.025), RMSE(change) 0.021, ECE 0.004, KL 1.557, R² 0.994.
- Average MAE reduction +0.025 across 1 studies.
- Mean R² 0.994.
- Mean RMSE(change) 0.021.
- Mean RMSE(change) delta 0.024.
- Mean calibration ECE 0.004.
- Mean KL divergence 1.557.
- Mean KL divergence delta 19.888.

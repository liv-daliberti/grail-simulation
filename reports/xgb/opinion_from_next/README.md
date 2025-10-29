# XGBoost Opinion Regression (Next-Video Config)

This section reuses the selected next-video configuration to estimate post-study opinion change.

- Dataset: `/n/fs/similarity/grail-simulation/data/cleaned_grail`
- Split: validation
- Metrics track MAE, RMSE, R², directional accuracy, MAE(change), RMSE(change), calibration slope/intercept, calibration ECE, and KL divergence versus the no-change baseline.
- Δ columns capture improvements relative to that baseline when available.

| Study | Participants | Accuracy ↑ | Baseline ↑ | Δ Accuracy ↑ | MAE ↓ | Δ vs baseline ↓ | RMSE ↓ | R² ↑ | MAE (change) ↓ | RMSE (change) ↓ | Δ RMSE (change) ↓ | Calib slope | Calib intercept | ECE ↓ | Δ ECE ↓ | KL div ↓ | Δ KL ↓ | Baseline MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Study 1 – Gun Control (MTurk) | 162 | 0.759 | 0.074 | +0.685 | 0.026 | +0.011 | 0.035 | 0.985 | 0.026 | 0.035 | +0.010 | 0.744 | 0.005 | 0.008 | — | 4.400 | +17.046 | 0.037 |
| Study 2 – Minimum Wage (MTurk) | 165 | 0.558 | 0.061 | +0.497 | 0.090 | +0.006 | 0.124 | 0.798 | 0.090 | 0.124 | +0.014 | 0.984 | 0.009 | 0.029 | — | 5.175 | +13.194 | 0.096 |
| Study 3 – Minimum Wage (YouGov) | 257 | 0.549 | 0.058 | +0.490 | 0.083 | +0.001 | 0.113 | 0.811 | 0.083 | 0.113 | +0.012 | 0.912 | 0.001 | 0.020 | — | 2.759 | +13.821 | 0.084 |

## Cross-Study Diagnostics

### Weighted Summary

- Weighted MAE 0.069 across 584 participants.
- Weighted baseline MAE 0.074 (+0.005 vs. final).
- Weighted directional accuracy 0.610 across 584 participants.
- Weighted baseline accuracy 0.063 (+0.546 vs. final).
- Weighted RMSE (change) 0.095 across 584 participants.
- Weighted baseline RMSE (change) 0.107 (+0.012 vs. final).
- Weighted calibration ECE 0.019 across 584 participants.
- Weighted KL divergence 3.897 across 584 participants.
- Weighted baseline KL divergence 18.435 (+14.538 vs. final).
- Largest MAE reduction: Study 1 – Gun Control (MTurk) (+0.011).
- Lowest MAE: Study 1 – Gun Control (MTurk) (0.026); Highest MAE: Study 2 – Minimum Wage (MTurk) (0.090).
- Highest directional accuracy: Study 1 – Gun Control (MTurk) (0.759).
- Lowest directional accuracy: Study 3 – Minimum Wage (YouGov) (0.549).
- Largest directional-accuracy gain: Study 1 – Gun Control (MTurk) (+0.685).
- Largest RMSE(change) reduction: Study 2 – Minimum Wage (MTurk) (+0.014).
- Lowest RMSE(change): Study 1 – Gun Control (MTurk) (0.035); Highest: Study 2 – Minimum Wage (MTurk) (0.124).
- Lowest calibration ECE: Study 1 – Gun Control (MTurk) (0.008); Highest: Study 2 – Minimum Wage (MTurk) (0.029).
- Largest KL divergence drop: Study 1 – Gun Control (MTurk) (+17.046).
- Lowest KL divergence: Study 3 – Minimum Wage (YouGov) (2.759); Highest: Study 2 – Minimum Wage (MTurk) (5.175).

- Unweighted MAE 0.067 (σ 0.029, range 0.026 – 0.090).
- MAE delta mean 0.006 (σ 0.004, range 0.001 – 0.011).
- Directional accuracy mean 0.622 (σ 0.097, range 0.549 – 0.759).
- Accuracy delta mean 0.557 (σ 0.090, range 0.490 – 0.685).
- RMSE(change) mean 0.091 (σ 0.040, range 0.035 – 0.124).
- RMSE(change) delta mean 0.012 (σ 0.001, range 0.010 – 0.014).
- Calibration ECE mean 0.019 (σ 0.009, range 0.008 – 0.029).
- KL divergence mean 4.111 (σ 1.007, range 2.759 – 5.175).
- KL divergence delta mean 14.687 (σ 1.687, range 13.194 – 17.046).

## Observations

- Study 1 – Gun Control (MTurk): MAE 0.026 (Δ vs. baseline +0.011), RMSE(change) 0.035, ECE 0.008, KL 4.400, R² 0.985.
- Study 2 – Minimum Wage (MTurk): MAE 0.090 (Δ vs. baseline +0.006), RMSE(change) 0.124, ECE 0.029, KL 5.175, R² 0.798.
- Study 3 – Minimum Wage (YouGov): MAE 0.083 (Δ vs. baseline +0.001), RMSE(change) 0.113, ECE 0.020, KL 2.759, R² 0.811.
- Average MAE reduction +0.006 across 3 studies.
- Mean R² 0.865.
- Mean RMSE(change) 0.091.
- Mean RMSE(change) delta 0.012.
- Mean calibration ECE 0.019.
- Mean KL divergence 4.111.
- Mean KL divergence delta 14.687.

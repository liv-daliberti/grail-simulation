# RESEARCH ARTICLE POLITICAL SCIENCES

## Short-term exposure to filter-bubble recommendation systems has limited polarization effects

This section replicates headline opinion-shift findings from _Short-term exposure to filter-bubble recommendation systems has limited polarization effects: Naturalistic experiments on YouTube_ (Liu et al., PNAS 2025) using the cleaned data in this repository.

### Opinion shift summary

| Study | Participants | Mean pre | Mean post | Mean change | Median change | Share ↑ | Share ↓ | Share \|Δ\| ≤ 0.05 |
| ------ | -------------- | ---------- | ----------- | ------------- | --------------- | --------- | --------- | ----------- |
| Study 1 – Gun Control (MTurk) | 1517 | 0.411 | 0.439 | 0.028 | 0.025 | 74.2% | 18.9% | 75.4% |
| Study 2 – Minimum Wage (MTurk) | 1607 | 0.264 | 0.300 | 0.036 | 0.011 | 54.0% | 39.5% | 49.0% |
| Study 3 – Minimum Wage (YouGov) | 2715 | 0.314 | 0.336 | 0.022 | 0.009 | 52.6% | 42.7% | 44.0% |

The minimal mean shifts and high share of small opinion changes (|Δ| ≤ 0.05 on a 0–1 scale) mirror the paper's conclusion that short-term algorithmic perturbations produced limited polarization in Studies 1–3.

### Pre/post opinion heatmaps

![Heatmap Study1 Gun Control](heatmap_study1_gun_control.png)

![Heatmap Study2 Minimum Wage](heatmap_study2_minimum_wage.png)

![Heatmap Study3 Minimum Wage](heatmap_study3_minimum_wage.png)

### Control vs. treatment shifts and pooled regression

![Mean opinion change](mean_opinion_change.png)

The first three panels separate mean opinion changes for the control and treatment arms of Studies 1–3 with 95% confidence intervals. The fourth panel reports the pooled regression coefficient comparing treatment versus control after adjusting for baseline opinion and study fixed effects.

Replication notes: opinion indices are scaled to [0, 1] and computed from the same survey composites used in the published study. Participants lacking a post-wave response are excluded from the relevant heatmap and summary.

### Control vs. treatment summary

| Study | Control Δ | Treatment Δ |
| ------ | ---------- | ------------ |
| Study 1 – Gun Control (MTurk) | 0.033 | 0.029 |
| Study 2 – Minimum Wage (MTurk) | -0.007 | 0.034 |
| Study 3 – Minimum Wage (YouGov) | 0.012 | 0.031 |

Pooled regression (control-adjusted) β̂ ≈ 0.018 with p ≈ 1.33e-11.

### Alignment with Liu et al. (2025)

- **Small average pre→post shifts:** Mean attitude changes in our Study 1–3 replications (+0.028, +0.036, +0.022 on a 0–1 index) match the paper’s qualitative takeaway that short-term perturbations yielded minimal opinion movement. This mirrors the across-study plots in Liu et al. (2025) that cluster around zero system effects.
- **Study coverage:** Our analyses focus on Studies 1–3 (gun control plus two minimum wage panels on MTurk and YouGov), which is the same filter-bubble subset summarized in Fig. 1 of the paper.

### Divergence from Liu et al. (2025)

- **Estimand and pooling vs. stratification:** We report a single pooled treatment coefficient (balanced 2/2 vs. slanted 3/1) with baseline opinion and study fixed effects, yielding β̂ ≈ +0.018 (p ≈ 1.3e-11). Liu et al. (2025) instead estimate algorithm effects separately for ideologues and for moderates split by liberal vs. conservative seed, then apply false-discovery-rate adjustments across outcome families. In their stratified framework most attitude effects are indistinguishable from zero; observable positives are small (≈ +0.03 among conservative ideologues in Study 3; ≈ +0.05 among moderates in Study 3 with a conservative seed).
- **Study-level pattern (Study 2):** Our Study 2 aggregation shows control Δ = −0.007 vs. treatment Δ = +0.034. After stratifying by ideology and seed, the published Study 2 effects sit near zero, with Study 3 driving the larger subgroup estimates.
- **Inference philosophy:** We emphasize a precise pooled p-value. The paper focuses on effect magnitudes and consistency across preregistered cells, highlighting minimum detectable effects of roughly 0.02–0.04 and concluding that algorithmic attitude shifts, if present, are modest.
- **Sample coverage:** Our analytic Ns (Study 1 = 1,517; Study 2 = 1,607/1,647; Study 3 = 2,715) fall slightly below the published counts (1,650; 1,679; 2,715), reflecting session drop-offs (e.g., missing slate metadata for Study 1). Without stratification, small composition differences can tilt the pooled estimate.

### Bottom line

Substantively, our replication agrees with Liu et al. (2025): short-term algorithmic slanting produced limited polarization. Numerically, we highlight a statistically precise pooled shift (+0.018), whereas the paper reports stratified, FDR-adjusted results that stay mostly null with small positives confined to Study 3 subgroups.

### Matching the paper’s reporting frame

To align perfectly with the published tables:

1. Re-estimate effects within {ideologues} and {moderates × seed ∈ {liberal, conservative}} for each study.
2. Include the preregistered covariates and apply the paper’s multiple-testing correction by outcome family.
3. Report confidence intervals alongside minimum detectable effects (≈ 0.02–0.04) to anchor interpretation.

The Methods, Results, and SI sections of Liu et al. (2025) document the required covariate set, cell definitions, and adjustment procedure.

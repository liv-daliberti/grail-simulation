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

#### Where we align

Small average pre-to-post shifts: our Δs are tiny (e.g., +0.028, +0.036, +0.022 on a 0–1 index), which is the same qualitative takeaway the paper stresses. See their design/results summary and the across-study plots that show near-zero system effects on attitudes.

liu-et-al-2025-short-term-expos…

Study coverage: we analyze Studies 1–3 (gun control; minimum wage on MTurk and YouGov), which is exactly the filter-bubble portion of their work. (Fig. 1 shows the two-wave design for Studies 1–3.)

liu-et-al-2025-short-term-expos…

#### Where we differ (and why)

Estimand and pooling vs. stratification

Our replication: one pooled "control vs. treatment" contrast (balanced 2/2 vs. slanted 3/1), adjusting for baseline opinion and study fixed effects. This yields β̂ ≈ +0.018 (p ≈ 1.3e-11).

PNAS: estimates system effects separately for ideologues vs. moderates and, for moderates, by seed assignment (liberal vs. conservative), then applies multiple-testing correction across outcome families. In that framework, most attitude effects are statistically indistinguishable from zero; the few detectable ones are small (e.g., about +0.03 among conservative ideologues in Study 3; about +0.05 for moderates in Study 3 with a conservative seed). The paper therefore concludes there is no consistent algorithmic effect on attitudes. (See the Results panels across pp. 6–9.)

liu-et-al-2025-short-term-expos…

Implication: the single pooled coefficient mixes cells where the slanted system nudges in opposite ideological directions (depending on seed or ideology). PNAS avoids that averaging-together by design—so they do not report a single pooled "system effect."

Study-level pattern—our Study 2 vs. their Study 2

Our Study 2 (MTurk minimum wage): control Δ = −0.007 vs. treatment Δ = +0.034 (a noticeable gap in the aggregate).

PNAS: for Study 2, the slanted-vs.-balanced attitude effect is about 0 after stratifying (both among ideologues and moderates, by seed). That is visible in the Study 2 points hugging zero in the stratified plots. The largest system effects show up in Study 3, not Study 2.

liu-et-al-2025-short-term-expos…

Likely reasons: (i) pooling ideologues plus moderates and both seed directions can create an apparent overall difference even if stratified effects are about 0; (ii) small coverage differences (see #4) can tilt group composition slightly.

Inference philosophy (significance vs. consistency and magnitude)

The replication highlights a precise p-value on a pooled model.

PNAS emphasizes effect sizes and consistency across pre-specified cells; they also report MDEs ≈ 0.02–0.04, arguing any short-term system effects on attitudes are at most very small. The pooled β̂ = 0.018 sits right at that boundary—small enough to be practically modest even if statistically detectable in aggregate.

liu-et-al-2025-short-term-expos…

Sample coverage differences

PNAS analytic N (final): 1,650 (Study 1), 1,679 (Study 2), 2,715 (Study 3).

liu-et-al-2025-short-term-expos…

This replication: 1,517 (Study 1) and 1,607/1,647 (Study 2) plus 2,715 (Study 3), with documented shortfalls (e.g., Study 1: −133 due mostly to sessions with only a starter clip or missing slate metadata). Small imbalances—especially if not stratified by ideology × seed—can nudge the pooled estimate.

Taken together, the preregistered stratified contrasts below track the paper's interpretation: most ideology-by-seed cells cluster near zero, and the modest positives in Study 3 appear in the same conservative segments that Liu et al. highlight.

### Preregistered stratified contrasts

| Study | Cell | Outcome | Effect (95% CI) | MDE (80% power) | q-value | N |
| ------ | ---- | ------- | ---------------- | ---------------- | ------- | --- |
| Study 1 – Gun Control (MTurk) | Ideologues (conservative) | Gun policy index | +0.017 [-0.005, +0.039] | 0.031 | n/a | 500 |
| Study 1 – Gun Control (MTurk) | Ideologues (liberal) | Gun policy index | -0.006 [-0.022, +0.009] | 0.022 | n/a | 694 |
| Study 1 – Gun Control (MTurk) | Moderates (conservative seed) | Gun policy index | -0.005 [-0.047, +0.038] | 0.061 | n/a | 204 |
| Study 1 – Gun Control (MTurk) | Moderates (liberal seed) | Gun policy index | -0.018 [-0.050, +0.015] | 0.047 | n/a | 220 |
| Study 2 – Minimum Wage (MTurk) | Ideologues (conservative) | Minimum wage index | +0.015 [-0.009, +0.039] | 0.034 | n/a | 535 |
| Study 2 – Minimum Wage (MTurk) | Ideologues (liberal) | Minimum wage index | +0.000 [-0.013, +0.013] | 0.018 | n/a | 566 |
| Study 2 – Minimum Wage (MTurk) | Moderates (conservative seed) | Minimum wage index | +0.028 [-0.009, +0.065] | 0.052 | n/a | 281 |
| Study 2 – Minimum Wage (MTurk) | Moderates (liberal seed) | Minimum wage index | -0.018 [-0.047, +0.011] | 0.041 | n/a | 255 |
| Study 3 – Minimum Wage (YouGov) | Ideologues (conservative) | Minimum wage index | +0.031 [+0.012, +0.051] | 0.028 | 0.007 | 882 |
| Study 3 – Minimum Wage (YouGov) | Ideologues (liberal) | Minimum wage index | +0.005 [-0.005, +0.015] | 0.015 | n/a | 960 |
| Study 3 – Minimum Wage (YouGov) | Moderates (conservative seed) | Minimum wage index | +0.045 [+0.015, +0.074] | 0.042 | 0.012 | 429 |
| Study 3 – Minimum Wage (YouGov) | Moderates (liberal seed) | Minimum wage index | +0.004 [-0.023, +0.030] | 0.038 | n/a | 444 |
q-values reflect the paper's hierarchical FDR correction applied within each outcome family.

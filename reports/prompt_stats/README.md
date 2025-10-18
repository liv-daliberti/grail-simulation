# Prompt Feature Report

- Output directory: `reports/prompt_stats`
- Figures: `reports/prompt_stats/figures`

## Profile availability

| Split | Rows | Missing profile | Share missing |
|-------|------|-----------------|---------------|
| train | 5631 | 0 | 0.00% |
| validation | 625 | 0 | 0.00% |

## Prior video counts

| Prior videos | Train | Validation |
|--------------|-------|------------|
| 0 | 5630 | 624 |
| 1 | 1 | 1 |

## Slate size distribution (`n_options`)

| Slate size | Train | Validation |
|------------|-------|------------|
| 1 | 482 | 46 |
| 2 | 302 | 37 |
| 3 | 303 | 35 |
| 4 | 3813 | 416 |
| 5 | 731 | 91 |

## Fallback profile usage

| Split | Count | Share of rows |
|-------|-------|---------------|
| train | 0 | 0.00% |
| validation | 0 | 0.00% |

## Rows missing all demographic fields

| Split | Count | Share of rows |
|-------|-------|---------------|
| train | 0 | 0.00% |
| validation | 0 | 0.00% |

## Unique content counts

| Split | Current videos | Gold videos | Unique slates | Unique state texts |
|-------|----------------|-------------|---------------|--------------------|
| train | 18 | 126 | 2718 | 5630 |
| validation | 16 | 94 | 493 | 625 |

## Unique participants per study and issue

| Split | Issue | Study | Participants |
|-------|-------|-------|--------------|
| train | gun_control | study1 | 1579 |
| train | minimum_wage | study3 | 4052 |
| train | all | all | 5631 |
| validation | gun_control | study1 | 178 |
| validation | minimum_wage | study3 | 447 |
| validation | all | all | 625 |

- Overall participants (all issues): 6256
- Overall participants for gun_control: 1757
- Overall participants for minimum_wage: 4499
- Overall participants in study1: 1757
- Overall participants in study3: 4499

## Feature figures

- `Viewer age` → `reports/prompt_stats/figures/age.png`
- `Supports assault weapons ban` → `reports/prompt_stats/figures/assault_ban.png`
- `College graduate flag` → `reports/prompt_stats/figures/college.png`
- `Believes concealed carry is safe` → `reports/prompt_stats/figures/concealed_safe.png`
- `Education level` → `reports/prompt_stats/figures/education.png`
- `Favorite channels text` → `reports/prompt_stats/figures/favorite_channels.png`
- `YouTube watch frequency code` → `reports/prompt_stats/figures/freq_youtube.png`
- `Gender` → `reports/prompt_stats/figures/gender.png`
- `Gun enthusiasm` → `reports/prompt_stats/figures/gun_enthusiasm.png`
- `Gun importance` → `reports/prompt_stats/figures/gun_importance.png`
- `Gun index` → `reports/prompt_stats/figures/gun_index.png`
- `Gun index (alternate)` → `reports/prompt_stats/figures/gun_index_2.png`
- `Gun ownership flag` → `reports/prompt_stats/figures/gun_ownership.png`
- `Supports handgun ban` → `reports/prompt_stats/figures/handgun_ban.png`
- `Household income bracket` → `reports/prompt_stats/figures/household_income.png`
- `Political ideology` → `reports/prompt_stats/figures/ideology.png`
- `$15 minimum wage support (wave 1)` → `reports/prompt_stats/figures/minwage15_w1.png`
- `$15 minimum wage support (wave 2)` → `reports/prompt_stats/figures/minwage15_w2.png`
- `Minimum wage stance (wave 1, inferred)` → `reports/prompt_stats/figures/minwage_text_r_w1.png`
- `Minimum wage stance (wave 2, inferred)` → `reports/prompt_stats/figures/minwage_text_r_w2.png`
- `Minimum wage stance (wave 1, survey)` → `reports/prompt_stats/figures/minwage_text_w1.png`
- `Minimum wage stance (wave 2, survey)` → `reports/prompt_stats/figures/minwage_text_w2.png`
- `Minimum wage support index (wave 1)` → `reports/prompt_stats/figures/mw_index_w1.png`
- `Minimum wage support index (wave 2)` → `reports/prompt_stats/figures/mw_index_w2.png`
- `Supports wage increase (wave 1)` → `reports/prompt_stats/figures/mw_support_w1.png`
- `Supports wage increase (wave 2)` → `reports/prompt_stats/figures/mw_support_w2.png`
- `Party identification` → `reports/prompt_stats/figures/party_identification.png`
- `Party lean` → `reports/prompt_stats/figures/party_lean.png`
- `Political interest` → `reports/prompt_stats/figures/political_interest.png`
- `Popular channels followed` → `reports/prompt_stats/figures/popular_channels_followed.png`
- `Race / ethnicity` → `reports/prompt_stats/figures/race_ethnicity.png`
- `Right to own importance` → `reports/prompt_stats/figures/right_to_own_importance.png`
- `Supports stricter gun laws` → `reports/prompt_stats/figures/stricter_laws.png`
- `prior_history_counts` → `reports/prompt_stats/figures/prior_history_counts.png`
- `slate_size_counts` → `reports/prompt_stats/figures/slate_size_counts.png`
- `profile_fallback_counts` → `reports/prompt_stats/figures/profile_fallback_counts.png`
- `demographic_missing_counts` → `reports/prompt_stats/figures/demographic_missing_counts.png`

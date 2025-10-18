# Prompt Feature Report

- Output directory: `reports/prompt_stats`
- Figures: `reports/prompt_stats/figures`

## Profile availability

| Split | Rows | Missing profile | Share missing |
|-------|------|-----------------|---------------|
| train | 30530 | 0 | 0.00% |
| validation | 3332 | 0 | 0.00% |

## Prior video counts

| Prior videos | Train | Validation |
|--------------|-------|------------|
| 0 | 8751 | 950 |
| 1 | 8017 | 859 |
| 2 | 7365 | 815 |
| 3 | 6323 | 704 |
| 4 | 52 | 4 |
| 5 | 12 | 0 |
| 6 | 7 | 0 |
| 7 | 3 | 0 |

## Slate size distribution (`n_options`)

| Slate size | Train | Validation |
|------------|-------|------------|
| 1 | 1346 | 128 |
| 2 | 3087 | 354 |
| 3 | 1153 | 129 |
| 4 | 19759 | 2187 |
| 5 | 5183 | 534 |
| 7 | 1 | 0 |
| 8 | 1 | 0 |

## Unique content counts

| Split | Current videos | Gold videos | Unique slates | Unique state texts |
|-------|----------------|-------------|---------------|--------------------|
| train | 277 | 452 | 17248 | 27045 |
| validation | 215 | 290 | 2650 | 3223 |

## Feature figures

- `Viewer age` → `reports/prompt_stats/figures/age.png`
- `Supports assault weapons ban` → `reports/prompt_stats/figures/assault_ban.png`
- `College graduate flag` → `reports/prompt_stats/figures/college.png`
- `Believes concealed carry is safe` → `reports/prompt_stats/figures/concealed_safe.png`
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

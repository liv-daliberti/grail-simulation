# Prompt Feature Report

- Output directory: `reports/prompt_stats`
- Figures: `reports/prompt_stats/figures`

## Profile availability

| Split | Rows | Missing profile | Share missing |
|-------|------|-----------------|---------------|
| train | 23919 | 0 | 0.00% |
| validation | 2657 | 0 | 0.00% |

## Prior video counts

| Prior videos | Train | Validation |
|--------------|-------|------------|
| 0 | 6454 | 709 |
| 1 | 6124 | 685 |
| 2 | 5851 | 649 |
| 3 | 5470 | 613 |
| 4 | 18 | 1 |
| 5 | 1 | 0 |
| 6 | 1 | 0 |

## Slate size distribution (`n_options`)

| Slate size | Train | Validation |
|------------|-------|------------|
| 1 | 1281 | 144 |
| 2 | 2081 | 245 |
| 3 | 1009 | 109 |
| 4 | 16706 | 1854 |
| 5 | 2841 | 305 |
| 8 | 1 | 0 |

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
| train | 249 | 410 | 15124 | 23820 |
| validation | 196 | 275 | 2207 | 2647 |

## Feature figures

- `Viewer age` → `reports/prompt_stats/figures/age.png`
- `Supports assault weapons ban` → `reports/prompt_stats/figures/assault_ban.png`
- `Biden approval` → `reports/prompt_stats/figures/biden_approval.png`
- `Binge-watches YouTube flag` → `reports/prompt_stats/figures/binge_youtube.png`
- `Children in household flag` → `reports/prompt_stats/figures/children_in_household.png`
- `City` → `reports/prompt_stats/figures/city.png`
- `Civic engagement` → `reports/prompt_stats/figures/civic_engagement.png`
- `College graduate flag` → `reports/prompt_stats/figures/college.png`
- `Believes concealed carry is safe` → `reports/prompt_stats/figures/concealed_safe.png`
- `County` → `reports/prompt_stats/figures/county.png`
- `Education level` → `reports/prompt_stats/figures/education.png`
- `Employment status` → `reports/prompt_stats/figures/employment_status.png`
- `Favorite channels text` → `reports/prompt_stats/figures/favorite_channels.png`
- `YouTube watch frequency code` → `reports/prompt_stats/figures/freq_youtube.png`
- `Gender` → `reports/prompt_stats/figures/gender.png`
- `Gun enthusiasm` → `reports/prompt_stats/figures/gun_enthusiasm.png`
- `Gun identity strength` → `reports/prompt_stats/figures/gun_identity.png`
- `Gun importance` → `reports/prompt_stats/figures/gun_importance.png`
- `Gun index` → `reports/prompt_stats/figures/gun_index.png`
- `Gun index (alternate)` → `reports/prompt_stats/figures/gun_index_2.png`
- `Gun ownership flag` → `reports/prompt_stats/figures/gun_ownership.png`
- `Gun policy stance` → `reports/prompt_stats/figures/gun_policy.png`
- `Gun policy priority` → `reports/prompt_stats/figures/gun_priority.png`
- `Supports handgun ban` → `reports/prompt_stats/figures/handgun_ban.png`
- `Household income bracket` → `reports/prompt_stats/figures/household_income.png`
- `Household size` → `reports/prompt_stats/figures/household_size.png`
- `Political ideology` → `reports/prompt_stats/figures/ideology.png`
- `Marital status` → `reports/prompt_stats/figures/marital_status.png`
- `Media diet description` → `reports/prompt_stats/figures/media_diet.png`
- `$15 minimum wage support (wave 1)` → `reports/prompt_stats/figures/minwage15_w1.png`
- `$15 minimum wage support (wave 2)` → `reports/prompt_stats/figures/minwage15_w2.png`
- `Minimum wage importance` → `reports/prompt_stats/figures/minwage_importance.png`
- `Minimum wage priority` → `reports/prompt_stats/figures/minwage_priority.png`
- `Minimum wage stance (wave 1, inferred)` → `reports/prompt_stats/figures/minwage_text_r_w1.png`
- `Minimum wage stance (wave 2, inferred)` → `reports/prompt_stats/figures/minwage_text_r_w2.png`
- `Minimum wage stance (wave 3, inferred)` → `reports/prompt_stats/figures/minwage_text_r_w3.png`
- `Minimum wage stance (wave 1, survey)` → `reports/prompt_stats/figures/minwage_text_w1.png`
- `Minimum wage stance (wave 2, survey)` → `reports/prompt_stats/figures/minwage_text_w2.png`
- `Minimum wage support index (wave 1)` → `reports/prompt_stats/figures/mw_index_w1.png`
- `Minimum wage support index (wave 2)` → `reports/prompt_stats/figures/mw_index_w2.png`
- `Supports wage increase (wave 1)` → `reports/prompt_stats/figures/mw_support_w1.png`
- `Supports wage increase (wave 2)` → `reports/prompt_stats/figures/mw_support_w2.png`
- `News consumption description` → `reports/prompt_stats/figures/news_consumption.png`
- `News consumption frequency` → `reports/prompt_stats/figures/news_frequency.png`
- `News sources list` → `reports/prompt_stats/figures/news_sources.png`
- `Top news sources` → `reports/prompt_stats/figures/news_sources_top.png`
- `News trust level` → `reports/prompt_stats/figures/news_trust.png`
- `Occupation text` → `reports/prompt_stats/figures/occupation.png`
- `Party identification` → `reports/prompt_stats/figures/party_identification.png`
- `Party lean` → `reports/prompt_stats/figures/party_lean.png`
- `Platform usage summary` → `reports/prompt_stats/figures/platform_use.png`
- `Political interest` → `reports/prompt_stats/figures/political_interest.png`
- `Popular channels followed` → `reports/prompt_stats/figures/popular_channels_followed.png`
- `Race / ethnicity` → `reports/prompt_stats/figures/race_ethnicity.png`
- `Religious affiliation` → `reports/prompt_stats/figures/religion.png`
- `Religious service attendance` → `reports/prompt_stats/figures/religious_attendance.png`
- `Right to own importance` → `reports/prompt_stats/figures/right_to_own_importance.png`
- `Social media use` → `reports/prompt_stats/figures/social_media_use.png`
- `State` → `reports/prompt_stats/figures/state.png`
- `Supports stricter gun laws` → `reports/prompt_stats/figures/stricter_laws.png`
- `Trump approval` → `reports/prompt_stats/figures/trump_approval.png`
- `Veteran status` → `reports/prompt_stats/figures/veteran.png`
- `2016 vote recall` → `reports/prompt_stats/figures/vote_2016.png`
- `2020 vote recall` → `reports/prompt_stats/figures/vote_2020.png`
- `2024 vote intention` → `reports/prompt_stats/figures/vote_2024.png`
- `ZIP3 prefix` → `reports/prompt_stats/figures/zip3.png`
- `prior_history_counts` → `reports/prompt_stats/figures/prior_history_counts.png`
- `slate_size_counts` → `reports/prompt_stats/figures/slate_size_counts.png`
- `profile_fallback_counts` → `reports/prompt_stats/figures/profile_fallback_counts.png`
- `demographic_missing_counts` → `reports/prompt_stats/figures/demographic_missing_counts.png`

# Prompt Feature Report

Generated with `python clean_data/prompt_stats.py` (wrapper around `clean_data.prompt.cli`).

- Output directory: `reports/prompt_stats`
- Figures: `reports/prompt_stats/figures`

## Profile availability

| Split | Rows | Missing profile | Share missing |
|-------|------|-----------------|---------------|
| train | 20342 | 0 | 0.00% |
| validation | 2272 | 0 | 0.00% |

## Prior video counts

| Prior videos | Train | Validation |
|--------------|-------|------------|
| 0 | 5309 | 588 |
| 1 | 5168 | 583 |
| 2 | 5043 | 562 |
| 3 | 4811 | 538 |
| 4 | 11 | 1 |

## Slate size distribution (`n_options`)

| Slate size | Train | Validation |
|------------|-------|------------|
| 1 | 1125 | 127 |
| 2 | 1776 | 218 |
| 3 | 857 | 83 |
| 4 | 14257 | 1535 |
| 5 | 2327 | 309 |

## Unique content counts

| Split | Current videos | Gold videos | Unique slates | Unique state texts |
|-------|----------------|-------------|---------------|--------------------|
| train | 241 | 405 | 13334 | 20341 |
| validation | 200 | 278 | 1895 | 2272 |

## Unique participants per study and issue

| Split | Issue | Study | Participants |
|-------|-------|-------|--------------|
| train | gun_control | study1 | 1365 |
| train | minimum_wage | study2 | 1505 |
| train | minimum_wage | study3 | 2439 |
| train | all | all | 5057 |
| validation | gun_control | study1 | 171 |
| validation | minimum_wage | study2 | 142 |
| validation | minimum_wage | study3 | 276 |
| validation | all | all | 586 |

- Overall participants (all issues): 5587
- Overall participants for gun_control: 1536
- Overall participants for minimum_wage: 4362
- Overall participants in study1: 1536
- Overall participants in study2: 1647
- Overall participants in study3: 2715

## Dataset coverage notes

Builder note: rows missing all survey demographics (age, gender, race, income, etc.) are dropped during cleaning so every retained interaction has viewer context for the prompt builder. This removes roughly 22% of the ~33k raw interactions.

> "The short answer is that sessions.json contains EVERYTHING. Every test run, every study. In addition to the studies that involved watching videos on the platform, it also contains sessions from the “First Impressions” study, which involved only rating thumbnails, and the “Shorts” study (Study 4 in the paper, I believe), which involved no user decisions (instead playing a sequence of predetermined videos that were either constant or increasing in their extremeness). All of these are differentiated by the topicId." — Emily Hu (University of Pennsylvania)

- Original study participants: 1,650 (Study 1 — gun rights), 1,679 (Study 2 — minimum wage MTurk), and 2,715 (Study 3 — minimum wage YouGov).
- Cleaned dataset participants captured here: 1536 (gun control) and 4362 (minimum wage). Study 4 (Shorts) is excluded because the released interaction logs do not contain recommendation slates.
- Shortfall summary (Studies 1–3 only):
  - Study 1 (gun control MTurk): 1650 expected vs. 1536 usable (-114). 98 sessions log only the starter clip (`vids` length = 1) and 15 log multiple clips but no recommendation slate (`displayOrders` empty).
  - Study 2 (minimum wage MTurk): 1679 expected vs. 1647 usable (-32). 14 sessions log only the starter clip; 17 have multiple clips but no slate metadata (`displayOrders` empty).
  - Study 3 (minimum wage YouGov): 2715 expected vs. 2715 usable (no gap). No gap — interaction logs are complete.
- Only gun-control and minimum-wage sessions (Studies 1–3) are retained; other topic IDs from the capsule are excluded.

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
- `demographic_missing_counts` → `reports/prompt_stats/figures/demographic_missing_counts.png`

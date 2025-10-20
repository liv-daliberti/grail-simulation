# Prompt feature report

## Dataset coverage notes

Statistics and charts focus on the core study sessions (study1–study3) covering the `gun_control` and `minimum_wage` issues.

- Train: 3992 of 4389 rows retained (91.0% coverage); excluded rows from study4: 42, unknown: 355
- Validation: 457 of 487 rows retained (93.8% coverage); excluded rows from study4: 9, unknown: 21

> "The short answer is that sessions.json contains EVERYTHING.
Every test run, every study.
In addition to the studies that involved watching videos on the platform,
it also contains sessions from the “First Impressions” study and the “Shorts” study
(Study 4 in the paper).
Those sessions involved no user decisions.
Instead they played predetermined videos that were
either constant or increasing in their extremeness.
All are differentiated by the topicId." — Emily Hu (University of Pennsylvania)

- Original study participants: 1,650 (Study 1 — gun rights)
  1,679 (Study 2 — minimum wage MTurk), and 2,715 (Study 3 — minimum wage YouGov).
- Cleaned dataset participants captured here: 88 (gun control) and 4361 (minimum wage).
  Study 4 (Shorts) is excluded because the released interaction logs
  do not contain recommendation slates.
- Shortfall summary (Studies 1–3 only):
  - Study 1 (gun control MTurk): 1650 expected vs. 88 usable (-1562).
    98 sessions log only the starter clip (`vids` length = 1) and 15 log multiple clips but no recommendation slate (`displayOrders` empty).
  - Study 2 (minimum wage MTurk): 1679 expected vs. 1646 usable (-33).
    14 sessions log only the starter clip; 17 have multiple clips but no slate metadata (`displayOrders` empty).
  - Study 3 (minimum wage YouGov): 2715 expected vs. 2715 usable (no gap).
    No gap — interaction logs are complete.
- Only gun-control and minimum-wage sessions (Studies 1–3) are retained;
  other topic IDs from the capsule are excluded.

Figures directory: `figures`

![Prior history distribution](figures/prior_history_counts.png)

![Slate size distribution](figures/slate_size_counts.png)

![Demographic coverage](figures/demographic_missing_counts.png)

## Demographic completeness

| Split | Rows | Missing all demographics | Share |
|-------|------|--------------------------|-------|
| train | 3992 | 0 | 0.00% |
| validation | 457 | 0 | 0.00% |
| overall | 4449 | 0 | 0.00% |

## Profile availability

| Split | Rows | Missing profile | Share missing |
|-------|------|-----------------|---------------|
| train | 3992 | 0 | 0.00% |
| validation | 457 | 0 | 0.00% |

## Prior video counts

| Prior videos | Train | Validation |
|--------------|-------|------------|
| 0 | 3967 | 454 |
| 1 | 19 | 3 |
| 2 | 2 | 0 |
| 3 | 4 | 0 |

## Slate size distribution (`n_options`)

| Slate size | Train | Validation |
|------------|-------|------------|
| 1 | 4 | 0 |
| 2 | 0 | 1 |
| 3 | 293 | 33 |
| 4 | 3695 | 423 |

## Unique content coverage

| Split | Current videos | Gold videos | Candidate videos | Unique slates | Prompt texts |
|-------|----------------|-------------|------------------|---------------|--------------|
| train | 16 | 77 | 100 | 2550 | 3990 |
| validation | 7 | 61 | 73 | 426 | 457 |
| overall | 16 | 77 | 100 | 2738 | 4447 |

## Unique participants

| Split | Participants (all issues) |
|-------|---------------------------|
| train | 3992 |
| validation | 457 |
| overall | 4449 |

## Features skipped due to missing data

- binge_youtube
- city
- civic_engagement
- county
- gun_identity
- gun_policy
- gun_priority
- household_size
- marital_status
- media_diet
- minwage_importance
- minwage_priority
- minwage_text_r_w3
- news_consumption
- news_sources
- news_sources_top
- occupation
- platform_use
- social_media_use
- veteran
- vote_2024
- zip3

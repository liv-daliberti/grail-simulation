# Prompt feature report

## Dataset coverage notes

Statistics and charts focus on the core study sessions (study1–study3) covering the `gun_control` and `minimum_wage` issues, using the full retained rows (no within-session subsetting). Session-ingestion details now live in `clean_data/sessions/README.md`.

- Train: 21460 rows (89.9% of dataset)
- Validation: 2419 rows (10.1% of dataset)

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
- Cleaned dataset participants captured here: 1517 (gun control) and 4362 (minimum wage).
  Study 4 (Shorts) is excluded because the released interaction logs
  do not contain recommendation slates.
- Shortfall summary (Studies 1–3 only):
  - Study 1 (gun control MTurk): 1650 expected vs. 1517 usable (-133).
    98 sessions log only the starter clip (`vids` length = 1) and 15 log multiple clips but no recommendation slate (`displayOrders` empty).
  - Study 2 (minimum wage MTurk): 1679 expected vs. 1647 usable (-32).
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
| train | 21460 | 0 | 0.00% |
| validation | 2419 | 0 | 0.00% |
| overall | 23879 | 0 | 0.00% |

## Profile availability

| Split | Rows | Missing profile | Share missing |
|-------|------|-----------------|---------------|
| train | 21460 | 0 | 0.00% |
| validation | 2419 | 0 | 0.00% |

## Prior video counts

| Prior videos | Train | Validation |
|--------------|-------|------------|
| 0 | 5673 | 630 |
| 1 | 5602 | 633 |
| 2 | 5279 | 596 |
| 3 | 4890 | 560 |
| 4 | 14 | 0 |
| 5 | 1 | 0 |
| 6 | 1 | 0 |

## Slate size distribution (`n_options`)

| Slate size | Train | Validation |
|------------|-------|------------|
| 1 | 14 | 0 |
| 2 | 1865 | 232 |
| 3 | 995 | 102 |
| 4 | 16227 | 1785 |
| 5 | 2358 | 300 |
| 8 | 1 | 0 |

## Unique content coverage

| Split | Current videos | Gold videos | Candidate videos | Unique slates | Prompt texts |
|-------|----------------|-------------|------------------|---------------|--------------|
| train | 198 | 393 | 406 | 14465 | 21361 |
| validation | 158 | 252 | 299 | 2068 | 2403 |
| overall | 199 | 399 | 412 | 15770 | 23764 |

## Unique participants

| Split | Participants (all issues) |
|-------|---------------------------|
| train | 5292 |
| validation | 587 |
| overall | 5879 |

### Participants by study

| Split | Study 1 | Study 2 | Study 3 |
|-------|---------|---------|---------|
| train | 1355 | 1479 | 2458 |
| validation | 162 | 168 | 257 |
| overall | 1517 | 1647 | 2715 |

_Study labels: study1 = gun control (MTurk), study2 = minimum wage (MTurk), study3 = minimum wage (YouGov)._

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

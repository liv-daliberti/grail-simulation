# Prompt Feature Report

## Dataset Coverage Notes

Builder note: rows missing all survey demographics (age, gender, race, income, etc.) are dropped during cleaning so every retained interaction has viewer context for the prompt builder. This removes roughly 22% of the ~33k raw interactions.

> "The short answer is that sessions.json contains EVERYTHING. Every test run, every study. In addition to the studies that involved watching videos on the platform, it also contains sessions from the “First Impressions” study, which involved only rating thumbnails, and the “Shorts” study (Study 4 in the paper, I believe), which involved no user decisions (instead playing a sequence of predetermined videos that were either constant or increasing in their extremeness). All of these are differentiated by the topicId." — Emily Hu (University of Pennsylvania)

- Original study participants: 1,650 (Study 1 — gun rights) and 5,326 (Studies 2–4 — minimum wage).
- Cleaned dataset participants captured here: 1,757 (gun control) and 4,499 (minimum wage).
- All statistics below operate on unique participants (one record per viewer). Only gun-control and minimum-wage sessions are retained; other topic IDs from the capsule are excluded.

## Unique Participants per Study and Issue

| Split | Issue | Study  | Participants |
|-------|-------|--------|--------------|
| train | gun_control | study1 | 1,579 |
| train | minimum_wage | study2 | _TBD_ |
| train | minimum_wage | study3 | _TBD_ |
| train | minimum_wage | study4 | _TBD_ |
| train | all | all | 5,631 |
| validation | gun_control | study1 | 178 |
| validation | minimum_wage | study2 | _TBD_ |
| validation | minimum_wage | study3 | 447 |
| validation | minimum_wage | study4 | _TBD_ |
| validation | all | all | 625 |
| total | gun_control | all | 1,757 |
| total | minimum_wage | all | 4,499 |

_The per-study breakdown will populate once `clean_data/prompt_stats.py` is re-run in an environment with the necessary plotting dependencies. The totals shown above already match the current cleaned dataset._

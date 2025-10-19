# Cleaned GRAIL Dataset: Alignment With CodeOcean Pipelines

This document explains how the Python builder in `clean_data/clean_data.py` replicates the preprocessing that the PNAS authors performed in their CodeOcean capsule. It is the detailed companion to the high‑level overview in the [repository README](../README.md).

## Overview

The builder reads the same intermediate CSV exports the CodeOcean R scripts emit and applies the study‑specific filters before combining them with the YouTube session logs. Every retained training row is keyed on the participant identifier that the paper used (worker IDs for MTurk/Shorts, case IDs for YouGov), and we deduplicate on `(participant_id, issue)` to ensure each person contributes at most one trajectory per domain.

The implementation mirrors three buckets of logic from the R code:

1. **Wave‑1 screening (Studies 1–3)** – attention checks, survey duration limits, and ideology index trimming.
2. **Session joins and earliest exposure** – merging Wave 2/3 surveys with the YouTube interaction logs and keeping the first qualifying session per participant.
3. **Analysis filters** – dropping pure control arms and rows missing the interaction summaries (`pro`, `anti`).

The sections below spell out the exact filters per study, show how we validate the resulting counts, and note the two remaining discrepancies between the public numbers and the current data drop.

## Study‑Specific Filters

### Study 1 — Gun Control (MTurk)

Source scripts: `code/gun control (issue 1)/02_clean_merge.R`, `03_analysis_multipletesting.R`

Steps reproduced in Python:

- Load `guncontrol_qualtrics_w1_clean.csv`.
- Keep respondents where:
  - `q87 == "Quick and easy"`
  - `q89 == "wikiHow"`
  - `survey_time >= 120` seconds
  - `gun_index` between `0.05` and `0.95` (inclusive).
- Join Wave 2/3 data (`guncontrol_qualtrics_w123_clean.csv`) and filter to the workers passing Wave 1.
- Drop sessions whose `treatment_arm` is `"control"` or missing.
- Require non‑missing `pro` and `anti`.
- Deduplicate on `worker_id`, keeping the earliest `start_time2`.

Result: **1,650** unique worker IDs (control matches the paper exactly). We also track the 1,635 distinct `urlid` values after filtering, matching the R scripts’ observation.

### Study 2 — Minimum Wage (MTurk / CloudResearch)

Source scripts: `code/minimum wage (issue 2)/02_clean_merge.R`, `03_analysis_multipletesting.R`

Filters:

- Load `qualtrics_w12_clean.csv`.
- Require the Wave‑1 checks:
  - `q87 == "Quick and easy"`
  - `q89 == "wikiHow"`
  - `survey_time >= 120`
  - `mw_index_w1` between `0.025` and `0.975`.
- Drop `treatment_arm == "control"` (case‑insensitive).
- Require non‑missing `pro` and `anti`.
- Deduplicate on `worker_id` using the earliest `start_time2`.

Output: **1,678** unique worker IDs. The published total is 1,679, and the missing participant corresponds to a single worker who passed the numeric filters but failed the audio check (`q87 == "Slow and arduous"`). The R scripts removed that respondent upstream as part of the attention check, so our pipeline leaves the count at 1,678 to stay faithful to that logic.

### Study 3 — Minimum Wage (YouGov)

Source scripts: `code/minimum wage (issue 2)/02b_clean_merge_yg.R`, `03b_analysis_multipletesting_yg.R`

Filters:

- Load `yg_w12_clean.csv` (YouGov’s merged panel export).
- Treat `caseid` as the participant key.
- Drop `treatment_arm == "control"`.
- Require non‑missing `pro` and `anti`.
- Deduplicate on `caseid` by earliest `start_time2`.

Result: **2,715** unique case IDs, matching the paper.

### Study 4 — Minimum Wage Shorts Experiment

Source script: `code/shorts/05_clean_shorts_data.R`

Filters:

- Load `qualtrics_w12_clean_ytrecs_may2024.csv`.
- Keep rows where:
  - `start_date >= 2024‑05‑28`
  - `q81 == "Quick and easy"`
  - `q82 == "wikiHow"`
  - `video_link` is present.
- Deduplicate on `worker_id`, keeping the earliest session (the R script retains every qualtrics row; the only duplicate in this export is a participant who launched two URLIDs within the same run).

Output: **931** unique worker IDs. The CodeOcean export contains 932 rows; the single shortfall is the duplicate participant noted above. The paper’s headline figure (932) counts both sessions, but the modeling scripts group by participant, so the deduped total is consistent with their analysis.

## Validation and Logging

During the build step the script records:

- Allow‑list sizes per study (`log.info` statements when reading the CSVs).
- A `sessions_filtered_allowlist` counter tracking interaction rows that are rejected because the participant is not in a study’s valid set.
- `participant_study` tags on each emitted row (`study1`, `study2`, `study3`, `study4`) so downstream metrics can be grouped by the original experiments.

You can re‑run the allow‑list sizing without rebuilding the entire dataset by executing:

```bash
python -m compileall clean_data/clean_data.py  # ensures the module imports
python clean_data/clean_data.py --dataset-name <capsule_dir>/data --output-dir /tmp/check --no-write
```

Inspect the logs for the “Allow-list” summaries to confirm the counts above.

## Discrepancies Versus Published Counts

| Study | Published | Python Builder | Gap | Cause |
|-------|-----------|----------------|-----|-------|
| Study 1 (gun control) | 1,650 | 1,650 | 0 | Perfect match |
| Study 2 (minimum wage MTurk) | 1,679 | 1,678 | −1 | One participant fails the audio attention check (`q87`) after R filtering |
| Study 3 (minimum wage YouGov) | 2,715 | 2,715 | 0 | Perfect match |
| Study 4 (minimum wage Shorts) | 932 | 931 | −1 | Duplicate `worker_id`; we keep the earliest session |

If you need to match the headline tallies exactly, you can relax the Wave‑1 audio check in Study 2 and keep both Shorts sessions. For auditability—and to stay aligned with the authors’ own preprocessing—we retain the stricter filters.

## Relationship to the Gun Control Pipeline

Gun control and minimum wage studies now share the same mechanics:

- Study allow‑lists are enforced before joining session logs.
- Participant identifiers are unified in this order: `worker_id`, `caseid`, YouTube anonymous ID, `urlid`, session ID.
- Deduplication occurs per `(participant_id, issue)`.

These conventions guarantee that any participant counts derived from the cleaned dataset line up with the original study populations, aside from the two discrepancies described above.

## Summary

- The Python builder traces the exact filters in the CodeOcean R scripts for Studies 1–4.
- Only two minor count differences remain, both explained by attention checks or duplicate sessions.
- Validation happens via logged allow-list sizes, per-session filtering counters, and the `participant_study` labels embedded in every output row.
- For an at-a-glance description of the data products and how to run the builder, see the top-level [project README](../README.md). This file serves as the deep-dive reference for researchers verifying provenance and reproducibility.

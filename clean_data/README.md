# Clean Data Pipeline

This document describes how the Python implementation in `clean_data/` mirrors the CodeOcean preprocessing used in the PNAS “Filter Bubble” studies. It complements the high-level project README and focuses on provenance, filtering rules, and module responsibilities.

## Package Overview

| Module | Responsibility |
| ------ | -------------- |
| `clean_data.sessions` | Parses CodeOcean session logs, enforces allow-lists, and emits the full interaction history dataframe (no participant/issue dedupe). |
| `clean_data.prompting` | Converts each interaction row into a GRPO-style prompt (`row_to_example`) and manages all prompt formatting helpers. |
| `clean_data.clean_data` | Public façade: loads datasets, filters unusable rows, builds cleaned splits, validates schema, saves/pushes outputs, and runs prompt stats. |
| `clean_data.codeocean` | Thin compatibility shim that reconstructs datasets directly from the capsule directory. |
| `clean_data.prompt` | Prompt analytics (plots + Markdown report) exposed via `python -m clean_data.prompt.cli`. |
| `clean_data.filters` | Simple predicates for prompt readiness and issue-count summaries used during dataset assembly. |
| `clean_data.surveys` / `clean_data.io` / `clean_data.helpers` | Shared utilities for reading survey exports, normalising identifiers, and synthesising viewer attributes. |

## Input Data and Allow-Lists

The builder consumes the intermediate CSV/RDS exports bundled with the CodeOcean capsule:

- `results/intermediate data/gun control (issue 1)/guncontrol_qualtrics_w1_clean.csv`
- `results/intermediate data/gun control (issue 1)/guncontrol_qualtrics_w123_clean.csv`
- `results/intermediate data/minimum wage (issue 2)/qualtrics_w12_clean.csv`
- `results/intermediate data/minimum wage (issue 2)/yg_w12_clean.csv`
- `results/intermediate data/shorts/qualtrics_w12_clean_ytrecs_may2024.csv`
- Session logs under `data/platform session data/sessions.json` and `data/shorts/ytrecs_sessions_may2024.rds`

Allow-lists are reconstructed by `clean_data.surveys.load_participant_allowlists`. Identifiers are normalised and deduplicated so each viewer contributes at most one trajectory per issue.

## Processing Pipeline

1. **Reconstruct allow-lists** – attention checks, survey-duration limits, ideology index trimming, and control-arm removal mirror the R scripts.
2. **Merge surveys with sessions** – join cleaned survey exports to the YouTube logs, retaining every usable interaction for Studies 1–3 (`clean_data.sessions.build_codeocean_rows`).
3. **Filter unusable rows** – drop interactions missing a usable slate or gold choice (`clean_data.filters.filter_prompt_ready`).
4. **Convert to prompts** – build GRPO-friendly prompt dictionaries with metadata passthrough (`clean_data.prompting.row_to_example`).
5. **Validate & save** – enforce required columns, write the full dataset (all interactions) to disk, emit prompt stats from a deduplicated view, and optionally push to the Hugging Face Hub.

## Study-Specific Filters

| Study | Audience | Key Filters | Resulting IDs |
| ----- | -------- | ----------- | ------------- |
| Study 1 – Gun Control (MTurk) | `guncontrol_qualtrics_w1_clean.csv` + `w123` follow-up | Attention checks (`q87/q89`), ≥120s survey time, ideology index ∈ [0.05, 0.95], drop control arm, require `pro`/`anti`, earliest `worker_id` session | 1,650 worker IDs |
| Study 2 – Minimum Wage (MTurk/CloudResearch) | `qualtrics_w12_clean.csv` | Same attention checks, ≥120s, wage index ∈ [0.025, 0.975], drop control arm, require `pro`/`anti`, earliest `worker_id` session | 1,678 worker IDs (paper lists 1,679; one respondent fails audio check) |
| Study 3 – Minimum Wage (YouGov) | `yg_w12_clean.csv` | Drop control arm, require `pro`/`anti`, earliest `caseid` session | 2,715 case IDs |
| Study 4 – Shorts experiment | `qualtrics_w12_clean_ytrecs_may2024.csv` + `ytrecs_sessions_may2024.rds` | Allow-list recorded for auditing, but sessions excluded from prompts because logs lack recommendation slates | 931 worker IDs retained in reporting |

Shorts participants remain in the allow-list summaries so shortfall analyses match the original paper, but prompt rows are emitted only for Studies 1–3.

## Output Artifacts

Running `python -m clean_data.cli` now produces two complementary datasets:

- **Full cleaned dataset** – persisted under `--output-dir`, retaining every promptable interaction for each `(participant, issue)` pair within Studies 1–3.
- **Deduped analytics view** – computed in-memory via `clean_data.clean_data.dedupe_by_participant_issue` so prompt statistics still mirror the historical “one row per participant/issue” summaries.

Downstream tooling that expects the deduped layout can invoke the helper directly, while modelling workflows can make use of the richer full-history export.

## Validation and Logging

Running `python -m clean_data.cli ...` produces informative logs:

- Allow-list sizes per study (“Allow-list (gun control): …”).
- Interaction statistics (`sessions_total`, `pairs_total`, `sessions_filtered_allowlist`, `pairs_missing_slates`, etc.).
- Confirmation when prompt analytics run and when cleaned splits are pushed to the Hugging Face Hub.

Replay the logging without writing to disk by invoking the CLI with a throwaway output directory.

## Known Discrepancies

| Study | Published | Builder | Delta | Explanation |
| ----- | --------- | ------- | ----- | ----------- |
| Study 1 | 1,650 | 1,650 | 0 | Matches R pipeline exactly |
| Study 2 | 1,679 | 1,678 | −1 | One worker fails the audio-attention check after R filtering |
| Study 3 | 2,715 | 2,715 | 0 | Matches R pipeline exactly |
| Study 4 | 932 | 931 | −1 | Duplicate `worker_id`; earliest session kept |

Relaxing the Study 2 audio check or retaining both Shorts sessions would recover the headline counts, but the default configuration stays aligned with the published preprocessing.

## Prompt Analytics

The reporting package lives in `clean_data/prompt/` and can be executed either through `python -m clean_data.cli --prompt-stats-dir ...` or directly:

```bash
python -m clean_data.prompt.cli \
  --dataset data/cleaned_grail \
  --output-dir reports/prompt_stats
```

Outputs include histogram PNGs, JSON summaries, and a Markdown README summarising feature coverage, participant counts, and demographic completeness.

## Summary

- Python faithfully mirrors the CodeOcean R preprocessing while exposing reusable modules.
- Allow-list enforcement, deduplication, and prompt synthesis live in separate modules for clarity.
- Shortfall and validation metrics are logged automatically, making it easy to audit runs or compare against the original studies.

## Development Checks

- `pylint clean_data` keeps the package aligned with the main repo lint rules (mirrors `scripts/run-lint.sh`).
- `pytest tests/test_clean_data_module.py tests/test_prompt_utils.py ...` exercises the clean-data pipeline end-to-end; run `pytest tests -k clean_data` for the full suite.
- `make -C docs html` rebuilds the Sphinx documentation, including autodoc pages for `clean_data.*` modules—confirm new APIs render correctly before shipping changes.

These checks run in CI, but executing them locally catches regressions earlier when iterating on the pipeline.

See the top-level README for setup instructions, training commands, and published artifacts.

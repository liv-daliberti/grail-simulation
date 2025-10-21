# Prompt Builder Package

`prompt_builder` turns a cleaned interaction row into both a narrative viewer
profile and the chat-style prompt consumed by baselines and RL trainers. The
package replaces the legacy `src/prompt_builder.py` single file with a modular
layout while preserving the same public API.

## Module layout

- `prompt.py` exposes the top-level `build_user_prompt(...)` helper that stitches
  together profile, current video, recently watched, and slate sections.
- `profiles.py` synthesizes viewer sentences (`render_profile`,
  `synthesize_viewer_sentence`) and keeps the friendly feature metadata in sync
  with `clean_data/prompt/question_mapping.py`.
- `formatters.py` houses generic text utilities (`clean_text`, `join_kv_section`)
  that normalize whitespace and convert structured data into readable strings.
- `parsers.py` provides convenience helpers (`truthy`, `secs`, `as_list_json`)
  used throughout the codebase to safely coerce survey values.
- `constants.py` defines prompt labels, ordering rules, and default limits (e.g.,
  maximum history length).

The package root re-exports the public entry points so existing imports such as
`from prompt_builder import build_user_prompt` continue to work unchanged.

## Data contract

The builder expects rows that already passed through `clean_data/clean_data.py`.
At a minimum the following columns should be present:

- `viewer_profile_sentence` or the fields required to synthesize one (see
  `profiles.py`).
- `watch_history_json` and/or `slate_items_json` for contextual sections.
- `gold_id` when a downstream consumer needs the clicked option.

Missing optional fields are tolerated—the prompt degrades gracefully and emits
fallback text such as “Profile information is unavailable.”

## Conventions & testing

- Keep dependencies optional; guard heavier imports (e.g., `pandas`) so the
  package remains lightweight when used in baselines.
- Avoid broad exception handlers; the callers rely on informative errors when
  schema mismatches occur.
- Update the high-level test `tests/test_prompt_builder_package.py` (and add
  package-specific fixtures) when you tweak rendering logic or introduce new
  schema columns.
- The Sphinx documentation under `docs/api/prompt_builder.rst` imports the
  modules directly—maintain docstrings for all public functions.

## Sample prompts

Use the helper CLI to generate markdown snippets for docs and reports:

```bash
PYTHONPATH=src python -m prompt_builder.samples \
  --dataset data/cleaned_grail \
  --issues gun_control,minimum_wage \
  --count 1 \
  --output reports/prompt_builder/README.md
```

The command above refreshes `reports/prompt_builder/README.md` with
fresh examples. The README carries a shorter preview—full prompts live in the
report.

### Gun control (validation split)

```text
PROFILE:
31-year-old, woman, Black or African-American (non-Hispanic), democrat liberal, $70,000-$79,999, college-educated, watches YouTube weekly. This viewer is a 31-year-old and a woman. They identify as Black or African-American (non-Hispanic). They are college educated. Their household income is reported as $70,000-$79,999. The survey was completed in English. Politics include party identification Democrat, party lean Not very strong Democrat, ideology Liberal, political interest 3, trump approval 30, and biden approval 80. Gun policy views include gun ownership does not own a gun, right-to-own importance no, supports assault weapons ban 0.25, supports handgun ban yes, believes concealed carry is safe no, supports stricter gun laws no, gun index 0.25, gun index (alt) 0.3125, gun enthusiasm no, and gun importance yes. Media habits include YouTube frequency weekly, favorite channels CBS,
CNN
NYT, popular channels followed Fox News,CNN,MSNBC, and news trust 0.6666666666666666.

CURRENT VIDEO:
Do We Need Stricter Gun Control? - The People Speak — channel: VICE News

RECENTLY WATCHED (NEWEST LAST):
(no recently watched videos available)

OPTIONS:
1. Piers Morgan Argues With Pro-Gun Campaigner About Orlando Shooting | Good Morning Britain — channel: Good Morning Britain
2. Why America Will NEVER Have Real Gun Control — channel: The Young Turks
3. Gun Control and The Violence in Chicago — channel: Colion Noir
4. Gun Banners Say the Darndest Things — channel: Rob Doar
5. How to Create a Gun-Free America in 5 Easy Steps
```

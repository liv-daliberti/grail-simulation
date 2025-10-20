# Prompt Builder Package

`prompt_builder` turns a cleaned interaction row into both a narrative viewer
profile and the chat-style prompt consumed by baselines and RL trainers. The
package replaces the legacy `src/prompt_builder.py` single file with a modular
layout while preserving the same public API.

## Module layout

- `prompt.py` exposes the top-level `build_user_prompt(...)` helper that stitches
  together profile, history, current video, and slate sections.
- `profiles.py` synthesises viewer sentences (`render_profile`,
  `synthesize_viewer_sentence`) and keeps the friendly feature metadata in sync
  with `clean_data/prompt/question_mapping.py`.
- `formatters.py` houses generic text utilities (`clean_text`, `join_kv_section`)
  that normalise whitespace and convert structured data into readable strings.
- `parsers.py` provides convenience helpers (`truthy`, `secs`, `as_list_json`)
  used throughout the codebase to safely coerce survey values.
- `constants.py` defines prompt labels, ordering rules, and default limits (e.g.,
  maximum history length).

The package root re-exports the public entry points so existing imports such as
`from prompt_builder import build_user_prompt` continue to work unchanged.

## Data contract

The builder expects rows that already passed through `clean_data/clean_data.py`.
At a minimum the following columns should be present:

- `viewer_profile_sentence` or the fields required to synthesise one (see
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

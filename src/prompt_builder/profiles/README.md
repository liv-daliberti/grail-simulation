# Profile Rendering Helpers

`prompt_builder.profiles` focuses on synthesising viewer profile text for the
prompt builder. Each module targets a specific slice of the survey data so
profiles stay consistent across issues.

## Modules

- `render.py` – top-level helpers (`render_profile`, etc.) that assemble the
  full viewer summary.
- `demographics.py` – phrasing for age, gender, race, income, education, and
  other demographic attributes.
- `media.py` – sentences describing media consumption habits (e.g., YouTube
  frequency).
- `politics.py` – language for party identification and ideology responses.
- `guns.py` / `wage.py` – issue-specific stance phrasing for gun control and
  minimum wage tasks.
- `__init__.py` – re-exports primary helpers.

Extend or adjust the issue-specific modules when new survey fields are added so
profiles stay informative without duplicating text logic elsewhere.

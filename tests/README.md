# Test Suite

This directory contains the automated checks for the project. Everything is written with `pytest`, and the shared `tests/conftest.py` bootstrap ensures the repository root and `src/` tree are importable while installing lightweight stubs for optional dependencies.

## Layout

### Top-level regression files
- `test_clean_data_module.py` exercises dataset validation, schema alignment, and serialization helpers in `clean_data`.
- `test_filters_reporting.py` checks aggregate issue statistics produced by the filtering/reporting utilities.
- `test_prompt_builder_package.py`, `test_prompt_utils.py`, and `test_prompt_smoke.py` cover prompt construction from multiple perspectives (unit behavior, helper functions, and smoke-level end-to-end assembly).
- `test_research_article_political_sciences.py` validates the replication analysis helpers, including dataframe preparation, statistics, markdown reporting, and plotting.
- `test_sessions_slate.py` and `test_surveys_allowlists.py` focus on the session slate reconstruction logic and survey allow-list ingestion (see `clean_data/sessions/README.md` for the helper overview).
- `test_visualization_recommendation_tree_viz.py` covers graph construction and rendering utilities for recommendation trees.

### Module-specific packages
- `common/` holds unit tests for shared utilities such as canonicalization helpers and logging configuration.
- `gpt4o/` targets the profile summarization and conversation-building logic, using fixtures to stub upstream title lookups.
- `knn/` and `xgb/` verify the recommendation baselines; each file uses `pytest.importorskip` so the tests only run when optional ML dependencies are installed.
- `open_r1/` confirms configuration parsing and reward helpers function with the provided dependency stubs.
- `integration/` currently contains `test_clean_prompt_flow.py`, which stitches together prompt cleaning and building to guard against regressions across packages.

### Shared fixtures and helpers
- `helpers/` provides stub modules (datasets, graphviz, openai, pandas, torch) used by `conftest.py` to allow the suite to run without the real heavy dependencies.
- `conftest.py` updates `sys.path` and activates the stubs before any test imports execute.

## Running the suite
- Run the full suite from the repository root with `pytest`.
- Use markers defined in `pytest.ini` (for example, `pytest -m clean_data` or `pytest -m "knn or xgb"`) to target specific subsystems; most files set a contextual `pytestmark` to make this convenient.

## Adding coverage
- Place new module-specific tests alongside existing ones: a new package under `src/foo` usually gets a matching `tests/foo/` directory or a `test_foo*.py` file at the top level.
- Prefer stubbing heavyweight services inside `tests/helpers/` when practical so the suite stays fast and hermetic.
- Remember to add or reuse a pytest marker when a test group corresponds to a subsystem that developers may want to run selectively.

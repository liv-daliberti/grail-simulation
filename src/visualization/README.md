# Visualization Package

Utilities in this directory power the recommendation-tree visualizations used
throughout the GRAIL reports. The public API lives at
`src/visualization/recommendation_tree_viz.py`, which re-exports helpers from
the `recommendation_tree` package so callers can import a single module.

## Recommendation Tree CLI
- Launch with `python -m visualization.recommendation_tree_viz` (ensure `PYTHONPATH=src`). The entry
  point enforces either `--tree` (CSV containing ranked recommendations) or
  `--cleaned-data` (a HuggingFace dataset saved via `clean_data/clean_data.py`).
- Always provide `--output path/to/figure.svg` or `--batch-output-dir`
  alongside the input source. The output file extension or the `--format`
  flag selects the rendering format (SVG, PNG, PDF, ...).
- When using `--tree`, pass optional `--metadata`, `--child-prefixes`, and
  `--highlight` arguments to control node labels, which columns are interpreted
  as children, and which viewer path is emphasized.
- With `--cleaned-data`, supply optional `--session-id`, `--split`, `--issue`,
  or `--max-steps` to focus the visualization on a particular trajectory. Use
  `--batch-output-dir` with `--batch-issues` to emit multiple sessions per
  issue automatically.

Example: render a single CSV tree to SVG.

```bash
export PYTHONPATH=src
python -m visualization.recommendation_tree_viz \
  --tree reports/visualized_recommendation_trees/minimum_wage/tree.csv \
  --metadata reports/visualized_recommendation_trees/metadata.csv \
  --output reports/visualized_recommendation_trees/minimum_wage/tree.svg
```

Example: visualize a cleaned dataset session as PNG.

```bash
export PYTHONPATH=src
python -m visualization.recommendation_tree_viz \
  --cleaned-data data/cleaned_grail \
  --session-id session_0001 \
  --highlight v0,v1,v2 \
  --output reports/visualized_recommendation_trees/session_0001.png
```

## Dependencies
- Install `graphviz` (both the Python package and system binaries) to enable
  rendering. During development the tests stub this dependency, but the CLI
  requires the real toolchain.
- Run the focused test coverage with
  `pytest tests/test_visualization_recommendation_tree_viz.py` whenever the
  visualization helpers change.

# Recommendation Tree Components

`visualization.recommendation_tree` provides the building blocks for rendering
recommendation trees from cleaned session data or precomputed CSV exports. The
package underpins the CLI documented in `src/visualization/README.md`.

## Modules

- `models.py` – dataclasses that describe nodes, edges, and viewer context used
  during rendering.
- `io.py` – loaders for CSV trees and cleaned datasets; normalises inputs into
  the `models` structures.
- `render.py` – Graphviz-based renderer that turns the structured models into
  SVG/PNG/PDF outputs.
- `cli.py` – command-line entry point for batch rendering (invoked via
  `python -m src.visualization.recommendation_tree.cli` or indirectly through
  `recommendation_tree_viz.py`).
- `__init__.py` – exports the public helpers.

Extend `models.py` and `render.py` when introducing new visual annotations so
both the CLI and programmatic callers benefit.

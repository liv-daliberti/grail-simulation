# Shared Visualisation Defaults

`common.visualization` hosts Matplotlib styling helpers shared by the report
builders. Centralising plot configuration keeps figures across the project
consistent.

## Modules

- `matplotlib.py` – context managers and helper functions that establish colour
  palettes, font sizes, and default figure dimensions. Import these utilities
  before creating plots in report builders.
- `__init__.py` – exposes the styling helpers.

Prefer extending this package instead of hard-coding plot styles in downstream
modules so new visual guidelines propagate automatically.

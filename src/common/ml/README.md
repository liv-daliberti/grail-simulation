# Shared ML Helpers

`common.ml` groups machine-learning utilities that do not belong to any single
baseline. The subpackages provide model-specific helpers that other modules can
import without creating circular dependencies.

## Structure

- `xgb/` – XGBoost-specific training helpers (see its README).
- `__init__.py` – exports shared helpers when they exist.

Add new subpackages here when introducing reusable ML utilities (e.g., shared
callbacks or model shims) so other modules can depend on them without duplicating code.

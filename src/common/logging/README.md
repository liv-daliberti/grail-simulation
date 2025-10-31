# Logging Utilities

`common.logging` provides structured logging helpers used across the pipelines
and trainers. Centralising the configuration keeps log formatting consistent
and makes it easy to toggle verbosity for debugging.

## Modules

- `utils.py` – convenience functions for configuring stdlib logging, attaching
  rich formatting, and emitting structured key/value pairs. These helpers are
  used by CLIs and pipeline executors before any custom logging occurs.
- `__init__.py` – exposes the top-level helpers.

Use these utilities whenever a new CLI or background worker needs logging so
the global log level and formatting remain coherent.

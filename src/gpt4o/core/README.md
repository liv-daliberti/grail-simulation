# GPT-4o Core Components

`gpt4o.core` implements the core primitives required to run GPT-4o slate and
opinion evaluations: client configuration, conversation assembly, scoring, and
utility helpers. Both the CLI and pipeline import from this package.

## Modules

- `client.py` – Azure OpenAI client wrapper that handles authentication,
  retries, and request throttling.
- `config.py` – dataclasses and helpers that translate CLI/pipeline options into
  evaluation settings.
- `conversation.py` – constructs the chat payloads sent to GPT-4o, wiring in
  prompt builder outputs and opinion context.
- `evaluate.py` – orchestrates evaluation loops, response parsing, and metric
  aggregation.
- `titles.py` – utilities for formatting candidate titles and injecting fallback
  metadata.
- `utils.py` – shared helpers (logging, deterministic caching, response schema
  checks).
- `opinion/` – opinion-specific helpers (see its README).

Extend this package when adding new evaluation behaviors or model backends so
the CLI/pipeline layers remain thin.

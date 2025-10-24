# Session Log Builders

Utilities under `clean_data/sessions/` transform the raw CodeOcean exports into
model-ready interaction rows. The package normalizes participant identifiers,
assembles recommendation slates, and enforces allow-lists before handing the
data off to prompt construction or downstream baselines.

## Package Layout

- `build.py` – orchestrates session ingestion via `build_codeocean_rows` and
  splits interaction dataframes with `split_dataframe`.
- `io_utils.py` – small helpers that load capsule JSON/RDS blobs and map survey
  exports to the layout expected by the builders.
- `models.py` – frozen dataclasses representing participant identifiers, timing
  metadata, session info, and allow-list configuration.
- `participants.py` – resolves survey candidates, applies allow-list checks, and
  selects canonical participant tokens (`participant_key`).
- `slates.py` – constructs recommendation slates, normalizes display orders, and
  surfaces convenience helpers such as `build_slate_items`.
- `watch.py` – parses the watched-video timeline, derives per-video metadata,
  and exposes lookup utilities shared across helpers.

Importing `clean_data.sessions` exposes the public API documented in
`clean_data/sessions/__init__.py`; legacy camelCase aliases remain available for
compatibility with older scripts.

## Quick Start

```python
import pandas as pd
from pathlib import Path

from clean_data.sessions import build_codeocean_rows, split_dataframe

capsule_root = Path("capsule-5416997")
rows, stats = build_codeocean_rows(
    capsule_root=capsule_root,
    data_root=capsule_root / "results",
    seed=13,
)
print(stats["sessions_total"])

train_df, eval_df = split_dataframe(
    pd.DataFrame(rows),
    eval_fraction=0.1,
    random_state=13,
)
```

`build_codeocean_rows` expects the CodeOcean capsule structure (session logs,
prompt metadata, and survey exports). It returns a list of interaction dictionaries
plus summary counters (`stats`). `split_dataframe` converts the rows into a
deterministic train/eval split when experiments need to hold out examples.

## Related Entry Points

- End-to-end dataset builds: `python -m clean_data.cli` runs the cleaning
  pipeline and delegates session ingestion to this package.
- Report refresh: `scripts/run-clean-data-suite.sh` stitches together session
  ingestion, prompt transforms, and the political-science replication.

See the top-level `README.md` and `clean_data/README.md` for how these pieces fit
into the broader pipeline. Tests covering the helpers above live in
`tests/test_sessions_slate.py` and the `clean_data` marked suite.

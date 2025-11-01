#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI shim that reuses the GRPO evaluation pipeline for GRAIL runs."""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from grpo.config import repo_root as _repo_root
# Import with a defensive fallback to avoid Sphinx autodoc failures when
# optional GRPO dependencies are missing during docs builds.
try:  # pragma: no cover - used during docs import
    from grpo.pipeline import (
        configure_logging as _configure_logging,
        main as _grpo_main,
    )
except (ImportError, ModuleNotFoundError) as _grpo_import_error:  # pragma: no cover
    def _configure_logging(_level: str) -> None:  # type: ignore
        return None

    def _grpo_main(_argv: Sequence[str] | None = None) -> None:  # type: ignore
        raise RuntimeError(
            "grpo.pipeline is unavailable during documentation import; "
            "install optional dependencies or build outside Sphinx."
        ) from _grpo_import_error

from .reports import DEFAULT_REGENERATE_HINT

__all__ = ["main"]


LOGGER = logging.getLogger("grail.pipeline")


def _flag_present(argv: Sequence[str], flag: str) -> bool:
    """Return ``True`` when the CLI already specifies ``flag``."""

    flag_eq = f"{flag}="
    return any(token == flag or token.startswith(flag_eq) for token in argv)


def _extract_log_level(argv: Sequence[str]) -> str:
    """Return the requested log level or the default when omitted."""

    for index, token in enumerate(argv):
        if token.startswith("--log-level="):
            return token.split("=", 1)[1]
        if token == "--log-level" and index + 1 < len(argv):
            return argv[index + 1]
    return "INFO"


def main(argv: Sequence[str] | None = None) -> None:
    """
    Entrypoint mirroring :mod:`grpo.pipeline` with GRAIL defaults.

    :param argv: Optional sequence of CLI tokens. When omitted, arguments are
        read from ``sys.argv[1:]``.
    :returns: ``None``. Delegates evaluation to :mod:`grpo.pipeline`.
    """

    user_args = list(argv) if argv is not None else sys.argv[1:]
    extra_args: list[str] = []
    defaults_log: list[str] = []
    log_level = _extract_log_level(user_args)

    repo_root = _repo_root()
    default_out_dir = str(repo_root / "models" / "grail")

    if not _flag_present(user_args, "--out-dir"):
        extra_args.extend(["--out-dir", default_out_dir])
        defaults_log.append(f"--out-dir {default_out_dir}")
    if not _flag_present(user_args, "--reports-subdir"):
        extra_args.extend(["--reports-subdir", "grail"])
        defaults_log.append("--reports-subdir grail")
    if not _flag_present(user_args, "--baseline-label"):
        extra_args.extend(["--baseline-label", "GRAIL"])
        defaults_log.append("--baseline-label GRAIL")
    if not _flag_present(user_args, "--regenerate-hint"):
        extra_args.extend(["--regenerate-hint", DEFAULT_REGENERATE_HINT])
        defaults_log.append("--regenerate-hint <default>")

    _configure_logging(log_level)

    LOGGER.info(
        "Launching GRAIL pipeline wrapper (log level %s).", str(log_level).upper()
    )
    if defaults_log:
        LOGGER.info("Applying default CLI args: %s", ", ".join(defaults_log))
    else:
        LOGGER.info("All CLI overrides supplied; no defaults applied.")
    LOGGER.debug("Forwarding arguments: %s", extra_args + user_args)
    LOGGER.info("Delegating execution to grpo.pipeline...")
    print(
        f"[grail.pipeline] defaults={defaults_log or ['<none>']}", flush=True
    )
    print(
        f"[grail.pipeline] argv={extra_args + user_args}",
        flush=True,
    )

    _grpo_main(extra_args + user_args)


if __name__ == "__main__":  # pragma: no cover
    main()

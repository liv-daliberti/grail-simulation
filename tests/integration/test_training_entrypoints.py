import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script_relative",
    (
        "src/grail/grail.py",
        "src/grpo/grpo.py",
    ),
)
def test_training_entrypoints_expose_help_without_pythonpath(script_relative: str) -> None:
    """Ensure entrypoints run as standalone scripts after path bootstrapping."""

    script_path = REPO_ROOT / script_relative
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        env=env,
    )
    assert "usage:" in process.stdout.lower()

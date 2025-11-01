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

"""RunnerState mixins for flag and config properties."""

from __future__ import annotations

from typing import Sequence

from .settings import VectorizerConfigs


class _FlagsMixin:
    # Flags
    @property
    def run_next_video(self) -> bool:  # type: ignore[override]
        """Whether to run the Next Video pipeline tasks."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("run_next_video", True)

    @run_next_video.setter
    def run_next_video(self, value: bool) -> None:  # type: ignore[override]
        """Enable or disable Next Video pipeline tasks."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["run_next_video"] = value

    @property
    def run_opinion(self) -> bool:  # type: ignore[override]
        """Whether to run the Opinion pipeline tasks."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("run_opinion", True)

    @run_opinion.setter
    def run_opinion(self, value: bool) -> None:  # type: ignore[override]
        """Enable or disable Opinion pipeline tasks."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["run_opinion"] = value

    @property
    def allow_incomplete(self) -> bool:  # type: ignore[override]
        """Whether incomplete intermediate results are allowed."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("allow_incomplete", True)

    @allow_incomplete.setter
    def allow_incomplete(self, value: bool) -> None:  # type: ignore[override]
        """Set whether incomplete intermediate results are allowed."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["allow_incomplete"] = value

    @property
    def stage(self) -> str:  # type: ignore[override]
        """The pipeline stage to execute."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("stage", "full")

    @stage.setter
    def stage(self, value: str) -> None:  # type: ignore[override]
        """Set the pipeline stage to execute."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["stage"] = value

    @property
    def reuse_sweeps(self) -> bool:  # type: ignore[override]
        """Whether to reuse completed hyperparameter sweeps if present."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("reuse_sweeps", False)

    @reuse_sweeps.setter
    def reuse_sweeps(self, value: bool) -> None:  # type: ignore[override]
        """Enable or disable reuse of completed sweeps."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["reuse_sweeps"] = value

    @property
    def reuse_final(self) -> bool:  # type: ignore[override]
        """Whether to reuse final evaluation artifacts when available."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("reuse_final", False)

    @reuse_final.setter
    def reuse_final(self, value: bool) -> None:  # type: ignore[override]
        """Enable or disable reuse of final evaluation artifacts."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["reuse_final"] = value

    @property
    def reuse_sweeps_source(self) -> str | None:  # type: ignore[override]
        """Optional path or run ID to source sweep reuse from."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("reuse_sweeps_source")

    @reuse_sweeps_source.setter
    def reuse_sweeps_source(self, value: str | None) -> None:  # type: ignore[override]
        """Set the source identifier for sweep reuse."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["reuse_sweeps_source"] = value

    @property
    def reuse_final_source(self) -> str | None:  # type: ignore[override]
        """Optional path or run ID to source final reuse from."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        return flags.get("reuse_final_source")

    @reuse_final_source.setter
    def reuse_final_source(self, value: str | None) -> None:  # type: ignore[override]
        """Set the source identifier for final stage reuse."""
        flags = self._state.setdefault("flags", {})  # type: ignore[attr-defined]
        flags["reuse_final_source"] = value


class _ConfigMixin:
    # Config
    @property
    def paths(self):  # type: ignore[override]
        """Resolved, commonly used filesystem paths for the run."""
        cfg = self._state.setdefault("cfg", {})  # type: ignore[attr-defined]
        return cfg.get("paths")

    @paths.setter
    def paths(self, value) -> None:  # type: ignore[override]
        """Set resolved filesystem paths for the run."""
        self._state.setdefault("cfg", {})["paths"] = value  # type: ignore[attr-defined]

    @property
    def jobs(self) -> int:  # type: ignore[override]
        """Number of parallel jobs used for sweeps and execution."""
        cfg = self._state.setdefault("cfg", {})  # type: ignore[attr-defined]
        return cfg.get("jobs", 1)

    @jobs.setter
    def jobs(self, value: int) -> None:  # type: ignore[override]
        """Set the number of parallel jobs used during execution."""
        self._state.setdefault("cfg", {})["jobs"] = int(value)  # type: ignore[attr-defined]

    @property
    def base_cli(self) -> Sequence[str]:  # type: ignore[override]
        """Base CLI for downstream tasks shared across stages."""
        cfg = self._state.setdefault("cfg", {})  # type: ignore[attr-defined]
        return cfg.get("base_cli", ())

    @base_cli.setter
    def base_cli(self, value: Sequence[str]) -> None:  # type: ignore[override]
        """Set the base CLI for downstream tasks shared across stages."""
        self._state.setdefault("cfg", {})["base_cli"] = value  # type: ignore[attr-defined]

    @property
    def vectorizers(self) -> VectorizerConfigs | None:  # type: ignore[override]
        """Vectorizer configuration bundle for text features."""
        cfg = self._state.setdefault("cfg", {})  # type: ignore[attr-defined]
        return cfg.get("vectorizers")

    @vectorizers.setter
    def vectorizers(self, value: VectorizerConfigs | None) -> None:  # type: ignore[override]
        """Set the vectorizer configuration bundle."""
        self._state.setdefault("cfg", {})["vectorizers"] = value  # type: ignore[attr-defined]


__all__ = ["_FlagsMixin", "_ConfigMixin"]

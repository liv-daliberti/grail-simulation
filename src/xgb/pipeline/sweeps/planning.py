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

"""Utilities for rendering sweep execution plans."""

from __future__ import annotations

from typing import List, Sequence

from ..context import OpinionSweepTask, SweepTask


def _emit_sweep_plan(tasks: Sequence[SweepTask]) -> None:
    """
    Print a concise sweep plan listing.

    :param tasks: Sweep tasks to display.
    :type tasks: Sequence[SweepTask]
    """

    print(f"TOTAL_TASKS={len(tasks)}")
    if not tasks:
        return
    print("INDEX\tSTUDY\tISSUE\tVECTORIZER\tLABEL")
    for display_index, task in enumerate(tasks):
        print(
            f"{display_index}\t{task.study.key}\t{task.study.issue}\t"
            f"{task.config.text_vectorizer}\t{task.config.label()}"
        )


def _format_sweep_task_descriptor(task: SweepTask) -> str:
    """
    Return a short descriptor for a sweep task.

    :param task: Sweep task to describe.
    :type task: SweepTask
    :returns: Concise descriptor combining study and configuration.
    :rtype: str
    """

    return f"{task.study.key}:{task.study.issue}:{task.config.label()}"


def _emit_combined_sweep_plan(
    *,
    slate_tasks: Sequence[SweepTask],
    opinion_tasks: Sequence[OpinionSweepTask],
) -> None:
    """Print a combined sweep plan covering next-video and opinion tasks."""

    total = len(slate_tasks) + len(opinion_tasks)
    sections: List[str] = []
    if slate_tasks:
        sections.append("next_video")
    if opinion_tasks:
        sections.append("opinion")
    if sections:
        print(",".join(sections))
    print(f"TOTAL_TASKS={total}")
    if slate_tasks:
        print("### NEXT_VIDEO")
        print("INDEX\tSTUDY\tISSUE\tVECTORIZER\tLABEL")
        for display_index, task in enumerate(slate_tasks):
            print(
                f"{display_index}\t{task.study.key}\t{task.study.issue}\t"
                f"{task.config.text_vectorizer}\t{task.config.label()}"
            )
    if opinion_tasks:
        print("### OPINION")
        print("INDEX\tSTUDY\tISSUE\tVECTORIZER\tLABEL")
        for display_index, task in enumerate(opinion_tasks):
            vectorizer = getattr(
                task,
                "feature_space",
                getattr(task.config, "text_vectorizer", "opinion"),
            )
            print(
                f"{display_index}\t{task.study.key}\t{task.study.issue}\t"
                f"{vectorizer}\t{task.config.label()}"
            )


def _format_opinion_sweep_task_descriptor(task: OpinionSweepTask) -> str:
    """
    Return a short descriptor for an opinion sweep task.

    :param task: Opinion sweep task to describe.
    :type task: ~xgb.pipeline.context.OpinionSweepTask
    :returns: Concise descriptor combining study and configuration.
    :rtype: str
    """

    return f"{task.study.key}:{task.study.issue}:{task.feature_space}:{task.config.label()}"


__all__ = [
    "_emit_combined_sweep_plan",
    "_emit_sweep_plan",
    "_format_opinion_sweep_task_descriptor",
    "_format_sweep_task_descriptor",
]

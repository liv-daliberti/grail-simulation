"""Selection logic to balance samples across issues and tasks.

This module selects a balanced set of samples for the two issues
"gun_control" and "minimum_wage" from two task pools: next_video (nv)
and opinion (op).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from .samples_types import Sample

# Type alias for the nested bins structure used throughout the selection.
IssueBins = Dict[str, Dict[str, List[Sample]]]


def _filter_samples(
    items: Sequence[Sample], *, issue: str | None = None, task: str | None = None
) -> List[Sample]:
    out: List[Sample] = []
    for sample in items:
        if issue and sample.issue != issue:
            continue
        if task and sample.task != task:
            continue
        out.append(sample)
    return out


def _build_bins(nv_samples: Sequence[Sample], op_samples: Sequence[Sample]) -> IssueBins:
    return {
        "gun_control": {
            "nv": _filter_samples(nv_samples, issue="gun_control"),
            "op": _filter_samples(op_samples, issue="gun_control"),
        },
        "minimum_wage": {
            "nv": _filter_samples(nv_samples, issue="minimum_wage"),
            "op": _filter_samples(op_samples, issue="minimum_wage"),
        },
    }


def _decide_op_takes(bins: Mapping[str, Mapping[str, List[Sample]]]) -> Tuple[int, int]:
    gun_op = bins["gun_control"]["op"]
    wage_op = bins["minimum_wage"]["op"]
    gun_op_quota, wage_op_quota = 2, 3
    if len(gun_op) >= 3 and (len(gun_op) >= len(wage_op) or len(wage_op) < 3):
        gun_op_quota, wage_op_quota = 3, 2
    elif len(wage_op) >= 3:
        gun_op_quota, wage_op_quota = 2, 3
    gun_take = min(gun_op_quota, len(gun_op))
    wage_take = min(wage_op_quota, len(wage_op))
    target_op_total = 5
    total = gun_take + wage_take
    if total < target_op_total:
        deficit = target_op_total - total
        gun_rem = len(gun_op) - gun_take
        wage_rem = len(wage_op) - wage_take
        while deficit > 0 and (gun_rem > 0 or wage_rem > 0):
            if gun_rem >= wage_rem and gun_rem > 0:
                gun_take += 1
                gun_rem -= 1
            elif wage_rem > 0:
                wage_take += 1
                wage_rem -= 1
            deficit -= 1
    return gun_take, wage_take


def _decide_nv_takes(
    bins: Mapping[str, Mapping[str, List[Sample]]],
    gun_op_take: int,
    wage_op_take: int,
    *,
    per_issue: int,
) -> Tuple[int, int]:
    gun_nv = bins["gun_control"]["nv"]
    wage_nv = bins["minimum_wage"]["nv"]
    gun_nv_quota = max(0, per_issue - gun_op_take)
    wage_nv_quota = max(0, per_issue - wage_op_take)
    gun_take = min(gun_nv_quota, len(gun_nv))
    wage_take = min(wage_nv_quota, len(wage_nv))
    target_nv_total = 5
    total = gun_take + wage_take
    if total < target_nv_total:
        deficit = target_nv_total - total
        gun_rem = len(gun_nv) - gun_take
        wage_rem = len(wage_nv) - wage_take
        while deficit > 0 and (gun_rem > 0 or wage_rem > 0):
            if gun_rem >= wage_rem and gun_rem > 0 and gun_take < per_issue:
                gun_take += 1
                gun_rem -= 1
            elif wage_rem > 0 and wage_take < per_issue:
                wage_take += 1
                wage_rem -= 1
            deficit -= 1
    return gun_take, wage_take


def _assemble_selection(
    bins: Mapping[str, Mapping[str, List[Sample]]],
    gun_op_take: int,
    wage_op_take: int,
    gun_nv_take: int,
    wage_nv_take: int,
    *,
    per_issue: int,
) -> Tuple[List[Sample], List[Sample]]:
    gun_op = bins["gun_control"]["op"]
    gun_nv = bins["gun_control"]["nv"]
    wage_op = bins["minimum_wage"]["op"]
    wage_nv = bins["minimum_wage"]["nv"]
    select_gun = (gun_op[:gun_op_take] + gun_nv[:gun_nv_take])[:per_issue]
    select_wage = (wage_op[:wage_op_take] + wage_nv[:wage_nv_take])[:per_issue]
    return select_gun, select_wage


def _top_up_selection(
    bins: Mapping[str, Mapping[str, List[Sample]]],
    select_gun: List[Sample],
    select_wage: List[Sample],
    op_takes: Tuple[int, int],
    nv_takes: Tuple[int, int],
    *,
    per_issue: int,
    nv_samples: Sequence[Sample],
    op_samples: Sequence[Sample],
) -> Tuple[List[Sample], List[Sample]]:
    if len(select_gun) + len(select_wage) >= 2 * per_issue:
        return select_gun, select_wage

    gun_op = bins["gun_control"]["op"]
    gun_nv = bins["gun_control"]["nv"]
    wage_op = bins["minimum_wage"]["op"]
    wage_nv = bins["minimum_wage"]["nv"]

    pool: List[Sample] = []
    if sum(1 for sample in (select_gun + select_wage) if sample.task == "opinion") < 5:
        pool.extend(gun_op[op_takes[0]:] + wage_op[op_takes[1]:])
    if sum(1 for sample in (select_gun + select_wage) if sample.task == "next_video") < 5:
        pool.extend(gun_nv[nv_takes[0]:] + wage_nv[nv_takes[1]:])
    if not pool:
        existing = set(select_gun + select_wage)
        pool.extend(
            [
                sample
                for sample in (list(nv_samples) + list(op_samples))
                if sample not in existing
            ]
        )

    for sample in pool:
        if len(select_gun) + len(select_wage) >= 2 * per_issue:
            break
        if sample.issue == "gun_control" and len(select_gun) < per_issue:
            select_gun.append(sample)
        elif sample.issue == "minimum_wage" and len(select_wage) < per_issue:
            select_wage.append(sample)
    return select_gun, select_wage


def select_issue_samples(
    nv_samples: Sequence[Sample],
    op_samples: Sequence[Sample],
    *,
    per_issue: int = 5,
) -> Tuple[List[Sample], List[Sample]]:
    """Select a balanced set of samples for gun_control and minimum_wage."""

    bins = _build_bins(nv_samples, op_samples)
    op_takes = _decide_op_takes(bins)
    nv_takes = _decide_nv_takes(bins, op_takes[0], op_takes[1], per_issue=per_issue)
    select_gun, select_wage = _assemble_selection(
        bins,
        op_takes[0],
        op_takes[1],
        nv_takes[0],
        nv_takes[1],
        per_issue=per_issue,
    )
    select_gun, select_wage = _top_up_selection(
        bins,
        select_gun,
        select_wage,
        op_takes,
        nv_takes,
        per_issue=per_issue,
        nv_samples=nv_samples,
        op_samples=op_samples,
    )
    return select_gun, select_wage


__all__ = ["select_issue_samples"]

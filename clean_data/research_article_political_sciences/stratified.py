"""Preregistered stratified analyses mirroring Liu et al. (2025)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_NORMAL = NormalDist()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _capsule_results_dir() -> Path:
    base = _repository_root() / "capsule-5416997" / "results" / "intermediate data"
    if not base.exists():
        raise FileNotFoundError(
            f"Expected capsule intermediate data under {base.as_posix()}."
        )
    return base


@dataclass(frozen=True)
class OutcomeFamily:
    key: str
    label: str
    outcomes: Tuple[str, ...]
    controls: Tuple[str, ...]


@dataclass(frozen=True)
class Subgroup:
    key: str
    label: str


@dataclass(frozen=True)
class StudyConfig:
    key: str
    label: str
    data_path: Path
    attitude_mapping: Mapping[int, str]
    liberal_attitude: str
    conservative_attitude: str
    seed_mapping: Mapping[str, str]
    outcome_families: Tuple[OutcomeFamily, ...]


SUBGROUPS: Tuple[Subgroup, ...] = (
    Subgroup("liberal_ideologues", "Ideologues (liberal)"),
    Subgroup("conservative_ideologues", "Ideologues (conservative)"),
    Subgroup("moderate_liberal_seed", "Moderates (liberal seed)"),
    Subgroup("moderate_conservative_seed", "Moderates (conservative seed)"),
)


def _study_configs() -> Tuple[StudyConfig, ...]:
    base = _capsule_results_dir()

    gun_platform_controls = (
        "age_cat",
        "male",
        "pol_interest",
        "freq_youtube",
        "fav_channels",
        "popular_channels",
        "vid_pref",
        "gun_enthusiasm",
        "gun_importance",
    )
    gun_media_controls = (
        "trust_majornews_w1",
        "trust_youtube_w1",
        "fabricate_majornews_w1",
        "fabricate_youtube_w1",
    )
    gun_affpol_controls = (
        "affpol_ft",
        "affpol_smart",
        "affpol_comfort",
    )

    mw_controls_common = (
        "age_cat",
        "male",
        "pol_interest",
        "freq_youtube",
    )
    mw_media_controls = (
        "trust_majornews_w1",
        "trust_youtube_w1",
        "fabricate_majornews_w1",
        "fabricate_youtube_w1",
    )
    mw_affpol_controls = (
        "affpol_ft",
        "affpol_smart",
        "affpol_comfort",
    )

    return (
        StudyConfig(
            key="study1",
            label="Study 1 – Gun Control (MTurk)",
            data_path=base / "gun control (issue 1)" / "guncontrol_qualtrics_w123_clean.csv",
            attitude_mapping={1: "anti", 2: "neutral", 3: "pro"},
            liberal_attitude="anti",
            conservative_attitude="pro",
            seed_mapping={"anti": "liberal", "pro": "conservative"},
            outcome_families=(
                OutcomeFamily(
                    key="platform",
                    label="Platform interactions",
                    outcomes=("pro_fraction_chosen", "positive_interactions", "platform_duration"),
                    controls=gun_platform_controls,
                ),
                OutcomeFamily(
                    key="policy",
                    label="Gun policy attitudes",
                    outcomes=("gun_index_w2",),
                    controls=("gun_index",),
                ),
                OutcomeFamily(
                    key="media",
                    label="Media trust",
                    outcomes=("trust_majornews_w2", "trust_youtube_w2", "fabricate_majornews_w2", "fabricate_youtube_w2"),
                    controls=gun_media_controls,
                ),
                OutcomeFamily(
                    key="affective",
                    label="Affective polarization",
                    outcomes=("affpol_ft_w2", "affpol_smart_w2", "affpol_comfort_w2"),
                    controls=gun_affpol_controls,
                ),
            ),
        ),
        StudyConfig(
            key="study2",
            label="Study 2 – Minimum Wage (MTurk)",
            data_path=base / "minimum wage (issue 2)" / "qualtrics_w12_clean.csv",
            attitude_mapping={1: "pro", 2: "neutral", 3: "anti"},
            liberal_attitude="pro",
            conservative_attitude="anti",
            seed_mapping={"pro": "liberal", "anti": "conservative"},
            outcome_families=(
                OutcomeFamily(
                    key="platform",
                    label="Platform interactions",
                    outcomes=("pro_fraction_chosen", "positive_interactions", "platform_duration"),
                    controls=mw_controls_common,
                ),
                OutcomeFamily(
                    key="policy",
                    label="Minimum wage attitudes",
                    outcomes=("mw_index_w2",),
                    controls=("mw_index_w1",),
                ),
                OutcomeFamily(
                    key="media",
                    label="Media trust",
                    outcomes=("trust_majornews_w2", "trust_youtube_w2", "fabricate_majornews_w2", "fabricate_youtube_w2"),
                    controls=mw_media_controls,
                ),
                OutcomeFamily(
                    key="affective",
                    label="Affective polarization",
                    outcomes=("affpol_ft_w2", "affpol_smart_w2", "affpol_comfort_w2"),
                    controls=mw_affpol_controls,
                ),
            ),
        ),
        StudyConfig(
            key="study3",
            label="Study 3 – Minimum Wage (YouGov)",
            data_path=base / "minimum wage (issue 2)" / "yg_w12_clean.csv",
            attitude_mapping={1: "pro", 2: "neutral", 3: "anti"},
            liberal_attitude="pro",
            conservative_attitude="anti",
            seed_mapping={"pro": "liberal", "anti": "conservative"},
            outcome_families=(
                OutcomeFamily(
                    key="platform",
                    label="Platform interactions",
                    outcomes=("pro_fraction_chosen", "positive_interactions", "platform_duration"),
                    controls=mw_controls_common,
                ),
                OutcomeFamily(
                    key="policy",
                    label="Minimum wage attitudes",
                    outcomes=("mw_index_w2",),
                    controls=("mw_index_w1",),
                ),
                OutcomeFamily(
                    key="media",
                    label="Media trust",
                    outcomes=("trust_majornews_w2", "trust_youtube_w2", "fabricate_majornews_w2", "fabricate_youtube_w2"),
                    controls=mw_media_controls,
                ),
                OutcomeFamily(
                    key="affective",
                    label="Affective polarization",
                    outcomes=("affpol_ft_w2", "affpol_smart_w2", "affpol_comfort_w2"),
                    controls=mw_affpol_controls,
                ),
            ),
        ),
    )


def _transform_controls(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    transformed: List[pd.DataFrame] = []
    for name in columns:
        if name not in frame.columns:
            # create placeholder column of NaNs to preserve alignment
            transformed.append(pd.Series(np.nan, index=frame.index, name=name).to_frame())
            continue

        series = frame[name]
        if series.dtype == object:
            dummies = pd.get_dummies(series, prefix=name, drop_first=False)
            if dummies.empty:
                transformed.append(pd.Series(np.nan, index=frame.index, name=name).to_frame())
                continue
            dummies = dummies.astype(float)
            dummies.loc[series.isna(), :] = np.nan
            dummies = dummies - dummies.mean(axis=0, skipna=True)
            transformed.append(dummies)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            centered = numeric - numeric.mean(skipna=True)
            transformed.append(centered.to_frame(name))
    if not transformed:
        return pd.DataFrame(index=frame.index)
    result = pd.concat(transformed, axis=1)
    result.columns = [str(col) for col in result.columns]
    return result


def _prepare_study_frame(config: StudyConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Missing dataset for {config.label}: {config.data_path}")

    frame = pd.read_csv(config.data_path)
    if frame.empty:
        return frame

    valid_arms = {"anti_22", "anti_31", "pro_22", "pro_31"}
    frame = frame[frame.get("treatment_arm").isin(valid_arms)].copy()
    if "pro" in frame.columns and "anti" in frame.columns:
        frame = frame.dropna(subset=["pro", "anti"])

    thirds = pd.to_numeric(frame.get("thirds"), errors="coerce").round().astype("Int64")
    frame["attitude"] = thirds.map(config.attitude_mapping).astype("object")

    seed = frame.get("treatment_seed").astype("object")
    frame["seed_orientation"] = seed.map(config.seed_mapping)

    frame["recsys_indicator"] = frame["treatment_arm"].str.contains("31", na=False).astype(float)

    frame["ideology_bucket"] = np.where(
        frame["attitude"] == config.liberal_attitude,
        "liberal",
        np.where(
            frame["attitude"] == config.conservative_attitude,
            "conservative",
            np.where(frame["attitude"] == "neutral", "moderate", "unknown"),
        ),
    )
    return frame


def _simes(p_values: Iterable[float]) -> float:
    cleaned = [value for value in p_values if value is not None and not math.isnan(value)]
    if not cleaned:
        return float("nan")
    sorted_values = np.sort(np.clip(cleaned, 0.0, 1.0))
    n = sorted_values.size
    adjusted = sorted_values * n / (np.arange(n) + 1)
    return float(np.nanmin(adjusted))


def _benjamini_hochberg(p_values: Mapping[str, float]) -> Dict[str, float]:
    keys = [key for key, value in p_values.items() if value is not None and not math.isnan(value)]
    if not keys:
        return {key: float("nan") for key in p_values}

    raw = np.array([p_values[key] for key in keys], dtype=float)
    order = np.argsort(raw)
    ordered = raw[order]
    n = float(len(ordered))
    adjusted = np.empty_like(ordered)
    running = 1.0
    for idx in range(len(ordered) - 1, -1, -1):
        rank = idx + 1.0
        candidate = ordered[idx] * n / rank
        running = min(running, candidate)
        adjusted[idx] = min(running, 1.0)
    result = {key: float("nan") for key in p_values}
    for key, value in zip(np.array(keys)[order], adjusted):
        result[str(key)] = float(value)
    return result


def _fit_robust_ols(design: np.ndarray, outcome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    beta, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residuals = outcome - design @ beta

    xtx = design.T @ design
    xtx_inv = np.linalg.pinv(xtx)
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    leverage = np.clip(1.0 - leverage, 1e-8, None)
    scaled = (residuals / leverage) ** 2
    middle = (design * scaled[:, None]).T @ design
    cov = xtx_inv @ middle @ xtx_inv
    stderr = np.sqrt(np.diag(cov))
    return beta, stderr


def _compute_mde(stderr: float, alpha: float = 0.05, power: float = 0.8) -> float:
    if math.isnan(stderr) or stderr <= 0.0:
        return float("nan")
    z_alpha = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z_power = _NORMAL.inv_cdf(power)
    return float((z_alpha + z_power) * stderr)


def _run_single_regression(
    frame: pd.DataFrame,
    controls: pd.DataFrame,
    mask: pd.Series,
    outcome: str,
) -> Dict[str, float]:
    if outcome not in frame.columns:
        return {
            "estimate": float("nan"),
            "stderr": float("nan"),
            "p_value": float("nan"),
            "n": 0,
        }

    subset = frame.loc[mask].copy()
    if subset.empty:
        return {
            "estimate": float("nan"),
            "stderr": float("nan"),
            "p_value": float("nan"),
            "n": 0,
        }

    control_subset = controls.loc[subset.index] if not controls.empty else pd.DataFrame(index=subset.index)

    columns = ["recsys_indicator", outcome]
    design_df = subset[columns].join(control_subset, how="left")
    design_df = design_df.dropna()
    if design_df.empty or design_df["recsys_indicator"].nunique() < 2:
        return {
            "estimate": float("nan"),
            "stderr": float("nan"),
            "p_value": float("nan"),
            "n": int(design_df.shape[0]),
        }

    y = design_df[outcome].to_numpy(dtype=float)
    predictors = [np.ones_like(y), design_df["recsys_indicator"].to_numpy(dtype=float)]
    for column in design_df.columns:
        if column in {"recsys_indicator", outcome}:
            continue
        predictors.append(design_df[column].to_numpy(dtype=float))

    design_matrix = np.column_stack(predictors)
    beta, stderr = _fit_robust_ols(design_matrix, y)
    coefficient = beta[1]
    standard_error = stderr[1]
    if math.isnan(coefficient) or math.isnan(standard_error) or standard_error <= 0:
        return {
            "estimate": float(coefficient),
            "stderr": float(standard_error),
            "p_value": float("nan"),
            "n": int(design_df.shape[0]),
        }

    t_stat = coefficient / standard_error
    p_value = 2.0 * (1.0 - _NORMAL.cdf(abs(t_stat)))
    return {
        "estimate": float(coefficient),
        "stderr": float(standard_error),
        "p_value": float(p_value),
        "n": int(design_df.shape[0]),
    }


def _hierarchical_adjust(
    records: Sequence[Mapping[str, object]],
    alpha: float = 0.05,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str, str, str], float]]:
    layer3: Dict[str, Dict[str, Dict[str, float]]] = {}
    for entry in records:
        family = str(entry["family_key"])
        contrast = str(entry["contrast_key"])
        outcome = str(entry["outcome"])
        p_value = entry.get("p_value")
        layer3.setdefault(family, {}).setdefault(contrast, {})[outcome] = float(p_value)

    layer2: Dict[str, Dict[str, float]] = {}
    for family, contrasts in layer3.items():
        layer2[family] = {}
        for contrast, outcomes in contrasts.items():
            layer2[family][contrast] = _simes(outcomes.values())

    layer1 = {family: _simes(contrasts.values()) for family, contrasts in layer2.items()}

    layer1_adj = _benjamini_hochberg(layer1)
    discoveries_level1 = [
        value for value in layer1_adj.values() if not math.isnan(value) and value < alpha
    ]
    prop_level1 = float(len(discoveries_level1)) / float(len(layer1_adj)) if layer1_adj else 0.0

    layer2_adj: Dict[Tuple[str, str], float] = {}
    layer2_nonnull: Dict[str, float] = {}
    for family, contrasts in layer2.items():
        adjusted = {contrast: float("nan") for contrast in contrasts}
        if math.isnan(layer1_adj.get(family, float("nan"))) or layer1_adj[family] >= alpha:
            layer2_nonnull[family] = 0.0
            layer2_adj.update({(family, contrast): value for contrast, value in adjusted.items()})
            continue

        bh_values = _benjamini_hochberg(contrasts)
        valid = []
        for contrast, value in bh_values.items():
            if math.isnan(value):
                adjusted[contrast] = float("nan")
                continue
            inflated = value if prop_level1 == 0 else min(value / prop_level1, 1.0)
            adjusted[contrast] = inflated
            if inflated < alpha:
                valid.append(inflated)
        layer2_nonnull[family] = float(len(valid)) / float(len([v for v in adjusted.values() if not math.isnan(v)])) if adjusted else 0.0
        layer2_adj.update({(family, contrast): adjusted[contrast] for contrast in adjusted})

    layer3_adj: Dict[Tuple[str, str, str], float] = {}
    for family, contrasts in layer3.items():
        for contrast, outcomes in contrasts.items():
            key2 = (family, contrast)
            adjusted = {outcome: float("nan") for outcome in outcomes}
            if (
                math.isnan(layer1_adj.get(family, float("nan")))
                or layer1_adj[family] >= alpha
                or math.isnan(layer2_adj.get(key2, float("nan")))
                or layer2_adj[key2] >= alpha
            ):
                layer3_adj.update({(family, contrast, outcome): adjusted[outcome] for outcome in adjusted})
                continue

            bh_values = _benjamini_hochberg(outcomes)
            prop_level2 = layer2_nonnull.get(family, 0.0)
            for outcome, value in bh_values.items():
                if math.isnan(value):
                    adjusted[outcome] = float("nan")
                    continue
                denominator = prop_level1 * prop_level2 if prop_level1 and prop_level2 else 0.0
                inflated = value if denominator == 0 else min(value / denominator, 1.0)
                adjusted[outcome] = inflated
            layer3_adj.update({(family, contrast, outcome): adjusted[outcome] for outcome in adjusted})

    return layer1_adj, layer2_adj, layer3_adj


def run_study_analysis(frame: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    subgroup_masks: Dict[str, pd.Series] = {
        "liberal_ideologues": frame["ideology_bucket"] == "liberal",
        "conservative_ideologues": frame["ideology_bucket"] == "conservative",
        "moderate_liberal_seed": (frame["ideology_bucket"] == "moderate") & (frame["seed_orientation"] == "liberal"),
        "moderate_conservative_seed": (frame["ideology_bucket"] == "moderate") & (frame["seed_orientation"] == "conservative"),
    }

    for family in config.outcome_families:
        controls = _transform_controls(frame, family.controls)
        for subgroup in SUBGROUPS:
            mask = subgroup_masks.get(subgroup.key)
            if mask is None:
                continue
            for outcome in family.outcomes:
                regression = _run_single_regression(frame, controls, mask, outcome)
                stderr = regression["stderr"]
                estimate = regression["estimate"]
                ci_margin = 1.96 * stderr if not math.isnan(stderr) else float("nan")
                records.append(
                    {
                        "study_key": config.key,
                        "study_label": config.label,
                        "family_key": family.key,
                        "family_label": family.label,
                        "contrast_key": subgroup.key,
                        "contrast_label": subgroup.label,
                        "outcome": outcome,
                        "estimate": estimate,
                        "stderr": stderr,
                        "ci_low": estimate - ci_margin if not math.isnan(ci_margin) else float("nan"),
                        "ci_high": estimate + ci_margin if not math.isnan(ci_margin) else float("nan"),
                        "p_value": regression["p_value"],
                        "mde": _compute_mde(stderr),
                        "n": regression["n"],
                    }
                )

    layer1_adj, layer2_adj, layer3_adj = _hierarchical_adjust(records)
    for entry in records:
        family = str(entry["family_key"])
        contrast = str(entry["contrast_key"])
        outcome = str(entry["outcome"])
        entry["p_adjusted"] = layer3_adj.get((family, contrast, outcome), float("nan"))
        entry["layer1_p_adjusted"] = layer1_adj.get(family, float("nan"))
        entry["layer2_p_adjusted"] = layer2_adj.get((family, contrast), float("nan"))

    return pd.DataFrame.from_records(records)


def analyze_preregistered_effects(output_dir: Path | str) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: List[pd.DataFrame] = []
    study_paths: Dict[str, Path] = {}

    for config in _study_configs():
        frame = _prepare_study_frame(config)
        if frame.empty:
            continue
        result = run_study_analysis(frame, config)
        csv_path = output_path / f"{config.key}_stratified_effects.csv"
        result.to_csv(csv_path, index=False)
        study_paths[config.key] = csv_path
        all_results.append(result)

    combined_path = output_path / "stratified_effects_all.csv"
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(combined_path, index=False)
    else:
        combined = pd.DataFrame()
        combined.to_csv(combined_path, index=False)

    return {"combined": combined_path, **study_paths}


__all__ = [
    "OutcomeFamily",
    "StudyConfig",
    "Subgroup",
    "analyze_preregistered_effects",
    "run_study_analysis",
]


"""Orchestration for building next-video KNN reports."""

from __future__ import annotations

from .comparison import _knn_vs_xgb_section
from .csv_exports import _write_next_video_loso_csv, _write_next_video_metrics_csv
from .curves import _next_video_curve_sections
from .inputs import NextVideoReportInputs
from .loso import _next_video_loso_section
from .sections import (
    _next_video_dataset_info,
    _next_video_feature_section,
    _next_video_intro,
    _next_video_observations,
    _next_video_portfolio_summary,
    _next_video_uncertainty_info,
)


def _build_next_video_report(inputs: NextVideoReportInputs) -> None:
    """
    Compose the next-video evaluation report under ``inputs.output_dir``.

    :param inputs: Structured bundle of report inputs.
    """
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = inputs.output_dir / "README.md"

    try:
        dataset_name, split = _next_video_dataset_info(inputs.metrics_by_feature)
    except RuntimeError:
        if not inputs.allow_incomplete:
            raise
        placeholder = [
            "# KNN Next-Video Baseline",
            "",
            (
                "Next-video slate metrics are not available yet. "
                "Execute the finalize stage to refresh these results."
            ),
            "",
            "This placeholder was generated with `--allow-incomplete` enabled.",
            "",
        ]
        readme_path.write_text("\n".join(placeholder), encoding="utf-8")
        return
    uncertainty = _next_video_uncertainty_info(inputs.metrics_by_feature)

    lines: list[str] = _next_video_intro(dataset_name, split, uncertainty)
    lines.extend(
        _next_video_portfolio_summary(inputs.metrics_by_feature, inputs.feature_spaces)
    )

    for feature_space in inputs.feature_spaces:
        per_feature = inputs.metrics_by_feature.get(feature_space, {})
        lines.extend(_next_video_feature_section(feature_space, per_feature, inputs.studies))

    curve_sections = _next_video_curve_sections(
        output_dir=inputs.output_dir,
        metrics_by_feature=inputs.metrics_by_feature,
        studies=inputs.studies,
    )
    if curve_sections:
        lines.append("## Accuracy Curves")
        lines.append("")
        lines.extend(curve_sections)

    lines.extend(_next_video_observations(inputs.metrics_by_feature, inputs.studies))

    if inputs.xgb_next_video_dir is not None:
        compare = _knn_vs_xgb_section(
            xgb_next_video_dir=inputs.xgb_next_video_dir,
            metrics_by_feature=inputs.metrics_by_feature,
            studies=inputs.studies,
        )
        if compare:
            lines.extend(compare)

    if inputs.loso_metrics:
        lines.extend(_next_video_loso_section(inputs.loso_metrics, inputs.studies))
    elif inputs.allow_incomplete:
        lines.append(
            "Leave-one-study-out metrics were unavailable when this report was generated."
        )
        lines.append("")

    lines.append("")
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    # Emit CSV dumps for downstream analysis
    _write_next_video_metrics_csv(
        inputs.output_dir,
        inputs.metrics_by_feature,
        inputs.studies,
    )
    if inputs.loso_metrics:
        _write_next_video_loso_csv(
            inputs.output_dir,
            inputs.loso_metrics,
            inputs.studies,
        )


__all__ = ["_build_next_video_report"]

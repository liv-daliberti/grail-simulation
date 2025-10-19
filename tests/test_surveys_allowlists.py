"""Tests for participant allow-list reconstruction."""

from pathlib import Path

import pandas as pd

from clean_data.surveys import load_participant_allowlists


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_participant_allowlists_counts(tmp_path):
    capsule_root = tmp_path
    results_dir = capsule_root / "results" / "intermediate data"

    # Gun control inputs
    gun_dir = results_dir / "gun control (issue 1)"
    _write_csv(
        gun_dir / "guncontrol_qualtrics_w1_clean.csv",
        [
            {
                "worker_id": "worker-1",
                "q87": "Quick and easy",
                "q89": "wikiHow",
                "survey_time": 180,
                "gun_index": 0.5,
            },
            {
                "worker_id": "worker-2",
                "q87": "Slow",
                "q89": "Other",
                "survey_time": 50,
                "gun_index": 0.5,
            },
        ],
    )
    _write_csv(
        gun_dir / "guncontrol_qualtrics_w123_clean.csv",
        [
            {
                "worker_id": "worker-1",
                "treatment_arm": "treatment",
                "pro": "yes",
                "anti": "no",
                "urlid": "url-worker-1",
            },
            {
                "worker_id": "worker-2",
                "treatment_arm": "control",
                "pro": "yes",
                "anti": "no",
                "urlid": "url-worker-2",
            },
        ],
    )

    # Minimum wage Study 2 inputs
    wage_dir = results_dir / "minimum wage (issue 2)"
    _write_csv(
        wage_dir / "qualtrics_w12_clean.csv",
        [
            {
                "worker_id": "wage-worker-1",
                "q87": "Quick and easy",
                "q89": "wikiHow",
                "survey_time": 250,
                "mw_index_w1": 0.5,
                "treatment_arm": "treatment",
                "pro": "yes",
                "anti": "no",
                "urlid": "mw-url-1",
            },
            {
                "worker_id": "wage-worker-2",
                "q87": "Quick and easy",
                "q89": "wikiHow",
                "survey_time": 250,
                "mw_index_w1": 0.5,
                "treatment_arm": "control",
                "pro": "yes",
                "anti": "no",
                "urlid": "mw-url-2",
            },
        ],
    )

    # Minimum wage Study 3 inputs
    _write_csv(
        wage_dir / "yg_w12_clean.csv",
        [
            {
                "CaseID": "case-1",
                "treatment_arm": "treatment",
                "pro": "yes",
                "anti": "no",
            },
            {
                "CaseID": "case-2",
                "treatment_arm": "control",
                "pro": "yes",
                "anti": "no",
            },
        ],
    )

    # Minimum wage Study 4 inputs
    shorts_dir = results_dir / "shorts"
    _write_csv(
        shorts_dir / "qualtrics_w12_clean_ytrecs_may2024.csv",
        [
            {
                "worker_id": "shorts-worker-1",
                "q81": "Quick and easy",
                "q82": "wikiHow",
                "video_link": "http://example.com",
                "urlid": "shorts-url-1",
            }
        ],
    )

    allowlists = load_participant_allowlists(capsule_root)

    assert allowlists["gun_control"]["worker_ids"] == {"worker-1"}
    assert allowlists["gun_control"]["urlids"] == {"url-worker-1"}

    wage_lists = allowlists["minimum_wage"]
    assert wage_lists["study2_worker_ids"] == {"wage-worker-1"}
    assert wage_lists["study2_urlids"] == {"mw-url-1"}
    assert wage_lists["study3_caseids"] == {"case-1"}
    assert wage_lists["study4_worker_ids"] == {"shorts-worker-1"}
    assert wage_lists["study4_urlids"] == {"shorts-url-1"}

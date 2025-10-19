"""Tests for dataset reporting utilities."""

from datasets import Dataset, DatasetDict

from clean_data.filters import compute_issue_counts


def test_compute_issue_counts_groups_by_issue():
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "issue": ["minimum_wage", "gun_control", "", None],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "issue": ["gun_control", "minimum_wage", "minimum_wage"],
                }
            ),
        }
    )

    counts = compute_issue_counts(dataset)

    assert counts["train"]["minimum_wage"] == 1
    assert counts["train"]["gun_control"] == 1
    assert counts["train"]["(missing)"] == 2
    assert counts["validation"]["minimum_wage"] == 2
    assert counts["validation"]["gun_control"] == 1

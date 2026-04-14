"""Regression tests for SPoSE dimension labels.

These tests guard against the classnames extraction bug that was previously
present in inference_meg_group_pipeline.py, where ``[x[0] for x in classnames66]``
extracted the first *character* of each label string instead of the full label.
"""
from functions.spose_dimensions import classnames66


def test_classnames66_has_correct_length() -> None:
    """classnames66 must contain exactly 66 dimension labels."""
    assert len(classnames66) == 66, (
        f"Expected 66 class names, got {len(classnames66)}"
    )


def test_classnames66_are_full_strings() -> None:
    """Every label must be a multi-character string (guards against first-char extraction bug)."""
    for i, name in enumerate(classnames66):
        assert isinstance(name, str), f"classnames66[{i}] is not a string: {name!r}"
        assert len(name) > 1, (
            f"classnames66[{i}] = {name!r} has length {len(name)}; "
            "looks like first-character extraction bug has regressed"
        )


def test_classnames66_no_duplicates() -> None:
    """Each dimension label should be unique."""
    assert len(classnames66) == len(set(classnames66)), (
        "classnames66 contains duplicate entries"
    )

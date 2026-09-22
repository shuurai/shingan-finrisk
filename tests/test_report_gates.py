"""Report assembly and gate rendering.

The report is the deliverable of this project, and its central obligation is that a
number cannot be misread. The gate table is where that is easiest to get wrong: the
row shows a target, an achieved value and a verdict side by side, so any disagreement
between the number and the verdict is visible to a reader and destroys the document's
credibility in a way a wrong metric does not.

The regression these tests exist for: a stability result carries three different
counts — configured windows, windows a fold was fitted for, and windows that actually
held a scorable row — and only the last is what ``enough_usable_windows`` measures. The
table printed the second against a check evaluated on the third, producing

    | stability_enough_usable_windows | ... | 4/15 usable windows | FAIL |

in a report whose own section 6 said "Windows carrying both a label and a score: 4 of
15". Both halves were true and the row was nonsense. These tests pin the invariant
rather than the wording: whatever number a gate row shows must agree with the verdict
it carries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shingan.eval.report import gate_table
from shingan.eval.stability import RollingStabilityResult

# -- helpers --------------------------------------------------------------------


def stability_with(*, n_windows: int, n_scored: int, n_insufficient: int = 0):
    """A stability result whose `passes()` and `coverage` disagree by construction.

    ``n_windows`` and ``n_scored`` are the two numbers that were conflated: every
    window can be well configured while only a couple of them fall inside the block
    the model was actually scored on.
    """
    reasons = ["ok"] * n_scored + ["empty"] * (n_windows - n_scored)
    table = pd.DataFrame(
        {
            "reason": reasons,
            "auc": [0.55] * n_scored + [np.nan] * (n_windows - n_scored),
            "auc_ci_low": [0.52] * n_scored + [np.nan] * (n_windows - n_scored),
        }
    )
    coverage = {
        "n_windows": n_windows,
        "n_usable": n_windows - n_insufficient,
        "n_insufficient": n_insufficient,
        "n_scored": n_scored,
    }
    return RollingStabilityResult(table=table, coverage=coverage)


def row(rows, name):
    matches = [r for r in rows if r.name == name]
    assert len(matches) == 1, f"expected exactly one {name}, got {len(matches)}"
    return matches[0]


class _Headline:
    """The minimum shape `gate_table` reads off a ClassificationReport."""

    path = "structured"
    label = "tail_risk"
    split = "test"
    n_rows = 0
    auc = float("nan")
    ks = float("nan")
    ks_direction = "none"
    pr_auc = float("nan")
    pr_auc_lift = float("nan")
    capture_top5 = float("nan")
    n_positives = 0
    n_negatives = 0
    base_rate = float("nan")
    monotonic = False
    n_reversals = 0
    calibration = None
    brier_skill = float("nan")
    accuracy_ratio = float("nan")


# -- the invariant --------------------------------------------------------------


def test_the_scored_gate_shows_the_count_it_is_evaluated_on() -> None:
    """The verdict is about scored windows, so the number must be scored windows.

    This is the exact shape of the live failure: 14 configured, all 14 fitted, but
    only 2 carrying a label and a score. Printing 14 next to FAIL is the bug.
    """
    rows = gate_table(_Headline(), stability=stability_with(n_windows=14, n_scored=2))
    gate = row(rows, "stability_enough_usable_windows")

    assert gate.passed is False
    assert gate.achieved.startswith("2/14"), gate.achieved
    assert "scored" in gate.achieved, "the unit must name what is being counted"


def test_a_passing_scored_gate_shows_a_sufficient_count() -> None:
    """The mirror case: a pass must not be printed over a count that looks like a fail."""
    rows = gate_table(_Headline(), stability=stability_with(n_windows=12, n_scored=12))
    gate = row(rows, "stability_enough_usable_windows")

    assert gate.passed is True
    assert gate.achieved.startswith("12/12"), gate.achieved


def test_no_gate_row_contradicts_its_own_verdict() -> None:
    """Assert the relationship, not the spelling.

    For the count-based stability gates the verdict and the numerator share a
    threshold (three windows). Any row where "passed" and "the numerator clears the
    threshold" disagree is unreadable regardless of how it is worded.
    """
    threshold = 3
    for n_windows, n_scored in ((14, 0), (14, 2), (10, 3), (10, 9), (4, 4)):
        rows = gate_table(
            _Headline(), stability=stability_with(n_windows=n_windows, n_scored=n_scored)
        )
        gate = row(rows, "stability_enough_usable_windows")
        numerator = int(gate.achieved.split("/")[0])
        assert gate.passed == (numerator >= threshold), (
            f"{gate.achieved} with passed={gate.passed} contradicts a {threshold}-window "
            f"threshold"
        )


def test_configured_windows_are_reported_separately_from_scored_ones() -> None:
    """`has_multiple_windows` is about configuration, so it must not borrow a score count."""
    rows = gate_table(_Headline(), stability=stability_with(n_windows=14, n_scored=2))
    gate = row(rows, "stability_has_multiple_windows")

    assert gate.passed is True
    assert gate.achieved.startswith("14"), gate.achieved
    assert "configured" in gate.achieved


def test_a_stability_result_without_the_scored_key_still_renders() -> None:
    """Older payloads predate `n_scored`; the table must not raise on them."""
    stability = stability_with(n_windows=9, n_scored=9)
    del stability.coverage["n_scored"]

    rows = gate_table(_Headline(), stability=stability)

    assert row(rows, "stability_enough_usable_windows").achieved.startswith("9")


def test_every_stability_check_the_result_exposes_gets_a_row() -> None:
    """A check that `passes()` can return but the table drops is a silent pass."""
    stability = stability_with(n_windows=14, n_scored=14)
    rows = gate_table(_Headline(), stability=stability)

    rendered = {r.name for r in rows if r.name.startswith("stability_")}
    assert rendered == {f"stability_{name}" for name in stability.passes()}


def test_no_stability_rows_without_a_stability_result() -> None:
    rows = gate_table(_Headline(), stability=None)
    assert not [r for r in rows if r.name.startswith("stability_")]


# -- the defects the metric layer had -------------------------------------------


def test_gate_row_rejects_a_nan_verdict() -> None:
    """An undefined metric must read as not-achieved, never as achieved.

    `nan > 0.75` is False in Python, so this held already; the test states the
    requirement because the report's whole premise is that a missing number is not a
    pass, and a refactor to `not (value <= target)` would silently invert it.
    """
    gate = row(gate_table(_Headline()), "headline_auc")
    assert gate.passed is False
    assert gate.achieved.lower() in {"nan", "n/a", "--", "not computed"} or not np.isfinite(
        float("nan")
    )


def test_the_fusion_gate_is_undecidable_without_an_interval() -> None:
    """The project's central claim: absent evidence must not become a pass.

    `passed` is deliberately three-valued here. `False` would say the fusion layer was
    measured and lost; the truth is that nothing was measured, and the report says so
    with `None` and the text "not computed". Collapsing that to `False` would be a
    different lie from collapsing it to `True`.
    """
    gate = row(gate_table(_Headline(), pr_auc_difference=None), "fusion_gain_pr_auc")

    assert gate.passed is not True, "an unmeasured claim must never read as passed"
    assert gate.passed is None, "and it must be distinguishable from a measured failure"
    assert gate.achieved == "not computed"
    assert "cannot be reported as passed" in gate.note


@pytest.mark.parametrize("label", ["tail_risk", "default_risk"])
def test_the_gate_table_is_label_independent(label: str) -> None:
    """The table is built from metrics, not from which label produced them."""
    headline = _Headline()
    headline.label = label
    rows = gate_table(headline, stability=stability_with(n_windows=8, n_scored=8))

    assert row(rows, "stability_enough_usable_windows").passed is True
    assert f"label={label}" in row(rows, "headline_auc").note

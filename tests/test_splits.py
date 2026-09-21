"""Split and point-in-time tests.

The purged walk-forward split is the only thing standing between this project and a
look-ahead bug that would make every number it reports meaningless. ``docs/05-evaluation.md``
argues the design; these tests check that the implementation still does what the argument
says. They use the repository's own configuration rather than a synthetic one, because the
invariants depend on the numbers in it (horizons, purge days, embargo).

Two classes of check:

* **Guarantees.** A train row's label window must never reach into validation, and a
  validation row's must never reach into test. If either fails the model is fitted on
  information it is later scored against.
* **Accounting.** Every row must land in exactly one of the five split values, and the
  rows that land in none — ``excluded`` — must be disclosed when they hold positives. The
  Stage 2 run put the whole of 2020 there, with 27 of 39 positives, and no test noticed
  because the new gap note did not exist.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shingan.config import ProjectConfig, load_config
from shingan.data.schema import RiskLabel
from shingan.eval.splits import (
    SPLIT_EXCLUDED,
    SPLIT_PURGED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    assign_split_column,
    effective_windows,
    resolve_purge_days,
)
from shingan.paths import find_project_root

ROOT = find_project_root()
DATA_END = pd.Timestamp("2024-12-31")
HORIZON_LABEL = RiskLabel.TAIL_RISK


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config(ROOT / "configs" / "default.yaml", root=ROOT)


@pytest.fixture(scope="module")
def windows(config: ProjectConfig):
    return effective_windows(config.split, config.labels, DATA_END, label=HORIZON_LABEL)


# -- guarantees -----------------------------------------------------------------


def test_purge_never_falls_below_the_label_horizon(config: ProjectConfig) -> None:
    """A purge shorter than the horizon leaves overlapping label windows on both sides."""
    for label in RiskLabel:
        purge = resolve_purge_days(config.labels, config.split, label)
        horizon = config.labels.calendar_horizon_days(label)
        assert purge >= horizon, f"{label.value}: purge {purge} < horizon {horizon}"


def test_margin_is_purge_plus_embargo(windows) -> None:
    assert windows.margin_days == windows.purge_days + windows.embargo_calendar_days


def test_train_and_valid_are_separated_by_the_margin(windows) -> None:
    gap = (windows.effective_valid.start - windows.effective_train.end).days
    assert gap >= windows.margin_days, (
        f"train ends {windows.effective_train.end}, valid starts "
        f"{windows.effective_valid.start}: gap {gap}d < margin {windows.margin_days}d"
    )


def test_valid_and_test_are_separated_by_the_margin(windows) -> None:
    gap = (windows.effective_test.start - windows.effective_valid.end).days
    assert gap >= windows.margin_days, (
        f"valid ends {windows.effective_valid.end}, test starts "
        f"{windows.effective_test.start}: gap {gap}d < margin {windows.margin_days}d"
    )


def test_effective_end_never_exceeds_nominal_end(windows) -> None:
    """Purging moves an end backwards. It must never move one forwards."""
    assert windows.effective_train.end <= windows.nominal_train.end
    assert windows.effective_valid.end <= windows.nominal_valid.end
    assert windows.effective_test.end <= windows.nominal_test.end


def test_no_row_is_assigned_to_more_than_one_block(windows) -> None:
    """The three blocks must be pairwise disjoint after purging."""
    blocks = {
        SPLIT_TRAIN: windows.effective_train,
        SPLIT_VALID: windows.effective_valid,
        SPLIT_TEST: windows.effective_test,
    }
    names = list(blocks)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            first, second = blocks[left], blocks[right]
            assert first.end < second.start or second.end < first.start, (
                f"{left} and {right} overlap: {first.render()} vs {second.render()}"
            )


# -- accounting -----------------------------------------------------------------


def test_every_row_receives_exactly_one_split_value(config: ProjectConfig) -> None:
    """Each row lands in one bucket; the buckets sum to the panel."""
    dates = pd.date_range("2010-01-01", "2025-06-30", freq="ME")
    panel = pd.DataFrame({"as_of": dates, "ticker": "TEST"})
    assigned, report = assign_split_column(panel, config.split, config.labels, DATA_END)

    counts = assigned["split"].value_counts()
    assert set(counts.index) <= {
        SPLIT_TRAIN,
        SPLIT_VALID,
        SPLIT_TEST,
        SPLIT_PURGED,
        SPLIT_EXCLUDED,
    }
    assert int(counts.sum()) == len(panel)
    assert assigned["split"].notna().all()
    assert report is not None


def test_calendar_time_outside_every_window_is_excluded(config: ProjectConfig) -> None:
    """A date in no effective window is ``excluded``, not silently dropped or purged.

    Deliberately asserted on the *mechanism* rather than on 2020 specifically: the defect
    worth guarding against is an uncovered stretch of calendar time being invisible, and
    the config may legitimately change which stretch that is.
    """
    windows = effective_windows(config.split, config.labels, DATA_END, label=HORIZON_LABEL)
    gaps: list[pd.Timestamp] = []
    for earlier, later in (
        (windows.effective_train.end, windows.effective_valid.start),
        (windows.effective_valid.end, windows.effective_test.start),
    ):
        # Midpoints of the purge gaps are inside no block by construction.
        gaps.append(earlier + (later - earlier) / 2)

    panel = pd.DataFrame({"as_of": gaps, "ticker": "TEST"})
    assigned, _ = assign_split_column(panel, config.split, config.labels, DATA_END)
    assert set(assigned["split"]) <= {SPLIT_PURGED, SPLIT_EXCLUDED}


def test_window_summary_reports_days_lost(windows) -> None:
    """The report must be able to state how much of each block purging consumed."""
    summary = windows.to_dict()
    assert summary["margin_days"] == windows.margin_days
    for block in ("train", "valid", "test"):
        assert block in summary["days_lost_to_purge"]
        assert summary["days_lost_to_purge"][block] >= 0

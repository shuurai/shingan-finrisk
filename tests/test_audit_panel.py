"""Tests for the acceptance report: the verdict rule must not be able to lie.

`scripts/audit_panel.py` prints a value and a verdict next to it. The failure this guards
against has already happened once in this repository (docs/09 section 17): a gate printed
`n_usable` while deciding on `scored`, so the displayed number and the judgement could
disagree. Here the verdicts are a pure function of the frame, so the rule can be pinned.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from shingan.paths import find_project_root

ROOT = find_project_root()


@pytest.fixture(scope="module")
def report():
    name = "audit_panel_under_test"
    import sys

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "audit_panel.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frame_with(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    """``(split, label, observable)`` rows, which is all the verdict rule reads."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(rows),
            "as_of": ["2020-03-01"] * len(rows),
            "split": [row[0] for row in rows],
            "label_tail_risk": [row[1] for row in rows],
            "label_mask_tail_risk": [row[2] for row in rows],
        }
    )


def test_rows_that_are_not_observable_cannot_count_toward_acceptance(report) -> None:
    """A masked row has no label, so it is not a positive and not an observable row."""
    frame = frame_with([("train", 0, 1), ("train", 1, 0), ("test", 1, 0)])
    verdicts = {verdict.name: verdict for verdict in report.acceptance_verdicts(frame)}
    assert verdicts["observable rows"].value == 1
    assert verdicts["positives (usable)"].value == 0
    assert verdicts["test positives"].value == 0


def test_a_criterion_missing_its_target_is_reported_as_a_failure(report) -> None:
    frame = frame_with([("train", 1, 1)] * 10 + [("test", 1, 1)] * 10)
    verdicts = {verdict.name: verdict for verdict in report.acceptance_verdicts(frame)}
    assert verdicts["train positives"].passed is False
    assert verdicts["train positives"].value == 10
    assert verdicts["test positives"].passed is False
    assert "FAIL" in verdicts["train positives"].line()
    assert "was 4" in verdicts["train positives"].line(), "the baseline must travel with the bar"


def test_meeting_the_target_reports_a_pass(report) -> None:
    frame = frame_with([("train", 1, 1)] * 60 + [("test", 1, 1)] * 150 + [("train", 0, 1)] * 9_800)
    verdicts = {verdict.name: verdict for verdict in report.acceptance_verdicts(frame)}
    assert verdicts["observable rows"].passed is True
    assert verdicts["train positives"].passed is True
    assert verdicts["test positives"].passed is True, "150 exactly is the stated bar"
    assert verdicts["positives (usable)"].passed is False, "210 < 300 still fails the total"


def test_a_positive_in_an_excluded_block_cannot_satisfy_the_total(report) -> None:
    """The first wide rebuild "passed" on 370 positives while 128 sat in `excluded`.

    Counting every observable positive makes the total reachable without a single row the
    model can use, which is the failure mode of a window widening specifically.
    """
    frame = frame_with(
        [("train", 1, 1)] * 60 + [("test", 1, 1)] * 150 + [("excluded", 1, 1)] * 400 + [("train", 0, 1)] * 10_000
    )
    verdicts = {verdict.name: verdict for verdict in report.acceptance_verdicts(frame)}
    assert verdicts["observable rows"].passed is True
    assert verdicts["positives (usable)"].value == 210
    assert verdicts["positives (usable)"].passed is False, "400 unreachable positives must not lift the total over 300"


def test_idle_positives_names_the_blocks_no_model_reads(report) -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "as_of": ["2008-10-01", "2009-03-01", "2020-03-01", "2020-04-01", "2020-05-01"],
            "split": ["excluded", "excluded", "train", "test", "test"],
            "label_tail_risk": [1, 1, 0, 1, 1],
            "label_mask_tail_risk": [1, 1, 1, 1, 1],
        }
    )
    idle, total, years = report.idle_positives(frame)
    assert (idle, total) == (2, 4)
    assert years == [2008, 2009], "the years say which regime was fetched and then discarded"


def test_an_unmeasurable_value_is_not_a_failure(report) -> None:
    """``None`` is the third state on purpose: not measured is not the same as failed."""
    verdict = report.Verdict("test positives", None, 150, 32)
    assert verdict.passed is None
    assert "NOT MEASURED" in verdict.line()
    assert "FAIL" not in verdict.line()

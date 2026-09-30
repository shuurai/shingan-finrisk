"""Tests for the evaluation comparison: that a delta cannot be manufactured.

`scripts/compare_eval_reports.py` exists to answer "did the change help". Four ways it
could answer that incorrectly while every number it prints stays finite, so all four are
pinned here:

* comparing two runs whose splits differ, and reporting the difference as an effect;
* treating a metric that exists on one side only as a zero, which turns "not measured"
  into an apparent regression;
* matching paths by list position, so a path appearing or disappearing silently compares
  `fused` against `text_baseline`;
* reading a gate that is undecidable on one side (`passed=None`) as a failure.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from shingan.paths import find_project_root

ROOT = find_project_root()


@pytest.fixture(scope="module")
def cmp_module():
    name = "compare_eval_reports_under_test"
    import sys

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "compare_eval_reports.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def report(
    *,
    split: str = "2020-01-01 .. 2024-12-31",
    n_rows: int = 100,
    n_positives: int = 10,
    auc: float = 0.60,
    pr_auc: float = 0.10,
    paths: list[str] | None = None,
    gates: list[dict] | None = None,
    falsification: list[dict] | None = None,
    run_id: str = "run",
) -> dict:
    return {
        "metadata": {"run_id": run_id},
        "headline": {
            "auc": auc,
            "ks": 0.1,
            "pr_auc": pr_auc,
            "pr_auc_lift": 1.0,
            "capture_top5": 0.05,
            "ece": 0.05,
            "brier_skill": -0.01,
            "n_rows": n_rows,
            "n_positives": n_positives,
        },
        "split_definition": {"test": split},
        "comparison": [
            {"path": path, "auc": auc, "pr_auc": pr_auc, "ece": 0.05, "brier_skill": -0.01, "capture_top5": 0.05}
            for path in (paths or ["structured", "fused"])
        ],
        "gates": gates or [],
        "falsification": falsification or [],
    }


def eval_panel(rows: list[tuple[str, int, float | None, str]]) -> pd.DataFrame:
    """``(year-date, label, structured score, split)`` rows."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(rows),
            "as_of": pd.to_datetime([row[0] for row in rows]),
            "split": [row[3] for row in rows],
            "label_tail_risk": [row[1] for row in rows],
            "label_mask_tail_risk": [1] * len(rows),
            "score_structured_tail_risk": [row[2] for row in rows],
            "score_fused_tail_risk": [row[2] for row in rows],
        }
    )


# -- 1. comparability ---------------------------------------------------------


def test_a_split_change_makes_the_pair_incomparable(cmp_module) -> None:
    """The wide rebuild moved `train.start` while the test block stayed put. A delta
    against the prior report would have read as a clean improvement."""
    reasons = cmp_module.comparability(
        report(split="2004-01-01 .. 2024-12-31"), report(split="2010-01-01 .. 2024-12-31")
    )
    assert any("split_definition" in reason for reason in reasons)


def test_a_positive_count_change_is_reported(cmp_module) -> None:
    """A different sample is not an improvement, and at ~200 positives it usually matters
    more than several decimal places."""
    reasons = cmp_module.comparability(report(n_positives=12), report(n_positives=7))
    assert any("positives differ" in reason for reason in reasons)


def test_identical_splits_and_counts_are_comparable(cmp_module) -> None:
    assert cmp_module.comparability(report(), report(run_id="later")) == []


# -- 2. unmeasured is not zero ------------------------------------------------


def test_a_metric_missing_on_one_side_is_not_a_zero_delta(cmp_module) -> None:
    baseline = {"auc": 0.6}
    candidate = {"auc": 0.6, "pr_auc": None}
    rows = {row["metric"]: row for row in cmp_module.scalar_deltas(baseline, candidate, ("auc", "pr_auc"))}
    assert rows["pr_auc"]["delta"] is None, "an absent metric has no delta"
    assert rows["auc"]["delta"] == pytest.approx(0.0)
    rendered = cmp_module._signed(rows["pr_auc"]["delta"])
    assert rendered == "not measured"
    assert "0.0000" not in rendered


# -- 3. paths are matched by name ---------------------------------------------


def test_a_path_that_disappears_does_not_shift_the_comparison(cmp_module) -> None:
    baseline = report(paths=["structured", "text_baseline", "fused"], auc=0.70)
    candidate = report(paths=["structured", "fused"], auc=0.60)
    table = {row["path"]: row for row in cmp_module.per_path_table(baseline, candidate)}
    assert table["text_baseline"]["present_candidate"] is False
    assert table["text_baseline"]["auc_delta"] is None
    assert table["structured"]["auc_delta"] == pytest.approx(-0.10)
    assert table["fused"]["auc_delta"] == pytest.approx(-0.10)


# -- 4. gates keep their third value ------------------------------------------


def test_an_undecidable_gate_is_not_a_regression(cmp_module) -> None:
    """`passed=None` means the inputs to decide were absent. Reporting it as FAIL would
    invent a failure out of missing data."""
    baseline = report(gates=[{"name": "headline_auc", "target": "> 0.75", "achieved": "n/a", "passed": None}])
    candidate = report(gates=[{"name": "headline_auc", "target": "> 0.75", "achieved": "0.60", "passed": False}])
    row = cmp_module.gates_diff(baseline, candidate)[0]
    assert row["change"] == "undecidable"
    assert cmp_module._verdict_word(row["candidate_passed"]) == "FAIL"
    assert cmp_module._verdict_word(None) == "undecidable"


def test_a_gate_that_is_undecidable_on_both_sides_does_not_change(cmp_module) -> None:
    gate = {"name": "calibration_ece", "target": "< 0.05", "achieved": "n/a", "passed": None}
    row = cmp_module.gates_diff(report(gates=[gate]), report(gates=[dict(gate)]))[0]
    assert row["change"] == "undecidable"


def test_a_flipped_gate_is_named_as_improved_or_regressed(cmp_module) -> None:
    failing = {"name": "headline_auc", "target": "> 0.75", "achieved": "0.60", "passed": False}
    passing = {"name": "headline_auc", "target": "> 0.75", "achieved": "0.80", "passed": True}
    assert cmp_module.gates_diff(report(gates=[failing]), report(gates=[passing]))[0]["change"] == "improved"
    assert cmp_module.gates_diff(report(gates=[passing]), report(gates=[failing]))[0]["change"] == "regressed"


# -- 5. falsification ---------------------------------------------------------


def test_a_falsification_verdict_change_is_visible(cmp_module) -> None:
    baseline = report(falsification=[{"code": "F1", "verdict": "triggered", "detail": "old"}])
    candidate = report(falsification=[{"code": "F1", "verdict": "not_triggered", "detail": "new"}])
    row = cmp_module.falsification_diff(baseline, candidate)[0]
    assert row["changed"] is True
    assert row["candidate_detail"] == "new"


def test_a_condition_absent_from_one_side_reads_as_absent(cmp_module) -> None:
    baseline = report(falsification=[{"code": "F1", "verdict": "triggered", "detail": "old"}])
    candidate = report(falsification=[])
    row = cmp_module.falsification_diff(baseline, candidate)[0]
    assert row["baseline_verdict"] == "triggered"
    assert row["candidate_verdict"] == "absent"
    assert row["changed"] is True


def test_the_fused_gain_is_read_from_f1_rather_than_recomputed(cmp_module) -> None:
    """A second implementation of the same arithmetic is a second thing to disagree with
    the report."""
    payload = report(falsification=[{"code": "F1", "verdict": "triggered", "detail": "CI crosses zero"}])
    assert cmp_module.fused_gain(payload)["detail"] == "CI crosses zero"
    assert cmp_module.fused_gain(report())["verdict"] == "absent"


# -- 6. per-year AUC ----------------------------------------------------------


def test_a_year_without_a_score_says_so_rather_than_scoring_zero(cmp_module) -> None:
    """`train` rows are never scored. A year that cannot be measured must not read as 0.5."""
    frame = eval_panel(
        [
            ("2008-06-01", 1, None, "train"),
            ("2008-07-01", 0, None, "train"),
            ("2020-03-01", 1, 0.9, "test"),
            ("2020-04-01", 1, 0.8, "test"),
            ("2020-05-01", 0, 0.1, "test"),
            ("2020-06-01", 0, 0.2, "test"),
        ]
    )
    table = cmp_module.year_auc(frame, "score_structured_tail_risk").set_index("year")
    assert table.loc[2008, "reason"] == "no score"
    assert table.loc[2008, "auc"] is None, "None, not nan: a nan AUC is a number to a later reader"
    assert table.loc[2020, "auc"] == pytest.approx(1.0)


def test_a_year_with_one_class_has_no_auc_not_a_perfect_one(cmp_module) -> None:
    frame = eval_panel([("2021-03-01", 0, 0.9, "test"), ("2021-04-01", 0, 0.1, "test")])
    table = cmp_module.year_auc(frame, "score_structured_tail_risk").set_index("year")
    assert table.loc[2021, "reason"] == "needs 2 positives, has 0"
    assert table.loc[2021, "auc"] is None


def test_a_thin_class_is_not_scored_here_when_the_metric_module_refuses(cmp_module) -> None:
    """`roc_auc` returns NaN below its own minimum class counts, and the report prints
    "not measured" on the strength of it. A weaker rule here would display a number for a
    fold the report calls undecidable."""
    from shingan.eval.metrics import MIN_POSITIVES

    assert MIN_POSITIVES > 1, "the fixture below only means anything if the floor is above 1"
    frame = eval_panel(
        [
            ("2022-03-01", 1, 0.9, "test"),
            ("2022-04-01", 0, 0.5, "test"),
            ("2022-05-01", 0, 0.2, "test"),
            ("2022-06-01", 0, 0.1, "test"),
        ]
    )
    table = cmp_module.year_auc(frame, "score_structured_tail_risk").set_index("year")
    assert table.loc[2022, "auc"] is None
    assert table.loc[2022, "reason"] == f"needs {MIN_POSITIVES} positives, has 1"


def two_class_panel(
    year: str, scores: tuple[float, float, float, float], scored: bool = True
) -> list[tuple[str, int, float | None, str]]:
    """2 positives then 2 negatives — the smallest year `roc_auc` will score at all.

    Scores are given explicitly rather than derived from a base, because the point of
    these fixtures is which way the ranking goes: a fixture that quietly ranks correctly
    cannot tell an inverted year from an aligned one.
    """
    split = "test" if scored else "train"
    days = ("03-01", "04-01", "05-01", "06-01")
    labels = (1, 1, 0, 0)
    return [
        (f"{year}-{day}", label, score if scored else None, split)
        for day, label, score in zip(days, labels, scores, strict=True)
    ]


ALIGNED = (0.9, 0.8, 0.2, 0.1)
INVERTED = (0.1, 0.2, 0.8, 0.9)


def test_the_year_join_keeps_both_sides_and_the_delta(cmp_module) -> None:
    baseline = eval_panel(two_class_panel("2020", INVERTED) + two_class_panel("2021", ALIGNED))
    candidate = eval_panel(two_class_panel("2020", ALIGNED) + two_class_panel("2021", ALIGNED, scored=False))
    table = cmp_module.year_comparison(baseline, candidate, "structured").set_index("year")
    assert table.loc[2020, "auc_baseline"] == pytest.approx(0.0)
    assert table.loc[2020, "auc_candidate"] == pytest.approx(1.0)
    assert table.loc[2020, "auc_delta"] == pytest.approx(1.0)
    assert table.loc[2021, "auc_candidate"] is None
    assert table.loc[2021, "auc_delta"] is None, "a year missing on one side has no delta"


def test_a_year_present_in_only_one_panel_is_still_described(cmp_module) -> None:
    """A dropped year must print its rows and positives from whichever panel has them,
    not a row of blanks that reads as zero."""
    baseline = eval_panel(two_class_panel("2020", INVERTED) + two_class_panel("2021", ALIGNED))
    candidate = eval_panel(two_class_panel("2020", ALIGNED))
    table = cmp_module.year_comparison(baseline, candidate, "structured").set_index("year")
    assert int(table.loc[2021, "rows"]) == 4
    assert int(table.loc[2021, "positives"]) == 2
    assert table.loc[2021, "reason_candidate"] == "absent from panel"
    assert table.loc[2021, "auc_baseline"] is not None


# -- 7. rendering -------------------------------------------------------------


def test_the_rendered_report_shouts_when_the_pair_is_incomparable(cmp_module) -> None:
    baseline = report(split="2010-01-01 .. 2024-12-31", run_id="old")
    candidate = report(split="2004-01-01 .. 2024-12-31", run_id="new")
    text = cmp_module.render(
        baseline,
        candidate,
        baseline_name="old.json",
        candidate_name="new.json",
        incomparable=cmp_module.comparability(candidate, baseline),
        headlines=cmp_module.scalar_deltas(baseline["headline"], candidate["headline"], ("auc",)),
        per_path=cmp_module.per_path_table(baseline, candidate),
        gates=cmp_module.gates_diff(baseline, candidate),
        falsification=cmp_module.falsification_diff(baseline, candidate),
        year_tables={},
        focus_year=None,
    )
    assert "**No.**" in text
    assert "Do not quote any of them as the effect of the code change." in text


def test_an_unscored_year_is_summarised_instead_of_listed_as_noise(cmp_module) -> None:
    baseline = eval_panel([("2008-06-01", 1, None, "train"), *two_class_panel("2020", INVERTED)])
    candidate = baseline.copy()
    tables = {"structured": cmp_module.year_comparison(baseline, candidate, "structured")}
    text = cmp_module.render(
        report(),
        report(),
        baseline_name="a",
        candidate_name="b",
        incomparable=[],
        headlines=cmp_module.scalar_deltas(report()["headline"], report()["headline"], ("auc",)),
        per_path=[],
        gates=[],
        falsification=[],
        year_tables=tables,
        focus_year=None,
    )
    assert "Years with no score in either run (1): 2008" in text
    assert "| 2008 |" not in text, "an unscored year is summarised once, not printed as a row per path"
    assert "| 2020 |" in text


def test_a_focus_year_is_called_out_with_its_positive_count(cmp_module) -> None:
    baseline = eval_panel(two_class_panel("2020", INVERTED))
    candidate = eval_panel(two_class_panel("2020", ALIGNED))
    tables = {"structured": cmp_module.year_comparison(baseline, candidate, "structured")}
    text = cmp_module.render(
        report(),
        report(),
        baseline_name="a",
        candidate_name="b",
        incomparable=[],
        headlines=cmp_module.scalar_deltas(report()["headline"], report()["headline"], ("auc",)),
        per_path=[],
        gates=[],
        falsification=[],
        year_tables=tables,
        focus_year=2020,
    )
    assert "### Focus year: 2020" in text
    assert "2 positives" in text


def test_rendering_an_empty_comparison_does_not_crash(cmp_module) -> None:
    text = cmp_module.render(
        {},
        {},
        baseline_name="a",
        candidate_name="b",
        incomparable=[],
        headlines=[],
        per_path=[],
        gates=[],
        falsification=[],
        year_tables={},
        focus_year=2025,
    )
    assert "absent from both panels" in text


def test_a_nan_metric_is_not_rendered_as_a_number(cmp_module) -> None:
    """pandas produces NaN from a genuinely absent value, so NaN reads as unmeasured too."""
    assert cmp_module._number(float("nan")) == "not measured"
    assert cmp_module._number(None) == "not measured"
    assert cmp_module._signed(float("nan")) == "not measured"
    assert cmp_module._number(0.123456, 3) == "0.123"
    assert cmp_module._signed(0.5) == "+0.5000"
    assert cmp_module._number("see backtest table") == "see backtest table"

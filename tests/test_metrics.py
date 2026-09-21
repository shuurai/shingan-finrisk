"""Metric tests, concentrated on the rare-event behaviour the report depends on.

Two things this file exists to pin down.

**Undefined is a distinct answer from bad.** With five positives in a 492-row test block,
an AUC is a very noisy thing, and a metric computed on too few positives is not a weak
measurement — it is not a measurement. The metrics return NaN below a minimum class
count, and the audit's cluster bootstrap then has to treat NaN as unusable rather than as
a low value. That contract is asserted here because it was previously enforced only by
reading the code.

**A ranking metric may not depend on row order.** ``average_precision`` resolved tied
scores by input order, so a constant score — a model with no signal at all — reported
1.0 when the positives happened to be listed first and 0.27 when they were listed last,
while ``roc_auc`` reported 0.5 for both. Any panel sorted in a way that correlates with
the label would have inflated the headline PR-AUC. Ties are now grouped, and the
implementation is checked against sklearn's on randomly tied scores rather than against
its own past output.
"""

from __future__ import annotations

import numpy as np
import pytest

from shingan.eval.metrics import (
    MIN_NEGATIVES,
    MIN_POSITIVES,
    average_precision,
    average_precision_lift,
    brier_score,
    brier_skill_score,
    calibration_curve_table,
    capture_at_fraction,
    expected_calibration_error,
    fbeta,
    ks_direction,
    ks_statistic,
    roc_auc,
    select_fbeta_threshold,
)

sklearn_metrics = pytest.importorskip("sklearn.metrics", reason="reference implementation")


# -- the reference cross-check ----------------------------------------------------


def tied_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels and heavy-tie scores, so ties rather than ranking dominate the value."""
    rng = np.random.default_rng(seed)
    size = int(rng.integers(40, 200))
    scores = np.round(rng.random(size), int(rng.integers(1, 3)))
    labels = (rng.random(size) < 0.25).astype(int)
    return labels, scores


@pytest.mark.parametrize("seed", range(12))
def test_average_precision_agrees_with_sklearn_under_ties(seed: int) -> None:
    labels, scores = tied_dataset(seed)
    if labels.sum() < 2 or (labels == 0).sum() < 2:
        pytest.skip("fold too degenerate for either implementation")
    reference = sklearn_metrics.average_precision_score(labels, scores)
    assert average_precision(labels, scores) == pytest.approx(reference, abs=1e-12)


@pytest.mark.parametrize("seed", range(12))
def test_roc_auc_agrees_with_sklearn_under_ties(seed: int) -> None:
    labels, scores = tied_dataset(seed)
    reference = sklearn_metrics.roc_auc_score(labels, scores)
    assert roc_auc(labels, scores) == pytest.approx(reference, abs=1e-12)


# -- rank metrics are invariant to row order ---------------------------------------


def test_average_precision_does_not_depend_on_row_order() -> None:
    """Regression: ties were broken by input order, so sorting the panel moved the metric.

    A constant score gave 1.0 with the positives first and 0.27 with them last. Row order
    is not information; this is the assertion that would have caught it.
    """
    labels = np.array([1, 1, 0, 0, 0, 0])
    scores = np.full(6, 2 / 6)
    rng = np.random.default_rng(3)
    values = [
        average_precision(labels[order], scores[order])
        for order in (rng.permutation(6) for _ in range(20))
    ]
    assert len(set(np.round(values, 12))) == 1, f"the value moved with row order: {set(values)}"


def test_a_constant_score_reports_the_base_rate() -> None:
    """The no-signal model's PR-AUC is the prevalence, and nothing else."""
    labels = np.array([1, 1, 0, 0, 0, 0])
    assert average_precision(labels, np.full(6, 2 / 6)) == pytest.approx(2 / 6)
    assert roc_auc(labels, np.full(6, 2 / 6)) == pytest.approx(0.5)


def test_a_constant_score_is_not_a_perfect_score_for_any_ordering() -> None:
    """The failure mode in the shape it actually appeared: perfect separation of nothing."""
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.full(10, 0.3)
    assert average_precision(labels, scores) == pytest.approx(0.3), (
        "a model with no signal must not score above the base rate"
    )


def test_roc_auc_does_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(4)
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.round(rng.random(8), 1)
    values = [roc_auc(labels[o], scores[o]) for o in (rng.permutation(8) for _ in range(10))]
    assert len(set(np.round(values, 12))) == 1


# -- perfect separation and its reverse -------------------------------------------


def separated() -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([1, 1, 0, 0, 0, 0])
    return labels, np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4])


def test_separation_and_its_reverse_are_distinguishable() -> None:
    labels, scores = separated()
    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert roc_auc(labels, 1 - scores) == pytest.approx(0.0)
    assert average_precision(labels, scores) == pytest.approx(1.0)
    assert average_precision(labels, 1 - scores) < 0.5, "a reversed model must score below base"


def test_ks_reports_the_direction_of_the_separation() -> None:
    labels, scores = separated()
    assert ks_statistic(labels, scores) == pytest.approx(1.0)
    assert ks_direction(labels, scores) == "positives_higher"
    assert ks_direction(labels, 1 - scores) == "negatives_higher"
    assert ks_direction(labels, np.full(6, 0.5)) == "tied"


# -- undefined is not the same as bad ---------------------------------------------


def test_auc_is_undefined_below_the_minimum_class_count() -> None:
    """A single-positive fold cannot produce an AUC; NaN is the honest answer."""
    one_positive = np.array([1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4])
    assert np.isnan(roc_auc(one_positive, scores))

    no_negatives = np.ones(MIN_POSITIVES + 1)
    assert np.isnan(roc_auc(no_negatives, np.linspace(0, 1, MIN_POSITIVES + 1)))

    assert MIN_POSITIVES >= 2 and MIN_NEGATIVES >= 2


def test_a_missing_pair_is_dropped_with_its_row() -> None:
    """A NaN label or score removes the row, not just the cell."""
    labels = np.array([1.0, 0.0, np.nan, 0.0, 1.0, 0.0])
    scores = np.array([0.9, 0.1, 0.5, 0.2, 0.8, 0.3])
    assert roc_auc(labels, scores) == pytest.approx(1.0)

    scores_with_nan = np.array([0.9, 0.1, 0.5, np.nan, 0.8, 0.3])
    assert roc_auc(labels, scores_with_nan) == pytest.approx(1.0)


def test_an_empty_fold_is_undefined_rather_than_an_exception() -> None:
    assert np.isnan(roc_auc(np.array([]), np.array([])))
    assert np.isnan(average_precision(np.array([]), np.array([])))


# -- lift, calibration and thresholds ---------------------------------------------


def test_lift_is_the_value_of_a_random_ranker() -> None:
    labels = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4])
    base_rate = labels.mean()
    assert average_precision_lift(labels, scores) == pytest.approx(1.0 / base_rate)


def test_lift_is_undefined_without_a_positive() -> None:
    assert np.isnan(average_precision_lift(np.zeros(6), np.linspace(0, 1, 6)))


def test_the_base_rate_forecast_has_no_skill_by_construction() -> None:
    """The reference point every skill score is measured against."""
    labels = np.array([1, 1, 0, 0, 0, 0])
    assert brier_skill_score(labels, np.full(6, 2 / 6)) == pytest.approx(0.0)
    assert brier_score(labels, np.full(6, 2 / 6)) > 0


def test_calibration_error_is_zero_for_a_perfectly_calibrated_forecast() -> None:
    labels = np.array([1, 1, 0, 0])
    result = expected_calibration_error(labels, np.array([1.0, 1.0, 0.0, 0.0]))
    assert result.ece == pytest.approx(0.0)
    assert result.mce == pytest.approx(0.0)


def test_a_constant_forecast_at_the_base_rate_is_one_bin_and_no_error() -> None:
    """Regression: the quantile binner ranked scores with ``method="first"``.

    Ranking broke ties by row order, so four identical scores landed in four different
    bins, each a singleton whose observed rate is 0 or 1. A forecast of exactly the base
    rate then reported an ECE of 0.5 — the worst possible score for the best possible
    forecast — and every tie in a real panel invented its own reliability gap.
    """
    labels = np.array([1, 1, 0, 0])
    table = calibration_curve_table(labels, np.full(4, 0.5))
    assert len(table) == 1, f"identical scores are one bin, got {len(table)}"
    assert table.iloc[0]["n"] == 4
    assert table.iloc[0]["abs_gap"] == pytest.approx(0.0)
    assert expected_calibration_error(labels, np.full(4, 0.5)).ece == pytest.approx(0.0)


def test_ties_do_not_invent_a_reliability_gap() -> None:
    """Two score values, two bins — however many rows carry each."""
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    table = calibration_curve_table(labels, scores)
    assert len(table) == 2
    assert set(table["n"]) == {3}


def test_calibration_error_grows_when_confidence_is_unearned() -> None:
    """Data calibrated by construction, then the forecast distorted either way."""
    rng = np.random.default_rng(11)
    size = 20_000
    forecast = rng.random(size)
    outcomes = (rng.random(size) < forecast).astype(int)

    honest = expected_calibration_error(outcomes, forecast).ece
    over = expected_calibration_error(outcomes, np.clip(forecast**0.5, 0, 1)).ece
    under = expected_calibration_error(outcomes, np.clip(forecast**2, 0, 1)).ece

    assert honest < 0.02, "a forecast calibrated by construction must score near zero"
    assert over > 0.1 and under > 0.1
    assert over > honest and under > honest


def test_a_calibration_table_never_drops_rows() -> None:
    """Every row must land in a bin: a dropped row silently reweights the whole curve."""
    rng = np.random.default_rng(12)
    labels = (rng.random(120) < 0.2).astype(int)
    scores = np.round(rng.random(120), 1)
    table = calibration_curve_table(labels, scores)
    assert table["n"].sum() == 120


def test_capture_never_falls_as_the_fraction_widens() -> None:
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.linspace(0.95, 0.05, 10)
    captures = [capture_at_fraction(labels, scores, fraction) for fraction in (0.1, 0.3, 0.5, 1.0)]
    assert captures == sorted(captures), "capturing more of the list cannot capture fewer events"
    assert captures[-1] == pytest.approx(1.0)


def test_fbeta_at_the_best_threshold_is_at_least_fbeta_at_a_fixed_one() -> None:
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    scores = np.linspace(0.9, 0.1, 10)
    chosen = select_fbeta_threshold(labels, scores)
    best = fbeta(labels, scores, threshold=chosen, beta=1.0)["fbeta"]
    for threshold in (0.2, 0.5, 0.8):
        assert best >= fbeta(labels, scores, threshold=threshold, beta=1.0)["fbeta"] - 1e-12

"""The matched-information baseline is a row of the standard report, not an extra command.

Two things are guarded here.

**The control has to be in the table the reader actually reads.** ``text_only_lora``'s
prompt carries :data:`PROMPT_SIGNAL_COLUMNS` — twelve structured signals — so subtracting
it from ``structured`` (the full feature set) measures two information sets as well as two
models. ``structured_matched`` is the estimator restricted to those same twelve columns,
and it only does its job if it sits next to the rows a reader is about to subtract. It used
to exist solely inside ``shingan eval lora``, which meant the standard report shipped three
rows that look complete and invite exactly the mismatched subtraction the control exists to
replace.

**The control is not a track.** ``PATH_ORDER`` decides which score columns are written onto
the panel and which two series the fusion stacks, so putting the control in it would leak a
measurement instrument into drift, rolling stability, the stress suite and the ablation
table. ``COMPARISON_ORDER`` exists so the table can print four rows while the tracks stay
three.

The wiring is tested at the seam — the heavy estimators are stubbed — because the claim is
about *what the pipeline asks for and registers*, not about the arithmetic of a booster.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from shingan.config import ProjectConfig, load_config
from shingan.data.builder import PROMPT_SIGNAL_COLUMNS
from shingan.paths import find_project_root
from shingan.pipeline import (
    COMPARISON_ORDER,
    PATH_FUSED,
    PATH_MATCHED,
    PATH_ORDER,
    PATH_STRUCTURED,
    PATH_TEXT,
    LabelOutcome,
    evaluate_label,
    matched_signal_report,
    run_pipeline,
)

ROOT = find_project_root()


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config(ROOT / "configs" / "default.yaml", root=ROOT)


def matched_panel(*, rows_per_split: int = 16) -> pd.DataFrame:
    """A panel holding only what the matched baseline reads: splits, a label, the signals."""
    frames: list[dict[str, object]] = []
    for split, offset in (("train", 0), ("valid", 400), ("test", 800)):
        for i in range(rows_per_split):
            row: dict[str, object] = {
                "ticker": f"T{i % 4}",
                "as_of": pd.Timestamp("2015-01-01") + pd.Timedelta(days=offset + i),
                "split": split,
                "label_mask_tail_risk": True,
                "label_tail_risk": int(i % 4 == 0),
                "f1": float(i),
            }
            for position, name in enumerate(PROMPT_SIGNAL_COLUMNS):
                row[name] = float(i + position) / 10.0
            frames.append(row)
    panel = pd.DataFrame(frames)
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    return panel


def stub_report(path: str) -> types.SimpleNamespace:
    """The subset of :class:`ClassificationReport` the comparison table reads."""
    return types.SimpleNamespace(
        path=path,
        n_rows=16,
        n_positives=4,
        base_rate=0.25,
        auc=0.5,
        ks=0.0,
        ks_direction="positives_higher",
        pr_auc=0.2,
        pr_auc_lift=1.0,
        capture_top5=0.5,
        calibration=None,
        # The table records the *headline* path's gate verdicts on every row, so every fake
        # report has to answer the call even though none of them carry gates of their own.
        gates=dict,
    )


# -- the KS magnitude is not the KS verdict ----------------------------------------


def test_the_comparison_table_carries_the_ks_direction_on_every_row() -> None:
    """A KS column without its direction cannot be read correctly on its own."""
    from shingan.pipeline import _comparison_table

    outcome = LabelOutcome(
        label="tail_risk",
        fitted=True,
        reports={path: stub_report(path) for path in COMPARISON_ORDER},
    )
    config = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    table = _comparison_table(outcome, config)

    assert "ks_direction" in table.columns
    assert set(table["ks_direction"]) == {"positives_higher"}


def test_an_inverted_ranking_is_visible_in_the_row() -> None:
    """The real-data case, as arithmetic rather than as a fixture.

    The 2026-09-23 real run scored `text_baseline` with AUC 0.3281 and KS 0.5692. The KS
    is direction-agnostic, so taken alone it reads as the best separation in the table
    while the AUC says the ordering is inverted. Recomputing both from one score vector
    pins the two definitions together: a reader who trusts the KS column alone is wrong,
    and the artifact now says so in the row itself.
    """
    from shingan.eval.metrics import ks_direction, ks_statistic

    # Negatives strictly above positives: the worst possible ranking, hence KS at its
    # maximum and AUC below 0.5.
    y_true = np.array([1, 1, 0, 0], dtype=float)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    assert ks_statistic(y_true, y_score) == pytest.approx(1.0)
    assert ks_direction(y_true, y_score) == "negatives_higher"


def test_the_console_prints_the_direction_next_to_the_ks() -> None:
    """Two renderings of one run must not disagree on what the KS means."""
    import io

    from rich.console import Console

    from shingan.cli import _label_table

    outcome = LabelOutcome(
        label="tail_risk",
        fitted=True,
        reports={path: stub_report(path) for path in COMPARISON_ORDER},
    )
    console = Console(file=io.StringIO(), width=200, no_color=True)
    with console.capture() as capture:
        console.print(_label_table(outcome))
    rendered = capture.get()

    assert "KS dir" in rendered
    assert rendered.count("positives_higher") == len(COMPARISON_ORDER)


# -- the ordering contract ---------------------------------------------------------


def test_the_control_is_not_a_track() -> None:
    """``PATH_ORDER`` drives score columns and the fusion; the control must stay out."""
    assert PATH_MATCHED not in PATH_ORDER
    assert PATH_ORDER == (PATH_STRUCTURED, PATH_TEXT, PATH_FUSED)


def test_the_table_prints_four_rows_with_the_control_second() -> None:
    """Reading order is part of the contract: the control sits under the path it shadows."""
    from shingan.pipeline import _comparison_table

    outcome = LabelOutcome(
        label="tail_risk",
        fitted=True,
        reports={path: stub_report(path) for path in COMPARISON_ORDER},
    )
    config = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    table = _comparison_table(outcome, config)

    assert list(table["path"]) == list(COMPARISON_ORDER)
    assert list(table["path"])[1] == PATH_MATCHED


def test_a_label_without_the_control_prints_three_rows_rather_than_a_blank_one() -> None:
    """A missing control is omitted, not printed as a row of NaNs under a real name."""
    from shingan.pipeline import _comparison_table

    outcome = LabelOutcome(
        label="tail_risk",
        fitted=True,
        reports={path: stub_report(path) for path in PATH_ORDER},
    )
    config = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    table = _comparison_table(outcome, config)

    assert list(table["path"]) == list(PATH_ORDER)


def test_the_console_order_matches_the_artifact_order() -> None:
    """Two renderings of one run disagreeing on row order read as two different tables."""
    import io

    from rich.console import Console

    from shingan.cli import _label_table

    outcome = LabelOutcome(
        label="tail_risk",
        fitted=True,
        reports={path: stub_report(path) for path in COMPARISON_ORDER},
    )
    console = Console(file=io.StringIO(), width=200, no_color=True)
    with console.capture() as capture:
        console.print(_label_table(outcome))
    rendered = capture.get()

    positions = [rendered.index(path) for path in COMPARISON_ORDER]
    assert positions == sorted(positions)


# -- the wiring --------------------------------------------------------------------


class FakeEstimator:
    """Stands in for the structured and text estimators: fit does nothing, scores do."""

    def __init__(self, config: object) -> None:
        del config
        self.calibration_report = {"calibrator": "stub"}

    def fit(self, *args: object, **kwargs: object) -> FakeEstimator:
        return self

    def predict_proba(self, frame: object) -> np.ndarray:
        return np.full(len(frame), 0.5)


class FakeFusion:
    """The stacker: its scores are never read by these assertions."""

    def __init__(self, config: object) -> None:
        del config

    def fit(self, *args: object, **kwargs: object) -> FakeFusion:
        return self

    def predict_proba(self, structured: pd.Series, text: pd.Series, **kwargs: object) -> pd.Series:
        return pd.Series(0.5, index=structured.index)


def stub_the_three_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything ``evaluate_label`` touches that is not the control under test."""
    monkeypatch.setattr("shingan.pipeline.StructuredRiskModel", FakeEstimator)
    monkeypatch.setattr("shingan.pipeline.TextBaselineModel", FakeEstimator)
    monkeypatch.setattr("shingan.pipeline.RiskFusion", FakeFusion)
    monkeypatch.setattr(
        "shingan.pipeline.evaluate_classification",
        lambda *args, **kwargs: stub_report(str(kwargs.get("path", ""))),
    )
    monkeypatch.setattr("shingan.pipeline.paired_bootstrap_difference", lambda *a, **k: None)
    monkeypatch.setattr("shingan.pipeline._information_coefficient", lambda *a, **k: None)
    monkeypatch.setattr("shingan.pipeline._comparison_table", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("shingan.pipeline._ablation_table", lambda *a, **k: pd.DataFrame())


def test_evaluate_label_fits_the_control_on_the_prompt_signals(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row is only a control if it is fitted on the columns the prompt shows."""
    stub_the_three_tracks(monkeypatch)
    panel = matched_panel()
    seen: list[list[str]] = []

    def spy(panel, config, label, signals):
        seen.append(list(signals))
        return pd.Series(0.5, index=panel.loc[panel["split"] == "test"].index)

    monkeypatch.setattr("shingan.pipeline._matched_signal_scores", spy)
    text = pd.Series("prompt", index=panel.index, dtype=object)

    outcome = evaluate_label(panel, config, "tail_risk", ["f1"], text, n_boot=10)

    assert seen == [list(PROMPT_SIGNAL_COLUMNS)]
    assert PATH_MATCHED in outcome.reports
    assert PATH_MATCHED in outcome.scores.columns
    assert outcome.matched_reason == ""


def test_a_refused_control_is_recorded_rather_than_raised(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A panel without the signal columns loses the row and says why — it does not crash.

    The three tracks do not depend on the control, so refusing the whole run over it would
    be wrong. Dropping it silently would be worse: the table would still read as complete.
    """
    stub_the_three_tracks(monkeypatch)

    def refuse(panel, config, label, signals):
        raise ValueError("asked for the matched baseline on columns the panel does not have")

    monkeypatch.setattr("shingan.pipeline._matched_signal_scores", refuse)
    panel = matched_panel()
    text = pd.Series("prompt", index=panel.index, dtype=object)

    outcome = evaluate_label(panel, config, "tail_risk", ["f1"], text, n_boot=10)

    assert PATH_MATCHED not in outcome.reports
    assert PATH_MATCHED not in outcome.scores.columns
    assert "columns the panel does not have" in outcome.matched_reason


def test_the_run_notes_name_the_label_whose_control_is_missing(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three rows under the ordinary headings look exactly like a complete comparison."""
    build = types.SimpleNamespace(
        panel=pd.DataFrame(
            {"ticker": ["A", "B"], "as_of": pd.to_datetime(["2020-01-01", "2021-01-01"])}
        ),
        split_report=types.SimpleNamespace(folds=[]),
        text_context=[],
        metadata={},
    )
    monkeypatch.setattr("shingan.pipeline.build_panel", lambda config, write=False: build)
    monkeypatch.setattr("shingan.pipeline.select_feature_columns", lambda panel, config: ["f1"])
    monkeypatch.setattr(
        "shingan.pipeline.text_inputs",
        lambda panel, result, config, label: pd.Series("prompt", index=panel.index, dtype=object),
    )

    def stub_evaluate(panel, config, label, features, text, *, n_boot):
        reason = "no signal columns" if label == "tail_risk" else ""
        return LabelOutcome(label=label, fitted=True, matched_reason=reason)

    monkeypatch.setattr("shingan.pipeline.evaluate_label", stub_evaluate)
    monkeypatch.setattr("shingan.pipeline._run_rolling", lambda *a, **k: None)
    monkeypatch.setattr("shingan.pipeline._run_drift", lambda *a, **k: [])
    monkeypatch.setattr("shingan.pipeline._run_stress", lambda *a, **k: [])
    monkeypatch.setattr("shingan.pipeline._run_backtest", lambda *a, **k: None)

    result = run_pipeline(config, labels=["default_risk", "tail_risk"], write=False)

    noted = [note for note in result.notes if PATH_MATCHED in note]
    assert len(noted) == 1
    assert "tail_risk" in noted[0]
    assert "no signal columns" in noted[0]
    assert "default_risk" not in noted[0]


def test_both_callers_go_through_one_implementation(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`eval run` and `eval lora` publish this row, so they must fit it the same way."""
    stub_the_three_tracks(monkeypatch)
    panel = matched_panel()

    def fixed(panel, config, label, signals):
        return pd.Series(
            np.linspace(0.1, 0.9, int((panel["split"] == "test").sum())),
            index=panel.loc[panel["split"] == "test"].index,
            name=PATH_MATCHED,
        )

    monkeypatch.setattr("shingan.pipeline._matched_signal_scores", fixed)
    text = pd.Series("prompt", index=panel.index, dtype=object)

    outcome = evaluate_label(panel, config, "tail_risk", ["f1"], text, n_boot=10)
    _, standalone = matched_signal_report(panel, config, "tail_risk")

    pd.testing.assert_series_equal(outcome.scores[PATH_MATCHED], standalone)

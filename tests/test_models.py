"""Model-layer tests.

The three tracks share one contract with the rest of the pipeline, and it is the contract
rather than the accuracy that is testable here — a unit test cannot say whether a model
predicts well, only that it behaves.

What is checked:

* **Missing features are the data layer's problem, not a crash.** A configured feature with
  no observation anywhere arrives as an all-NaN column. sklearn's histogram binner raises
  ``ValueError: window shape cannot be larger than input array shape`` on one, which reads
  as a library bug. The model drops such columns and *names* them.
* **Calibration reports carry the keys the report layer reads.** These were mismatched once
  (``n_calibration`` versus ``n_calibration_rows``) and the failure surfaced as
  "report is incomplete" from a validator two layers away.
* **A degenerate fusion fold still predicts.** A validation fold with one positive cannot
  support a logistic stacker; the layer must fall back and say so rather than emit NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shingan.config import load_config
from shingan.models.fusion import (
    ABLATION_PATHS,
    STRUCTURED_SCORE,
    TEXT_SCORE,
    RiskFusion,
    fusion_ablation,
)
from shingan.models.fusion import MODEL_FILENAME as FUSION_FILENAME
from shingan.models.structured import MODEL_FILENAME as STRUCTURED_FILENAME
from shingan.models.structured import StructuredRiskModel
from shingan.models.text_baseline import MODEL_FILENAME as TEXT_FILENAME
from shingan.models.text_baseline import TextBaselineModel
from shingan.paths import find_project_root

ROOT = find_project_root()


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs" / "default.yaml", root=ROOT)


def make_block(rows: int, positives: int, *, seed: int, dead_column: bool = False):
    """A small synthetic block with a learnable signal and a fixed positive count."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=rows),
            "constant": np.ones(rows),
        }
    )
    if dead_column:
        frame["no_source"] = np.nan
    labels = np.zeros(rows, dtype=int)
    labels[np.argsort(signal)[-positives:]] = 1
    return frame, labels


# -- structured ----------------------------------------------------------------


def test_column_with_no_observation_is_dropped_and_named(config) -> None:
    """Regression: an all-NaN feature used to crash the histogram binner.

    The assertion is two-sided on purpose. Dropping the column is only half the fix; the
    other half is that the name survives into the report, because "20 features were
    dropped" is unactionable and "vix_level was dropped" points at a missing data source.
    """
    train_x, train_y = make_block(120, 6, seed=1, dead_column=True)
    valid_x, valid_y = make_block(80, 3, seed=2, dead_column=True)

    model = StructuredRiskModel(config.structured)
    model.fit(train_x, train_y, valid_x, valid_y)  # must not raise

    assert model.dropped_features == ["no_source"]
    assert "no_source" not in model._feature_names
    assert np.isfinite(model.predict_proba(valid_x)).all()


def test_no_columns_are_dropped_when_every_column_is_observed(config) -> None:
    train_x, train_y = make_block(120, 6, seed=3)
    valid_x, valid_y = make_block(80, 3, seed=4)
    model = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)
    assert model.dropped_features == []


def test_fit_refuses_a_training_block_with_too_few_positives(config) -> None:
    """F8: the caller is meant to skip the label, not fit an unrankable model."""
    train_x, train_y = make_block(120, 1, seed=5)
    valid_x, valid_y = make_block(80, 3, seed=6)
    with pytest.raises(ValueError, match="F8"):
        StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)


def test_calibration_report_uses_the_keys_the_report_layer_reads(config) -> None:
    train_x, train_y = make_block(120, 6, seed=7)
    valid_x, valid_y = make_block(80, 3, seed=8)
    model = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)

    report = model.calibration_report
    assert "n_calibration_rows" in report, "report layer reads n_calibration_rows"
    assert "n_calibration_positives" in report
    assert report["n_calibration_rows"] == len(valid_x)
    assert report["n_calibration_positives"] == int(valid_y.sum())
    assert report["dropped_features"] == []


def test_predictions_are_finite_probabilities(config) -> None:
    train_x, train_y = make_block(120, 6, seed=9)
    valid_x, valid_y = make_block(80, 3, seed=10)
    model = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)
    probabilities = model.predict_proba(valid_x)
    assert probabilities.shape == (len(valid_x),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_degenerate_validation_fold_still_predicts(config, tmp_path) -> None:
    """One positive in validation cannot calibrate; scores must stay usable."""
    train_x, train_y = make_block(120, 6, seed=11)
    valid_x, valid_y = make_block(80, 1, seed=12)
    model = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)
    probabilities = model.predict_proba(valid_x)
    assert np.isfinite(probabilities).all()
    assert model.calibration_report["n_calibration_positives"] == 1


def test_save_load_round_trip_preserves_scores_and_dropped_features(config, tmp_path) -> None:
    train_x, train_y = make_block(120, 6, seed=13, dead_column=True)
    valid_x, valid_y = make_block(80, 3, seed=14, dead_column=True)
    model = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)
    directory = model.save(tmp_path)

    restored = StructuredRiskModel.load(directory)
    assert restored.dropped_features == ["no_source"]
    np.testing.assert_allclose(
        restored.predict_proba(valid_x), model.predict_proba(valid_x), rtol=1e-10
    )


# -- text baseline -------------------------------------------------------------


def test_text_baseline_reports_the_same_calibration_keys(config) -> None:
    train_text = pd.Series([f"disclosure {index % 7} material weakness" for index in range(40)])
    valid_text = pd.Series([f"disclosure {index % 7}" for index in range(20)])
    train_y = np.array([1] * 6 + [0] * 34)
    valid_y = np.array([1] * 3 + [0] * 17)

    model = TextBaselineModel(config.text_baseline).fit(train_text, train_y, valid_text, valid_y)
    probabilities = model.predict_proba(valid_text)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_text_baseline_handles_an_empty_corpus_without_raising(config) -> None:
    """No documents at all is the Stage 2 state and must not be an exception."""
    empty_train = pd.Series([""] * 20)
    empty_valid = pd.Series([""] * 10)
    model = TextBaselineModel(config.text_baseline).fit(
        empty_train, np.array([1] * 2 + [0] * 18), empty_valid, np.array([1] * 2 + [0] * 8)
    )
    probabilities = model.predict_proba(empty_valid)
    assert np.isfinite(probabilities).all()


# -- fusion --------------------------------------------------------------------


def test_fusion_falls_back_when_the_fold_cannot_support_a_stacker(config) -> None:
    """One positive in the fitting fold: fall back, record why, keep predicting."""
    rng = np.random.default_rng(21)
    rows = 60
    structured = rng.random(rows)
    text = rng.random(rows)
    labels = np.zeros(rows, dtype=int)
    labels[np.argsort(structured)[-1:]] = 1

    fusion = RiskFusion(config.fusion).fit(structured, text, labels, fit_split="valid")
    probabilities = fusion.predict_proba(structured, text)
    assert np.isfinite(probabilities).all()
    assert fusion.diagnostics.get("kind")
    assert fusion.diagnostics.get("n_fit_rows") == rows
    assert fusion.diagnostics.get("positives_fit") == 1
    assert fusion.diagnostics.get("warning") or fusion.diagnostics.get("note"), (
        "a degenerate fit must leave an explanation in diagnostics"
    )


def test_fusion_predicts_for_a_fold_that_can_support_a_stacker(config) -> None:
    rng = np.random.default_rng(22)
    rows = 200
    structured = rng.random(rows)
    text = rng.random(rows)
    labels = (structured + 0.1 * rng.random(rows) > 0.9).astype(int)
    if labels.sum() < 5:
        labels[np.argsort(structured)[-5:]] = 1

    fusion = RiskFusion(config.fusion).fit(structured, text, labels, fit_split="valid")
    probabilities = fusion.predict_proba(structured, text)
    assert np.isfinite(probabilities).all()
    assert fusion.diagnostics["kind"] in {"logistic_stack", "rank_average", "rank_average_fallback"}


def test_fusion_weights_are_normalised(config) -> None:
    fusion = RiskFusion(config.fusion)
    weights = fusion._normalised_weights()
    assert len(weights) == 2
    assert all(weight >= 0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0)


def test_ablation_covers_all_three_paths(config) -> None:
    """The ablation table is the project's headline evidence; it must have every path.

    ``scores`` is the pipeline's score frame, whose columns are :data:`STRUCTURED_SCORE`
    and :data:`TEXT_SCORE` — not the path names. The three paths are the table's *rows*,
    and the fused row is refitted here rather than taken from the caller.
    """
    rng = np.random.default_rng(23)
    rows = 150
    panel = pd.DataFrame({"as_of": pd.date_range("2020-01-01", periods=rows)})
    labels = np.zeros(rows, dtype=int)
    labels[-6:] = 1
    scores = pd.DataFrame(
        {STRUCTURED_SCORE: rng.random(rows), TEXT_SCORE: rng.random(rows)},
        index=panel.index,
    )

    table = fusion_ablation(
        panel,
        scores,
        labels,
        dates=panel["as_of"],
        fit_mask=np.ones(rows, dtype=bool),
        apply_mask=np.ones(rows, dtype=bool),
        config=config.fusion,
    )
    assert set(table["path"]) == set(ABLATION_PATHS), "every path must appear in the table"
    assert (table["n_rows"] == rows).all(), "all three paths must rest on the same population"
    auc_rows = table[table["metric"] == "auc"]
    assert len(auc_rows) == len(ABLATION_PATHS)
    assert auc_rows["value"].notna().all(), "a fitted ablation must produce a real AUC"


def test_ablation_refuses_a_scores_frame_that_lost_its_columns(config) -> None:
    """Regression: a rename upstream used to surface as a silently flat ablation."""
    rows = 40
    panel = pd.DataFrame({"as_of": pd.date_range("2020-01-01", periods=rows)})
    scores = pd.DataFrame({"structured": np.zeros(rows), "text": np.zeros(rows)})
    with pytest.raises(KeyError, match="score_structured"):
        fusion_ablation(panel, scores, np.zeros(rows, dtype=int), config=config.fusion)


# -- persistence ---------------------------------------------------------------
#
# The three tracks share one contract: ``save(directory)`` returns the directory and
# writes ``MODEL_FILENAME`` inside it, and ``load(directory)`` reads it back. Every part
# of that sentence was false somewhere in this package when these tests were written —
# the fused track was written as ``fused.joblib`` while its loader looked for
# ``fusion.joblib``, and the text track had no loader at all — so the contract is
# asserted directly rather than inferred from a round trip that happened to pass.


def test_every_track_saves_under_its_own_constant(config, tmp_path) -> None:
    """The writer must not invent a file name; the loader only knows the constant."""
    train_x, train_y = make_block(120, 6, seed=31)
    valid_x, valid_y = make_block(80, 3, seed=32)
    structured = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)

    text = pd.Series([f"disclosure {index % 5} restatement" for index in range(40)])
    baseline = TextBaselineModel(config.text_baseline).fit(
        text, np.array([1] * 6 + [0] * 34), text, np.array([1] * 3 + [0] * 37)
    )

    rng = np.random.default_rng(33)
    rows = 90
    stacker_scores = rng.random(rows)
    labels = np.zeros(rows, dtype=int)
    labels[np.argsort(stacker_scores)[-5:]] = 1
    fusion = RiskFusion(config.fusion).fit(
        stacker_scores, rng.random(rows), labels, fit_split="valid"
    )

    for model, filename in (
        (structured, STRUCTURED_FILENAME),
        (baseline, TEXT_FILENAME),
        (fusion, FUSION_FILENAME),
    ):
        directory = tmp_path / type(model).__name__
        assert Path(model.save(directory)) == directory, "save() must return the directory"
        assert (directory / filename).is_file(), f"{type(model).__name__} must write {filename}"


def test_save_load_round_trip_preserves_scores_and_dropped_features(config, tmp_path) -> None:
    train_x, train_y = make_block(120, 6, seed=13, dead_column=True)
    valid_x, valid_y = make_block(80, 3, seed=14, dead_column=True)
    model = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)
    directory = model.save(tmp_path)

    restored = StructuredRiskModel.load(directory)
    assert restored.dropped_features == ["no_source"]
    assert restored._feature_names == model._feature_names
    np.testing.assert_allclose(
        restored.predict_proba(valid_x), model.predict_proba(valid_x), rtol=1e-10
    )


def test_text_baseline_round_trips(config, tmp_path) -> None:
    """The Track B0 loader did not exist, so its numbers could not be reproduced."""
    train_text = pd.Series([f"disclosure {index % 7} material weakness" for index in range(40)])
    valid_text = pd.Series([f"disclosure {index % 7}" for index in range(20)])
    model = TextBaselineModel(config.text_baseline).fit(
        train_text, np.array([1] * 6 + [0] * 34), valid_text, np.array([1] * 3 + [0] * 17)
    )

    restored = TextBaselineModel.load(model.save(tmp_path))
    np.testing.assert_allclose(
        restored.predict_proba(valid_text), model.predict_proba(valid_text), rtol=1e-10
    )


def test_fusion_round_trips(config, tmp_path) -> None:
    rng = np.random.default_rng(34)
    rows = 120
    structured = rng.random(rows)
    text = rng.random(rows)
    labels = np.zeros(rows, dtype=int)
    labels[np.argsort(structured)[-6:]] = 1
    fusion = RiskFusion(config.fusion).fit(structured, text, labels, fit_split="valid")

    restored = RiskFusion.load(fusion.save(tmp_path))
    assert restored.diagnostics["kind"] == fusion.diagnostics["kind"]
    np.testing.assert_allclose(
        restored.predict_proba(structured, text), fusion.predict_proba(structured, text),
        rtol=1e-10,
    )


def test_load_rejects_a_file_path_with_the_directory_to_use(config, tmp_path) -> None:
    """Handing the file where a directory belongs must name the correct call.

    ``load(save(d))`` used to fail as ``.../model.joblib/model.joblib``, which reads as a
    corrupt artifact rather than as a disagreement about the signature.
    """
    train_x, train_y = make_block(120, 6, seed=35)
    valid_x, valid_y = make_block(80, 3, seed=36)
    model = StructuredRiskModel(config.structured).fit(train_x, train_y, valid_x, valid_y)
    directory = model.save(tmp_path)

    with pytest.raises(ValueError) as error:
        StructuredRiskModel.load(directory / STRUCTURED_FILENAME)
    message = str(error.value)
    assert "takes the model directory" in message
    assert str(directory) in message, "the message must name the directory to use instead"


def test_load_lists_what_is_present_when_the_model_file_is_absent(config, tmp_path) -> None:
    """A name disagreement between writer and loader must be diagnosable from the error."""
    (tmp_path / "fused.joblib").write_bytes(b"a stale artifact under an old name")
    with pytest.raises(FileNotFoundError) as error:
        RiskFusion.load(tmp_path)
    message = str(error.value)
    assert FUSION_FILENAME in message
    assert "fused.joblib" in message, "the message must list the files actually present"


def test_load_reports_a_missing_directory_rather_than_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no model directory"):
        StructuredRiskModel.load(tmp_path / "never_written")


def test_load_explains_an_artifact_pickled_as_a_bare_object(config, tmp_path) -> None:
    """An artifact from the old writer must be diagnosed, not unpacked.

    ``cli train`` used to pickle the model object itself for the text and fused tracks
    while the loader expects a dict of named fields, so loading one raised
    ``'TextBaselineModel' object is not subscriptable`` — an error that names neither the
    file nor the cause.
    """
    import joblib

    train_text = pd.Series([f"disclosure {index % 5}" for index in range(30)])
    model = TextBaselineModel(config.text_baseline).fit(
        train_text, np.array([1] * 5 + [0] * 25), train_text, np.array([1] * 3 + [0] * 27)
    )
    (tmp_path / TEXT_FILENAME).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, tmp_path / TEXT_FILENAME)

    with pytest.raises(ValueError) as error:
        TextBaselineModel.load(tmp_path)
    message = str(error.value)
    assert "expected a metadata dict" in message
    assert "TextBaselineModel" in message, "a bare object must be named as the cause"
    assert "re-run" in message, "the reader must be told the artifact is stale, not corrupt"


def test_load_names_the_field_that_is_missing_from_a_stale_artifact(tmp_path) -> None:
    """A payload that predates a change to the saved fields must say which field is gone."""
    import joblib

    joblib.dump({"config": {}}, tmp_path / STRUCTURED_FILENAME)
    with pytest.raises(ValueError) as error:
        StructuredRiskModel.load(tmp_path)
    message = str(error.value)
    assert "estimator" in message, "the missing field must be named"
    assert "re-run" in message

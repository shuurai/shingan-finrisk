"""The prompt asks one specific question, and the text track must be asked that one.

One regression lives here. ``text_inputs()`` renders the string the text tracks read, and
that string states which question it is asking::

    <TASK>label=default_risk horizon_days=365</TASK>

The label was hard-coded to ``default_risk`` inside the function, and ``run_pipeline``
called it **once**, before the per-label loop. So every label except the first was scored
on numbers produced by a model that had been asked a different question — ``tail_risk``
(30 trading days) was predicted from a prompt headed ``horizon_days=365``. Nothing raised:
both labels produce a float, and the report reads the same either way. The recorded
``text_baseline`` rows for ``tail_risk`` in ``artifacts/reports/20260923T055845Z.json`` and
``artifacts/stage2/20260921T073120Z.json`` carry this defect and are superseded.

The horizons differ by an order of magnitude (365 / 730 calendar days, 30 trading days), so
the mislabelling is not cosmetic.

These tests are written against the call structure and the rendered prompt rather than
against any particular metric, because the failure was silent in every downstream number.
"""

from __future__ import annotations

import datetime as dt
import inspect
import types

import pandas as pd
import pytest

from shingan.config import ProjectConfig, load_config
from shingan.paths import find_project_root
from shingan.pipeline import LabelOutcome, text_inputs
from shingan.prompts import PromptContext, build_user_prompt

ROOT = find_project_root()


@pytest.fixture(scope="module")
def config() -> ProjectConfig:
    return load_config(ROOT / "configs" / "default.yaml", root=ROOT)


def context_for(label: str, horizon_days: int) -> PromptContext:
    return PromptContext(
        as_of=dt.date(2020, 1, 1),
        label=label,
        horizon_days=horizon_days,
    )


def test_the_prompt_states_the_question_it_is_asking() -> None:
    """The task tag is the only place the question is named, so it has to be right."""
    rendered = build_user_prompt(context_for("tail_risk", 30))

    assert "<TASK>label=tail_risk horizon_days=30</TASK>" in rendered


def test_the_horizons_actually_differ_between_labels(config: ProjectConfig) -> None:
    """A mislabelled prompt is only a defect because the horizons are not interchangeable.

    Included so that the guard above is not mistaken for cosmetics: if all three labels
    shared a horizon, asking about the wrong one would be harmless.
    """
    horizons = {label: config.labels.horizon_days(label) for label in config.labels.targets}

    assert len(set(horizons.values())) == len(horizons), horizons


def test_the_label_argument_has_no_default() -> None:
    """A default would restore the bug quietly, which is how it got in.

    The parameter is required on purpose: the caller always knows which label it is
    scoring, and a default that is right only for the first label is worse than no
    default at all.
    """
    parameter = inspect.signature(text_inputs).parameters["label"]

    assert parameter.default is inspect.Parameter.empty


def test_text_inputs_renders_the_label_it_is_given(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label must reach the prompt builder, not be swallowed on the way."""
    seen: list[str] = []

    def spy(panel, result, config, label, budget):
        seen.append(label)
        return {}

    monkeypatch.setattr("shingan.pipeline.build_prompt_contexts", spy)

    panel = pd.DataFrame({"ticker": ["TEST"], "as_of": [pd.Timestamp("2020-01-01")]})
    rendered = text_inputs(panel, result=None, config=config, label="fraud_risk")

    assert seen == ["fraud_risk"]
    assert "default_risk" not in seen
    assert list(rendered) == [""]


def test_the_pipeline_asks_each_label_about_itself(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression guard: one prompt per label, each carrying its own label.

    The expensive stages are stubbed — panel construction and the label fits — because
    the claim under test is about *wiring*: which label each rendered prompt was built
    for. Stubbing keeps this honest rather than weaker; a real panel would exercise the
    arithmetic of a different module and still not observe the call.
    """
    seen: list[str] = []

    build = types.SimpleNamespace(
        panel=pd.DataFrame(
            {"ticker": ["A", "B"], "as_of": pd.to_datetime(["2020-01-01", "2021-01-01"])}
        ),
        split_report=types.SimpleNamespace(folds=[]),
        text_context=[],
    )
    monkeypatch.setattr("shingan.pipeline.build_panel", lambda config, write=False: build)
    monkeypatch.setattr("shingan.pipeline.select_feature_columns", lambda panel, config: ["f1"])

    def spy(panel, result, config, label):
        seen.append(label)
        return pd.Series(["prompt"] * len(panel), index=panel.index, dtype=object)

    monkeypatch.setattr("shingan.pipeline.text_inputs", spy)
    monkeypatch.setattr(
        "shingan.pipeline.evaluate_label",
        lambda panel, config, label, features, text, *, n_boot: LabelOutcome(
            label=label, fitted=False, reason="stubbed"
        ),
    )
    monkeypatch.setattr("shingan.pipeline._run_drift", lambda *a, **k: [])
    monkeypatch.setattr("shingan.pipeline._run_stress", lambda *a, **k: [])
    monkeypatch.setattr("shingan.pipeline._run_backtest", lambda *a, **k: None)

    from shingan.pipeline import run_pipeline

    result = run_pipeline(config, labels=["default_risk", "tail_risk"], write=False)

    # Called once per label rather than once per run, and each call names its own label.
    assert seen == ["default_risk", "tail_risk"]
    assert list(result.outcomes) == ["default_risk", "tail_risk"]

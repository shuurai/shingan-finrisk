"""Tests for the report's data caveats.

The caveats are the part of the output a reader cannot check against the tables. "An AUC
of 0.769" is verifiable; "15 of 49 configured features were dropped" is a claim the report
makes about its own inputs, and if it is wrong or vague the reader has no way to tell.

The requirement these tests encode: **an absent feature is advice, not a disclaimer.**
A count alone invites the reader to assume the data is unobtainable. Several of the
Stage 2 gaps were one fetch away, so each one has to name its source and its remedy, and
no feature may be silently omitted from the accounting.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from shingan.data.schema import (
    FEATURE_SOURCES,
    NON_FEATURE_COLUMNS,
    PANEL_COLUMNS,
    SOURCE_ADVICE,
    diagnose_gaps,
)
from shingan.pipeline import (
    PATH_FUSED,
    PATH_STRUCTURED,
    PATH_TEXT,
    _data_gap_notes,
    _window_gap_notes,
)

#: The feature columns the panel can carry: everything that is not an id, a split value,
#: a label, a forward-looking column or a weight.
FEATURE_COLUMNS = tuple(name for name in PANEL_COLUMNS if name not in set(NON_FEATURE_COLUMNS))


# -- the feature -> source map ---------------------------------------------------


def test_every_feature_column_has_a_declared_source() -> None:
    """An unmapped feature produces the placeholder remedy, which is not advice.

    Found by this test: ``ratios_missing_frac`` was excluded from the RATIO_COLUMNS
    comprehension and assigned nowhere, so a gap in that one column could not name its
    source. Every other gap in the Stage 2 run named a fetch the reader could act on.
    """
    undeclared = sorted(name for name in FEATURE_COLUMNS if name not in FEATURE_SOURCES)
    assert not undeclared, f"feature columns with no declared source: {undeclared}"


def test_no_feature_source_entry_is_stale() -> None:
    """A mapping key that is not a panel column is a rename that was never followed up."""
    stale = sorted(name for name in FEATURE_SOURCES if name not in set(PANEL_COLUMNS))
    assert not stale, f"FEATURE_SOURCES names that no longer exist as columns: {stale}"


# -- diagnose_gaps ---------------------------------------------------------------


def test_gaps_are_grouped_by_source_and_ordered_by_size() -> None:
    gaps = diagnose_gaps(["vix_level", "risk_factor_token_share", "going_concern_hits", "vol_60d"])
    assert [gap["count"] for gap in gaps] == sorted((gap["count"] for gap in gaps), reverse=True)
    for gap in gaps:
        assert gap["features"] == sorted(gap["features"])
        assert gap["count"] == len(gap["features"])


def test_every_diagnosed_feature_is_accounted_for() -> None:
    """Requirement: no absent feature may vanish from the caveat accounting."""
    requested = [
        "vix_level",
        "credit_spread_chg_20d",
        "risk_factor_token_share",
        "going_concern_hits",
        "some_feature_nobody_declared",
        "vol_60d",
    ]
    reported = [name for gap in diagnose_gaps(requested) for name in gap["features"]]
    assert sorted(reported) == sorted(requested)
    assert len(reported) == len(set(reported)), "a feature must appear under exactly one source"


def test_an_undeclared_feature_is_shown_as_unmapped_not_dropped() -> None:
    gaps = diagnose_gaps(["some_feature_nobody_declared"])
    assert gaps[0]["source"] == "unmapped"
    assert gaps[0]["features"] == ["some_feature_nobody_declared"]
    assert "FEATURE_SOURCES" in gaps[0]["advice"], "the remedy is where to declare the source"


def test_a_declared_source_always_has_a_remedy() -> None:
    """A source with no advice renders as the placeholder, which is not advice."""
    declared = set(FEATURE_SOURCES.values())
    undocumented = sorted(source for source in declared if source not in SOURCE_ADVICE)
    assert not undocumented, f"sources with no remedy text: {undocumented}"


def test_advice_says_what_to_do_not_that_data_is_missing() -> None:
    """A remedy must name a source or a command, not restate the symptom."""
    for source, advice in SOURCE_ADVICE.items():
        assert len(advice) > 20, f"{source} has a stub remedy"
        assert not advice.lower().startswith("missing"), f"{source} restates the symptom"


def test_no_gaps_for_an_empty_set() -> None:
    assert diagnose_gaps([]) == []


def test_text_features_are_owed_by_the_document_fetch() -> None:
    """The Stage 2 gaps were mostly this, and the remedy is the downloader this repo now has."""
    gaps = {gap["source"]: gap for gap in diagnose_gaps(list(FEATURE_SOURCES))}
    assert "fetch_sec_docs.py" in gaps["sec_filing_text"]["advice"]


# -- _data_gap_notes -------------------------------------------------------------


def structured_with(dropped: list[str]) -> SimpleNamespace:
    return SimpleNamespace(dropped_features=dropped)


def outcome_with(*, dropped=(), text_warnings=(), fusion_diagnostics=None) -> SimpleNamespace:
    models: dict[str, object] = {PATH_STRUCTURED: structured_with(list(dropped))}
    if text_warnings:
        models[PATH_TEXT] = SimpleNamespace(warnings=list(text_warnings))
    if fusion_diagnostics is not None:
        models[PATH_FUSED] = SimpleNamespace(diagnostics=fusion_diagnostics)
    return SimpleNamespace(models=models)


def test_gap_notes_name_the_source_and_the_remedy() -> None:
    notes = _data_gap_notes(outcome_with(dropped=["vix_level", "vix_chg_5d"]))
    assert len(notes) == 1, "both features belong to the same source, so one line covers them"
    note = notes[0]
    assert "2 configured feature(s)" in note, "the count must be stated"
    assert "vix_level" in note and "vix_chg_5d" in note, "the names must be stated"
    assert "market_volatility_index" in note, "the source the reader has to go and fix"
    assert "Remedy:" in note


def test_gap_notes_separate_features_by_source() -> None:
    """One line per source: a reader acts on sources, not on a pooled count."""
    notes = _data_gap_notes(
        outcome_with(dropped=["vix_level", "risk_factor_token_share", "going_concern_hits"])
    )
    assert len(notes) == 2, "the filing-text pair shares a line; the VIX feature gets its own"
    joined = "\n".join(notes)
    assert "market_volatility_index" in joined
    assert "sec_filing_text" in joined


def test_no_gap_notes_when_nothing_was_dropped() -> None:
    assert _data_gap_notes(outcome_with(dropped=[])) == []


def test_no_gap_notes_without_an_outcome() -> None:
    assert _data_gap_notes(None) == []


def test_text_warnings_are_surfaced_as_caveats() -> None:
    notes = _data_gap_notes(
        outcome_with(text_warnings=["no document text in the training block; text features are NaN"])
    )
    assert any("no document text" in note for note in notes)


def test_the_fusion_note_reports_the_fitted_kind_not_the_requested_one() -> None:
    """`fusion.kind` is a request; the fold decides. The report must say which was fitted."""
    notes = _data_gap_notes(
        outcome_with(
            fusion_diagnostics={
                "kind": "rank_average_fallback",
                "fit_split": "valid",
                "n_fit_rows": 412,
                "positives_fit": 1,
                "warning": "one positive cannot fit a stacker",
            }
        )
    )
    joined = "\n".join(notes)
    assert "rank_average_fallback" in joined
    assert "412" in joined and "1 positive" in joined
    assert "one positive cannot fit a stacker" in joined


def test_no_fusion_note_when_the_layer_left_no_diagnostics() -> None:
    assert _data_gap_notes(outcome_with(fusion_diagnostics={})) == []



# -- _window_gap_notes -----------------------------------------------------------


def panel(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows)
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    return frame


def test_window_gap_note_names_the_year_the_share_and_the_remedy() -> None:
    """The Stage 2 shape: 2020 in no window, holding most of the positives."""
    rows = [{"as_of": "2019-06-03", "split": "valid", "label_tail_risk": 0, "label_mask_tail_risk": True}]
    rows += [
        {"as_of": f"2020-03-{day:02d}", "split": "excluded", "label_tail_risk": 1, "label_mask_tail_risk": True}
        for day in range(1, 10)
    ]
    notes = _window_gap_notes(panel(rows))

    assert len(notes) == 1
    note = notes[0]
    assert "9 of 9" in note, "the note must say how many positives are outside every window"
    assert "100%" in note
    assert "2020" in note
    assert "Remedy:" in note
    assert "valid.end" in note or "test.start" in note, "the remedy must name a concrete setting"


def test_no_window_gap_note_when_the_excluded_block_holds_no_positives() -> None:
    rows = [
        {"as_of": "2019-06-03", "split": "valid", "label_tail_risk": 0, "label_mask_tail_risk": True},
        {"as_of": "2020-03-02", "split": "excluded", "label_tail_risk": 0, "label_mask_tail_risk": True},
    ]
    assert _window_gap_notes(panel(rows)) == []


def test_no_window_gap_note_when_nothing_is_excluded() -> None:
    rows = [{"as_of": "2019-06-03", "split": "valid", "label_tail_risk": 0, "label_mask_tail_risk": True}]
    assert _window_gap_notes(panel(rows)) == []


def test_an_unobservable_label_produces_no_window_gap_note() -> None:
    """A row whose label window has not closed holds no positive, so it is not a gap."""
    rows = [
        {"as_of": "2019-06-03", "split": "valid", "label_tail_risk": 0, "label_mask_tail_risk": True},
        {"as_of": "2020-03-02", "split": "excluded", "label_tail_risk": 1, "label_mask_tail_risk": False},
    ]
    assert _window_gap_notes(panel(rows)) == []


def test_window_gap_notes_tolerate_a_panel_without_splits() -> None:
    assert _window_gap_notes(pd.DataFrame({"as_of": pd.to_datetime(["2020-01-01"])})) == []
    assert _window_gap_notes(pd.DataFrame()) == []


def test_every_label_gets_its_own_window_gap_note() -> None:
    rows = [{"as_of": "2019-06-03", "split": "valid", "label_tail_risk": 0, "label_mask_tail_risk": True}]
    rows += [
        {"as_of": f"2020-03-{day:02d}", "split": "excluded", "label_tail_risk": 1, "label_mask_tail_risk": True}
        for day in range(1, 5)
    ]
    frame = panel(rows)
    frame["label_default_risk"] = np.where(frame["split"] == "excluded", 1, 0)
    frame["label_mask_default_risk"] = True

    notes = _window_gap_notes(frame)
    assert len(notes) == 2, "each label's coverage gap must be reported separately"
    assert any("tail_risk" in note for note in notes)
    assert any("default_risk" in note for note in notes)

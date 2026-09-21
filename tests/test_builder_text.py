"""Builder tests: the text-feature pass and its disclosure construction.

One regression lives here. `_text_features_and_context` concatenates a filing's section
rows into one disclosure string and hands it to `build_text_features`, which re-segments
it. That round trip only works if the separator keeps each section's ``Item`` heading at
the start of a line, and the code used to join with a single space — which silently zeroed
``risk_factor_token_share``, ``mdna_token_share`` and ``neg_kw_density_mdna`` across an
entire 2221-row panel with no error anywhere.

The test is written against the *observable output* (the feature values), not against the
separator string, so it keeps working if the join is refactored — it just has to keep
producing non-zero shares.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shingan.data.builder import _text_features_and_context

RISK_BODY = " ".join(
    ["We face substantial uncertainty and our results depend on customer demand."] * 25
)
MDNA_BODY = " ".join(
    ["Revenue increased and we believe margins will improve, though we cannot be certain."] * 25
)

RISK_SECTION = "Item 1A. Risk Factors\n" + RISK_BODY
MDNA_SECTION = "Item 7. Management's Discussion and Analysis\n" + MDNA_BODY


def grid_for(as_of: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["TEST"], "as_of": [pd.Timestamp(as_of)]})


def filings_for(*dates_and_sections: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "cik": "0000000001",
                "filed": pd.Timestamp(filed),
                "doc_type": "10-K",
                "section": section.split(".")[0],
                "text": section,
                "accession": f"acc-{filed}-{index}",
            }
            for index, (filed, section) in enumerate(dates_and_sections)
        ]
    )


def test_section_features_survive_the_concatenation() -> None:
    """A two-section filing must produce non-zero section shares.

    Zero is the failure signature, and it is invisible: nothing raises, the panel schema is
    intact, and the column simply carries no information.
    """
    features, _ = _text_features_and_context(
        grid_for("2011-06-30"),
        filings_for(("2011-01-31", RISK_SECTION), ("2011-01-31", MDNA_SECTION)),
        pd.DataFrame(),
    )
    row = features.iloc[0]
    assert row["risk_factor_token_share"] > 0, "risk-factor share collapsed to zero"
    assert pd.notna(row["neg_kw_density_mdna"]), "MD&A density is NaN: MD&A was not found"
    assert row["disclosure_len_tokens"] > 0


def test_disclosure_length_change_needs_a_prior_filing() -> None:
    """With one filing there is no comparable prior, so the change is NaN, not 0%."""
    single, _ = _text_features_and_context(
        grid_for("2011-06-30"), filings_for(("2011-01-31", RISK_SECTION)), pd.DataFrame()
    )
    assert pd.isna(single.iloc[0]["disclosure_len_chg"])

    longer = "Item 1A. Risk Factors\n" + RISK_BODY * 2
    two, _ = _text_features_and_context(
        grid_for("2012-06-30"),
        filings_for(("2011-01-31", RISK_SECTION), ("2012-01-31", longer)),
        pd.DataFrame(),
    )
    assert pd.notna(two.iloc[0]["disclosure_len_chg"])
    assert two.iloc[0]["disclosure_len_chg"] > 0


def test_filings_after_as_of_are_invisible() -> None:
    """The core no-lookahead rule for the text leg, at the builder level."""
    features, context = _text_features_and_context(
        grid_for("2011-06-30"),
        filings_for(("2011-01-31", RISK_SECTION), ("2011-12-31", RISK_SECTION)),
        pd.DataFrame(),
    )
    assert features.iloc[0]["disclosure_len_tokens"] > 0
    shown = context[0]["filings"]
    assert len(shown) == 1
    assert shown[0]["filed"] == "2011-01-31", "a filing filed after as_of reached the context"


def test_no_filing_at_all_yields_no_document_features() -> None:
    """Absent text must read as NaN, never as a zero-valued document."""
    features, context = _text_features_and_context(
        grid_for("2001-06-30"), filings_for(("2011-01-31", RISK_SECTION)), pd.DataFrame()
    )
    assert pd.isna(features.iloc[0]["disclosure_len_tokens"])
    assert context[0]["filings"] == []


def test_context_keeps_the_accession_for_evidence_citation() -> None:
    _, context = _text_features_and_context(
        grid_for("2011-06-30"), filings_for(("2011-01-31", RISK_SECTION)), pd.DataFrame()
    )
    assert context[0]["filings"][0]["accession"] == "acc-2011-01-31-0"


@pytest.mark.parametrize("section", [RISK_SECTION, MDNA_SECTION])
def test_each_section_carries_its_own_heading(section: str) -> None:
    """Section bodies are stored with their heading, which is what makes re-segmentation work."""
    assert section.splitlines()[0].lower().startswith("item ")

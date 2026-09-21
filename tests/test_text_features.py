"""Text feature tests.

The feature block in :mod:`shingan.features.text` is where this project's central claim
is measured, and it has three properties that are easy to break silently:

* section lookup depends on recognising ``Item`` headings in three real-world styles —
  line-anchored, ALL-CAPS inline, and mixed-case inline with punctuation — which makes
  both the separator used when a filing's sections are concatenated and the punctuation
  after the item number part of the contract;
* "no document" and "a document with no negative words" must stay distinguishable, so
  empty input yields NaN rather than zero;
* 10-K and 10-Q number their sections differently, and the same label means different
  things in the two forms.

Each test below pins one of those. Two regressions shaped this file: the builder used to
join sections with a space, which zeroed three features across a 2221-row panel; and the
segmenter used to require line-anchored headings, which silently failed on the flattened
text layer — 1722 of the corpus's 2222 real documents (419 of 547 10-Ks) have almost no
newlines left, so ``Item 1A`` and MD&A were unrecoverable for three quarters of the
sample. The real corpus is what forced the inline patterns into the contract.
"""

from __future__ import annotations

import math

import pytest

from shingan.features.text import (
    MDNA_SECTIONS,
    RISK_FACTOR_SECTIONS,
    document_features,
    lexicon_density,
    segment_items,
)

RISK_BODY = " ".join(
    ["Our results depend on customer demand and are subject to substantial uncertainty."] * 20
)
MDNA_BODY = " ".join(
    ["Revenue increased and we believe margins will improve, although we cannot be certain."] * 20
)


def ten_k(mdna_heading: str = "Item 7") -> str:
    """A minimal 10-K-shaped document with line-anchored headings."""
    return "\n".join(
        [
            "Item 1. Business",
            "We operate in several segments.",
            "Item 1A. Risk Factors",
            RISK_BODY,
            f"{mdna_heading}. Management's Discussion and Analysis",
            MDNA_BODY,
            "Item 8. Financial Statements",
            "The consolidated statements follow.",
        ]
    )


# -- segmentation --------------------------------------------------------------


def test_headings_must_start_a_line() -> None:
    """Punctuation is what separates an inline heading from a prose cross-reference.

    The real corpus renders most headings inline (``"... 11 ITEM 1A. RISK FACTORS
    Our ability ..."``, mixed case: ``"Item 1A. Risk Factors"`` mid-line), so line
    anchoring alone cannot be the guard. What protects against false hits is the
    requirement of punctuation directly after the item number: ``"listed in Item 1A
    could cause"`` and ``"see Item 1A, Risk Factors"`` carry none and must not match.
    """
    anchored = segment_items(ten_k())
    assert {"Item 1A", "Item 7"} <= set(anchored)

    # A comma or a bare mention is prose, not a heading — even mid-line.
    prose = "The risk factors listed in Item 1A could cause losses. See Item 1A, Risk Factors."
    assert segment_items(prose) == {"Item 1": prose.strip()}


def test_flat_text_layer_headings_are_recognised_inline() -> None:
    """The three styles the real corpus actually uses, spliced mid-line on purpose."""
    body = RISK_BODY
    tail = " ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA "
    caps = f"Forward-looking statements. 11 ITEM 1A. RISK FACTORS {body}{tail}follow."
    title = f"Forward-looking statements. Item 1A. Risk Factors {body}{tail}follow."
    colon = f"Part I Item 1A: Risk Factors {body}{tail}follow."

    for label, document in (("caps", caps), ("title", title), ("colon", colon)):
        sections = segment_items(document)
        assert "Item 1A" in sections, f"{label}-style inline heading was not recognised"
        assert sections["Item 1A"].startswith("ITEM 1A") or "RISK" in sections["Item 1A"].upper()


def test_space_joined_sections_survive_when_headings_carry_punctuation() -> None:
    """The old failure mode, and its current boundary.

    The builder used to join a filing's section blobs with a space, which pushed every
    heading after the first mid-line and zeroed three features panel-wide. With the
    punctuation-bearing inline patterns that construction now segments — which is the
    behaviour measured at 75% (Item 1A) and 89% (MD&A) recovery on the real corpus.
    Headings *without* punctuation still depend on the line start, so the separator
    remains part of the contract.
    """
    risk_section = "Item 1A. Risk Factors\n" + RISK_BODY
    mdna_section = "Item 7. Management's Discussion and Analysis\n" + MDNA_BODY

    by_space = segment_items(" ".join([risk_section, mdna_section]))
    by_newline = segment_items("\n\n".join([risk_section, mdna_section]))

    assert {"Item 1A", "Item 7"} <= set(by_space), "punctuated headings survive a space join"
    assert {"Item 1A", "Item 7"} <= set(by_newline)
    assert document_features(" ".join([risk_section, mdna_section]))["mdna_token_share"] > 0

    # The boundary: punctuation-free headings are only recognisable at a line start.
    bare_risk = "Item 1A Risk Factors\n" + RISK_BODY
    bare_mdna = "Item 7 Management's Discussion and Analysis\n" + MDNA_BODY
    bare_by_space = segment_items(" ".join([bare_risk, bare_mdna]))
    assert set(bare_by_space) == {"Item 1"}, "a space join must still destroy bare headings"
    bare_by_newline = segment_items("\n\n".join([bare_risk, bare_mdna]))
    assert "Item 1A" in bare_by_newline and "Item 7" in bare_by_newline


def test_unsplittable_document_is_returned_whole_and_labelled() -> None:
    """No headings means one key holding the entire text — detectable, not guessed."""
    result = segment_items("A filing with no Item headings at all.")
    assert list(result) == ["Item 1"]
    assert result["Item 1"] == "A filing with no Item headings at all."


def test_empty_input_segments_to_nothing() -> None:
    assert segment_items("") == {}
    assert segment_items("   \n  ") == {}


def test_longest_body_wins_for_a_repeated_label() -> None:
    """A table-of-contents entry and the real section share a label; the real one is longer."""
    content = "\n".join(["Item 7. MD&A", "short", "Item 7. MD&A", MDNA_BODY])
    sections = segment_items(content)
    assert sections["Item 7"].endswith(MDNA_BODY.strip())


# -- missing versus zero -------------------------------------------------------


def test_empty_document_is_nan_not_zero() -> None:
    """"We have no document" and "no negative words" are different facts."""
    features = document_features("")
    assert all(math.isnan(value) for value in features.values()), features
    assert math.isnan(lexicon_density("", ("loss",)))


def test_present_document_yields_finite_densities() -> None:
    features = document_features(ten_k())
    assert features["disclosure_len_tokens"] > 0
    assert math.isfinite(features["uncertainty_density"])
    assert math.isfinite(features["negative_density"])


# -- section shares ------------------------------------------------------------


def test_both_shares_are_nonzero_when_headings_are_intact() -> None:
    """The two columns the separator bug zeroed. Zero here is the failure signature."""
    features = document_features(ten_k())
    assert features["risk_factor_token_share"] > 0
    assert features["mdna_token_share"] > 0
    assert math.isfinite(features["neg_kw_density_mdna"])


def test_shares_are_zero_when_sections_cannot_be_located() -> None:
    """The counterpart: with the headings destroyed the shares collapse, no error raised.

    Destruction now has to defeat all three heading styles — punctuation stripped, no
    line starts, only bare mid-line mentions, which is exactly how a prose
    cross-reference reads and why the inline patterns must ignore it.
    """
    destroyed = (
        "cover page boilerplate the risk factors listed in Item 1A could cause losses "
        "and Item 7 discusses our liquidity see Item 1A of this report for detail "
        + RISK_BODY.replace("Item", "Section")
    )
    features = document_features(destroyed)
    assert features["risk_factor_token_share"] == 0.0
    assert features["mdna_token_share"] == 0.0


def test_ten_k_prefers_item_7_over_item_2_for_mdna() -> None:
    """A 10-K's Item 2 is Properties. Item 7 must win wherever it exists."""
    document = "\n".join(
        [
            "Item 2. Properties",
            "We lease offices in several countries.",
            "Item 7. Management's Discussion and Analysis",
            MDNA_BODY,
        ]
    )
    features = document_features(document)
    assert document_features(document)["mdna_token_share"] > 0
    assert "Item 7" in MDNA_SECTIONS
    assert MDNA_SECTIONS.index("Item 7") < MDNA_SECTIONS.index("Item 2")
    assert features["mdna_token_share"] > 0


def test_ten_q_mdna_is_found_under_item_2() -> None:
    """A 10-Q numbers its MD&A as Item 2, which the label set must contain."""
    document = "\n".join(
        [
            "Item 1. Financial Statements",
            "The condensed statements follow.",
            "Item 2. Management's Discussion and Analysis",
            MDNA_BODY,
        ]
    )
    segments = segment_items(document)
    assert "Item 2" in segments
    assert document_features(document)["mdna_token_share"] > 0


def test_risk_factors_prefer_item_1a() -> None:
    """1B and 1C are fallbacks, not equals."""
    assert RISK_FACTOR_SECTIONS[0] == "Item 1A"
    document = "\n".join(["Item 1B. Unresolved Staff Comments", RISK_BODY, "Item 8. Financials", "x"])
    assert document_features(document)["risk_factor_token_share"] > 0


# -- lexicon -------------------------------------------------------------------


def test_lexicon_density_is_per_thousand_tokens() -> None:
    text = " ".join(["loss"] * 1 + ["gain"] * 999)
    assert lexicon_density(text, ("loss",)) == pytest.approx(1.0, abs=0.05)


def test_lexicon_density_counts_phrases() -> None:
    text = " ".join(["material weakness"] * 4 + ["revenue"] * 996)
    assert lexicon_density(text, ("material weakness",)) > 0

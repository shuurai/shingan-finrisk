"""The instruction contract must document every closed set the parser enforces.

These tests exist because of one measured failure. ``SYSTEM_PROMPT`` named
``source_type`` inside the evidence object but never said which values it accepts,
while ``EvidenceSpan.source_type`` is a strict enum. Asked without an adapter, the
base model guessed the only taxonomy it could see — the block names
``FILING_EXCERPTS`` / ``NEWS`` / ``STRUCTURED_SIGNALS`` — and ``parse_assessment``
rejected 60 of 64 generations, including every positive. No metric could have shown
this: a row the model answered wrongly and a row it never answered both arrive
downstream as a row with no score.

The expectations are derived from ``data/schema.py`` rather than restated here, and
equality is asserted rather than containment, so both directions are covered at
once: an enum member the prompt forgot, and a value the prompt offers that no enum
accepts.
"""

from __future__ import annotations

import re
from datetime import date

from shingan.data.schema import EvidenceSpan, RiskLabel, Severity, SourceType
from shingan.prompts import (
    CHARS_PER_TOKEN,
    PROMPT_OVERHEAD_CHARS,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    FilingExcerpt,
    NewsItem,
    PromptContext,
    build_user_prompt,
    chars_budget_for_seq_length,
    truncate_context,
)

#: Characters allowed for the values substituted into the one-line task block
#: (``label=`` / ``horizon_days=``) when the reserve is checked.
_TASK_VALUES_ALLOWANCE = 48

#: The sequence limit the reserve is dimensioned against; the EvalConfig default.
_SEQ_LIMIT = 4096


def _documented_for(field: str) -> set[str]:
    """The values the prompt names for ``field``, read out of the contract itself."""
    match = re.search(rf"^\s*{field}\s*:\s*one of (.+)$", SYSTEM_PROMPT, re.MULTILINE)
    assert match is not None, f"{field} is not documented as a closed set in SYSTEM_PROMPT"
    return {item.strip() for item in match.group(1).split(",") if item.strip()}


def _long_filing_context() -> PromptContext:
    """A context whose filing is far larger than any budget, so truncation runs."""
    return PromptContext(
        as_of=date(2019, 2, 26),
        label=str(RiskLabel.TAIL_RISK),
        horizon_days=30,
        structured_signals={"debt_to_equity": 1.25, "current_ratio": 0.98},
        filings=[
            FilingExcerpt(
                doc_type="10-K",
                filed=date(2019, 2, 26),
                section="Item 1A",
                text="Risk factor sentence about liquidity. " * 4_000,
            )
        ],
    )


def test_every_source_type_the_parser_accepts_is_documented() -> None:
    assert _documented_for("source_type") == {member.value for member in SourceType}


def test_every_label_the_parser_accepts_is_documented() -> None:
    assert _documented_for("label") == {member.value for member in RiskLabel}


def test_every_severity_the_parser_accepts_is_documented() -> None:
    assert _documented_for("severity") == {member.value for member in Severity}


def test_the_evidence_block_names_all_three_of_its_keys() -> None:
    """A prompt that asks for evidence but not for its shape invites an invented one."""
    evidence_line = re.search(r"^\s*evidence\s*:(.+)$", SYSTEM_PROMPT, re.MULTILINE)
    assert evidence_line is not None
    for key in ("source_type", "source_ref", "quote"):
        assert key in evidence_line.group(1)


def test_the_evidence_ref_examples_match_what_the_renderers_emit() -> None:
    """The prompt's examples must be copied from the code that builds the headers.

    The prompt tells the model to build ``source_ref`` from the block header, so the
    examples are a contract with :class:`FilingExcerpt` and :class:`NewsItem`. When
    they drift, the model produces references that look plausible and name documents
    that do not exist — and nothing downstream validates ``source_ref``, so it is
    only visible here.
    """
    filing_ref = FilingExcerpt(
        doc_type="10-K",
        filed=date(2019, 2, 26),
        section="Item 1A",
        text="Liquidity risk factors.",
    ).source_ref
    news_ref = NewsItem(published=date(2020, 3, 1), source="reuters", title="t").source_ref

    assert filing_ref == "10-K:2019-02-26:Item 1A"
    assert news_ref == "news:reuters:2020-03-01"
    assert filing_ref in SYSTEM_PROMPT
    assert news_ref in SYSTEM_PROMPT


def test_the_schema_field_description_agrees_with_the_renderers() -> None:
    """The in-code documentation of ``source_ref`` drifted from the renderers.

    It advertised ``10-K:JPM:2019-02-26:Item 1A`` and ``news:reuters:2020-03-01:0``,
    neither of which any renderer produces: there is no ticker, and no article index.
    Asserting the exact rendered shapes rather than a prefix is what makes the stale
    example fail.
    """
    description = EvidenceSpan.model_fields["source_ref"].description or ""
    assert "10-K:2019-02-26:Item 1A" in description
    assert "news:reuters:2020-03-01" in description
    assert "JPM" not in description


def test_prompt_overhead_covers_the_real_scaffolding() -> None:
    """The reserve must exceed the fixed text it exists to pay for.

    Checked as an inequality on the real components rather than a hardcoded number,
    so growing the system prompt without growing the reserve fails here instead of
    showing up as a silent context overflow at scoring time.
    """
    assert len(SYSTEM_PROMPT) + len(USER_TEMPLATE) + _TASK_VALUES_ALLOWANCE <= (
        PROMPT_OVERHEAD_CHARS
    )


def test_a_prompt_that_fills_the_budget_fits_the_sequence_limit() -> None:
    fitted = truncate_context(
        _long_filing_context(),
        budget_chars=chars_budget_for_seq_length(_SEQ_LIMIT),
    )
    assert fitted.truncated, "the fixture must actually be truncated, or this proves nothing"

    user = build_user_prompt(fitted)
    assert (len(SYSTEM_PROMPT) + len(user)) / CHARS_PER_TOKEN <= _SEQ_LIMIT


def test_the_previous_reserve_would_have_overflowed() -> None:
    """The reserve this change raised was not hypothetical.

    Under 1,500 the budget handed out more characters than the limit could hold once
    the contract enumerated ``source_type``, and the failure would have been a prompt
    that quietly overran ``max_seq_length`` rather than an exception.
    """
    context = _long_filing_context()
    outcomes = {}
    for overhead in (PROMPT_OVERHEAD_CHARS, 1_500):
        budget = chars_budget_for_seq_length(_SEQ_LIMIT, overhead=overhead)
        user = build_user_prompt(truncate_context(context, budget_chars=budget))
        outcomes[overhead] = (len(SYSTEM_PROMPT) + len(user)) / CHARS_PER_TOKEN <= _SEQ_LIMIT

    assert outcomes[1_500] is False
    assert outcomes[PROMPT_OVERHEAD_CHARS] is True

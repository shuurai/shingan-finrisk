"""Evidence citations must name documents the prompt actually contained.

The schema has accepted any string as ``source_ref`` since it was written, and the
quote-verification the repository does run is performed against documents *the model
named* — so a fabricated citation passes every gate. These tests pin the audit that
closes that gap: every document-type citation a parsed assessment carries is matched
against the refs derived from the very context rendered into that row's prompt.

History, kept because it is why the target-side test below exists: the first SFT
corpus minted news references as the bare source name and filing references without
their section, while the system prompt taught the ``news:source:date`` shape — so an
adapter faithfully reproducing its training targets was scored as citing documents
the prompt never contained. ``pipeline._sft_evidence`` now mints the taught shape,
and :func:`test_sft_targets_cite_in_the_taught_shape` pins target and contract to
the same ``source_ref`` property so they cannot drift apart again.
"""

from __future__ import annotations

from datetime import date

from shingan.data.schema import EvidenceSpan, RiskAssessment, RiskLabel, SourceType
from shingan.eval.lora import (
    CitationAudit,
    audit_citations,
    citation_summary,
    document_refs,
)
from shingan.prompts import FilingExcerpt, NewsItem, PromptContext


def _context(*, with_accession: bool = False) -> PromptContext:
    filing = FilingExcerpt(
        doc_type="10-K",
        filed=date(2019, 2, 26),
        section="Item 1A",
        text="Liquidity risk factors.",
    )
    if with_accession:
        filing.accession = "0000123456-19-000012"
    return PromptContext(
        as_of=date(2019, 2, 26),
        label=str(RiskLabel.TAIL_RISK),
        horizon_days=30,
        filings=[filing],
        news=[NewsItem(published=date(2020, 3, 1), source="reuters", title="Markets fall", body="b")],
    )


def _assessment(*refs: tuple[str, str]) -> RiskAssessment:
    """An assessment whose evidence carries the given ``(source_type, source_ref)`` pairs."""
    return RiskAssessment(
        label=RiskLabel.TAIL_RISK,
        severity="high",
        score=0.7,
        horizon_days=30,
        reasons=["r"],
        evidence=[
            EvidenceSpan(source_type=SourceType(source_type), source_ref=ref, quote="q")
            for source_type, ref in refs
        ],
    )


def test_a_citation_in_the_taught_shape_resolves() -> None:
    """Every shape a model can legitimately emit is in the valid set.

    ``document_refs`` returns separator-normalised refs; the literal colon spellings
    are exercised for resolution in :func:`test_separator_style_does_not_change_resolution`.
    """
    refs = document_refs(_context(with_accession=True))
    assert "10-K 2019-02-26 Item 1A" in refs
    assert "10-K 2019-02-26 Item 1A 0000123456-19-000012" in refs
    assert "reuters 2020-03-01" in refs
    assert "news reuters 2020-03-01" in refs


def test_the_sectionless_filing_form_resolves() -> None:
    """The SFT targets cite filings as ``doc_type date``; that names the document."""
    assert "10-K 2019-02-26" in document_refs(_context())


def test_a_bare_source_name_does_not_resolve() -> None:
    """A source names an outlet, not an article — a citation that vague must not count."""
    assert "reuters" not in document_refs(_context())


def test_sft_targets_cite_in_the_taught_shape() -> None:
    """The SFT evidence mints refs the audit resolves, for both document types.

    Regression pin for the contract gap the first zero-shot run exposed: the targets
    used to carry a bare outlet name and a section-less filing ref, shapes the audit
    counts as unresolved. Target and prompt must be one shape.
    """
    from shingan.pipeline import _sft_evidence

    context = _context(with_accession=True)
    context.news[0].body = ""  # headline-only: the quote comes from the title
    context.filings[0].text = "Liquidity risk factors threaten the credit profile."
    context.news[0].title = "Markets fall sharply as credit spreads widen"

    refs = document_refs(context)
    from shingan.eval.lora import _normalize_ref

    for source_type, source_ref, _quote in _sft_evidence(context):
        # Compare through the audit's own normalisation path, exactly as
        # ``citation_summary`` does — the taught colon form must resolve.
        assert _normalize_ref(source_ref) in refs, (source_type, source_ref)


def test_sft_quote_verification_survives_headline_only_news() -> None:
    """A headline-only article must not make the verbatim check fail its own quote.

    The quote is sliced from ``body or title``; the haystack used to be ``body`` only,
    so an empty body made ``quotes_are_verbatim`` fail — which the builder escalates
    to a hard error, i.e. the real corpus could not have been rebuilt at all.
    """
    from shingan.data.schema import quotes_are_verbatim
    from shingan.pipeline import _sft_evidence
    from shingan.prompts import assessment_from_labels

    context = _context()
    context.news[0].body = ""
    context.news[0].title = "Markets fall sharply as credit spreads widen beyond warning levels"
    context.filings = []  # news-only row

    spans = _sft_evidence(context)
    assert spans, "a headline-only article must still yield one quote"
    assessment = assessment_from_labels(
        "tail_risk",
        score=0.0,
        horizon_days=30,
        reasons=["r"],
        evidence=spans,
    )
    documents = [item.body or item.title for item in context.news]
    assert quotes_are_verbatim(assessment, documents)


def test_a_fabricated_ref_is_counted_unresolved_with_an_example() -> None:
    assessment = _assessment(("filing", "10-K:2099-01-01:Item 7"))
    audit = audit_citations([assessment], [_context()], row_ids=["JPM-20190226"])

    assert audit.n_citations == 1
    assert audit.n_resolved == 0
    assert audit.n_unresolved == 1
    assert audit.n_rows_with_unresolved == 1
    assert audit.examples == [
        {"row": "JPM-20190226", "source_type": "filing", "source_ref": "10-K:2099-01-01:Item 7"}
    ]


def test_separator_style_does_not_change_resolution() -> None:
    """The prompt teaches colon-joined refs; the SFT targets used spaces. Same document."""
    for ref in ("10-K:2019-02-26:Item 1A", "10-K 2019-02-26 Item 1A", "10-K: 2019-02-26  Item 1A"):
        summary = citation_summary(_assessment(("filing", ref)), _context())
        assert summary["n_unresolved"] == 0, ref
        assert summary["n_resolved"] == 1, ref


def test_an_accession_bearing_citation_resolves() -> None:
    context = _context(with_accession=True)
    ref = context.filings[0].source_ref
    summary = citation_summary(_assessment(("filing", ref)), context)
    assert summary["n_resolved"] == 1


def test_non_document_citations_are_counted_separately() -> None:
    """``structured``/``price``/``other`` spans have no prompt document to resolve to."""
    summary = citation_summary(
        _assessment(("structured", "unattributed"), ("other", "unattributed")),
        _context(),
    )
    assert summary["n_citations"] == 0
    assert summary["n_non_document"] == 2
    assert summary["n_unresolved"] == 0


def test_an_unparsed_row_is_reported_not_hidden() -> None:
    audit = audit_citations([None, _assessment(("news", "news:reuters:2020-03-01"))],
                            [_context(), _context()])
    assert audit.n_rows_not_audited == 1
    assert audit.n_rows_audited == 1
    assert audit.n_unresolved == 0


def test_a_row_without_evidence_is_audited_and_counted() -> None:
    assessment = _assessment()
    assessment = assessment.model_copy(update={"evidence": []})
    audit = audit_citations([assessment], [_context()])
    assert audit.n_rows_audited == 1
    assert audit.n_rows_with_evidence == 0
    assert audit.n_citations == 0


def test_the_audit_is_a_disclosure_not_a_gate() -> None:
    """Unresolved citations must never drop rows or scores — that would bias the metrics."""
    audit = CitationAudit()
    payload = audit.as_dict()
    assert "ok" not in payload, "an ok/failed verdict would invite use as a gate"

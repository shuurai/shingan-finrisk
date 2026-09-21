"""Count-based features from disclosure and news text (the "text_counts" group).

This module is the bridge between the two tracks. The semantic reading of the text
is the language model's job; what lives here is the shallow, auditable layer that
runs on a CPU in milliseconds and often carries most of the signal — document
length, how much of it is the risk-factors section, how often going-concern
language appears, how negative the news flow has been.

**On the lexicons.** They are abridged, hand-maintained stand-ins for the
Loughran-McDonald master dictionary, which is the standard in this literature. The
full dictionary is not bundled because its licence does not permit redistribution in
a permissively licensed repository. The word lists here are therefore *not*
interchangeable with LM counts: they are shorter, they omit the finance-specific
inflections, and they will under-count relative to published work. Any result that
depends on the exact counts should be re-run against the licensed dictionary. The
lexicons are versioned with the dataset through ``LEXICON_VERSION``.

**On heading detection.** :func:`segment_items` finds 10-K/10-Q ``Item`` headings
with a regular expression. Filings format these inconsistently — some use
``ITEM 1A.``, some ``Item 1A - Risk Factors``, some put the heading on its own line
and some inline. The parser is deliberately tolerant and, when it finds fewer than
two headings, returns the whole document as a single unnamed segment rather than
guessing, because a mis-split document produces a risk-factor share that looks
plausible and is meaningless.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd

from shingan.data.schema import TEXT_COUNT_COLUMNS
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Bump when any lexicon below changes, so a dataset built under a different word
#: list is distinguishable. Recorded in run metadata.
LEXICON_VERSION = "0.1.0"

#: Window lengths for the news aggregates.
NEWS_WINDOWS: tuple[int, ...] = (30, 90)

#: Sentences in the risk-factor section are long and dense; the tokenizer should
#: not try to be clever about hyphenation or possessives.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")

#: Matches "Item 1A", "ITEM 1A.", "Item 7A -" etc. Anchored to reduce false hits
#: from cross-references like "see Item 1A of this report".
#:
#: Items 2 and 4 are present only because 10-Q filings use a different numbering: a 10-Q's
#: MD&A is "Item 2", where a 10-K's is "Item 7". Without them 1675 of the Stage 2 corpus's
#: 2222 documents had no MD&A section at all, and ``mdna_token_share`` and
#: ``neg_kw_density_mdna`` were structurally zero for three quarters of the sample. The
#: collision is real — a 10-K's "Item 2" is Properties — and is handled by ordering in
#: :data:`MDNA_SECTIONS`, where Item 7 is tried first.
_ITEM_HEADING_RE = re.compile(
    r"(?m)^[ \t]*item[ \t]+(1a|1b|1c|1|2|3|4|5|7a|7|8|9a|9b|9|11|13|15)\b[ \t]*[.:\-)\u2013]?",
    re.IGNORECASE,
)

#: Matches headings in the flattened text layer. A large share of EDGAR primary
#: documents — 419 of this corpus's 547 10-Ks — convert to text with almost no
#: newlines left (one sampled document: 5 newlines in 504,154 characters), so
#: ``(?m)^`` can never fire and every such filing fell back to the unsplit
#: document. What survives the flattening is capitalisation: headings are ALL
#: CAPS (``"... 11 ITEM 1A. RISK FACTORS Our ability ..."``) while prose
#: cross-references are mixed case (``"the risk factors listed in Item 1A could
#: cause"``). Matching uppercase only, with no line anchor, recovers the
#: headings without dragging in the cross-references.
_ITEM_CAPS_HEADING_RE = re.compile(
    r"\bITEM[ \t]+(1A|1B|1C|1|2|3|4|5|7A|7|8|9A|9B|9|11|13|15)\b[ \t]*[.:\-)\u2013]?"
)

#: Matches mixed-case inline headings, which the flattened text layer renders as
#: e.g. ``"... Page Item 1A. Risk Factors Our ability ..."`` — Title case, no
#: line start. This pattern is the riskiest of the three, because prose
#: cross-references are also mixed case; requiring punctuation directly after
#: the item number is the filter that keeps them out ("see Item 1A, Risk
#: Factors" and "listed in Item 1A could cause" carry no punctuation and do not
#: match). Table-of-contents lines do match, but :func:`segment_items` keeps the
#: longest text per label, and a real body heading always spans more than the
#: TOC entry before it. In a 400-document sample of the flattened corpus this
#: recovered Item 1A for all but 21 stylistic outliers.
_ITEM_INLINE_HEADING_RE = re.compile(
    r"\bItem[ \t]+(1A|1B|1C|1|2|3|4|5|7A|7|8|9A|9B|9|11|13|15)\b[ \t]*[.:\-)\u2013\u2014]"
)


def _heading_matches(raw_text: str) -> list[re.Match[str]]:
    """Union of both heading patterns, ordered by position, overlaps resolved.

    A line-anchored ``ITEM 1A`` also matches the caps pattern at the same
    position, so de-duplication by start offset is what keeps them from
    double-counting; a position can only be claimed once, first pattern wins.
    """
    claimed: dict[int, re.Match[str]] = {}
    for pattern in (
        _ITEM_HEADING_RE,
        _ITEM_CAPS_HEADING_RE,
        _ITEM_INLINE_HEADING_RE,
    ):
        for match in pattern.finditer(raw_text):
            claimed.setdefault(match.start(), match)
    return [claimed[position] for position in sorted(claimed)]

#: Canonical formatting for a section label, e.g. "1a" -> "Item 1A".
_ITEM_LABELS: dict[str, str] = {
    "1": "Item 1",
    "1a": "Item 1A",
    "1b": "Item 1B",
    "1c": "Item 1C",
    "2": "Item 2",
    "3": "Item 3",
    "4": "Item 4",
    "5": "Item 5",
    "7": "Item 7",
    "7a": "Item 7A",
    "8": "Item 8",
    "9": "Item 9",
    "9a": "Item 9A",
    "9b": "Item 9B",
    "11": "Item 11",
    "13": "Item 13",
    "15": "Item 15",
}

#: Uncertainty and modal hedging. Dense use of these around a specific topic is a
#: documented correlate of subsequent bad news.
UNCERTAINTY_TERMS: tuple[str, ...] = (
    "approximately",
    "anticipate",
    "anticipates",
    "appear",
    "appears",
    "assume",
    "assumes",
    "believe",
    "believes",
    "can",
    "conceivable",
    "conditional",
    "could",
    "depend",
    "depends",
    "depending",
    "estimate",
    "estimates",
    "expect",
    "expects",
    "fluctuate",
    "fluctuates",
    "indefinite",
    "indicate",
    "indicates",
    "intend",
    "intends",
    "likely",
    "may",
    "might",
    "possible",
    "possibly",
    "predict",
    "predicts",
    "preliminary",
    "presume",
    "probable",
    "probably",
    "project",
    "projects",
    "risk",
    "risks",
    "seek",
    "seeks",
    "should",
    "suggest",
    "suggests",
    "uncertain",
    "uncertainty",
    "unclear",
    "undefined",
    "undetermined",
    "unknown",
    "unpredictable",
    "unproven",
    "variable",
    "vary",
    "varies",
    "would",
)

#: Litigation vocabulary. A step change in these words precedes restatements and
#: enforcement actions often enough to be worth counting.
LITIGIOUS_TERMS: tuple[str, ...] = (
    "accusation",
    "accuse",
    "allegation",
    "allege",
    "alleges",
    "appeal",
    "arbitration",
    "attorney",
    "claim",
    "claims",
    "class action",
    "complaint",
    "consent decree",
    "counterclaim",
    "court",
    "damages",
    "defendant",
    "deposition",
    "discovery",
    "enforcement",
    "fine",
    "grand jury",
    "indictment",
    "injunction",
    "investigation",
    "judgment",
    "lawsuit",
    "litigation",
    "penalty",
    "plaintiff",
    "proceeding",
    "proceedings",
    "regulatory action",
    "sanction",
    "settlement",
    "subpoena",
    "sue",
    "testimony",
    "trial",
    "verdict",
    "violation",
    "violations",
    "wrongdoing",
)

#: Contractual restriction vocabulary: covenants, guarantees and liens.
CONSTRAINING_TERMS: tuple[str, ...] = (
    "covenant",
    "covenants",
    "collateral",
    "cross-default",
    "default",
    "encumbered",
    "guarantee",
    "guarantees",
    "lien",
    "liens",
    "mandatory prepayment",
    "maturity",
    "non-compliance",
    "pledge",
    "refinance",
    "repurchase obligation",
    "restrict",
    "restricted",
    "restriction",
    "restrictions",
    "security interest",
    "waiver",
)

NEGATIVE_TERMS: tuple[str, ...] = (
    "adverse",
    "adversely",
    "breach",
    "challenging",
    "decline",
    "declined",
    "declines",
    "decrease",
    "decreased",
    "deficit",
    "delay",
    "delayed",
    "deteriorate",
    "deterioration",
    "difficulty",
    "dilution",
    "disruption",
    "downturn",
    "fail",
    "failed",
    "failure",
    "impair",
    "impairment",
    "inability",
    "loss",
    "losses",
    "negative",
    "pled",
    "poor",
    "reduction",
    "shortfall",
    "slowdown",
    "terminate",
    "terminated",
    "termination",
    "unfavorable",
    "volatile",
    "weak",
    "weakness",
    "worse",
    "worsened",
)

POSITIVE_TERMS: tuple[str, ...] = (
    "achieve",
    "achieved",
    "benefit",
    "benefits",
    "exceed",
    "exceeded",
    "expansion",
    "favorable",
    "gain",
    "gains",
    "grew",
    "growth",
    "improve",
    "improved",
    "improvement",
    "increase",
    "increased",
    "outperform",
    "positive",
    "profit",
    "profitable",
    "progress",
    "strong",
    "strength",
    "success",
    "successful",
)

#: Multi-word phrases whose presence is a direct going-concern signal. Phrase
#: matching is done on whitespace-normalised text so line breaks do not hide them.
GOING_CONCERN_PHRASES: tuple[str, ...] = (
    "substantial doubt",
    "going concern",
    "going-concern",
    "ability to continue as a going concern",
    "liquidity to fund",
    "insufficient liquidity",
    "unable to meet",
    "covenant violation",
    "covenant waiver",
    "default on",
    "accelerate the maturity",
    "raise substantial doubt",
)

RESTATEMENT_PHRASES: tuple[str, ...] = (
    "restatement",
    "restated",
    "restate",
    "revised financial statements",
    "correction of an error",
    "material weakness",
    "significant deficiency",
    "out-of-period adjustment",
    "error in previously issued",
    "non-reliance",
    "should no longer be relied upon",
)

#: Priority for the risk-factor share feature: Item 1A is the canonical location,
#: 1B and 1C are adjacent and occasionally used instead.
RISK_FACTOR_SECTIONS: tuple[str, ...] = ("Item 1A", "Item 1B", "Item 1C")

#: MD&A, the second-most informative narrative section.
#:
#: Order is load-bearing. A 10-K's "Item 2" is Properties, not MD&A, so Item 7 must win
#: wherever it exists; "Item 2" is listed last and is reached only by filings that have no
#: Item 7 at all — which in practice means 10-Qs. If a future corpus ever contains a 10-K
#: with no Item 7 heading, this fallback would silently measure Properties as MD&A, so
#: ``scripts/fetch_sec_docs.py --report`` prints the per-form resolution rates that would
#: expose it.
MDNA_SECTIONS: tuple[str, ...] = ("Item 7", "Item 7A", "Item 2")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens.

    Numeral and currency tokens are dropped: ``$1,234`` and ``2,021`` are not words,
    and counting them as tokens inflates the density of every lexicon feature in
    numbers-heavy filings.
    """
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def count_terms(tokens: Sequence[str], terms: Iterable[str]) -> int:
    """Count occurrences of ``terms`` in a pre-tokenised sequence.

    Single-word terms are matched against the token set; multi-word terms are joined
    from adjacent tokens so that ``going concern`` is found after tokenisation.

    Args:
        tokens: Output of :func:`tokenize`.
        terms: Terms to count. Multi-word entries are matched on the token stream.

    Returns:
        Total number of occurrences, counting repeats.
    """
    term_set = {term.lower() for term in terms}
    single = {term for term in term_set if " " not in term}
    multi = [tuple(term.split()) for term in term_set if " " in term]

    hits = sum(1 for token in tokens if token in single)
    if multi:
        joined = tokens
        for ngram in multi:
            width = len(ngram)
            for start in range(len(joined) - width + 1):
                if tuple(joined[start : start + width]) == ngram:
                    hits += 1
    return hits


def count_phrases(text: str, phrases: Iterable[str]) -> int:
    """Count phrase occurrences on whitespace-normalised text.

    Used where the phrase may span a line break in the raw filing, which tokens
    joined from a line-oriented parse would miss.
    """
    normalised = " ".join(text.lower().split())
    return sum(normalised.count(phrase.lower()) for phrase in phrases)


def lexicon_density(text: str, terms: Iterable[str], *, per: int = 1000) -> float:
    """Occurrences of ``terms`` per ``per`` tokens. NaN for an empty document."""
    tokens = tokenize(text)
    if not tokens:
        return float("nan")
    return count_terms(tokens, terms) / len(tokens) * per


def segment_items(raw_text: str) -> dict[str, str]:
    """Split a filing into ``Item`` sections.

    Args:
        raw_text: Filing text with HTML already stripped.

    Returns:
        Mapping of canonical section label (``"Item 1A"``) to section text. Sections
        before the first heading are discarded, since they are cover-page boilerplate.
        When fewer than two headings are found the whole text is returned under the
        single key ``"Item 1"`` — a deliberate fallback rather than a guess, and the
        caller can detect it because that key will hold the entire document.
    """
    if not raw_text or not raw_text.strip():
        return {}

    matches = _heading_matches(raw_text)
    if len(matches) < 2:
        logger.debug("found %d Item headings; returning the document unsplit", len(matches))
        return {"Item 1": raw_text}

    segments: dict[str, str] = {}
    for position, match in enumerate(matches):
        label = _ITEM_LABELS.get(match.group(1).lower())
        if label is None:  # pragma: no cover - regex and dict keys are kept in sync
            continue
        start = match.start()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        # Keep the longest text for a repeated label: filings legitimately contain
        # "Item 7" both as a heading and as a cross-reference in the table of contents.
        if label not in segments or len(body) > len(segments[label]):
            segments[label] = body
    return segments


def section_text(segments: Mapping[str, str], candidates: Iterable[str]) -> str:
    """Concatenate the first present section from a priority list."""
    for name in candidates:
        if name in segments and segments[name].strip():
            return segments[name]
    return ""


def document_features(
    raw_text: str, *, segments: Mapping[str, str] | None = None
) -> dict[str, float]:
    """Shallow features for a single disclosure document.

    Args:
        raw_text: Full document text, HTML stripped.
        segments: Pre-computed output of :func:`segment_items`. Computed here when
            omitted; pass it in when the caller needs the segments for other reasons.

    Returns:
        Mapping with token and sentence counts, lexical densities, and the risk-factor
        and MD&A token shares. Empty or whitespace-only input yields a mapping of
        NaNs rather than zeros, because "we have no document" and "we have a document
        with no negative words" are different facts.
    """
    if not raw_text or not raw_text.strip():
        return {
            "disclosure_len_tokens": float("nan"),
            "n_sentences": float("nan"),
            "avg_sentence_tokens": float("nan"),
            "numeric_token_ratio": float("nan"),
            "risk_factor_token_share": float("nan"),
            "mdna_token_share": float("nan"),
            "neg_kw_density_mdna": float("nan"),
            "uncertainty_density": float("nan"),
            "litigious_density": float("nan"),
            "constraining_density": float("nan"),
            "negative_density": float("nan"),
            "positive_density": float("nan"),
            "going_concern_hits": float("nan"),
            "restatement_hits": float("nan"),
        }

    sections = dict(segments) if segments is not None else segment_items(raw_text)
    tokens = tokenize(raw_text)
    n_tokens = len(tokens)
    sentences = [chunk for chunk in _SENTENCE_RE.split(raw_text) if chunk.strip()]
    n_sentences = len(sentences)
    numeric_tokens = len(re.findall(r"\d", raw_text))

    risk_text = section_text(sections, RISK_FACTOR_SECTIONS)
    mdna_text = section_text(sections, MDNA_SECTIONS)

    risk_tokens = len(tokenize(risk_text))
    mdna_tokens = len(tokenize(mdna_text))

    return {
        "disclosure_len_tokens": float(n_tokens),
        "n_sentences": float(n_sentences),
        "avg_sentence_tokens": float(n_tokens / n_sentences) if n_sentences else float("nan"),
        "numeric_token_ratio": float(numeric_tokens / max(len(raw_text), 1)),
        "risk_factor_token_share": float(risk_tokens / n_tokens) if n_tokens else float("nan"),
        "mdna_token_share": float(mdna_tokens / n_tokens) if n_tokens else float("nan"),
        "neg_kw_density_mdna": lexicon_density(mdna_text, NEGATIVE_TERMS),
        "uncertainty_density": lexicon_density(raw_text, UNCERTAINTY_TERMS),
        "litigious_density": lexicon_density(raw_text, LITIGIOUS_TERMS),
        "constraining_density": lexicon_density(raw_text, CONSTRAINING_TERMS),
        "negative_density": lexicon_density(raw_text, NEGATIVE_TERMS),
        "positive_density": lexicon_density(raw_text, POSITIVE_TERMS),
        "going_concern_hits": float(count_phrases(raw_text, GOING_CONCERN_PHRASES)),
        "restatement_hits": float(count_phrases(raw_text, RESTATEMENT_PHRASES)),
    }


def aggregate_news_features(
    news: pd.DataFrame,
    as_of: date | pd.Timestamp,
    *,
    windows: Sequence[int] = NEWS_WINDOWS,
    sentiment_column: str = "sentiment",
    negative_threshold: float = -0.2,
    date_column: str = "published",
) -> dict[str, float]:
    """Aggregate news flow up to and including ``as_of``.

    Only articles published at or before ``as_of`` are considered — a news feature
    that saw tomorrow's headlines is the most common leakage in sentiment pipelines,
    and it is worth restating here because the bug is usually in the caller, not in
    this function.

    Args:
        news: Frame with ``date_column`` and optionally ``sentiment_column``. Rows
            after ``as_of`` are ignored rather than raising, so that the caller can
            pass the full history safely.
        as_of: Cutoff date, inclusive.
        windows: Lookback lengths in calendar days.
        sentiment_column: Name of the sentiment score column.
        negative_threshold: Sentiment at or below this counts as negative.
        date_column: Name of the publication date column.

    Returns:
        Mapping with ``n_news_<window>d`` counts, ``sent_mean_<window>d``,
        ``sent_std_<window>d`` and ``sent_neg_share_<window>d`` for each window. The
        shortest window's keys are the canonical ones consumed by the panel; the
        longer ones are extra context for the report. Empty flow yields zero counts
        and NaN sentiment statistics.
    """
    cutoff = pd.Timestamp(as_of)
    features: dict[str, float] = {}

    if news.empty:
        for window in windows:
            features[f"n_news_{window}d"] = 0.0
            features[f"sent_mean_{window}d"] = float("nan")
            features[f"sent_std_{window}d"] = float("nan")
            features[f"sent_neg_share_{window}d"] = float("nan")
        return features

    published = pd.to_datetime(news[date_column])
    history = news.loc[published <= cutoff].copy()
    history["_published"] = published[published <= cutoff]
    has_sentiment = sentiment_column in history.columns

    for window in windows:
        lower = cutoff - pd.Timedelta(days=window)
        window_rows = history.loc[history["_published"] > lower]
        features[f"n_news_{window}d"] = float(len(window_rows))
        if has_sentiment and len(window_rows):
            sentiment = pd.to_numeric(window_rows[sentiment_column], errors="coerce")
            features[f"sent_mean_{window}d"] = float(sentiment.mean())
            features[f"sent_std_{window}d"] = (
                float(sentiment.std(ddof=1)) if sentiment.notna().sum() > 1 else float("nan")
            )
            features[f"sent_neg_share_{window}d"] = float((sentiment <= negative_threshold).mean())
        else:
            features[f"sent_mean_{window}d"] = float("nan")
            features[f"sent_std_{window}d"] = float("nan")
            features[f"sent_neg_share_{window}d"] = float("nan")
    return features


def build_text_features(
    disclosures: Sequence[str],
    news: pd.DataFrame,
    as_of: date | pd.Timestamp,
    *,
    prior_disclosure_len_tokens: float | None = None,
) -> dict[str, float]:
    """Assemble the canonical ``text_counts`` feature block.

    Args:
        disclosures: Disclosure documents available at ``as_of``, most recent last.
            The last one is treated as the current filing.
        news: News frame; rows after ``as_of`` are ignored.
        as_of: Cutoff date.
        prior_disclosure_len_tokens: Token count of the same company's previous
            filing, for ``disclosure_len_chg``. Pass None when there is no prior
            filing, which yields NaN rather than a spurious 100% growth.

    Returns:
        Mapping containing exactly the columns in
        :data:`shingan.data.schema.TEXT_COUNT_COLUMNS`. Any column whose input is
        unavailable is NaN, so the panel schema does not depend on data availability.
    """
    document = document_features(disclosures[-1]) if disclosures else document_features("")
    news_features = aggregate_news_features(news, as_of)

    current_tokens = document.get("disclosure_len_tokens", float("nan"))
    # Change in filing length against the previous filing, or NaN when there is no
    # comparable prior disclosure. Written as an explicit branch rather than a
    # conditional expression so that the denominator is narrowed to a float before
    # it is divided by: a missing prior filing must produce NaN, not a TypeError.
    length_change = float("nan")
    if (
        prior_disclosure_len_tokens is not None
        and np.isfinite(prior_disclosure_len_tokens)
        and prior_disclosure_len_tokens > 0
        and np.isfinite(current_tokens)
    ):
        length_change = current_tokens / prior_disclosure_len_tokens - 1.0

    values: dict[str, float] = {
        "n_news_30d": news_features.get("n_news_30d", float("nan")),
        "n_news_90d": news_features.get("n_news_90d", float("nan")),
        "sent_mean_30d": news_features.get("sent_mean_30d", float("nan")),
        "sent_std_30d": news_features.get("sent_std_30d", float("nan")),
        "sent_neg_share_30d": news_features.get("sent_neg_share_30d", float("nan")),
        "neg_kw_density_mdna": document.get("neg_kw_density_mdna", float("nan")),
        "risk_factor_token_share": document.get("risk_factor_token_share", float("nan")),
        "disclosure_len_tokens": current_tokens,
        "disclosure_len_chg": length_change,
        "going_concern_hits": document.get("going_concern_hits", float("nan")),
        "uncertainty_hits": document.get("uncertainty_density", float("nan")),
        "restatement_hits": document.get("restatement_hits", float("nan")),
    }

    missing = [column for column in TEXT_COUNT_COLUMNS if column not in values]
    if missing:  # pragma: no cover - guards against a schema addition
        raise RuntimeError(
            f"build_text_features does not produce these schema columns: {missing}. "
            "Add them here and to docs/02-data.md section 5.4."
        )

    cleaned: dict[str, float] = {}
    for column in TEXT_COUNT_COLUMNS:
        value = values[column]
        if isinstance(value, float) and not np.isfinite(value):
            cleaned[column] = float("nan")
        else:
            cleaned[column] = float(value)
    return cleaned


def lexicon_summary() -> pd.DataFrame:
    """Describe the bundled lexicons.

    Used by the report so that a reader can see exactly how many terms each feature
    is built from, which is the honest way to present counts derived from an abridged
    dictionary.

    Returns:
        A frame with one row per lexicon: name, term count and version.
    """
    rows = [
        ("uncertainty", len(UNCERTAINTY_TERMS)),
        ("litigious", len(LITIGIOUS_TERMS)),
        ("constraining", len(CONSTRAINING_TERMS)),
        ("negative", len(NEGATIVE_TERMS)),
        ("positive", len(POSITIVE_TERMS)),
        ("going_concern_phrases", len(GOING_CONCERN_PHRASES)),
        ("restatement_phrases", len(RESTATEMENT_PHRASES)),
    ]
    return pd.DataFrame(rows, columns=["lexicon", "n_terms"]).assign(version=LEXICON_VERSION)


__all__ = [
    "CONSTRAINING_TERMS",
    "GOING_CONCERN_PHRASES",
    "LEXICON_VERSION",
    "LITIGIOUS_TERMS",
    "MDNA_SECTIONS",
    "NEGATIVE_TERMS",
    "NEWS_WINDOWS",
    "POSITIVE_TERMS",
    "RESTATEMENT_PHRASES",
    "RISK_FACTOR_SECTIONS",
    "UNCERTAINTY_TERMS",
    "aggregate_news_features",
    "build_text_features",
    "count_phrases",
    "count_terms",
    "document_features",
    "lexicon_density",
    "lexicon_summary",
    "section_text",
    "segment_items",
    "tokenize",
]

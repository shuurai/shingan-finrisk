"""Risk taxonomy, output contract and the processed-panel data dictionary.

This module is the single place where the project's vocabulary is defined. Two
things live here that everything else depends on:

1. **The output contract.** A risk assessment is not a number. It is a label, a
   severity, a calibrated score, the horizon it applies to, and — critically —
   the verbatim evidence the model relied on. In a regulated setting "the model
   says 0.71" is not actionable and not reviewable; "the model says 0.71 because
   of these three quoted passages from these two documents" is.

2. **The panel data dictionary, as code.** ``PANEL_COLUMNS`` and its groups must
   match section 5 of ``docs/02-data.md`` exactly. The builder asserts against
   them, the feature selector reads them, and the leakage guard uses them to
   decide what may never enter a feature matrix.

Logically-independent note on naming: the length prefixes matter. ``label_``,
``fwd_``, ``event_``, ``source_of_record_`` and ``horizon_days_`` are all
forbidden as model inputs, and that is enforced, not conventional. See
:mod:`shingan.leakage`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shingan.logging_utils import get_logger

logger = get_logger(__name__)


class RiskLabel(str, Enum):
    """The three risk labels this POC predicts.

    The string values are the canonical identifiers used in every JSONL file, CSV
    column and report. Do not rename them without bumping the dataset version.
    """

    DEFAULT_RISK = "default_risk"
    FRAUD_RISK = "fraud_risk"
    TAIL_RISK = "tail_risk"

    def __str__(self) -> str:
        return self.value


class OutOfScopeLabel(str, Enum):
    """Risks the project acknowledges but does not predict in the POC.

    Kept as an explicit enumeration rather than omitted, so that the taxonomy is
    visibly incomplete by design rather than incomplete by oversight.
    """

    LIQUIDITY_RISK = "liquidity_risk"
    EVENT_DRIVEN_RISK = "event_driven_risk"
    MACRO_CONTAGION_RISK = "macro_contagion_risk"


class Severity(str, Enum):
    """Ordinal bucket for a risk score, for human consumption only.

    Never used as a target: bucketing a calibrated probability into four levels
    throws away the information that calibration exists to preserve.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Ordinal position, 0 for ``low``."""
        return _SEVERITY_RANK[self]

    @classmethod
    def from_score(cls, score: float) -> Severity:
        """Bucket a score in [0, 1] into a severity.

        The edges (0.10 / 0.25 / 0.50) are presentation choices, not statistical
        ones. At a 1% base rate even a well-behaved model rarely exceeds 0.5, so
        most genuinely elevated rows will read as ``medium`` — which is honest,
        and is why the score, not the bucket, is the deliverable.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        if score < 0.10:
            return cls.LOW
        if score < 0.25:
            return cls.MEDIUM
        if score < 0.50:
            return cls.HIGH
        return cls.CRITICAL


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class SourceType(str, Enum):
    """Where a piece of evidence came from."""

    FILING = "filing"
    NEWS = "news"
    PRICE = "price"
    STRUCTURED = "structured"
    OTHER = "other"


class EvidenceSpan(BaseModel):
    """A quoted passage the assessment depends on.

    ``quote`` must appear verbatim in the document identified by ``source_ref``.
    That is checked by :func:`quotes_are_verbatim`; a model that paraphrases its
    evidence is producing an unverifiable claim.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_ref: str = Field(
        description="Identifier of the quoted document, e.g. '10-K:2019-02-26:Item 1A' "
        "for a filing excerpt or 'news:reuters:2020-03-01' for a news item. These are "
        "the shapes FilingExcerpt.source_ref and NewsItem.source_ref render, and the "
        "shapes SYSTEM_PROMPT asks the model to reproduce."
    )
    quote: str = Field(min_length=1)
    section: str | None = None
    published: date | None = None


class RiskAssessment(BaseModel):
    """The text track's structured output, and the schema the LLM must emit.

    The model is instruction-tuned to produce exactly this object as JSON. Parsing
    is strict: an assessment whose evidence cannot be verified is dropped rather
    than silently accepted, because a risk report with fabricated citations is
    worse than no report.
    """

    model_config = ConfigDict(extra="forbid")

    label: RiskLabel
    severity: Severity
    score: float = Field(ge=0.0, le=1.0)
    horizon_days: int = Field(ge=1)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    catalysts: list[str] = Field(
        default_factory=list,
        description="Forward-looking triggers the documents imply but do not state as facts.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="What the provided documents do NOT establish. Required to be "
        "non-empty when severity is low because the documents were uninformative, "
        "as opposed to informative and reassuring.",
    )

    @field_validator("score")
    @classmethod
    def _finite(cls, value: float) -> float:
        if value != value:  # NaN
            raise ValueError("score must be a finite number, got NaN")
        return value

    @model_validator(mode="after")
    def _reasons_present(self) -> RiskAssessment:
        if not self.reasons:
            raise ValueError("an assessment with no reasons is not reviewable")
        return self

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise to canonical JSON, sorted keys, ASCII-safe."""
        return self.model_dump_json(indent=indent)


@dataclass(slots=True)
class SampleRecord:
    """One training or evaluation example for the text track.

    Deliberately a dataclass rather than a pydantic model: these are produced in
    the hundreds of thousands, and the validation cost of a model per row is not
    worth paying for a structure the builder has already checked.
    """

    sample_id: str
    ticker: str
    company_name: str
    as_of: date
    label: RiskLabel
    label_binary: int
    horizon_days: int
    text: str
    structured: dict[str, float]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping suitable for JSONL."""
        return {
            "sample_id": self.sample_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "as_of": self.as_of.isoformat(),
            "label": str(self.label),
            "label_binary": self.label_binary,
            "horizon_days": self.horizon_days,
            "text": self.text,
            "structured": self.structured,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Panel data dictionary.
#
# These tuples are the code form of docs/02-data.md section 5. Column order in a
# written parquet follows the order here, which keeps diffs of the dataset
# readable. Adding a column means adding it here AND to the doc, in one change.
# ---------------------------------------------------------------------------

ID_COLUMNS: tuple[str, ...] = ("ticker", "cik", "company_name", "sector", "as_of")

SPLIT_COLUMNS: tuple[str, ...] = ("split",)

META_COLUMNS: tuple[str, ...] = (
    "is_synthetic",
    "data_version",
    "n_sources",
    "insufficient_history",
)

RATIO_COLUMNS: tuple[str, ...] = (
    "debt_to_equity",
    "debt_short_term_ratio",
    "current_ratio",
    "interest_coverage",
    "net_margin",
    "roa",
    "roe",
    "fcf_margin",
    "altman_z",
    "accruals_ratio",
    "revenue_yoy",
    "asset_growth",
    "goodwill_to_assets",
    "ratios_missing_frac",
)

TECHNICAL_COLUMNS: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_252d",
    "vol_20d",
    "vol_60d",
    "downside_vol_60d",
    "skew_60d",
    "kurt_60d",
    "max_drawdown_60d",
    "dist_52w_high",
    "rsi_14",
    "macd_hist",
    "atr_14",
    "adx_14",
    "amihud_illiq_20d",
    "turnover_20d",
    "beta_252d",
    "abnormal_volume_20d",
    "vix_level",
    "vix_chg_5d",
    "credit_spread_chg_20d",
)

TEXT_COUNT_COLUMNS: tuple[str, ...] = (
    "n_news_30d",
    "n_news_90d",
    "sent_mean_30d",
    "sent_std_30d",
    "sent_neg_share_30d",
    "neg_kw_density_mdna",
    "risk_factor_token_share",
    "disclosure_len_tokens",
    "disclosure_len_chg",
    "going_concern_hits",
    "uncertainty_hits",
    "restatement_hits",
)

LABEL_COLUMNS: tuple[str, ...] = tuple(f"label_{label.value}" for label in RiskLabel)
LABEL_MASK_COLUMNS: tuple[str, ...] = tuple(f"label_mask_{label.value}" for label in RiskLabel)
EVENT_COLUMNS: tuple[str, ...] = tuple(f"event_date_{label.value}" for label in RiskLabel)
SOURCE_OF_RECORD_COLUMNS: tuple[str, ...] = tuple(
    f"source_of_record_{label.value}" for label in RiskLabel
)
HORIZON_COLUMNS: tuple[str, ...] = tuple(f"horizon_days_{label.value}" for label in RiskLabel)

FORWARD_COLUMNS: tuple[str, ...] = ("fwd_ret_21d", "fwd_realized_vol_21d", "fwd_max_drawdown_30d")

WEIGHT_COLUMNS: tuple[str, ...] = ("sample_weight",)

#: The data leg that supplies each feature. A feature that is absent everywhere is
#: otherwise indistinguishable from a feature whose source was never wired, and the two
#: call for opposite responses: the first is a bug in this repository, the second is a
#: fetch that has not been run. Naming the source turns "15 features were dropped" into
#: a work item.
SOURCE_XBRL = "sec_xbrl_fundamentals"
SOURCE_PRICES = "prices_ohlcv"
SOURCE_PRICES_SHARES = "prices_ohlcv+shares_outstanding"
SOURCE_PRICES_BENCHMARK = "prices_ohlcv+index_benchmark"
SOURCE_MARKET_VOL = "market_volatility_index"
SOURCE_MACRO_CREDIT = "macro_credit_spread"
SOURCE_FILING_TEXT = "sec_filing_text"
SOURCE_NEWS = "news_feed"

#: What to do about a gap, per source. Written as an instruction rather than an apology:
#: a caveat that only says "this was missing" leaves the reader to guess whether it is
#: recoverable, and the answers here range from "one more symbol" to "no source exists".
SOURCE_ADVICE: dict[str, str] = {
    SOURCE_XBRL: (
        "the XBRL fetch already runs; a gap here means the concept mapping in "
        "scripts/fetch_real.py lacks a tag this filer uses, so extend CONCEPTS_BY_COLUMN"
    ),
    SOURCE_PRICES: (
        "the price fetch already runs; a gap here means the ticker returned no series "
        "(delisted or unavailable) and the universe entry should be replaced or dropped"
    ),
    SOURCE_PRICES_SHARES: (
        "recoverable for free: XBRL dei:EntityCommonStockSharesOutstanding carries a filed "
        "date per observation, so turnover_20d needs no new vendor"
    ),
    SOURCE_PRICES_BENCHMARK: (
        "one extra symbol: fetch ^GSPC alongside the universe and regress on it"
    ),
    SOURCE_MARKET_VOL: "one extra symbol: fetch ^VIX alongside the universe",
    SOURCE_MACRO_CREDIT: (
        "one series from FRED (BAA10Y or the HY OAS index); not wired at all today"
    ),
    SOURCE_FILING_TEXT: (
        "run scripts/fetch_sec_docs.py — the archive is reachable and the accession "
        "numbers to address it are already in the filings table"
    ),
    SOURCE_NEWS: (
        "no source wired. FNSPID is a snapshot, not a feed; a live vendor or an "
        "archive are both open decisions, so treat every news-derived column as absent"
    ),
}

#: Per-feature source. Groups first, then the features that do not follow their group.
FEATURE_SOURCES: dict[str, str] = {
    **{name: SOURCE_XBRL for name in RATIO_COLUMNS if name != "ratios_missing_frac"},
    **{name: SOURCE_PRICES for name in TECHNICAL_COLUMNS},
    **{name: SOURCE_NEWS for name in ("n_news_30d", "n_news_90d")},
}
FEATURE_SOURCES.update(
    {
        "turnover_20d": SOURCE_PRICES_SHARES,
        "beta_252d": SOURCE_PRICES_BENCHMARK,
        "vix_level": SOURCE_MARKET_VOL,
        "vix_chg_5d": SOURCE_MARKET_VOL,
        "credit_spread_chg_20d": SOURCE_MACRO_CREDIT,
        # Not a ratio, but it is the fraction of the ratio block that is absent, so it is
        # owed by the same fetch and the same concept mapping. It used to be excluded from
        # the RATIO_COLUMNS comprehension above and never assigned anywhere else, which left
        # it with no declared source: a gap in this column produced the placeholder remedy
        # instead of the XBRL one, and `test_every_feature_column_has_a_declared_source`
        # now fails if that recurs.
        "ratios_missing_frac": SOURCE_XBRL,
        "sent_mean_30d": SOURCE_NEWS,
        "sent_std_30d": SOURCE_NEWS,
        "sent_neg_share_30d": SOURCE_NEWS,
        **{
            name: SOURCE_FILING_TEXT
            for name in (
                "neg_kw_density_mdna",
                "risk_factor_token_share",
                "disclosure_len_tokens",
                "disclosure_len_chg",
                "going_concern_hits",
                "uncertainty_hits",
                "restatement_hits",
            )
        },
    }
)


def diagnose_gaps(features: Iterable[str]) -> list[dict[str, Any]]:
    """Group absent features by the source that owes them, with the remedy for each.

    Args:
        features: Feature names that were dropped or that held no observation.

    Returns:
        One record per source, ordered by how many features it accounts for: ``source``,
        ``features`` (sorted), ``count``, ``advice``. Unknown names are grouped under
        ``"unmapped"`` rather than dropped, because a feature with no declared source is
        itself a defect worth seeing.
    """
    grouped: dict[str, list[str]] = {}
    for name in features:
        grouped.setdefault(FEATURE_SOURCES.get(name, "unmapped"), []).append(name)
    records = [
        {
            "source": source,
            "features": sorted(names),
            "count": len(names),
            "advice": SOURCE_ADVICE.get(
                source, "no declared source — add it to FEATURE_SOURCES in schema.py"
            ),
        }
        for source, names in grouped.items()
    ]
    records.sort(key=lambda record: (-record["count"], record["source"]))
    return records

#: Every column of the processed panel, in the order it is written.
PANEL_COLUMNS: tuple[str, ...] = (
    *ID_COLUMNS,
    *SPLIT_COLUMNS,
    *META_COLUMNS,
    *RATIO_COLUMNS,
    *TECHNICAL_COLUMNS,
    *TEXT_COUNT_COLUMNS,
    *LABEL_COLUMNS,
    *LABEL_MASK_COLUMNS,
    *EVENT_COLUMNS,
    *SOURCE_OF_RECORD_COLUMNS,
    *HORIZON_COLUMNS,
    *FORWARD_COLUMNS,
    *WEIGHT_COLUMNS,
)

#: Column groups used by the structured track's feature selection and by the
#: ablation table. The key is the name used in ``structured.feature_groups``.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "ratios": RATIO_COLUMNS,
    "technical": TECHNICAL_COLUMNS,
    "text_counts": TEXT_COUNT_COLUMNS,
}

#: Prefixes that must never appear in a feature matrix. Anything derived from a
#: label, a forward return or an event date is future information by construction.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "label_",
    "fwd_",
    "event_",
    "source_of_record_",
    "horizon_days_",
)

#: Columns that are not features but are legitimately needed at training time.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    *ID_COLUMNS,
    *SPLIT_COLUMNS,
    *META_COLUMNS,
    *LABEL_COLUMNS,
    *LABEL_MASK_COLUMNS,
    *EVENT_COLUMNS,
    *SOURCE_OF_RECORD_COLUMNS,
    *HORIZON_COLUMNS,
    *FORWARD_COLUMNS,
    *WEIGHT_COLUMNS,
)


def label_column(label: RiskLabel | str) -> str:
    """``RiskLabel.TAIL_RISK`` -> ``'label_tail_risk'``."""
    return f"label_{RiskLabel(label).value}"


def mask_column(label: RiskLabel | str) -> str:
    """``RiskLabel.TAIL_RISK`` -> ``'label_mask_tail_risk'``."""
    return f"label_mask_{RiskLabel(label).value}"


def event_column(label: RiskLabel | str) -> str:
    """``RiskLabel.TAIL_RISK`` -> ``'event_date_tail_risk'``."""
    return f"event_date_{RiskLabel(label).value}"


def horizon_column(label: RiskLabel | str) -> str:
    """``RiskLabel.TAIL_RISK`` -> ``'horizon_days_tail_risk'``."""
    return f"horizon_days_{RiskLabel(label).value}"


def source_of_record_column(label: RiskLabel | str) -> str:
    """``RiskLabel.FRAUD_RISK`` -> ``'source_of_record_fraud_risk'``."""
    return f"source_of_record_{RiskLabel(label).value}"


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def write_jsonl(records: Iterable[dict[str, Any]], path: Path | str) -> int:
    """Write an iterable of mappings as JSONL.

    Parent directories are created. The file is written with UTF-8 and LF line
    endings explicitly, so a Windows run produces byte-identical output to a Linux
    run — which matters because these files are hashed into run metadata.

    Args:
        records: Mappings to serialise. Values must be JSON-serialisable; use
            ``.isoformat()`` on dates before calling.
        path: Destination file, overwritten if present.

    Returns:
        The number of records written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: Path | str, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield mappings from a JSONL file.

    Blank lines are skipped. Malformed lines raise rather than being ignored: in a
    training set, silently dropping a line is a silent change to the label
    distribution.

    Args:
        path: File to read.
        limit: Optional cap on the number of records yielded.

    Yields:
        One mapping per non-blank line.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {lineno} of {source}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {lineno} of {source} is not a JSON object")
            yield payload
            if limit is not None and lineno >= limit:
                break


def quotes_are_verbatim(assessment: RiskAssessment, documents: Sequence[str]) -> bool:
    """Whether every quoted span appears verbatim in the supplied documents.

    Whitespace is normalised on both sides before comparison, because extracted
    filing text routinely contains non-breaking spaces, hard wraps and repeated
    spaces that a model will reproduce as ordinary spaces. Nothing else is
    normalised: case, punctuation and wording must match exactly, since the whole
    point is to catch paraphrase and fabrication.

    Args:
        assessment: The parsed model output.
        documents: The full text(s) that were placed in the prompt.

    Returns:
        True when every evidence quote is found. An assessment with no evidence
        returns False — absence of evidence is not verification.
    """
    if not assessment.evidence:
        return False
    haystacks = [_normalise_ws(document) for document in documents]
    return all(
        any(_normalise_ws(span.quote) in haystack for haystack in haystacks)
        for span in assessment.evidence
    )


def _normalise_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return " ".join(text.split())


def panel_subset(groups: Sequence[str]) -> tuple[str, ...]:
    """Return the columns belonging to the named feature groups.

    Args:
        groups: Any of ``"ratios"``, ``"technical"``, ``"text_counts"``.

    Returns:
        The concatenated column tuple, in group order, without duplicates.

    Raises:
        KeyError: If a group name is unknown. Raising is preferable to returning a
            short list, because a silently empty feature set trains a model that
            looks like it works and predicts the base rate.
    """
    columns: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group {group!r}; known: {sorted(FEATURE_GROUPS)}")
        for column in FEATURE_GROUPS[group]:
            if column not in columns:
                columns.append(column)
    return tuple(columns)


__all__ = [
    "EVENT_COLUMNS",
    "FEATURE_GROUPS",
    "FEATURE_SOURCES",
    "FORBIDDEN_PREFIXES",
    "FORWARD_COLUMNS",
    "HORIZON_COLUMNS",
    "ID_COLUMNS",
    "LABEL_COLUMNS",
    "LABEL_MASK_COLUMNS",
    "META_COLUMNS",
    "NON_FEATURE_COLUMNS",
    "PANEL_COLUMNS",
    "RATIO_COLUMNS",
    "SOURCE_ADVICE",
    "SOURCE_FILING_TEXT",
    "SOURCE_MACRO_CREDIT",
    "SOURCE_MARKET_VOL",
    "SOURCE_NEWS",
    "SOURCE_OF_RECORD_COLUMNS",
    "SOURCE_PRICES",
    "SOURCE_PRICES_BENCHMARK",
    "SOURCE_PRICES_SHARES",
    "SOURCE_XBRL",
    "TECHNICAL_COLUMNS",
    "TEXT_COUNT_COLUMNS",
    "WEIGHT_COLUMNS",
    "diagnose_gaps",
    "EvidenceSpan",
    "OutOfScopeLabel",
    "RiskAssessment",
    "RiskLabel",
    "SampleRecord",
    "Severity",
    "SourceType",
    "event_column",
    "horizon_column",
    "label_column",
    "mask_column",
    "panel_subset",
    "quotes_are_verbatim",
    "read_jsonl",
    "source_of_record_column",
    "write_jsonl",
]

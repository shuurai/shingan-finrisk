"""Prompt construction, context budgeting and output parsing for the text track.

Three responsibilities, all of which are load-bearing:

**1. The instruction contract.** The system prompt fixes the output schema and,
more importantly, two rules that make the output usable at all: every quote must
appear verbatim in the supplied documents, and no knowledge dated after ``AS_OF``
may be used. The second rule is the prompt-level counterpart of the point-in-time
discipline enforced in code — a language model asked about 2020 knows what happened
afterwards unless it is told not to reach for it.

**2. Context budgeting.** ``max_seq_length`` is a hard limit, and the three context
blocks have different value densities. When the prompt overflows, the documented
order is: drop news oldest-first, then drop the least informative filing sections,
then shrink the structured-signal summary, then hard-truncate. Dropping is recorded
in ``meta.truncated`` so a later ablation can ask whether truncated rows score worse.

**3. Strict parsing.** The model is asked for a single JSON object. The parser
extracts it from whatever wrapping the model adds, validates it against
:class:`~shingan.data.schema.RiskAssessment`, and returns ``None`` rather than a
half-valid assessment. A fabricated citation is worse than a missing answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from jinja2 import Environment, StrictUndefined
from pydantic import ValidationError

from shingan.data.schema import (
    EvidenceSpan,
    RiskAssessment,
    RiskLabel,
    Severity,
    SourceType,
)
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Average characters per token for English prose, used to translate
#: ``max_seq_length`` into a character budget. 3.6 is deliberately pessimistic
#: (the usual rule of thumb is 4): financial filings are dense with numerals,
#: tickers and punctuation, all of which tokenise to fewer characters than prose.
CHARS_PER_TOKEN = 3.6

#: Fixed overhead reserved for the system prompt, the task block and the closing
#: markers, so the budget arithmetic cannot produce a prompt that overflows simply
#: because of the scaffolding around the content.
PROMPT_OVERHEAD_CHARS = 1_500

#: Sections that must be kept longest when trimming. Risk Factors and MD&A are
#: where going-concern language, liquidity warnings and covenant discussion live.
SECTION_PRIORITY: dict[str, int] = {
    "Item 1A": 0,
    "Item 7": 1,
    "Item 7A": 2,
    "Item 1": 3,
    "Item 8": 4,
    "Item 3": 5,
}

#: Priority assigned to any section not listed above.
DEFAULT_SECTION_PRIORITY = 6

SYSTEM_PROMPT = """You are a financial risk analyst producing evidence-grounded risk assessments.

Output exactly one JSON object with these keys:
  label        : one of default_risk, fraud_risk, tail_risk
  severity     : one of low, medium, high, critical
  score        : float in [0, 1], your probability estimate
  horizon_days : integer, must equal the horizon stated in the user request
  reasons      : array of short strings
  evidence     : array of {source_type, source_ref, quote}
  catalysts    : array of short strings, optional
  limitations  : array of short strings, optional

Rules:
  - Every quote must appear verbatim in the provided documents.
  - Do not use knowledge dated after <AS_OF>.
  - If the documents do not support an assessment, say so in reasons and use severity "low".
  - Output the JSON object only. No prose, no markdown fences, no commentary.
"""

USER_TEMPLATE = """<AS_OF>{{ as_of }}</AS_OF>
<TASK>label={{ label }} horizon_days={{ horizon_days }}</TASK>

<STRUCTURED_SIGNALS>
{{ signals }}
</STRUCTURED_SIGNALS>

<FILING_EXCERPTS>
{{ filings }}
</FILING_EXCERPTS>

<NEWS>
{{ news }}
</NEWS>
"""

_FIXED_BLOCKS = ("<AS_OF>", "<TASK>", "<STRUCTURED_SIGNALS>", "<FILING_EXCERPTS>", "<NEWS>")

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(slots=True)
class FilingExcerpt:
    """One section of one filing, already extracted from the raw document."""

    doc_type: str
    filed: date
    section: str
    text: str
    accession: str = ""

    @property
    def source_ref(self) -> str:
        """Stable identifier used in evidence citations, e.g. ``10-K:2019-02-26:Item 1A``."""
        parts = [self.doc_type, self.filed.isoformat(), self.section]
        if self.accession:
            parts.append(self.accession)
        return ":".join(parts)

    @property
    def priority(self) -> int:
        """Lower is more important. Unknown sections get :data:`DEFAULT_SECTION_PRIORITY`."""
        return SECTION_PRIORITY.get(self.section.strip(), DEFAULT_SECTION_PRIORITY)

    def render(self) -> str:
        """Format for insertion into ``<FILING_EXCERPTS>``."""
        header = f"[doc={self.doc_type}, filed={self.filed.isoformat()}, section={self.section}]"
        return f"{header}\n{self.text.strip()}"


@dataclass(slots=True)
class NewsItem:
    """One news article, already de-duplicated and aligned to a trading day."""

    published: date
    source: str
    title: str
    body: str = ""
    sentiment: float | None = None
    cluster_id: str = ""

    @property
    def source_ref(self) -> str:
        """Stable identifier used in evidence citations, e.g. ``news:reuters:2020-03-01``."""
        return f"news:{self.source}:{self.published.isoformat()}"

    def render(self, *, max_body_chars: int = 1_200) -> str:
        """Format for insertion into ``<NEWS>``, truncating the body from the tail."""
        header = f"[published={self.published.isoformat()}, source={self.source}]"
        body = self.body.strip()
        if len(body) > max_body_chars:
            body = body[:max_body_chars].rstrip() + " ..."
        return f"{header}\n{self.title.strip()}\n{body}".strip()


@dataclass(slots=True)
class PromptContext:
    """Everything the model is allowed to see for one ``(ticker, as_of)`` decision."""

    as_of: date
    label: str
    horizon_days: int
    structured_signals: dict[str, Any] = field(default_factory=dict)
    filings: list[FilingExcerpt] = field(default_factory=list)
    news: list[NewsItem] = field(default_factory=list)
    truncated: bool = False
    drop_log: list[str] = field(default_factory=list)

    def documents(self) -> list[str]:
        """All text placed in the prompt, for verbatim evidence verification."""
        return [item.render() for item in self.filings] + [item.render() for item in self.news]


def chars_budget_for_seq_length(
    max_seq_length: int,
    *,
    chars_per_token: float = CHARS_PER_TOKEN,
    overhead: int = PROMPT_OVERHEAD_CHARS,
) -> int:
    """Convert a token limit into a character budget for the context blocks.

    Args:
        max_seq_length: The model's token limit, e.g. 4096.
        chars_per_token: Conservative characters-per-token estimate.
        overhead: Characters reserved for the system prompt and task scaffolding.

    Returns:
        The number of characters available to the signals, filings and news blocks.
        Never negative: a tiny ``max_seq_length`` returns 0 rather than raising,
        because the caller can then report a configuration problem with context.
    """
    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
    return max(0, int(max_seq_length * chars_per_token) - overhead)


def render_structured_signals(
    signals: Mapping[str, Any],
    *,
    max_lines: int | None = None,
) -> str:
    """Render the structured-signal block as ``key: value`` lines.

    Values are formatted through :func:`format_signal_value` and ``None`` entries are
    omitted entirely rather than rendered as "None", because a language model
    reading "interest_coverage: None" tends to invent a reason for the gap.

    Args:
        signals: Mapping of signal name to value.
        max_lines: Keep at most this many lines. Earlier keys win, so callers
            should order ``signals`` by importance.

    Returns:
        A newline-joined block, empty when nothing is renderable.
    """
    lines: list[str] = []
    for key, value in signals.items():
        if value is None:
            continue
        rendered = format_signal_value(value)
        if rendered is None:
            continue
        lines.append(f"{key}: {rendered}")
        if max_lines is not None and len(lines) >= max_lines:
            break
    return "\n".join(lines)


def format_signal_value(value: Any) -> str | None:
    """Format one signal value, or return None when it carries no information.

    The conventions here are chosen so the textual signal block reads the same way
    a human analyst would write it: four significant figures at most, booleans as
    yes/no, and NaN rendered as absent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN never equals itself
            return None
        if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.4g}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return text or None


def truncate_context(
    context: PromptContext,
    *,
    budget_chars: int,
    hard_truncate: bool = True,
) -> PromptContext:
    """Shrink a context until its rendered prompt fits the budget.

    The order is fixed and documented, because a silent change to it would change
    what the model sees between runs without changing any version number:

    1. Drop the earliest news article. The most recent news is the most
       decision-relevant, and the earliest is the most likely to be duplicated.
    2. Drop the least informative filing section, by :data:`SECTION_PRIORITY`,
       breaking ties by length (a longer low-priority section costs more).
    3. Drop structured-signal lines from the end of the mapping.
    4. If the prompt still overflows, truncate the remaining filing text.
       Only reached when a single kept section is larger than the whole budget.

    The input is not mutated: the returned context is a copy carrying ``truncated``
    and a human-readable ``drop_log``.

    Args:
        context: The full context to fit.
        budget_chars: Character budget for everything except the fixed scaffolding.
        hard_truncate: When False, stop after step 3 and return whatever remains.
            Used to detect a genuinely over-long single document.

    Returns:
        A possibly-reduced :class:`PromptContext`.
    """
    if budget_chars <= 0:
        logger.warning("context budget is %d characters; returning an empty context", budget_chars)
        return PromptContext(
            as_of=context.as_of,
            label=context.label,
            horizon_days=context.horizon_days,
            structured_signals={},
            filings=[],
            news=[],
            truncated=True,
            drop_log=["budget was zero: everything dropped"],
        )

    signals = dict(context.structured_signals)
    filings = list(context.filings)
    news = list(context.news)
    drop_log: list[str] = []

    def rendered_chars() -> int:
        return _context_chars(signals, filings, news)

    while rendered_chars() > budget_chars:
        if news:
            # News arrives sorted oldest-first from the builder; pop from the front.
            dropped = news.pop(0)
            drop_log.append(f"dropped news {dropped.published.isoformat()} ({dropped.source})")
            continue
        if len(filings) > 1:
            victim = max(filings, key=lambda item: (item.priority, len(item.text)), default=None)
            if victim is not None:
                filings.remove(victim)
                drop_log.append(
                    f"dropped filing section {victim.section} from {victim.filed.isoformat()}"
                )
                continue
        if signals:
            last_key = list(signals)[-1]
            del signals[last_key]
            drop_log.append(f"dropped signal {last_key}")
            continue
        if hard_truncate and filings:
            excessive = rendered_chars() - budget_chars
            only = filings[0]
            keep = max(200, len(only.text) - excessive - 32)
            logger.debug(
                "hard-truncating %s from %d to %d chars", only.section, len(only.text), keep
            )
            filings[0] = FilingExcerpt(
                doc_type=only.doc_type,
                filed=only.filed,
                section=only.section,
                text=only.text[:keep],
                accession=only.accession,
            )
            drop_log.append(f"hard-truncated section {only.section} to {keep} characters")
            continue
        break

    if drop_log:
        logger.debug("context truncated with %d drop actions", len(drop_log))

    return PromptContext(
        as_of=context.as_of,
        label=context.label,
        horizon_days=context.horizon_days,
        structured_signals=signals,
        filings=filings,
        news=news,
        truncated=bool(drop_log),
        drop_log=drop_log,
    )


def _context_chars(
    signals: Mapping[str, Any],
    filings: Sequence[FilingExcerpt],
    news: Sequence[NewsItem],
) -> int:
    """Characters contributed by the variable blocks of a render."""
    total = len(render_structured_signals(signals))
    total += sum(len(item.render()) + 1 for item in filings)
    total += sum(len(item.render()) + 1 for item in news)
    return total


def _jinja_env() -> Environment:
    """A strict Jinja environment: an undefined variable is an error, not a blank."""
    return Environment(
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def build_user_prompt(context: PromptContext, *, include_news: bool = True) -> str:
    """Render the user turn.

    Args:
        context: The (already truncated) context.
        include_news: Set False to emit an empty news block. Used by the ablation
            that removes news while keeping filings, so that the only difference
            between the two prompts is the content.

    Returns:
        The rendered user prompt.
    """
    filings_block = "\n\n".join(item.render() for item in context.filings)
    # Emitting an empty block rather than dropping the tag keeps the prompt shape
    # identical between the with-news and without-news ablation arms.
    news_block = "\n\n".join(item.render() for item in context.news) if include_news else ""
    template = _jinja_env().from_string(USER_TEMPLATE)
    return template.render(
        as_of=context.as_of.isoformat(),
        label=context.label,
        horizon_days=context.horizon_days,
        signals=render_structured_signals(context.structured_signals),
        filings=filings_block,
        news=news_block,
    )


def build_chat_messages(
    context: PromptContext, *, include_news: bool = True
) -> list[dict[str, str]]:
    """Build the ``messages`` list for a chat-format model.

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context, include_news=include_news)},
    ]


def build_sft_example(
    context: PromptContext,
    assessment: RiskAssessment,
    *,
    include_news: bool = True,
) -> dict[str, Any]:
    """Build one supervised fine-tuning example.

    The assistant turn is the canonical JSON serialisation of the assessment, with
    sorted keys so that two runs of the data builder produce byte-identical
    training files.

    Args:
        context: Input context.
        assessment: Target output.
        include_news: Passed through to the prompt builder.

    Returns:
        ``{"messages": [...], "meta": {...}}``. The metadata is not part of the
        loss; it is carried for filtering, ablation and debugging.
    """
    messages = build_chat_messages(context, include_news=include_news)
    messages.append({"role": "assistant", "content": render_target(assessment)})
    return {
        "messages": messages,
        "meta": {
            "as_of": context.as_of.isoformat(),
            "label": context.label,
            "horizon_days": context.horizon_days,
            "truncated": context.truncated,
            "n_filings": len(context.filings),
            "n_news": len(context.news),
            "drop_log": list(context.drop_log),
        },
    }


def render_target(assessment: RiskAssessment) -> str:
    """Canonical JSON for the assistant turn.

    Args:
        assessment: The target assessment.

    Returns:
        A single-line JSON object with sorted keys and no trailing whitespace.
    """
    payload = assessment.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))


def extract_json_object(text: str) -> str | None:
    """Pull the first JSON object out of a model response.

    Handles the three things models do despite being told not to: wrap the object
    in a markdown fence, prefix it with a sentence, and append a sentence. Braces
    inside strings are not specially handled — if the output is that malformed, the
    result will fail validation, which is the correct outcome.

    Args:
        text: Raw model output.

    Returns:
        The candidate JSON substring, or None when no braces are present.
    """
    if not text:
        return None
    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text
    match = _JSON_OBJECT_RE.search(candidate)
    if match is None:
        return None
    return match.group(0)


def parse_assessment(
    text: str,
    *,
    expected_label: str | None = None,
    expected_horizon_days: int | None = None,
) -> RiskAssessment | None:
    """Parse and validate a model response.

    Returns None rather than raising, because a single malformed generation must
    not abort a scoring run over thousands of rows. The reason is logged at debug
    level and counted by the caller, which reports a parse-failure rate.

    Args:
        text: Raw model output.
        expected_label: If given, the parsed label must match; a mismatch means the
            model answered a different question than the one asked.
        expected_horizon_days: If given, the parsed horizon must match.

    Returns:
        The validated assessment, or None if the output was unusable.
    """
    payload_text = extract_json_object(text)
    if payload_text is None:
        logger.debug("no JSON object found in model output")
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        logger.debug("model output is not valid JSON: %s", exc)
        return None
    if not isinstance(payload, dict):
        logger.debug("model output JSON is not an object")
        return None

    # Tolerate the two most common schema slips before validation: a severity that
    # does not match the score, and evidence given as bare strings.
    payload.setdefault("catalysts", [])
    payload.setdefault("limitations", [])
    if isinstance(payload.get("evidence"), list):
        payload["evidence"] = [
            {"source_type": "other", "source_ref": "unattributed", "quote": item}
            if isinstance(item, str)
            else item
            for item in payload["evidence"]
        ]

    try:
        assessment = RiskAssessment.model_validate(payload)
    except ValidationError as exc:
        logger.debug("model output failed schema validation: %s", exc.errors()[:2])
        return None

    if expected_label is not None and str(assessment.label) != expected_label:
        logger.debug(
            "model answered label %s but %s was requested", assessment.label, expected_label
        )
        return None
    if expected_horizon_days is not None and assessment.horizon_days != expected_horizon_days:
        logger.debug(
            "model answered horizon %d but %d was requested",
            assessment.horizon_days,
            expected_horizon_days,
        )
        return None
    return assessment


def score_to_risk(score: float) -> Severity:
    """Convenience re-export of :meth:`Severity.from_score` for call sites that only
    import this module."""
    return Severity.from_score(score)


def assessment_from_labels(
    label: str,
    *,
    score: float,
    horizon_days: int,
    reasons: Sequence[str],
    evidence: Iterable[tuple[str, str, str]] = (),
    catalysts: Sequence[str] = (),
    limitations: Sequence[str] = (),
) -> RiskAssessment:
    """Build an assessment from loose arguments, mainly for the data builder.

    The synthetic generator and the label builder both need to mint target
    assessments, and neither should have to know the pydantic field names.

    Args:
        label: One of the three label names.
        score: Probability in [0, 1].
        horizon_days: Must match the label's definition.
        reasons: Non-empty list of short strings.
        evidence: Iterable of ``(source_type, source_ref, quote)`` triples.
        catalysts: Optional forward-looking triggers.
        limitations: Optional statements of what the documents do not establish.

    Returns:
        A validated :class:`RiskAssessment`.
    """
    spans = [
        EvidenceSpan(source_type=SourceType(source_type), source_ref=source_ref, quote=quote)
        for source_type, source_ref, quote in evidence
    ]
    return RiskAssessment(
        # Coerced rather than passed through: callers hold the label either as a
        # `RiskLabel` or as its string value, and validating here means an
        # unrecognised name fails at construction instead of producing an
        # assessment whose label silently matches nothing downstream.
        label=RiskLabel(label),
        severity=Severity.from_score(score),
        score=score,
        horizon_days=horizon_days,
        reasons=list(reasons),
        evidence=spans,
        catalysts=list(catalysts),
        limitations=list(limitations),
    )


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_SECTION_PRIORITY",
    "PROMPT_OVERHEAD_CHARS",
    "SECTION_PRIORITY",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "FilingExcerpt",
    "NewsItem",
    "PromptContext",
    "assessment_from_labels",
    "build_chat_messages",
    "build_sft_example",
    "build_user_prompt",
    "chars_budget_for_seq_length",
    "extract_json_object",
    "format_signal_value",
    "parse_assessment",
    "render_structured_signals",
    "render_target",
    "score_to_risk",
    "truncate_context",
]

"""Pipeline orchestration: panel, three tracks, evaluation, report.

This module is the layer between the CLI and the library. It exists so that the
end-to-end flow is *testable without a terminal*: everything the CLI does is a call
into a function here, and every function here is a pure-ish transformation of a
:class:`~shingan.config.ProjectConfig` plus a
:class:`~shingan.paths.ProjectPaths`.

Three rules govern what the evaluation is allowed to claim, and they are enforced
here rather than left to the writer of the report:

1. **One fit/apply split for every track.** The fusion comparison is only meaningful
   when the structured track, the text track and the fusion are fitted on the same rows
   and scored on the same rows. This module builds those index sets once and passes the
   same sets to all three, instead of letting each track pick its own.
2. **A metric that cannot be computed is reported as not applicable, with a reason.**
   Ten companies cannot form a meaningful cross-sectional quintile, so the backtest is
   withheld rather than allowed to produce a number nobody should read.
3. **A label with too few positives is reported, not scored.** The label is kept in the
   report with its counts and an explicit reason; it is not dropped, because a missing
   label reads as "not evaluated" while a present label with counts reads as "cannot be
   decided yet". Those are different claims.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shingan.__about__ import DATA_SCHEMA_VERSION, __version__
from shingan.config import ProjectConfig
from shingan.data.builder import PROMPT_SIGNAL_COLUMNS, BuildResult, build_panel
from shingan.data.schema import (
    RiskLabel,
    SourceType,
    diagnose_gaps,
    panel_subset,
    quotes_are_verbatim,
)
from shingan.eval.backtest import quantile_returns
from shingan.eval.metrics import (
    BootstrapCI,
    ClassificationReport,
    ICResult,
    average_precision,
    evaluate_classification,
    information_coefficient,
    newey_west_tstat,
    paired_bootstrap_difference,
)
from shingan.eval.report import (
    EvaluationReport,
    build_run_metadata,
    default_caveats,
    falsification_table,
    gate_table,
)
from shingan.eval.splits import SplitReport, walk_forward_folds
from shingan.eval.stability import (
    DriftReport,
    RollingStabilityResult,
    drift_report,
    historical_stress_suite,
    rolling_stability,
)
from shingan.labeling.definitions import build_label_definitions
from shingan.logging_utils import get_logger
from shingan.models.fusion import (
    STRUCTURED_SCORE,
    TEXT_SCORE,
    RiskFusion,
    fusion_ablation,
)
from shingan.models.structured import StructuredRiskModel
from shingan.models.text_baseline import TextBaselineModel
from shingan.paths import ProjectPaths
from shingan.prompts import (
    FilingExcerpt,
    NewsItem,
    PromptContext,
    assessment_from_labels,
    build_sft_example,
    build_user_prompt,
    chars_budget_for_seq_length,
    truncate_context,
)

logger = get_logger(__name__)

#: Names of the three evaluation paths, in the order the report prints them.
PATH_STRUCTURED = "structured"
PATH_TEXT = "text_baseline"
PATH_FUSED = "fused"
PATH_ORDER: tuple[str, ...] = (PATH_STRUCTURED, PATH_TEXT, PATH_FUSED)

#: A structured model restricted to the signals the text track's prompt actually shows.
#:
#: The LoRA prompt carries :data:`PROMPT_SIGNAL_COLUMNS`, so its score is produced from
#: text *and* a dozen structured signals. Subtracting it from :data:`PATH_STRUCTURED`
#: (the full feature set) measures the gap between two information sets as well as two
#: models, and the resulting number cannot be read as "what the text added". This fourth
#: path is the baseline that makes the subtraction mean one thing.
PATH_MATCHED = "structured_matched"

#: Rows the comparison table prints, in reading order. The matched baseline sits second:
#: directly below the path it shadows and directly above the text path it is the control
#: for, because those are the two subtractions a reader is meant to make.
#:
#: It is deliberately *not* in :data:`PATH_ORDER`. That tuple is the list of **tracks** —
#: it decides which reports get fitted, which score columns are written onto the panel, and
#: which two series the fusion stacks. The matched baseline is a control, not a track: it
#: has no fusion of its own, and nothing about it should reach the panel. Three tracks,
#: four comparison rows.
COMPARISON_ORDER: tuple[str, ...] = (PATH_STRUCTURED, PATH_MATCHED, PATH_TEXT, PATH_FUSED)

#: The continuous forward quantity the information coefficient is computed against,
#: per label, with the sign that makes "larger means more risk".
#:
#: IC is deliberately **not** computed against the binary label: on a 0/1 series it is
#: a restatement of AUC carrying less information, and its sign convention would be
#: arbitrary. It is also not computed against ``fwd_ret_21d``: ranking forward returns
#: is an alpha claim, and this is a risk screen. The quantities below are risk
#: quantities, so a positive IC means the score ranks future risk correctly.
RISK_FORWARD_SPEC: dict[str, tuple[str, float]] = {
    "default_risk": ("fwd_realized_vol_21d", 1.0),
    "fraud_risk": ("fwd_realized_vol_21d", 1.0),
    "tail_risk": ("fwd_max_drawdown_30d", -1.0),
}

#: Names of the text column in the panel, for the text track's input.
TEXT_COLUMN = "text_input"

#: Minimum distinct companies present on a single date for a cross-sectional quantile
#: sort to mean anything. Five names per bucket is already sparse; below it the
#: "quintile" is two companies and the long-short spread is an anecdote. Ten companies
#: against five quantiles is therefore not a quintile backtest, and this module says so
#: instead of producing one — see ``docs/07-roadmap.md`` on Stage 2 scope.
MIN_NAMES_PER_DATE = 25

#: Columns that describe provenance or bookkeeping and must never reach a model.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "as_of",
        "cik",
        "company_name",
        "data_version",
        "insufficient_history",
        "is_synthetic",
        "n_sources",
        "sample_weight",
        "sector",
        "split",
        "ticker",
    }
)


def select_feature_columns(panel: pd.DataFrame, config: ProjectConfig) -> list[str]:
    """Numeric panel columns belonging to the configured feature groups.

    The selection goes through :func:`shingan.data.schema.panel_subset`, so a group
    name that does not exist raises rather than silently contributing no columns —
    which would train a model on nothing and report the base rate as a success.

    Args:
        panel: The assembled panel.
        config: Project configuration carrying ``structured.feature_groups``.

    Returns:
        Column names, in group order, restricted to columns actually present.
    """
    wanted = panel_subset(config.structured.feature_groups)
    return [
        column for column in wanted if column in panel.columns and column not in NON_FEATURE_COLUMNS
    ]


def label_split_frames(panel: pd.DataFrame, label: str) -> dict[str, pd.DataFrame]:
    """Rows of each split whose label is observable.

    Restricted to observable rows on purpose. A row whose label window has not closed
    has no label, and including it as a negative is the single most common way a
    backtest overstates a risk model.

    Args:
        panel: The assembled panel.
        label: Label name, e.g. ``default_risk``.

    Returns:
        Mapping with keys ``train``, ``valid``, ``test``.
    """
    observable = panel[f"label_mask_{label}"].astype(bool)
    frames: dict[str, pd.DataFrame] = {}
    for name in ("train", "valid", "test"):
        frames[name] = panel.loc[(panel["split"] == name) & observable]
    return frames


def text_inputs(
    panel: pd.DataFrame, result: BuildResult, config: ProjectConfig, label: str
) -> pd.Series:
    """Render the text the LoRA sees, so the bag-of-words baseline sees the same thing.

    The baseline's job is to answer "does the fine-tuned LLM beat TF-IDF on the same
    input?". Feeding it anything other than the identical prompt would make the
    comparison depend on the difference between two inputs as well as two models.

    ``label`` is required rather than defaulted. The prompt states the question it is
    asking (``<TASK>label=... horizon_days=...</TASK>``), so a row scored against
    ``tail_risk`` whose prompt asks about ``default_risk`` is being measured on a
    question it was never asked. Until this argument existed the function hard-coded
    ``default_risk`` and was called once per run from outside the label loop, which
    put that wrong task tag in front of every label except the first.

    Args:
        panel: The assembled panel.
        result: The build result, for ``text_context``.
        config: Project configuration, for the token budget.
        label: The label the prompt is written for. Must be the same label the
            resulting scores are compared against.

    Returns:
        One rendered user prompt per panel row, aligned to ``panel.index``.
    """
    budget = chars_budget_for_seqlength(config)
    contexts = build_prompt_contexts(panel, result, config, label, budget)
    rendered = pd.Series("", index=panel.index, dtype=object)
    for position, context in contexts.items():
        rendered.iloc[position] = build_user_prompt(context)
    return rendered


def chars_budget_for_seqlength(config: ProjectConfig) -> int:
    """Character budget for the prompt context blocks, from the configured token limit."""
    return chars_budget_for_seq_length(config.lora.max_seq_length)


def build_prompt_contexts(
    panel: pd.DataFrame,
    result: BuildResult,
    config: ProjectConfig,
    label: str,
    budget_chars: int,
) -> dict[int, PromptContext]:
    """One :class:`PromptContext` per panel row, truncated to the budget.

    The context reports its own truncation (``context.truncated`` and
    ``drop_log``), so the fraction of prompts that had to be cut is available to the
    report rather than being an invisible preprocessing step.

    Args:
        panel: The assembled panel.
        result: The build result, for ``text_context``.
        config: Project configuration, for the horizon.
        label: Label the prompt is written for; recorded in the context so a parsed
            answer can be checked against it.
        budget_chars: Character budget for the context blocks.

    Returns:
        Mapping from panel position to :class:`PromptContext`.
    """
    horizon = config.labels.horizon_days(label)
    payloads = {str(entry["sample_id"]): entry for entry in result.text_context}
    signal_frame = panel[list(PROMPT_SIGNAL_COLUMNS)].to_dict(orient="records")
    stamps = pd.to_datetime(panel["as_of"])
    tickers = panel["ticker"].astype(str).to_numpy()

    contexts: dict[int, PromptContext] = {}
    dropped = 0
    for position, (ticker, stamp, signals) in enumerate(
        zip(tickers, stamps, signal_frame, strict=True)
    ):
        sample_id = f"{ticker}-{stamp.strftime('%Y%m%d')}"
        payload = payloads.get(sample_id, {})
        context = PromptContext(
            as_of=stamp.date(),
            label=label,
            horizon_days=horizon,
            structured_signals={k: signals.get(k) for k in PROMPT_SIGNAL_COLUMNS},
            filings=[_filing_excerpt(item) for item in payload.get("filings", [])],
            news=[_news_item(item) for item in payload.get("news", [])],
        )
        trimmed = truncate_context(context, budget_chars=budget_chars)
        dropped += int(trimmed.truncated)
        contexts[position] = trimmed

    if dropped:
        logger.warning(
            "%d of %d prompts exceeded the %d-character budget and were truncated; the "
            "count is recorded in the report rather than applied silently",
            dropped,
            len(panel),
            budget_chars,
        )
    return contexts


def _filing_excerpt(payload: dict[str, Any]) -> FilingExcerpt:
    """Build a :class:`FilingExcerpt` from a ``text_context`` entry."""
    return FilingExcerpt(
        doc_type=str(payload.get("doc_type", "")),
        filed=pd.Timestamp(payload["filed"]).date(),
        section=str(payload.get("section", "")),
        text=str(payload.get("text", "")),
        accession=str(payload.get("accession", "")),
    )


def _news_item(payload: dict[str, Any]) -> NewsItem:
    """Build a :class:`NewsItem` from a ``text_context`` entry."""
    sentiment = payload.get("sentiment")
    return NewsItem(
        published=pd.Timestamp(payload["published"]).date(),
        source=str(payload.get("source", "")),
        title=str(payload.get("title", "")),
        body=str(payload.get("body", "")),
        sentiment=None if sentiment is None else float(sentiment),
    )


#: Splits that may be turned into supervised examples. ``test`` is deliberately absent.
#: The test block is what the report measures on; writing it into the training file
#: invites exactly the leakage the purge margin exists to prevent, and the temptation
#: would be silent — the file would look like one more output.
SFT_SPLITS: tuple[str, ...] = ("train", "valid")

#: Shortest document fragment that may be used as an evidence quote. Below this the
#: "quote" is a phrase that would match almost any document and verifies nothing.
MIN_QUOTE_CHARS = 24

#: Longest evidence quote taken from a document, in characters.
QUOTE_MAX_CHARS = 240


def _first_verbatim_quote(document: str, *, limit: int = QUOTE_MAX_CHARS) -> str | None:
    """A quote sliced out of ``document``, or None when the text is too short.

    Slicing an existing string is the whole point. ``docs/04-training.md`` requires
    ``evidence[].quote`` to appear in the source character for character, and notes
    that the training data satisfies this "naturally" because the quote is cut from the
    source; this function is where that is made true rather than assumed. The result is
    re-checked per example by :func:`~shingan.data.schema.quotes_are_verbatim`, which
    normalises whitespace on both sides — so the collapsed form returned here still
    verifies against the raw document text.
    """
    collapsed = " ".join(document.split())
    if len(collapsed) < MIN_QUOTE_CHARS:
        return None
    snippet = collapsed[:limit]
    # Trim back to a sentence boundary where one exists, so the quote reads as
    # language. Trimming only ever shortens, so the substring property survives.
    for terminator in (". ", "; "):
        cut = snippet.rfind(terminator)
        if cut >= MIN_QUOTE_CHARS:
            return snippet[: cut + 1]
    return snippet


def _sft_evidence(context: PromptContext) -> list[tuple[str, str, str]]:
    """One verbatim quote from the context's first filing, then its first news item.

    ``(source_type, source_ref, quote)`` triples, in the shape
    :func:`~shingan.prompts.assessment_from_labels` expects.
    """
    spans: list[tuple[str, str, str]] = []
    for excerpt in context.filings[:1]:
        quote = _first_verbatim_quote(excerpt.text)
        if quote is not None:
            reference = excerpt.accession or f"{excerpt.doc_type} {excerpt.filed.isoformat()}"
            spans.append((SourceType.FILING.value, reference, quote))
    for item in context.news[:1]:
        quote = _first_verbatim_quote(item.body or item.title)
        if quote is not None:
            spans.append((SourceType.NEWS.value, item.source or "news", quote))
    return spans


def _sft_reasons(
    label: str,
    *,
    positive: bool,
    horizon_days: int,
    as_of: Any,
    event_date: Any,
    source_of_record: Any,
) -> list[str]:
    """Short factual reasons, stated no more strongly than the panel supports."""
    stamp = pd.Timestamp(as_of).date().isoformat()
    if positive:
        reasons = [f"a {label} event was recorded within {horizon_days} days of {stamp}"]
        event_stamp = pd.to_datetime(event_date, errors="coerce")
        if not pd.isna(event_stamp):
            reasons.append(f"event date {pd.Timestamp(event_stamp).date().isoformat()}")
        if isinstance(source_of_record, str) and source_of_record:
            reasons.append(f"source of record: {source_of_record}")
        return reasons[:3]
    return [
        f"no {label} event was recorded in the {horizon_days} days after {stamp}",
        f"the recorded outcome for this window is negative (source: {source_of_record})"
        if isinstance(source_of_record, str) and source_of_record
        else "the recorded outcome for this window is negative",
    ]


def sft_examples(
    panel: pd.DataFrame,
    build: BuildResult,
    config: ProjectConfig,
    *,
    labels: list[str] | None = None,
    include_news: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Supervised fine-tuning records, per split, for every requested label.

    One example per (observable row, label): the prompt is the same one the text
    baseline sees, and the assistant turn is the canonical JSON of the row's own label.

    **The target score is the binary label, not a soft one.** "Where did 0.37 come
    from?" is a fair question about the example in ``docs/04-training.md`` §2.3, which
    shows a score of 0.37 for a ``high`` severity; the answer is that 0.37 there is
    illustrative formatting, not a target. Only two candidate targets exist in the
    panel — :attr:`~shingan.data.schema.SampleRecord.label_binary`, which is what the
    text track is defined against, and the continuous forward-risk columns, which are
    *not* the same quantity the structured track predicts and would therefore break the
    fusion stack, whose whole premise is that both inputs estimate one thing. The
    honest consequence is that ``severity`` is degenerate on these targets: it resolves
    to ``low`` or ``critical`` and carries no information the score does not. Ranking
    metrics (AUC, PR-AUC, IC) are unaffected, because they depend on order only;
    probability *calibration* of the text track will be poor until a calibrator is
    applied to it, and the report's calibration section concerns the structured track.

    Args:
        panel: The assembled panel.
        build: The build result, for ``text_context``.
        config: Project configuration.
        labels: Labels to emit. Defaults to ``config.labels.targets``.
        include_news: Whether news items are rendered into the prompt.
        provenance: The ``data`` block recorded in the manifest. Computed by the
            caller through :func:`shingan.data.provenance.data_provenance`, because
            only the caller knows which overlay selected the sources and which paths
            were involved. When omitted, a minimal block is derived from the panel
            and config alone — sources and shape, no file identities — so a manifest
            never has to be silent about its provenance.

    Returns:
        ``(records_by_split, manifest)``. The manifest records the target rule, the
        counts, and the reason ``test`` is absent, so the choice travels with the file
        instead of living only in this docstring.
    """
    wanted = [str(name) for name in (labels or [str(item) for item in config.labels.targets])]
    unknown = [name for name in wanted if name not in {str(item) for item in RiskLabel}]
    if unknown:
        raise ValueError(
            f"unknown label(s) {unknown}; expected a subset of {sorted(str(x) for x in RiskLabel)}"
        )

    budget = chars_budget_for_seqlength(config)
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in SFT_SPLITS}
    per_label: dict[str, dict[str, int]] = {}
    quote_failures = 0
    truncated = 0

    for label in wanted:
        contexts = build_prompt_contexts(panel, build, config, label, budget)
        horizon = config.labels.horizon_days(label)
        mask = panel[f"label_mask_{label}"].astype(bool).to_numpy()
        split_values = panel["split"].astype(str).to_numpy()
        labels_binary = (
            pd.to_numeric(panel[f"label_{label}"], errors="coerce").fillna(0).to_numpy(dtype=int)
        )
        event_dates = panel[f"event_date_{label}"].to_numpy()
        sources = panel[f"source_of_record_{label}"].to_numpy()
        tickers = panel["ticker"].astype(str).to_numpy()
        stamps = panel["as_of"].to_numpy()

        counts = {
            "train": 0,
            "valid": 0,
            "positives_train": 0,
            "positives_valid": 0,
            "truncated": 0,
        }
        for name in SFT_SPLITS:
            for position in np.flatnonzero((split_values == name) & mask):
                context = contexts[int(position)]
                positive = bool(labels_binary[position])
                target = assessment_from_labels(
                    label,
                    score=float(labels_binary[position]),
                    horizon_days=horizon,
                    reasons=_sft_reasons(
                        label,
                        positive=positive,
                        horizon_days=horizon,
                        as_of=stamps[position],
                        event_date=event_dates[position],
                        source_of_record=sources[position],
                    ),
                    evidence=_sft_evidence(context),
                    limitations=(
                        []
                        if positive
                        else [
                            "the window closed without an event; this is one negative "
                            "observation, not evidence that no risk was present"
                        ]
                    ),
                )
                documents = [excerpt.text for excerpt in context.filings] + [
                    item.body for item in context.news
                ]
                if target.evidence and not quotes_are_verbatim(target, documents):
                    # Impossible while `_first_verbatim_quote` slices from these very
                    # strings; asserted because the format's whole credibility rests on
                    # the quotes being real, and a silent failure here would train the
                    # model to fabricate.
                    quote_failures += 1
                    logger.error(
                        "a generated quote did not verify for %s at %s", label, stamps[position]
                    )

                example = build_sft_example(context, target, include_news=include_news)
                example["meta"].update(
                    {
                        "sample_id": f"{tickers[position]}-{pd.Timestamp(stamps[position]).strftime('%Y%m%d')}",
                        "ticker": tickers[position],
                        "split": name,
                        "label_binary": int(labels_binary[position]),
                    }
                )
                records[name].append(example)
                counts[name] += 1
                if positive:
                    counts["positives_" + name] += 1
                counts["truncated"] += int(context.truncated)
                truncated += int(context.truncated)
        per_label[label] = counts

    if quote_failures:
        raise ValueError(
            f"{quote_failures} generated evidence quotes did not appear verbatim in their "
            "source documents. Refusing to write a training file that teaches fabrication."
        )

    manifest: dict[str, Any] = {
        "target_rule": "score = label_binary (0.0 or 1.0); severity is derived and degenerate",
        "score_is_binary": True,
        "severity_carries_information": False,
        "labels": wanted,
        "include_news": include_news,
        "splits": {name: len(records[name]) for name in SFT_SPLITS},
        "test_split_emitted": False,
        "test_split_absent_because": (
            "the test block is the evaluation sample; emitting it as training data is the "
            "leakage the purge margin exists to prevent"
        ),
        "per_label": per_label,
        "prompts_truncated": truncated,
        "max_seq_length": config.lora.max_seq_length,
        "character_budget": budget,
        # The data block travels with the split counts, not beside them: a manifest
        # that says "189 train rows" without saying which panel those rows came from
        # cannot distinguish a synthetic-SFT run from a real one a month later.
        "data": dict(provenance) if provenance is not None else _minimal_provenance(panel, config),
    }
    return records, manifest


def _minimal_provenance(panel: pd.DataFrame, config: ProjectConfig) -> dict[str, Any]:
    """Sources and shape only, for a caller that passed no provenance block."""
    return {
        "sources": [str(source) for source in config.data.sources],
        "is_synthetic": bool(panel["is_synthetic"].any())
        if "is_synthetic" in panel.columns and len(panel)
        else None,
        "n_rows": len(panel),
        "n_tickers": int(panel["ticker"].nunique()) if "ticker" in panel.columns else None,
        "data_version": str(config.project.data_version),
        "data_schema_version": DATA_SCHEMA_VERSION,
    }


@dataclass(slots=True)
class LabelOutcome:
    """Everything one label's evaluation produced, including why it produced nothing."""

    label: str
    fitted: bool
    reason: str = ""
    #: Why the matched-information control row is missing from the comparison table, when
    #: it is. Empty means the row was produced. Recorded rather than swallowed: a table
    #: that is quietly missing its control still reads as a complete comparison, and the
    #: reader then makes the subtraction against the wrong baseline.
    matched_reason: str = ""
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    positives_train: int = 0
    positives_valid: int = 0
    positives_test: int = 0
    reports: dict[str, ClassificationReport] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    calibrator: str = "not measured"
    pr_auc_difference: BootstrapCI | None = None
    ic: ICResult | None = None
    ic_tstat: float | None = None
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    ablation: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The fitted estimators, keyed by path name. Carried on the outcome so that
    #: ``shingan train structured`` can persist exactly the model this evaluation
    #: measured, rather than re-fitting and quietly shipping a second, unmeasured one.
    #: Empty when the label could not be fitted.
    models: dict[str, Any] = field(default_factory=dict)

    def headline(self) -> ClassificationReport | None:
        """The fused report when available, else the structured one.

        Falling back is deliberate and is recorded in the metadata: a report whose
        "headline" is silently a different path than the reader assumes is worse than
        one that says which path it used.
        """
        return self.reports.get(PATH_FUSED) or self.reports.get(PATH_STRUCTURED)


def _test_scores(report: ClassificationReport | None) -> tuple[float, float]:
    """``(auc, pr_auc)`` from a report, or NaNs."""
    if report is None:
        return float("nan"), float("nan")
    return report.auc, report.pr_auc


def evaluate_label(
    panel: pd.DataFrame,
    config: ProjectConfig,
    label: str,
    features: list[str],
    text: pd.Series,
    *,
    n_boot: int,
) -> LabelOutcome:
    """Fit all three tracks for one label and evaluate them on the test block.

    Track A (structured) is fitted on train and calibrated on valid. Track B0 (text
    baseline) sees the same rows. Track C (fusion) is fitted on the *validation* fold's
    scores only — fitting it on train scores would calibrate the fusion on
    optimistically-biased inputs.

    Args:
        panel: The assembled panel.
        config: Project configuration.
        label: Label name.
        features: Structured feature columns.
        text: Rendered text inputs, aligned to ``panel.index``.
        n_boot: Bootstrap resamples for the interval and the paired difference. The
            block length is each label's own horizon, taken from the configuration.

    Returns:
        The :class:`LabelOutcome`. ``fitted`` is False, with a reason, when the label
        cannot support a model rather than raising — the caller reports it either way.
    """
    frames = label_split_frames(panel, label)
    outcome = LabelOutcome(
        label=label,
        fitted=False,
        n_train=len(frames["train"]),
        n_valid=len(frames["valid"]),
        n_test=len(frames["test"]),
        positives_train=int(frames["train"][f"label_{label}"].sum()),
        positives_valid=int(frames["valid"][f"label_{label}"].sum()),
        positives_test=int(frames["test"][f"label_{label}"].sum()),
    )

    if outcome.positives_train < 2 or outcome.positives_valid < 1 or outcome.positives_test < 1:
        outcome.reason = (
            f"positives are train={outcome.positives_train}, valid={outcome.positives_valid}, "
            f"test={outcome.positives_test}. A model needs at least two training positives, "
            f"one validation positive to calibrate on, and one test positive to measure "
            f"anything. This is a property of the sample and the label horizon, not a bug: "
            f"see docs/05-evaluation.md section 10 (F8)."
        )
        logger.warning("%s: not evaluated. %s", label, outcome.reason)
        return outcome

    y_train = frames["train"][f"label_{label}"].to_numpy(dtype=int)
    y_valid = frames["valid"][f"label_{label}"].to_numpy(dtype=int)
    y_test = frames["test"][f"label_{label}"].to_numpy(dtype=int)

    weight_train = None
    if config.structured.use_sample_weight and "sample_weight" in frames["train"].columns:
        weight_train = frames["train"]["sample_weight"].to_numpy(dtype=float)

    structured = StructuredRiskModel(config.structured)
    structured.fit(
        frames["train"][features],
        y_train,
        frames["valid"][features],
        y_valid,
        sample_weight=weight_train,
    )

    baseline = TextBaselineModel(config.text_baseline)
    baseline.fit(
        text.loc[frames["train"].index],
        y_train,
        text.loc[frames["valid"].index],
        y_valid,
    )

    structured_valid = structured.predict_proba(frames["valid"][features])
    text_valid = baseline.predict_proba(text.loc[frames["valid"].index])
    fusion = RiskFusion(config.fusion).fit(
        pd.Series(structured_valid, index=frames["valid"].index),
        pd.Series(text_valid, index=frames["valid"].index),
        pd.Series(y_valid, index=frames["valid"].index),
        dates=frames["valid"]["as_of"],
        fit_split="valid",
    )

    scores = pd.DataFrame(
        {
            "ticker": frames["test"]["ticker"].to_numpy(),
            "as_of": frames["test"]["as_of"].to_numpy(),
            PATH_STRUCTURED: structured.predict_proba(frames["test"][features]),
            PATH_TEXT: baseline.predict_proba(text.loc[frames["test"].index]),
        },
        # `index=` is the panel's row labels, and it is load-bearing. `frames["test"]` is a
        # `.loc` selection off the panel, so its index is a scattered set of row positions —
        # the builder sorts by `["ticker", "as_of"]` before `reset_index(drop=True)`, which
        # leaves one contiguous run of test rows per ticker rather than one block overall.
        # Omitted here, the frame got a fresh `0..n_test-1` RangeIndex, and every downstream
        # `.loc[outcome.scores.index]` — the panel's `score_*` columns, which feed drift,
        # rolling stability, stress and the backtest, plus the ablation's stacked frame —
        # then wrote test scores onto whichever rows happened to sit at those positions.
        # Measured on the demo panel: test rows at 46.., scores claiming 0.., 8 of 64 rows
        # correct.
        index=frames["test"].index,
    )
    # `scores.index` is now the panel's index, so the fusion can be handed the columns
    # directly instead of re-wrapping them in fresh Series with an explicit index.
    scores[PATH_FUSED] = fusion.predict_proba(
        scores[PATH_STRUCTURED],
        scores[PATH_TEXT],
        dates=scores["as_of"],
    )

    # The matched-information control. Fitted here, in the one place that produces the
    # standard report, rather than left to a caller: `text_only_lora - structured_matched`
    # is the subtraction that means "what the text added", and a report that ships without
    # the control invites the reader to subtract against `structured` instead — the
    # mismatched comparison the control exists to prevent.
    matched_scores: pd.Series | None = None
    try:
        matched_scores = _matched_signal_scores(panel, config, label, PROMPT_SIGNAL_COLUMNS)
    except ValueError as error:
        # Not fatal: none of the three tracks depends on this control. Disclosed through
        # the run's notes instead of dropped silently.
        outcome.matched_reason = str(error)
        logger.warning("%s: matched-information baseline not fitted. %s", label, error)
    if matched_scores is not None:
        # `matched_scores` is indexed by the panel's row labels, the same index `scores`
        # was built with, so this aligns row-for-row rather than positionally.
        scores[PATH_MATCHED] = matched_scores

    for path in PATH_ORDER:
        outcome.reports[path] = evaluate_classification(
            y_test,
            scores[path].to_numpy(),
            path=path,
            label=label,
            split="test",
            calibration_bins=config.eval.n_bins,
        )
    if matched_scores is not None:
        outcome.reports[PATH_MATCHED] = evaluate_classification(
            y_test,
            scores[PATH_MATCHED].to_numpy(),
            path=PATH_MATCHED,
            label=label,
            split="test",
            calibration_bins=config.eval.n_bins,
        )
    outcome.calibration = dict(structured.calibration_report)
    outcome.calibrator = str(outcome.calibration.get("calibrator", "not measured"))

    outcome.pr_auc_difference = None
    if (
        outcome.reports[PATH_FUSED].n_positives >= 1
        and outcome.reports[PATH_STRUCTURED].n_positives >= 1
    ):
        outcome.pr_auc_difference = paired_bootstrap_difference(
            frames["test"]["as_of"].to_numpy(),
            scores[PATH_FUSED].to_numpy(),
            scores[PATH_STRUCTURED].to_numpy(),
            y_test,
            average_precision,
            n_boot=n_boot,
            # The label's *own* horizon, not the split's purge margin. The block length
            # exists to cover the window over which two rows share a label outcome, and
            # that window is the horizon. Using the purge margin (730 days, set by
            # fraud_risk) for a 365-day label would put the whole test block inside one
            # resampling block and collapse the interval to zero width.
            block_days=config.labels.calendar_horizon_days(label),
            alpha=1.0 - config.eval.rolling.confidence_level,
            seed=config.project.seed,
        )

    outcome.ic = _information_coefficient(frames["test"], scores[PATH_FUSED], label)
    if outcome.ic is not None and len(outcome.ic.series):
        outcome.ic_tstat = newey_west_tstat(outcome.ic.series.to_numpy(dtype=float))

    # `fitted` is set here, not at the end of the function. Everything below derives from
    # a booster that has already been fitted successfully, and `_ablation_table` guards on
    # this flag — so setting it last meant the guard read False on every single call and
    # the ablation table came back empty on every run. A flag that describes the model has
    # to be set when the model exists, not when the last reporting step happens to finish.
    outcome.fitted = True
    outcome.comparison = _comparison_table(outcome, config)
    outcome.ablation = _ablation_table(
        panel,
        config,
        label,
        frames,
        scores,
        # The ablation refits the stacker, and the stacker may only be fitted on the
        # validation fold — so the ablation needs the validation fold's path scores as well
        # as the test fold's. `scores` above holds test rows only.
        pd.DataFrame(
            {PATH_STRUCTURED: structured_valid, PATH_TEXT: text_valid},
            index=frames["valid"].index,
        ),
        outcome,
    )
    outcome.scores = scores
    # `PATH_MATCHED` is deliberately absent. `shingan train structured` persists exactly
    # what is in this dict, and the matched control is a measurement instrument, not a
    # deliverable: shipping it would put a file on disk whose name implies it is a
    # candidate model while its only defined use is to be subtracted from one.
    outcome.models = {
        PATH_STRUCTURED: structured,
        PATH_TEXT: baseline,
        PATH_FUSED: fusion,
    }
    return outcome


def matched_signal_report(
    panel: pd.DataFrame,
    config: ProjectConfig,
    label: str,
    *,
    signals: Sequence[str] = PROMPT_SIGNAL_COLUMNS,
) -> tuple[ClassificationReport, pd.Series]:
    """Fit a structured model on the signals the prompt shows, and score the test block.

    Why this exists rather than "just compare against ``structured``": the text track's
    prompt contains a structured-signal block, so the two paths being compared do not see
    the same inputs. The difference between them therefore mixes the value of the text
    with the value of the extra features, and a reader who subtracts them anyway gets a
    number that looks like an answer and is not one.

    Fitting the same estimator on the same dozen columns over the same splits, with the
    same calibration fold, removes that mismatch: ``text_only_lora - structured_matched``
    is now a difference between two models that were shown the same information.

    Args:
        panel: The assembled panel.
        config: Project configuration; the structured block supplies the estimator.
        label: Label to fit.
        signals: Columns to use. Defaults to the prompt's own signal list; an explicit
            list is accepted so a caller can ask "what if the prompt had held other
            columns?" without editing the prompt.

    Returns:
        ``(report, scores)`` where ``scores`` is aligned to the test frame's index, so a
        caller can pair it row-by-row with another path's scores.

    Raises:
        ValueError: If any requested signal is absent from the panel. Silently dropping
            one would make the "same information" claim false while keeping the row.
    """
    scores = _matched_signal_scores(panel, config, label, signals)
    frames = label_split_frames(panel, label)
    y_test = frames["test"][f"label_{label}"].to_numpy(dtype=int)
    report = evaluate_classification(
        y_test,
        scores.to_numpy(),
        path=PATH_MATCHED,
        label=label,
        split="test",
        # The same bin count every other row uses. The matched row is the one the LoRA row
        # is subtracted from, so a different binning here would put two different ECEs in
        # one table under two names that read as the same measurement.
        calibration_bins=config.eval.n_bins,
    )
    return report, scores


def _matched_signal_scores(
    panel: pd.DataFrame,
    config: ProjectConfig,
    label: str,
    signals: Sequence[str],
) -> pd.Series:
    """Fit the structured estimator on ``signals`` and score the test block.

    Split out of :func:`matched_signal_report` so that :func:`evaluate_label` — which is
    what writes the standard report — produces this row from the *same* code. Two
    independent implementations of "the same baseline" would eventually disagree, and the
    disagreement would be invisible: both rows are labelled ``structured_matched``, and
    neither carries a version of the formula it used.

    Args:
        panel: The assembled panel.
        config: Project configuration; the structured block supplies the estimator.
        label: Label to fit.
        signals: The columns to fit on. Not defaulted here on purpose — the caller states
            which information set it is claiming, and the docstring above records why.

    Returns:
        Test-block scores, indexed by the panel's row labels so a caller can pair them
        row-by-row with another path's scores.

    Raises:
        ValueError: If any requested signal is absent from the panel.
    """
    missing = [name for name in signals if name not in panel.columns]
    if missing:
        raise ValueError(
            f"asked for the matched baseline on columns the panel does not have: {missing}. "
            "The comparison against the LoRA row is only valid on the columns the prompt "
            "actually shows, so this is refused rather than reduced to whatever is present."
        )

    frames = label_split_frames(panel, label)
    y_train = frames["train"][f"label_{label}"].to_numpy(dtype=int)
    y_valid = frames["valid"][f"label_{label}"].to_numpy(dtype=int)

    weight_train = None
    if config.structured.use_sample_weight and "sample_weight" in frames["train"].columns:
        weight_train = frames["train"]["sample_weight"].to_numpy(dtype=float)

    columns = list(signals)
    model = StructuredRiskModel(config.structured)
    model.fit(
        frames["train"][columns],
        y_train,
        frames["valid"][columns],
        y_valid,
        sample_weight=weight_train,
    )
    return pd.Series(
        model.predict_proba(frames["test"][columns]),
        index=frames["test"].index,
        name=PATH_MATCHED,
    )


def _information_coefficient(
    test_frame: pd.DataFrame, fused: pd.Series, label: str
) -> ICResult | None:
    """IC of the fused score against the continuous risk quantity for this label."""
    column, sign = RISK_FORWARD_SPEC[label]
    if column not in test_frame.columns:
        logger.warning("IC not computed for %s: %s is absent from the panel", label, column)
        return None
    forward = pd.to_numeric(test_frame[column], errors="coerce").to_numpy(dtype=float) * sign
    result = information_coefficient(test_frame["as_of"].to_numpy(), fused.to_numpy(), forward)
    if result is None or not len(result.series):
        logger.warning(
            "IC not computed for %s: no date held enough scored rows with a defined %s",
            label,
            column,
        )
        return None
    return result


def _comparison_table(outcome: LabelOutcome, config: ProjectConfig) -> pd.DataFrame:
    """One row per path with its test metrics and the relevant gate verdicts."""
    rows: list[dict[str, Any]] = []
    # Bound to a name rather than called twice: `outcome.headline().gates() if
    # outcome.headline() else {}` reads as if it guards the call, but nothing lets a
    # reader (or a type checker) know the two calls return the same object.
    headline_report = outcome.headline()
    gates = headline_report.gates() if headline_report else {}
    # `COMPARISON_ORDER`, not `PATH_ORDER`: the table carries a row per path *and* the
    # matched-information control. `reports.get` still guards the row, so a label whose
    # control could not be fitted shows three rows and says why in the run's notes rather
    # than printing a row of NaNs under a name that claims to be a baseline.
    for path in COMPARISON_ORDER:
        report = outcome.reports.get(path)
        if report is None:
            continue
        auc, pr_auc = _test_scores(report)
        row: dict[str, Any] = {
            "path": path,
            "n_rows": report.n_rows,
            "n_positives": report.n_positives,
            "base_rate": report.base_rate,
            "auc": auc,
            "ks": report.ks,
            # `ks_statistic` takes an absolute gap, so a path whose negatives outrank its
            # positives shows a large KS over an inverted ranking. The real-data run of
            # 2026-09-23 produced exactly that (`text_baseline`: AUC 0.3281 with KS
            # 0.5692), and without this column the two numbers read as if they corroborated
            # each other. `ks_direction` is computed for every path already; `eval lora`
            # has been shipping it since it was written, so dropping it here made the
            # standard report the less honest of the two renderings.
            "ks_direction": report.ks_direction,
            "pr_auc": pr_auc,
            "pr_auc_lift": report.pr_auc_lift,
            "capture_top5": report.capture_top5,
            "iso_fraction": float("nan"),
        }
        if report.calibration is not None:
            row["ece"] = report.calibration.ece
            row["brier"] = report.calibration.brier
            row["brier_skill"] = report.calibration.brier_skill
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        # The gates are the headline path's; recorded per row so a reader can see that
        # the verdict and the numbers it rests on belong to the same path.
        table["headline_gates_failing"] = (
            ", ".join(sorted(name for name, value in gates.items() if value is False)) or "none"
        )
        table["headline_gates_undecidable"] = (
            ", ".join(sorted(name for name, value in gates.items() if value is None)) or "none"
        )
    return table


def _ablation_table(
    panel: pd.DataFrame,
    config: ProjectConfig,
    label: str,
    frames: dict[str, pd.DataFrame],
    scores: pd.DataFrame,
    valid_scores: pd.DataFrame,
    outcome: LabelOutcome,
) -> pd.DataFrame:
    """Structured-only / text-only / fused over one common fit and apply mask.

    The mask is what makes the comparison a comparison. Without it each path would be
    scored on its own subset and the difference would confound the models with the rows.

    Args:
        panel: The assembled panel, for labels and dates.
        config: Project configuration; its ``fusion`` block is passed through so the fused
            row describes the same stacker as the headline report.
        label: Label name.
        frames: Per-split frames from :func:`label_split_frames`.
        scores: Test-fold path scores, indexed by the panel's row labels.
        valid_scores: Validation-fold path scores, likewise. The stacker is fitted on the
            validation fold, so the ablation needs it: with only the test fold available
            the fit set was entirely NaN, which the fusion's design matrix imputes to 0.5 —
            fitting the stacker on a constant.
        outcome: The outcome, read for the ``fitted`` flag only.
    """
    if not outcome.fitted:
        return pd.DataFrame()
    fit_mask = panel.index.isin(frames["valid"].index)
    apply_mask = panel.index.isin(frames["test"].index)

    # `fusion_ablation` speaks the fusion module's vocabulary — its design matrix is keyed
    # by `score_structured` / `score_text` — while this pipeline calls the same two paths
    # `structured` / `text_baseline` (`PATH_ORDER`). Translate at this one boundary rather
    # than renaming either side: `PATH_*` keys `outcome.reports` and every `score_*` column,
    # while the fusion names are what the stacker consumes. Handing the pipeline names
    # straight through raised `KeyError: scores frame is missing 'score_structured'`.
    stacked = (
        pd.concat(
            [valid_scores[[PATH_STRUCTURED, PATH_TEXT]], scores[[PATH_STRUCTURED, PATH_TEXT]]]
        )
        .rename(
            columns={
                PATH_STRUCTURED: STRUCTURED_SCORE,
                PATH_TEXT: TEXT_SCORE,
            }
        )
        # Back onto the full panel: rows outside valid and test stay NaN and are excluded by
        # the masks, so the fused score is never computed for a row that has no inputs.
        .reindex(panel.index)
    )
    return fusion_ablation(
        panel,
        stacked,
        panel[f"label_{label}"].astype(float),
        dates=panel["as_of"],
        fit_mask=pd.Series(fit_mask, index=panel.index),
        apply_mask=pd.Series(apply_mask, index=panel.index),
        config=config.fusion,
        # `metric` is left at its default deliberately. It was passed `roc_auc`, which
        # returns a float while `MetricFn` is declared to return a Mapping —
        # `fusion_ablation` then calls `.items()` on that float. The default reports AUC
        # *and* PR-AUC per path, which is what an ablation table needs anyway: a single
        # metric cannot show that the fused path lost on one measure and won on another.
    )


@dataclass(slots=True)
class PipelineResult:
    """Everything the CLI, the report writer and the tests need from one run."""

    config: ProjectConfig
    build: BuildResult
    outcomes: dict[str, LabelOutcome]
    split_report: SplitReport
    folds: list[Any]
    drift: list[DriftReport]
    stability: dict[str, RollingStabilityResult]
    historical_stress: pd.DataFrame
    backtest: dict[str, Any]
    dataset_card_values: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def headline_outcome(self) -> LabelOutcome | None:
        """The outcome for the configured headline label, defaulting to ``default_risk``."""
        for name in ("default_risk", *(str(label) for label in self.outcomes)):
            if name in self.outcomes and self.outcomes[name].fitted:
                return self.outcomes[name]
        return next(iter(self.outcomes.values()), None)


def _run_rolling(
    panel: pd.DataFrame, label: str, score_column: str, config: ProjectConfig
) -> RollingStabilityResult | None:
    """Rolling-window stability for one label, or None when it cannot be computed."""
    try:
        return rolling_stability(
            panel,
            score_column=score_column,
            label_column=f"label_{label}",
            mask_column=f"label_mask_{label}",
            freq="YE",
            n_boot=config.eval.rolling.bootstrap_samples,
            seed=config.project.seed,
        )
    except Exception as exc:
        logger.warning("rolling stability unavailable for %s: %s", label, exc)
        return None


def run_pipeline(
    config: ProjectConfig,
    paths: ProjectPaths | None = None,
    *,
    labels: list[str] | None = None,
    write: bool = False,
) -> PipelineResult:
    """Build the panel and evaluate every configured label.

    Args:
        config: Validated configuration.
        paths: Resolved project paths. Defaults to the project root.
        labels: Labels to evaluate. Defaults to ``config.labels.targets``.
        write: Whether the builder writes its artifacts under ``data/processed``.

    Returns:
        The :class:`PipelineResult`.
    """
    resolved_paths = paths or ProjectPaths.from_root(config.project.root)
    # Propagated rather than ignored. The builder resolves its own output location from
    # `config.project.root`, so a `paths` argument pointing at a different root used to
    # be accepted and then have no effect at all — the build wrote to the config's root
    # regardless. Copying the resolved root into the config is what makes the parameter
    # mean what its name says, and it is a no-op when the two already agree.
    if config.project.root != resolved_paths.root:
        config = config.model_copy(
            update={"project": config.project.model_copy(update={"root": resolved_paths.root})}
        )
    build = build_panel(config, write=write)
    panel = build.panel
    definitions = build_label_definitions(config.labels)
    wanted = [str(name) for name in (labels or [str(label) for label in definitions])]

    features = select_feature_columns(panel, config)
    if not features:
        raise ValueError(
            "no feature columns were selected. Check structured.feature_groups in the "
            "configuration and that the builder produced those columns."
        )
    # Score columns are added to the panel before any per-label work so that the
    # rolling and stress sections, which read the panel rather than the test slice,
    # find them. NaN outside the test block is the correct state: those rows were not
    # scored, and a zero would look like "no risk".
    for label in wanted:
        panel[f"score_{PATH_FUSED}_{label}"] = np.nan
        panel[f"score_{PATH_STRUCTURED}_{label}"] = np.nan

    outcomes: dict[str, LabelOutcome] = {}
    for label in wanted:
        # Rendered inside the loop rather than once for the whole run. The prompt
        # carries the question it asks, so handing the text track for `tail_risk`
        # the prompt written for `default_risk` measures it on the wrong question —
        # and does so silently, because both are just strings.
        text = text_inputs(panel, build, config, label)
        outcome = evaluate_label(
            panel,
            config,
            label,
            features,
            text,
            n_boot=config.eval.rolling.bootstrap_samples,
        )
        outcomes[label] = outcome
        if outcome.fitted and not outcome.scores.empty:
            panel.loc[outcome.scores.index, f"score_{PATH_FUSED}_{label}"] = outcome.scores[
                PATH_FUSED
            ].to_numpy()
            panel.loc[outcome.scores.index, f"score_{PATH_STRUCTURED}_{label}"] = outcome.scores[
                PATH_STRUCTURED
            ].to_numpy()

    folds = walk_forward_folds(config.split, config.labels, config.data.end)
    # Attached to the split report as well as returned, so the three places that describe
    # the folds cannot disagree. `SplitReport.folds` was previously left empty while
    # `PipelineResult.folds` held the real list, and the report's own "split actually
    # used" section reads its `n_folds` from the former — so a run with four folds
    # printed `n_folds | 0`, which a reader takes to mean no walk-forward was done.
    build.split_report.folds = folds
    notes: list[str] = []
    if not folds:
        notes.append(
            "walk-forward produced no usable fold. A block of "
            f"{config.split.test_block_years} years is smaller than the "
            f"{config.split.margin_days}-day purge margin, so every fold's validation "
            "block is inverted by purging. Raise split.test_block_years."
        )

    stability: dict[str, RollingStabilityResult] = {}
    for label, outcome in outcomes.items():
        if not outcome.fitted:
            continue
        result = _run_rolling(panel, label, f"score_{PATH_FUSED}_{label}", config)
        if result is not None:
            stability[label] = result

    # A comparison table that is missing its matched-information control must say so.
    # Three rows under the ordinary headings look exactly like a complete comparison, and
    # the reader's next move — subtracting the text path from `structured` — is then the
    # mismatched subtraction the control exists to replace.
    for label, outcome in outcomes.items():
        if outcome.matched_reason:
            notes.append(
                f"{label}: the comparison table has no '{PATH_MATCHED}' row. "
                f"{outcome.matched_reason}"
            )

    drift = _run_drift(panel, outcomes, features)
    stress = _run_stress(panel, outcomes, config)
    backtest = _run_backtest(panel, outcomes, config)

    return PipelineResult(
        config=config,
        build=build,
        outcomes=outcomes,
        split_report=build.split_report,
        folds=folds,
        drift=drift,
        stability=stability,
        historical_stress=stress,
        backtest=backtest,
        notes=notes,
    )


def _run_drift(
    panel: pd.DataFrame, outcomes: dict[str, LabelOutcome], features: list[str]
) -> list[DriftReport]:
    """Train-versus-test drift for every fitted label's fused score.

    The baseline is the *training* block, fixed rather than rolling: a rolling baseline
    drifts along with the data and reports zero drift by construction.
    """
    reports: list[DriftReport] = []
    train = panel.loc[panel["split"] == "train"]
    test = panel.loc[panel["split"] == "test"]
    if train.empty or test.empty:
        return reports
    for label, outcome in outcomes.items():
        column = f"score_{PATH_FUSED}_{label}"
        if not outcome.fitted or column not in panel.columns:
            continue
        try:
            reports.append(
                drift_report(
                    train,
                    test,
                    feature_columns=[c for c in features if c in panel.columns],
                    score_column=column,
                    baseline_label="train",
                    comparison_label="test",
                )
            )
        except Exception as exc:
            logger.warning("drift report unavailable for %s: %s", label, exc)
    return reports


def _run_stress(
    panel: pd.DataFrame, outcomes: dict[str, LabelOutcome], config: ProjectConfig
) -> pd.DataFrame:
    """Historical stress-period slices, concatenated across labels."""
    periods = [(p.name, p.start.isoformat(), p.end.isoformat()) for p in config.eval.stress_periods]
    frames: list[pd.DataFrame] = []
    for label, outcome in outcomes.items():
        column = f"score_{PATH_FUSED}_{label}"
        if not outcome.fitted or column not in panel.columns:
            continue
        try:
            table = historical_stress_suite(
                panel,
                score_column=column,
                label_column=f"label_{label}",
                forward_column=RISK_FORWARD_SPEC[label][0],
                periods=periods,
                n_quantiles=config.eval.n_quantiles,
            )
        except Exception as exc:
            logger.warning("stress suite unavailable for %s: %s", label, exc)
            continue
        if not table.empty:
            table = table.assign(label=label)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_backtest(
    panel: pd.DataFrame, outcomes: dict[str, LabelOutcome], config: ProjectConfig
) -> dict[str, Any]:
    """Cross-sectional quantile backtest, or an explicit refusal.

    Ten companies against five quantiles is two names per bucket, and a long-short
    spread built on that is an anecdote with a Sharpe ratio attached. The refusal is
    recorded with its reason so the report can print "not applicable" rather than
    nothing, because a missing section reads as an omission and this is a decision.
    """
    names_per_date = int(panel.groupby("as_of")["ticker"].nunique().max()) if len(panel) else 0
    if names_per_date < MIN_NAMES_PER_DATE:
        return {
            "applicable": False,
            "reason": (
                f"the busiest date holds {names_per_date} distinct companies, below the "
                f"{MIN_NAMES_PER_DATE} needed for {config.eval.n_quantiles} quantiles to "
                f"mean anything. Stage 2 is scoped as data-pipeline validation rather than "
                f"performance validation and reports IC and classification metrics instead "
                f"— see docs/07-roadmap.md."
            ),
            "results": {},
        }

    results: dict[str, Any] = {}
    forward_column = "fwd_ret_21d"
    for label, outcome in outcomes.items():
        column = f"score_{PATH_FUSED}_{label}"
        if not outcome.fitted or column not in panel.columns:
            continue
        test = panel.loc[(panel["split"] == "test") & panel[f"label_mask_{label}"].astype(bool)]
        if test.empty:
            continue
        scored = test.loc[test[column].notna()]
        if scored.empty:
            continue
        results[label] = quantile_returns(
            scored,
            score_col=column,
            forward_col=forward_column,
            date_col="as_of",
            n_quantiles=config.eval.n_quantiles,
            bootstrap=True,
            n_boot=config.eval.rolling.bootstrap_samples,
            block_days=config.labels.calendar_horizon_days(label),
            seed=config.project.seed,
        )
    return {
        "applicable": bool(results),
        "reason": "" if results else "no fitted label had scored test rows",
        "results": results,
    }


def environment_snapshot() -> dict[str, Any]:
    """Interpreter and platform facts recorded in every report.

    Deliberately small and dependency-free: it must not import torch, because the
    core install has no torch and a report that cannot be written without the training
    stack is a report the CPU-only path cannot produce.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
    }


def assemble_report(result: PipelineResult, *, run_id: str | None = None) -> EvaluationReport:
    """Build the :class:`EvaluationReport` from a pipeline result.

    Every field is populated from the run's own artifacts. Nothing here estimates or
    fills in a number: a quantity that was not measured stays ``not measured``, which is
    what the falsification table's ``not_evaluated`` states mean.
    """
    outcome = result.headline_outcome()
    headline = outcome.headline() if outcome else None
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Provenance decides two of the standing caveats, so it is read from the panel
    # rather than assumed. `default_caveats()` defaults to the synthetic case, and
    # calling it with no arguments — as this function used to — made every real-data
    # report carry "the panel contains synthetic rows" and "no real-data evaluation has
    # been performed" underneath its own real-data numbers. A caveat that is false
    # about the run it is attached to is worse than a missing one: it teaches the
    # reader to skim the section that exists to stop them misreading the numbers.
    is_synthetic = bool(result.build.metadata.get("is_synthetic", True))
    if "is_synthetic" in result.build.panel.columns and len(result.build.panel):
        is_synthetic = bool(result.build.panel["is_synthetic"].astype(bool).any())
    # Guarded as a whole rather than per field: `calibration` was read via
    # `(outcome.calibration or {})`, which protects against an empty dict but not against
    # `outcome` itself being None — the same mistake the surrounding `if outcome else`
    # lines already avoid.
    calibration = (outcome.calibration if outcome else None) or {}

    metadata = build_run_metadata(
        config=result.config.model_dump(mode="json"),
        split_report=result.split_report,
        calibration_fold="valid",
        calibrator_type=outcome.calibrator if outcome else "not measured",
        n_calibration_rows=int(calibration.get("n_calibration_rows", 0)),
        n_calibration_positives=int(calibration.get("n_calibration_positives", 0)),
        seed=result.config.project.seed,
        code_version=__version__,
        environment=environment_snapshot(),
        run_id=identifier,
        headline_label=outcome.label if outcome else "not measured",
        regime_note=_regime_note(result),
    )

    stability = result.stability.get(outcome.label) if outcome else None
    backtest = None
    if result.backtest.get("applicable") and outcome:
        backtest = result.backtest["results"].get(outcome.label)

    gates = (
        gate_table(
            headline,
            pr_auc_difference=outcome.pr_auc_difference if outcome else None,
            backtest=backtest,
            stability=stability,
        )
        if headline
        else []
    )

    comparison = _combined_comparison(result)
    ablation = (
        pd.concat(
            [
                frame.assign(label=label)
                for label, item in result.outcomes.items()
                if not (frame := item.ablation).empty
            ],
            ignore_index=True,
        )
        if any(not item.ablation.empty for item in result.outcomes.values())
        else pd.DataFrame()
    )

    return EvaluationReport(
        metadata=metadata,
        headline=headline,
        comparison=comparison,
        gates=gates,
        falsification=falsification_table(
            headline=headline,
            pr_auc_difference=outcome.pr_auc_difference if outcome else None,
            # The fine-tuned path is absent from `eval run` on purpose: it needs the
            # `train` extra and a GPU. Passing None leaves F3 reported as undecidable,
            # which is the honest state — a blank cell would read as "passed".
            text_lora=None,
            text_tfidf=outcome.reports.get(PATH_TEXT) if outcome else None,
            # No placebo path is implemented yet. `docs/05-evaluation.md` calls the
            # placebo control a hard requirement, so this is a stated gap in the
            # evaluation rather than a check that silently passed.
            placebo=None,
            # The IC interval would need a block bootstrap over the per-period IC series;
            # only the Newey-West t statistic is computed here.
            ic_interval=None,
            stability=stability,
        ),
        backtest=backtest,
        # Carried into the prose report, not just the JSON payload. The refusal reason was
        # only ever written to `report.json`, so the Markdown — the artifact a reader
        # actually reads — dropped the section and left the headings numbered 4, 6, 7.
        backtest_note=(
            str(result.backtest.get("reason", "")) if not result.backtest.get("applicable") else ""
        ),
        stability=stability,
        drift=result.drift,
        historical_stress=result.historical_stress,
        synthetic_stress=pd.DataFrame(),
        ablation=ablation,
        rolling_folds=_folds_table(result),
        caveats=(
            default_caveats(
                contains_synthetic_data=is_synthetic,
                real_data_evaluation=not is_synthetic,
            )
            + _data_gap_notes(outcome)
            + _window_gap_notes(result.build.panel)
            + result.notes
            + _unfitted_notes(result)
        ),
        # Row counts of the raw text sources, so the section-9 note describes the
        # corpus this run actually had. The build's frames are the ground truth —
        # feature columns could be zero for reasons other than an empty source.
        text_corpus={
            "news_rows": len(result.build.news),
            "filing_sections": len(result.build.filings),
        },
    )


def _window_gap_notes(panel: pd.DataFrame) -> list[str]:
    """Disclose calendar time that no split window covers, and what it costs.

    A row outside every nominal window is assigned ``excluded`` and quietly leaves the
    sample. That is usually a handful of days at the panel's edges. When it is a whole
    calendar year it is not a detail: in the Stage 2 run 2020 sits in no window and holds
    27 of the 39 observable positives, so the reported base rate (1.0%) is half the
    panel's own (2.2%) and the biggest drawdown episode in the sample is neither trained
    on nor evaluated. A reader cannot reconcile those two numbers without being told.

    Only rows whose label window has closed are counted; rows beyond the end of the data
    are terminal truncation, which is expected and already recorded elsewhere.
    """
    if panel.empty or "split" not in panel.columns or "as_of" not in panel.columns:
        return []
    excluded = panel.loc[panel["split"] == "excluded"]
    if excluded.empty:
        return []

    notes: list[str] = []
    years = sorted({int(year) for year in excluded["as_of"].dt.year.dropna().unique()})
    for column in sorted(name for name in panel.columns if name.startswith("label_")):
        label = column[len("label_") :]
        mask_column = f"label_mask_{label}"
        if mask_column not in panel.columns:
            continue
        observable = panel[mask_column].astype(bool)
        excluded_observable = excluded[mask_column].astype(bool)
        total_positives = int(panel.loc[observable, column].sum())
        excluded_positives = int(excluded.loc[excluded_observable, column].sum())
        if excluded_positives == 0:
            continue
        share = excluded_positives / total_positives if total_positives else float("nan")
        notes.append(
            f"`{label}`: {excluded_positives} of {total_positives} observable positive(s) "
            f"({share:.0%}) fall in rows no split window covers — calendar year(s) "
            f"{', '.join(str(year) for year in years)}, base rate "
            f"{excluded.loc[excluded_observable, column].mean():.1%} inside that block "
            f"against {panel.loc[observable, column].mean():.1%} for the panel. Those rows "
            "are neither trained on nor evaluated. Remedy: make the windows adjacent in the "
            "data overlay (set `valid.end` to the day before `test.start`) or move "
            "`test.start` back to cover the gap; the purge margin will then decide the "
            "boundary instead of the calendar."
        )
    return notes


def _data_gap_notes(outcome: LabelOutcome | None) -> list[str]:
    """Caveats for inputs the run did not have, and for what the fusion actually fitted.

    Two things a reader cannot infer from the metric tables and would otherwise have to
    take on trust.

    First, a model that silently trains on 34 of 49 configured features: "the report
    shows a strong structured result" and "the report shows a strong structured result
    computed without 15 of the 49 configured features, six of which have no data source
    at all" are different claims and only one of them is true.

    Second, the fusion's *fitted* kind. ``fusion.kind`` in the configuration is a
    request; a validation fold holding a single positive cannot support a logistic
    stacker, so the layer falls back to a rank average and the headline "fused" row is
    then an average, not a fitted stack. Without this note the fused number reads as
    evidence about stacking when it is evidence about arithmetic.

    Everything here is read off the fitted models rather than the configuration: the
    configuration says what was *asked* for, and the whole point is to report what was
    *there*.
    """
    notes: list[str] = []
    if outcome is None:
        return notes

    structured = outcome.models.get(PATH_STRUCTURED)
    dropped = list(getattr(structured, "dropped_features", []) or [])
    if dropped:
        # One line per source, with the remedy. A bare count of dropped features invites
        # the reader to assume they were unobtainable; several are one fetch or one symbol
        # away, and the difference changes what the reader should do next.
        for gap in diagnose_gaps(dropped):
            names = ", ".join(gap["features"])
            notes.append(
                f"{gap['count']} configured feature(s) held no observation anywhere in the "
                f"training block and were dropped before fitting — source `{gap['source']}`: "
                f"{names}. Remedy: {gap['advice']}."
            )

    baseline = outcome.models.get(PATH_TEXT)
    for warning in list(getattr(baseline, "warnings", []) or []):
        notes.append(f"text track: {warning}")

    fusion = outcome.models.get(PATH_FUSED)
    diagnostics = dict(getattr(fusion, "diagnostics", {}) or {})
    if diagnostics:
        notes.append(
            f"the fusion layer fitted as `{diagnostics.get('kind', 'unknown')}` on the "
            f"`{diagnostics.get('fit_split', 'unknown')}` fold "
            f"({diagnostics.get('n_fit_rows', '?')} rows, "
            f"{diagnostics.get('positives_fit', '?')} positive(s)); `fusion.kind` in the "
            "configuration is a request, and what can be fitted is decided by the fold."
        )
        if diagnostics.get("note"):
            notes.append(f"fusion layer: {diagnostics['note']}")
        if diagnostics.get("warning"):
            notes.append(f"fusion layer: {diagnostics['warning']}")
    return notes


def _regime_note(result: PipelineResult) -> str:
    """Regime characteristics of the evaluation window, for the report header.

    ``docs/05-evaluation.md`` requires the regime alongside any headline number,
    because an AUC measured through a calm window and one measured through a violent
    window are not the same quantity. The window described is the **effective test
    block** — the one the headline metrics were computed on — falling back to the full
    panel span only when no test rows exist.

    The series handed to :func:`~shingan.eval.report.describe_regime` is an
    equal-weighted buy-and-hold portfolio over the panel universe, not the raw price
    table. ``describe_regime`` computes ``pct_change`` on whatever frame it is given,
    so passing the pooled long-format table would difference one company's close
    against the next company's and report a fictitious path: an initial attempt at this
    did exactly that and produced "realised vol 876%, max drawdown -99.9%", which is
    the shape of a bug rather than of a market.

    An earlier version called ``describe_regime`` with a window and no price frame at
    all, inside a bare ``except``. That raised ``TypeError`` on every run and the
    handler returned a date range, so every report carried a ``regime_note`` that
    looked like a finding and contained no regime information. Failure is now a logged
    warning plus ``NOT_MEASURED``: a quantity that was not measured must not be dressed
    as one that was.
    """
    from shingan.eval.backtest import buy_and_hold_returns
    from shingan.eval.report import NOT_MEASURED, describe_regime

    prices = result.build.prices
    if prices is None or prices.empty:
        logger.warning("no price table on the build result; regime_note is not measured")
        return NOT_MEASURED

    window = result.split_report.windows.effective_test
    if window.start > window.end:
        logger.warning("effective test window is empty; describing the full panel span")
        stamps = pd.to_datetime(result.build.panel["as_of"])
        start, end = stamps.min().date(), stamps.max().date()
    else:
        start, end = window.start, window.end

    try:
        portfolio = buy_and_hold_returns(prices)
    except (KeyError, ValueError) as exc:
        logger.warning("could not build the regime benchmark portfolio: %s", exc)
        return NOT_MEASURED
    if portfolio.empty:
        logger.warning("the regime benchmark portfolio is empty; regime_note is not measured")
        return NOT_MEASURED

    equity = (1.0 + portfolio).cumprod()
    frame = pd.DataFrame(
        {
            "date": portfolio.index,
            "close": equity.to_numpy(),
            "returns": portfolio.to_numpy(),
        }
    )
    try:
        return describe_regime(frame, start, end, returns_column="returns")
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("regime description failed for %s..%s: %s", start, end, exc)
        return NOT_MEASURED


def _combined_comparison(result: PipelineResult) -> pd.DataFrame:
    """Every label's path comparison, stacked with the label as a column."""
    frames = [
        item.comparison.assign(label=label)
        for label, item in result.outcomes.items()
        if not item.comparison.empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _folds_table(result: PipelineResult) -> pd.DataFrame:
    """Walk-forward folds as a frame, so the report can show the rolling window."""
    return pd.DataFrame([fold.to_dict() for fold in result.folds])


def _unfitted_notes(result: PipelineResult) -> list[str]:
    """A caveat line per label that could not be evaluated, quoting the reason."""
    return [
        f"{label} was not evaluated: {item.reason}"
        for label, item in result.outcomes.items()
        if not item.fitted
    ]


def write_outputs(
    result: PipelineResult, paths: ProjectPaths, *, run_id: str | None = None
) -> dict[str, Path]:
    """Write the panel, the per-label evaluation JSON and the Markdown report.

    Returns:
        Mapping from a short name to the path written.
    """
    paths.ensure()
    written: dict[str, Path] = {}
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    panel_path = paths.processed / "panel.csv"
    result.build.panel.to_csv(panel_path, index=False)
    written["panel"] = panel_path

    report = assemble_report(result, run_id=identifier)
    # Raises ReportValidationError rather than writing an incomplete report: a report
    # that lands on disk without its calibration metadata gets read later as if it were
    # complete, which is worse than no report at all.
    report.metadata.validate()
    markdown_path = paths.reports / f"{identifier}.md"
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    written["report_markdown"] = markdown_path

    payload = report.as_dict()
    payload["labels"] = {
        label: {
            "fitted": item.fitted,
            "reason": item.reason,
            "counts": {
                "train": item.n_train,
                "valid": item.n_valid,
                "test": item.n_test,
                "positives_train": item.positives_train,
                "positives_valid": item.positives_valid,
                "positives_test": item.positives_test,
            },
            "calibration": item.calibration,
            "ic": None if item.ic is None else item.ic.as_dict(),
            "ic_newey_west_t": item.ic_tstat,
            "pr_auc_difference": (
                None if item.pr_auc_difference is None else item.pr_auc_difference.as_dict()
            ),
        }
        for label, item in result.outcomes.items()
    }
    payload["backtest"] = {
        "applicable": result.backtest.get("applicable", False),
        "reason": result.backtest.get("reason", ""),
    }
    payload["notes"] = list(result.notes)
    json_path = paths.reports / f"{identifier}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    written["report_json"] = json_path

    return written


__all__ = [
    "MIN_NAMES_PER_DATE",
    "MIN_QUOTE_CHARS",
    "NON_FEATURE_COLUMNS",
    "PATH_FUSED",
    "PATH_MATCHED",
    "PATH_ORDER",
    "PATH_STRUCTURED",
    "PATH_TEXT",
    "QUOTE_MAX_CHARS",
    "RISK_FORWARD_SPEC",
    "SFT_SPLITS",
    "TEXT_COLUMN",
    "LabelOutcome",
    "PipelineResult",
    "assemble_report",
    "build_prompt_contexts",
    "chars_budget_for_seqlength",
    "environment_snapshot",
    "evaluate_label",
    "label_split_frames",
    "matched_signal_report",
    "run_pipeline",
    "select_feature_columns",
    "sft_examples",
    "text_inputs",
    "write_outputs",
]

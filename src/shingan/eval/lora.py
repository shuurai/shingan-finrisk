"""Assembling a scored run of the text track's adapter.

``shingan eval run`` scores three paths and none of them is the fine-tuned model, so
the repository contained no answer to its own central question. This module builds the
missing artifact: one command's worth of measurements, with the two disclosures that
make the LoRA row readable rather than merely quotable.

The first disclosure is that the prompt is not text-only. It carries
``<STRUCTURED_SIGNALS>`` — the twelve columns of
:data:`~shingan.data.builder.PROMPT_SIGNAL_COLUMNS` — so ``text_only_lora`` does not
mean "saw only text", and subtracting it from the full-feature ``structured`` row is
not a measurement of the text contribution. Every payload written here says so, and
carries a matched baseline trained on the *same twelve signals* as the row to subtract
from.

The second is the parse-failure rate. A generation that does not parse is dropped from
the metrics, and dropped rows are not a random sample: a model that fails on its
longest, most complex prompts will look better after the failures are removed. The
rate, the count and the number of dropped **positives** travel with the numbers.

Nothing here imports torch. Loading and generation live in
:mod:`shingan.models.lora_inference`, and the functions in this module take the
resulting scores as input.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shingan.eval.metrics import (
    ClassificationReport,
    average_precision,
    evaluate_classification,
    paired_bootstrap_difference,
    roc_auc,
)

logger = logging.getLogger(__name__)

#: The LoRA's path name in every comparison table. Kept as ``text_only_lora`` because
#: ADR-0003 and the model card template both name it, but it is *not* what the name
#: claims — see the module docstring and ``PROMPT_DISCLOSURE``.
PATH_LORA = "text_only_lora"

#: The baseline the LoRA row is allowed to be subtracted from lives in
#: :mod:`shingan.pipeline` as ``PATH_MATCHED``, next to the three paths it joins: it is
#: fitted by ``matched_signal_report`` there, so its name is defined where its model is.
#: ``lora - structured_matched`` is the text contribution at equal information;
#: ``lora - structured`` is not.

#: The disclosure attached to every payload. One sentence, because it is the one that
#: gets left out. The signal names themselves are in ``prompt.structured_signals``.
PROMPT_DISCLOSURE = (
    "the prompt carries a <STRUCTURED_SIGNALS> block holding twelve structured signal "
    "columns, so this row is prompt-conditioned, not text-only; subtract it from "
    "'structured_matched' (the same twelve signals) rather than from 'structured' "
    "(the full feature set)"
)


@dataclass(slots=True)
class PromptIntegrity:
    """Whether the prompts rebuilt for scoring are the prompts the adapter was trained on.

    This is the check that turns "the model scored badly" from an unexplained result
    into a diagnosable one. Training builds its prompts through
    ``build_prompt_contexts`` + ``build_chat_messages``; scoring rebuilds them the same
    way, but a divergence — a different character budget, a different label in the task
    tag, a truncation applied on one side only — is invisible in the metrics and would
    read as a weak model rather than as a broken comparison.

    The key is ``(sample_id, label)`` and not ``sample_id``. ``sft_examples`` writes one
    example per (row, label), and the sample id is formed from the ticker and date alone,
    so a three-label SFT file holds three records under each id whose only difference is
    the ``<TASK>`` block. Keyed on the id alone this check reports a mismatch on two
    rows out of three for a file that is entirely correct — which is how this was found.
    """

    n_compared: int = 0
    n_matching: int = 0
    n_rebuilt_missing: int = 0
    n_scanned: int = 0
    n_other_label: int = 0
    n_unkeyed: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every comparable row matched and none was missing.

        ``n_unkeyed`` is deliberately not part of the verdict. Those are records from a
        file that does not carry the metadata this check keys on, not records this check
        failed to reproduce; they are reported so the coverage is visible.
        """
        return (
            self.n_compared > 0
            and self.n_matching == self.n_compared
            and self.n_rebuilt_missing == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_scanned": self.n_scanned,
            "n_compared": self.n_compared,
            "n_matching": self.n_matching,
            "n_rebuilt_missing": self.n_rebuilt_missing,
            "n_other_label": self.n_other_label,
            "n_unkeyed": self.n_unkeyed,
            "ok": self.ok,
            "examples": self.examples[:5],
        }


def _sft_user_turn(record: Mapping[str, Any]) -> str | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def _sample_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(sample_id, label)`` for a record, or None when the file lacks either."""
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return None
    sample = meta.get("sample_id")
    label = meta.get("label")
    if sample is None or label is None:
        return None
    return str(sample), str(label)


def compare_prompts(
    records: Iterable[Mapping[str, Any]],
    prompts_by_row: Mapping[tuple[str, str], str],
) -> PromptIntegrity:
    """Compare an SFT file's user turns against prompts rebuilt from the panel.

    Args:
        records: Parsed SFT JSONL records, for any label.
        prompts_by_row: ``(sample_id, label) -> rendered user prompt``, rebuilt through the
            same functions training used.

    Returns:
        The :class:`PromptIntegrity`. ``n_rebuilt_missing > 0`` is a failure of the same
        severity as a mismatch: a row the adapter trained on that scoring cannot
        reproduce is a row scoring is not entitled to measure. ``n_other_label`` counts
        records for labels that were not rebuilt, which is expected when one label is
        being scored out of a file that holds three.
    """
    integrity = PromptIntegrity()
    for record in records:
        integrity.n_scanned += 1
        key = _sample_key(record)
        if key is None:
            integrity.n_unkeyed += 1
            continue
        expected = _sft_user_turn(record)
        if expected is None:
            integrity.n_unkeyed += 1
            continue
        rebuilt = prompts_by_row.get(key)
        if rebuilt is None:
            # Distinguishing "this label was not rebuilt" from "this row is missing"
            # matters: the first is the normal case when scoring one of three labels.
            labels_rebuilt = {label for _, label in prompts_by_row}
            if key[1] not in labels_rebuilt:
                integrity.n_other_label += 1
            else:
                integrity.n_rebuilt_missing += 1
            continue
        integrity.n_compared += 1
        if rebuilt == expected:
            integrity.n_matching += 1
        elif len(integrity.examples) < 5:
            integrity.examples.append(
                {
                    "sample_id": key[0],
                    "label": key[1],
                    "sft_chars": len(expected),
                    "rebuilt_chars": len(rebuilt),
                    "first_difference": _first_difference(expected, rebuilt),
                }
            )
    return integrity


def _first_difference(left: str, right: str) -> dict[str, Any]:
    """Where two prompts diverge, as a small JSON-safe dict (for the failure message)."""
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return {
                "offset": index,
                "sft": left[max(0, index - 30) : index + 30],
                "rebuilt": right[max(0, index - 30) : index + 30],
            }
    longer = left if len(left) > len(right) else right
    return {
        "offset": min(len(left), len(right)),
        "trailing": longer[min(len(left), len(right)) :][:60],
    }


@dataclass(slots=True)
class ScoredRun:
    """A metric row plus the accounting that decides whether it may be quoted."""

    report: ClassificationReport
    n_attempted: int
    n_parsed: int
    n_dropped: int
    n_dropped_positives: int
    failure_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        return self.n_dropped / self.n_attempted if self.n_attempted else 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = self.report.as_dict()
        payload.update(
            {
                "n_attempted": self.n_attempted,
                "n_parsed": self.n_parsed,
                "n_dropped": self.n_dropped,
                "n_dropped_positives": self.n_dropped_positives,
                "parse_failure_rate": self.failure_rate,
                "parse_failure_reasons": self.failure_reasons,
            }
        )
        return payload


def score_from_attempts(
    y_true: Sequence[int],
    scores: Sequence[float | None],
    reasons: Sequence[str],
    *,
    label: str,
    path: str = PATH_LORA,
    split: str = "test",
    calibration_bins: int = 10,
) -> ScoredRun:
    """Metric row for one path, with the dropped rows accounted for.

    A row whose generation did not parse has no score, and there is no honest default to
    give it. Dropping it is the lesser evil — filling in 0.0 asserts the model called it
    negative, and filling in the base rate invents an observation — but dropping is only
    acceptable when the count is disclosed, which is what :class:`ScoredRun` is for.

    Args:
        y_true: Binary outcomes, one per attempted row.
        scores: Parsed score per row, ``None`` where the generation was rejected.
        reasons: Rejection reason per row, empty string where none.
        label: Label being scored.
        path: Name of this path in the table.
        split: Split being scored.
        calibration_bins: Bins for the calibration curve.

    Returns:
        The :class:`ScoredRun`.
    """
    truth = np.asarray(y_true, dtype=float)
    parsed = np.array([value is not None for value in scores], dtype=bool)
    kept = np.array([value for value in scores if value is not None], dtype=float)

    report = evaluate_classification(
        truth[parsed],
        kept,
        path=path,
        label=label,
        split=split,
        calibration_bins=calibration_bins,
    )
    tally = Counter(reason for reason, ok in zip(reasons, parsed, strict=True) if not ok)
    return ScoredRun(
        report=report,
        n_attempted=int(truth.size),
        n_parsed=int(parsed.sum()),
        n_dropped=int((~parsed).sum()),
        n_dropped_positives=int(truth[~parsed].sum()),
        failure_reasons=dict(tally.most_common()),
    )


def paired_differences(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baselines: Sequence[str],
    n_boot: int,
    block_days: int,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired bootstrap of ``candidate - baseline`` for PR-AUC and AUC, per baseline.

    Paired, not two independent intervals: two intervals that overlap is not a test, and
    two that do not overlap is not one either. Resampling both arms on the same rows
    cancels the shared sampling noise, which is the difference between "the text track
    beat this baseline" and "both were measured".

    Args:
        frame: One row per evaluation row, with ``as_of``, ``y_true`` and one column per
            path. Rows where any path is missing are dropped before resampling, so every
            comparison is measured on identical rows.
        candidate: Column holding the path being defended, normally the LoRA row.
        baselines: Columns to subtract.
        n_boot: Resamples.
        block_days: Block length for the bootstrap, normally the label's own horizon.
        alpha: Two-sided level.
        seed: RNG seed.

    Returns:
        One record per (baseline, metric). A record with ``estimate=None`` means no
        interval could be formed — stated explicitly rather than as a NaN, because a
        difference that was never measured must not read as a difference of zero.

    Raises:
        ValueError: If a named column is absent, or if the candidate is also a baseline.
    """
    if candidate in baselines:
        raise ValueError(f"{candidate!r} cannot be its own baseline")
    missing = [name for name in (candidate, *baselines) if name not in frame.columns]
    if missing:
        raise ValueError(f"the comparison frame is missing column(s): {missing}")

    usable = frame.dropna(subset=[candidate, *baselines])
    if len(usable) < len(frame):
        logger.warning(
            "%d of %d rows dropped from the paired comparison: at least one path had no "
            "score there, and a difference between different row sets is not a difference",
            len(frame) - len(usable),
            len(frame),
        )

    truth = usable["y_true"].to_numpy(dtype=float)
    if truth.size == 0 or truth.min() == truth.max():
        reason = (
            "the paired rows are empty"
            if truth.size == 0
            else "the paired rows hold only one class, so neither AUC nor PR-AUC exists"
        )
        return [
            {
                "a": candidate,
                "b": baseline,
                "metric": name,
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "crosses_zero": None,
                "n_rows": int(truth.size),
                "note": reason,
            }
            for baseline in baselines
            for name in ("pr_auc", "auc")
        ]

    records: list[dict[str, Any]] = []
    for baseline in baselines:
        for name, metric in (("pr_auc", average_precision), ("auc", roc_auc)):
            interval = paired_bootstrap_difference(
                usable["as_of"].to_numpy(),
                usable[candidate].to_numpy(),
                usable[baseline].to_numpy(),
                truth,
                metric,
                n_boot=n_boot,
                block_days=block_days,
                alpha=alpha,
                seed=seed,
            )
            record = interval.as_dict()
            record.update(
                {
                    "a": candidate,
                    "b": baseline,
                    "metric": name,
                    "n_rows": int(truth.size),
                    "note": (
                        "no interval: the test span holds fewer than two blocks, so the "
                        "resample distribution is degenerate"
                        if interval.n_blocks < 2
                        else ""
                    ),
                }
            )
            records.append(record)
    return records


def build_payload(
    *,
    label: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    scored: ScoredRun,
    adapter: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt: Mapping[str, Any],
    data: Mapping[str, Any],
    split_definition: Mapping[str, Any] | None = None,
    predictions: Sequence[Mapping[str, Any]] = (),
    differences: Sequence[Mapping[str, Any]] = (),
    caveats: Sequence[str] = (),
) -> dict[str, Any]:
    """The artifact written by ``shingan eval lora``.

    Section order is reading order: what was asked, of which data, with which model, and
    only then the numbers.

    Args:
        label: Label scored.
        split: Split scored.
        rows: Every path's metric row, in :data:`COMPARISON_ORDER` where available.
        scored: The LoRA row's own run, for the parse accounting.
        adapter: ``AdapterFacts.as_dict()``.
        generation: Sampling parameters.
        prompt: Integrity plus the signals disclosure.
        data: Provenance of the panel being scored.
        split_definition: The window definition the split came from, when available.
        predictions: One record per scored row — identifier, outcome, score and raw
            generation. Present so a reader can recompute the metrics instead of
            trusting them, which is the only way an aggregate AUC can be checked.
        differences: Paired bootstrap results against the baselines.
        caveats: Free-text caveats, appended to the fixed ones.

    Returns:
        A JSON-serialisable payload.
    """
    fixed_caveats = [PROMPT_DISCLOSURE]
    if data.get("is_synthetic"):
        fixed_caveats.append(
            "the panel is synthetic: the generator planted a text-only component, so a "
            "positive text contribution here is a check on the wiring and exactly zero "
            "evidence about markets"
        )
    if scored.n_dropped:
        fixed_caveats.append(
            f"{scored.n_dropped} of {scored.n_attempted} generations did not parse and "
            "were dropped; the metrics are conditional on the model producing usable "
            "output"
        )
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "label": label,
            "split": split,
        },
        "data": dict(data),
        "split_definition": dict(split_definition or {}),
        "adapter": dict(adapter),
        "generation": dict(generation),
        "prompt": dict(prompt),
        "rows": [dict(row) for row in rows],
        "parse": {
            "n_attempted": scored.n_attempted,
            "n_parsed": scored.n_parsed,
            "n_dropped": scored.n_dropped,
            "n_dropped_positives": scored.n_dropped_positives,
            "parse_failure_rate": scored.failure_rate,
            "reasons": scored.failure_reasons,
        },
        "differences": [dict(item) for item in differences],
        "predictions": [dict(item) for item in predictions],
        "caveats": [*fixed_caveats, *caveats],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the payload as a short report. Numbers first, then what they are not."""
    metadata = payload["metadata"]
    lines = [
        f"# LoRA evaluation — {metadata['label']} / {metadata['split']}",
        "",
        f"Generated {metadata['generated_at']}.",
        "",
        "## Comparison",
        "",
        "| path | AUC | KS | PR-AUC | base rate | positives | n rows |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {path} | {auc} | {ks} | {pr_auc} | {base} | {pos} | {n} |".format(
                path=row.get("path", "?"),
                auc=_number(row.get("auc")),
                ks=_number(row.get("ks")),
                pr_auc=_number(row.get("pr_auc")),
                base=_number(row.get("base_rate")),
                pos=row.get("n_positives", "?"),
                n=row.get("n_rows", "?"),
            )
        )

    if payload.get("differences"):
        lines += [
            "",
            "## Paired differences",
            "",
            "| comparison | metric | estimate | 95% CI | crosses zero | note |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in payload["differences"]:
            lines.append(
                "| {a} - {b} | {metric} | {est} | [{low}, {high}] | {cross} | {note} |".format(
                    a=item.get("a", "?"),
                    b=item.get("b", "?"),
                    metric=item.get("metric", "?"),
                    est=_number(item.get("estimate")),
                    low=_number(item.get("ci_low")),
                    high=_number(item.get("ci_high")),
                    cross=item.get("crosses_zero"),
                    note=item.get("note", ""),
                )
            )

    parse = payload["parse"]
    adapter = payload["adapter"]
    prompt = payload["prompt"]
    lines += [
        "",
        "## What produced these numbers",
        "",
        f"- adapter `{adapter.get('weights_file')}` sha256 `{str(adapter.get('weights_sha256'))[:16]}…` "
        f"({adapter.get('weights_bytes')} bytes), rank {adapter.get('rank')}, base `{adapter.get('base_model')}`",
        f"- tokenizer source: {adapter.get('tokenizer_source')}",
        f"- generation: {payload['generation']}",
        f"- prompt integrity against the SFT file: {prompt.get('integrity', {})}",
        f"- parse: {parse['n_parsed']}/{parse['n_attempted']} usable, "
        f"failure rate {parse['parse_failure_rate']:.4f}, dropped positives {parse['n_dropped_positives']}",
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {caveat}" for caveat in payload["caveats"]]
    return "\n".join(lines) + "\n"


def _number(value: Any) -> str:
    if value is None:
        return "not measured"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "not measured" if not np.isfinite(number) else f"{number:.4f}"


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``None``, recursively.

    ``json.dumps`` writes bare ``NaN`` and ``Infinity`` by default. Neither is JSON, and a
    strict parser rejects the whole file — so an artifact whose AUC was unmeasurable would
    be unreadable rather than merely incomplete. NaN in these metrics always means "not
    measurable on this sample" (too few positives of either class), which ``null`` states
    exactly, and which the renderer already prints as ``not measured``.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_artifact(payload: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Write ``payload`` as JSON plus a Markdown rendering.

    Both, not one: the JSON is what a later step reads a number out of, and the Markdown
    is what a person reads before quoting it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "lora_eval.json"
    markdown_path = out_dir / "lora_eval.md"
    # `allow_nan=False` makes the guard above load-bearing: if some future field carries a
    # non-finite number past `_json_safe`, this raises instead of writing a file that only
    # a lenient parser can read.
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"lora_eval_json": json_path, "lora_eval_markdown": markdown_path}


__all__ = [
    "PATH_LORA",
    "PROMPT_DISCLOSURE",
    "PromptIntegrity",
    "ScoredRun",
    "build_payload",
    "compare_prompts",
    "paired_differences",
    "render_markdown",
    "score_from_attempts",
    "write_artifact",
]

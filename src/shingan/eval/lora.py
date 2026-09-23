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
from shingan.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

#: The LoRA's path name in every comparison table. Kept as ``text_only_lora`` because
#: ADR-0003 and the model card template both name it, but it is *not* what the name
#: claims — see the module docstring and ``PROMPT_DISCLOSURE``.
PATH_LORA = "text_only_lora"

#: The base model before any fine-tuning, on the same prompt and through the same parser.
#: docs/05 defines this row as "the increment fine-tuning bought"; a row that does not
#: exist cannot play that role, and a row measured in a different process cannot either.
PATH_ZERO_SHOT = "text_only_zero_shot"

#: Which rows each mode scores, in the order they must be *generated*.
#:
#: Order is load-bearing, not cosmetic: scores come from one model object, and the
#: adapter arm attaches its weights to that object. The adapter arm is therefore last in
#: every tuple here, and :func:`arms_for_mode` is what the caller iterates.
ARMS_BY_MODE: dict[str, tuple[str, ...]] = {
    "adapter": (PATH_LORA,),
    "zero_shot": (PATH_ZERO_SHOT,),
    "both": (PATH_ZERO_SHOT, PATH_LORA),
}

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

#: The disclosure that only a run with a zero-shot arm needs. It is the same sentence the
#: adapter row needs in reverse: there, the prompt is not text-only; here, the model is
#: not fine-tuned *and* is being shown a template published by neither Qwen nor the
#: project's own config — it is the one the adapter saved.
ZERO_SHOT_DISCLOSURE = (
    "'text_only_zero_shot' is the base model with no adapter attached, shown the same "
    "prompt and parsed by the same parser as the adapter arm; it is also rendered by the "
    "tokenizer stored *with the adapter* rather than by the base model's own template, so "
    "the two text arms differ in their weights and not in their tokenisation"
)


def arms_for_mode(mode: str) -> tuple[str, ...]:
    """The paths a scoring mode produces, in generation order.

    Raises:
        ValueError: On an unknown mode. The command takes this string from the user, so
            the error has to name the alternatives rather than fail later with a KeyError
            from inside a loop.
    """
    try:
        return ARMS_BY_MODE[mode]
    except KeyError:
        raise ValueError(
            f"unknown scoring mode {mode!r}; choose one of {sorted(ARMS_BY_MODE)}"
        ) from None


def difference_plan(
    arms: Sequence[str],
    baselines: Sequence[str],
) -> list[dict[str, Any]]:
    """Which candidate is subtracted from what, as a list of groups.

    Two arms are not two independent results: the interesting quantity is the difference,
    and a difference between two rows measured in two processes, on two prompt sets, is
    not a paired measurement. Every arm is therefore compared against the arm it has to
    beat *and* against the fitted baselines — in **separate groups**, which is the part
    that is easy to get wrong.

    Separate, because :func:`paired_differences` restricts a comparison to the rows where
    every arm it was handed has a score. Passing the other arm and the fitted baselines in
    one group silently removes the rows where the other arm failed to parse from the
    comparison against baselines that do not involve it at all. Measured on this
    repository: adding a zero-shot arm that parsed 4 of 64 rows collapsed the adapter
    arm's `lora - structured_matched` from 64 rows to 4, and `paired_differences` then
    reported "not measured" for a comparison that is perfectly measurable without the
    zero-shot arm.

    Args:
        arms: The paths scored in this run, in generation order.
        baselines: The fitted baselines available to compare against, most important
            first. Passed in rather than derived, because ``structured_matched`` is
            absent whenever the panel lacks a prompt signal and a difference against an
            all-NaN arm would be reported as "not measured" beside two real ones.

    Returns:
        One ``{"candidate": path, "baselines": (...)}`` per comparison, in reading order:
        for each arm, the arms already scored before it, then the fitted baselines. The
        candidate is never in its own baseline list, and groups with nothing to compare
        are omitted.

    Raises:
        ValueError: If ``arms`` is empty, or if a name appears twice.
    """
    if not arms:
        raise ValueError("no arms to compare: a run that scores nothing has no plan")
    if len(set(arms)) != len(arms):
        raise ValueError(f"an arm was listed twice: {list(arms)}")
    if set(arms) & set(baselines):
        raise ValueError(
            f"{sorted(set(arms) & set(baselines))} cannot be both an arm and a baseline"
        )

    plan: list[dict[str, Any]] = []
    for index, arm in enumerate(arms):
        # Earlier arms first: in a two-arm run the base model is the subtraction that
        # answers "what did fine-tuning buy", and the reading order should not bury it
        # behind the fitted baselines.
        others = tuple(arms[:index])
        if others:
            plan.append({"candidate": arm, "baselines": others})
        if baselines:
            plan.append({"candidate": arm, "baselines": tuple(baselines)})
    return plan


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

    Two turns, two mechanisms. The user turn differs per row, so it is compared row by
    row. The system turn is a constant — every example carries the same one — and it was
    **not checked at all** until the zero-shot work surfaced why that matters: the
    instruction contract (`SYSTEM_PROMPT`) is not compared against anything, so editing it
    changes what the adapter is shown at scoring time relative to what it was trained on,
    silently, while this object still reports ``ok``. It is now compared, and any
    mismatch is fatal, because "the prompts are the ones the adapter trained on" is the
    sentence that licenses every number downstream.
    """

    n_compared: int = 0
    n_matching: int = 0
    n_rebuilt_missing: int = 0
    n_scanned: int = 0
    n_other_label: int = 0
    n_unkeyed: int = 0
    n_system_checked: int = 0
    n_system_mismatch: int = 0
    system_example: dict[str, Any] | None = None
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
            and self.n_system_mismatch == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_scanned": self.n_scanned,
            "n_compared": self.n_compared,
            "n_matching": self.n_matching,
            "n_rebuilt_missing": self.n_rebuilt_missing,
            "n_other_label": self.n_other_label,
            "n_unkeyed": self.n_unkeyed,
            "n_system_checked": self.n_system_checked,
            "n_system_mismatch": self.n_system_mismatch,
            "system_example": self.system_example,
            "ok": self.ok,
            "examples": self.examples[:5],
        }


def _turns(record: Mapping[str, Any]) -> dict[str, str]:
    """The ``role -> content`` mapping of an SFT record's messages, for the two roles
    this check compares."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return {}
    found: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user"} and isinstance(content, str) and role not in found:
            found[role] = content
    return found


def _sft_user_turn(record: Mapping[str, Any]) -> str | None:
    return _turns(record).get("user")


def _sft_system_turn(record: Mapping[str, Any]) -> str | None:
    return _turns(record).get("system")


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
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> PromptIntegrity:
    """Compare an SFT file's turns against prompts rebuilt from the panel.

    Args:
        records: Parsed SFT JSONL records, for any label.
        prompts_by_row: ``(sample_id, label) -> rendered user prompt``, rebuilt through the
            same functions training used.
        system_prompt: The instruction contract as it stands *now*. Every SFT example
            carries the one that was in force when the file was written, so comparing it
            is how an edit to `SYSTEM_PROMPT` between training and scoring is caught. It
            costs one string comparison per record and it is the difference between a
            check that covers the prompt and one that covers half of it.

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
        system = _sft_system_turn(record)
        if system is not None:
            integrity.n_system_checked += 1
            if system != system_prompt:
                integrity.n_system_mismatch += 1
                if integrity.system_example is None:
                    integrity.system_example = {
                        "sft_chars": len(system),
                        "current_chars": len(system_prompt),
                        "first_difference": _first_difference(system, system_prompt),
                    }
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

    def parse_dict(self) -> dict[str, Any]:
        """The parse accounting alone, without the metrics."""
        return {
            "n_attempted": self.n_attempted,
            "n_parsed": self.n_parsed,
            "n_dropped": self.n_dropped,
            "n_dropped_positives": self.n_dropped_positives,
            "parse_failure_rate": self.failure_rate,
            "reasons": self.failure_reasons,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.report.as_dict()
        payload.update(self.parse_dict())
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
    # The row set is part of the result, not a detail of it. A comparison measured on a
    # subset of the rows another comparison used is not the same measurement, and the
    # count alone does not say that the subset was chosen by which arm failed to parse.
    shortfall = (
        f"{len(frame) - len(usable)} of {len(frame)} rows are outside this comparison "
        "because at least one arm has no score there"
        if len(usable) < len(frame)
        else ""
    )
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
                "note": "; ".join(part for part in (reason, shortfall) if part),
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
            notes = [shortfall] if shortfall else []
            if interval.n_blocks < 2:
                notes.append(
                    "no interval: the test span holds fewer than two blocks, so the "
                    "resample distribution is degenerate"
                )
            record.update(
                {
                    "a": candidate,
                    "b": baseline,
                    "metric": name,
                    "n_rows": int(truth.size),
                    "note": "; ".join(notes),
                }
            )
            records.append(record)
    return records


def build_payload(
    *,
    label: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    scored: Mapping[str, ScoredRun],
    arms: Sequence[str],
    model: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt: Mapping[str, Any],
    data: Mapping[str, Any],
    split_definition: Mapping[str, Any] | None = None,
    predictions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    differences: Sequence[Mapping[str, Any]] = (),
    caveats: Sequence[str] = (),
) -> dict[str, Any]:
    """The artifact written by ``shingan eval lora``.

    Section order is reading order: what was asked, of which data, with which model, and
    only then the numbers.

    Three sections moved shape when the zero-shot arm arrived, and the reason is worth
    stating: an artifact that can describe one arm cannot describe two. ``adapter``
    became ``model``, because a run with no adapter has no adapter to name; ``parse`` and
    ``predictions`` became keyed by arm, because two arms have two parse rates and
    "the parse rate" stopped being a single number. Each section is still one thing —
    what changed is that the thing is now plural.

    Args:
        label: Label scored.
        split: Split scored.
        rows: Every path's metric row, in :data:`COMPARISON_ORDER` where available.
        scored: The text track's own runs, keyed by arm path, for the parse accounting.
        arms: The arm paths in reading order. Must match ``scored`` exactly: an arm in one
            and not the other is a payload whose table and whose accounting disagree.
        model: ``ModelIdentity.as_dict()`` for the weights that produced the scores.
        generation: Sampling parameters.
        prompt: Integrity plus the signals disclosure.
        data: Provenance of the panel being scored.
        split_definition: The window definition the split came from, when available.
        predictions: One record per scored row **per arm** — identifier, outcome, score
            and raw generation. Present so a reader can recompute the metrics instead of
            trusting them, which is the only way an aggregate AUC can be checked.
        differences: Paired bootstrap results against the baselines.
        caveats: Free-text caveats, appended to the fixed ones.

    Returns:
        A JSON-serialisable payload.

    Raises:
        ValueError: If ``arms`` and ``scored`` disagree, or if either is empty.
    """
    if set(arms) != set(scored):
        raise ValueError(
            f"the arms scored {sorted(scored)} and the arms reported {sorted(arms)} are "
            "not the same set; one of them would be an unaccounted row in the table"
        )
    if not arms:
        raise ValueError("no arm was scored, so there is nothing to report")

    fixed_caveats = [PROMPT_DISCLOSURE]
    if PATH_ZERO_SHOT in scored:
        fixed_caveats.append(ZERO_SHOT_DISCLOSURE)
    if data.get("is_synthetic"):
        fixed_caveats.append(
            "the panel is synthetic: the generator planted a text-only component, so a "
            "positive text contribution here is a check on the wiring and exactly zero "
            "evidence about markets"
        )
    for arm in arms:
        run = scored[arm]
        if run.n_dropped:
            fixed_caveats.append(
                f"{arm}: {run.n_dropped} of {run.n_attempted} generations did not parse "
                "and were dropped; that arm's metrics are conditional on the model "
                "producing usable output"
            )
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "label": label,
            "split": split,
        },
        "data": dict(data),
        "split_definition": dict(split_definition or {}),
        "model": dict(model),
        "arms": list(arms),
        "generation": dict(generation),
        "prompt": dict(prompt),
        "rows": [dict(row) for row in rows],
        "parse": {arm: scored[arm].parse_dict() for arm in arms},
        "differences": [dict(item) for item in differences],
        "predictions": {
            arm: [dict(item) for item in (predictions or {}).get(arm, ())] for arm in arms
        },
        "caveats": [*fixed_caveats, *caveats],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the payload as a short report. Numbers first, then what they are not."""
    metadata = payload["metadata"]
    arms = list(payload.get("arms", []))
    lines = [
        f"# Text-track evaluation — {metadata['label']} / {metadata['split']}",
        "",
        f"Generated {metadata['generated_at']}.",
        "",
        "## Comparison",
        "",
        "| path | role | AUC | KS | KS dir | PR-AUC | base rate | positives | n rows |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {path} | {role} | {auc} | {ks} | {ks_dir} | {pr_auc} | {base} | {pos} | {n} |".format(
                # The role is in the table rather than in a sentence below it: which rows
                # came from a model and which were fitted is the first thing a reader has
                # to know to subtract anything at all.
                path=row.get("path", "?"),
                role="arm" if row.get("path") in arms else "baseline",
                auc=_number(row.get("auc")),
                ks=_number(row.get("ks")),
                ks_dir=row.get("ks_direction") or "?",
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

    model = payload["model"]
    parse = payload.get("parse", {})
    adapter = model.get("adapter")
    prompt = payload["prompt"]
    if adapter is None:
        identity_line = (
            f"- weights: base model `{model.get('base_model')}` with **no adapter**, "
            f"{model.get('quantization')}"
        )
    else:
        identity_line = (
            f"- weights: base model `{model.get('base_model')}` plus adapter "
            f"`{adapter.get('weights_file')}` sha256 "
            f"`{str(adapter.get('weights_sha256'))[:16]}…` ({adapter.get('weights_bytes')} bytes), "
            f"rank {adapter.get('rank')}, {model.get('quantization')}"
        )
    lines += [
        "",
        "## What produced these numbers",
        "",
        f"- mode: {model.get('mode')} — arms scored: {', '.join(f'`{arm}`' for arm in arms)}",
        identity_line,
        f"- base model source: {model.get('base_model_source')}",
        f"- tokenizer: {model.get('tokenizer_source')} at `{model.get('tokenizer_path')}`",
        f"- generation: {payload['generation']}",
        f"- prompt integrity against the SFT file: {prompt.get('integrity', {})}",
    ]
    for arm in arms:
        accounting = parse.get(arm)
        if accounting is None:
            continue
        lines.append(
            f"- parse ({arm}): {accounting['n_parsed']}/{accounting['n_attempted']} usable, "
            f"failure rate {accounting['parse_failure_rate']:.4f}, "
            f"dropped positives {accounting['n_dropped_positives']}"
        )
    lines += [
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
    "ARMS_BY_MODE",
    "PATH_LORA",
    "PATH_ZERO_SHOT",
    "PROMPT_DISCLOSURE",
    "ZERO_SHOT_DISCLOSURE",
    "PromptIntegrity",
    "ScoredRun",
    "arms_for_mode",
    "build_payload",
    "compare_prompts",
    "difference_plan",
    "paired_differences",
    "render_markdown",
    "score_from_attempts",
    "write_artifact",
]

"""Report assembly, gate evaluation and card rendering.

The report is the deliverable, and its job is to make a number hard to misuse. Three
mechanisms, all of them from ``docs/05-evaluation.md``:

**Required metadata, enforced.** A metric vector without the split windows that
produced it, the calibration fold, and the class counts cannot be interpreted and
cannot be compared with anything. :class:`RunMetadata` declares those fields and
:meth:`RunMetadata.validate` raises when one is absent, so a report missing them is
never written rather than written and ignored.

**Gates reported as target-versus-achieved.** The gate values in
:data:`shingan.eval.metrics.GATES` are *targets*. The report prints both columns. A
report that lists only achieved values reads as though the targets were chosen to
match, which is the opposite of the point.

**Falsification conditions, evaluated where the inputs allow.** Section 10 of the
evaluation document lists eight conditions that would refute the project's central
claim. :func:`falsification_table` computes the ones that are decidable from the
report's own inputs and marks the rest ``not_evaluated`` — with the reason. An
unevaluated condition must not be readable as "not triggered".

Card rendering is here rather than in the publishing code because the templates live
in the repository and the values they need are the same ones the report holds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shingan.__about__ import (
    DATA_SCHEMA_VERSION,
    GITHUB_URL,
    HF_DATASET_ID,
    HF_MODEL_ID,
    PROJECT_DISPLAY_NAME,
    PROJECT_TAGLINE,
)
from shingan.eval.backtest import QuantileBacktestResult
from shingan.eval.metrics import GATES, BootstrapCI, ClassificationReport
from shingan.eval.splits import SplitReport
from shingan.eval.stability import DriftReport, RollingStabilityResult
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Metadata without which a metric vector is not interpretable. Every entry is
#: checked by :meth:`RunMetadata.validate`.
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "run_id",
    "created_utc",
    "config_snapshot",
    "split_definition",
    "test_window",
    "calibrator_type",
    "calibration_fold",
    "n_calibration_rows",
    "n_calibration_positives",
    "seed",
    "code_version",
    "environment",
)

#: Value written for anything that was not measured. Spelled the same everywhere so
#: that a reader can grep for it.
NOT_MEASURED = "not measured"

#: Written when a stress period falls outside the data span.
NOT_AVAILABLE = "not available: data span does not cover this period"


class ReportValidationError(ValueError):
    """Raised when a report is missing metadata that makes it interpretable."""


@dataclass(slots=True)
class RunMetadata:
    """Everything needed to trace a number back to the run that produced it."""

    run_id: str
    created_utc: str
    config_snapshot: Mapping[str, Any]
    split_definition: Mapping[str, Any]
    test_window: Mapping[str, str]
    calibrator_type: str
    calibration_fold: str
    n_calibration_rows: int
    n_calibration_positives: int
    seed: int
    code_version: str
    environment: Mapping[str, Any] = field(default_factory=dict)
    regime_note: str = NOT_MEASURED
    per_label_split: bool = False
    headline_label: str = "default_risk"
    n_seeds: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise unless every required field carries a real value.

        The empty-string and ``not measured`` checks are deliberate: a field present
        but blank passes a naive key check while carrying no information, which is the
        failure mode this guard exists for.

        Raises:
            ReportValidationError: Listing every offending field at once, so the
                caller does not have to fix them one report run at a time.
        """
        problems: list[str] = []
        for key in REQUIRED_METADATA_KEYS:
            value = getattr(self, key, None)
            if value is None:
                problems.append(f"{key}: missing")
            elif isinstance(value, str) and (not value.strip() or value == NOT_MEASURED):
                problems.append(f"{key}: blank or '{NOT_MEASURED}'")
            elif isinstance(value, Mapping) and not value:
                problems.append(f"{key}: empty mapping")
        if self.n_calibration_rows <= 0:
            problems.append("n_calibration_rows: must be positive")
        if self.n_calibration_positives < 0:
            problems.append("n_calibration_positives: must not be negative")
        if self.n_calibration_positives > self.n_calibration_rows:
            problems.append("n_calibration_positives: exceeds n_calibration_rows")
        if problems:
            raise ReportValidationError(
                "the report cannot be written because its metadata is incomplete; a "
                "metric vector without these fields is not interpretable:\n  - "
                + "\n  - ".join(problems)
            )

    def as_dict(self) -> dict[str, Any]:
        payload = {key: getattr(self, key) for key in REQUIRED_METADATA_KEYS}
        payload.update(
            {
                "regime_note": self.regime_note,
                "per_label_split": self.per_label_split,
                "headline_label": self.headline_label,
                "n_seeds": self.n_seeds,
            }
        )
        payload.update(dict(self.extra))
        return payload


def build_run_metadata(
    *,
    config: Mapping[str, Any],
    split_report: SplitReport | None = None,
    calibration_fold: str = "valid",
    calibrator_type: str = "not measured",
    n_calibration_rows: int = 0,
    n_calibration_positives: int = 0,
    seed: int = 0,
    code_version: str = "unknown",
    environment: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    headline_label: str = "default_risk",
    regime_note: str = NOT_MEASURED,
) -> RunMetadata:
    """Assemble :class:`RunMetadata` from a run's own artefacts.

    The defaults are *invalid on purpose*: ``calibrator_type`` defaults to
    ``not measured`` and the calibration counts to zero, so a caller that forgets to
    pass them gets a validation error rather than a report asserting a mean of zero
    calibrator. Defaults that quietly fabricate plausible values are how the required
    metadata requirement gets satisfied on paper and not in fact.
    """
    split_definition: dict[str, Any] = {}
    test_window: dict[str, str] = {}
    per_label = False
    if split_report is not None:
        split_definition = {
            "train": split_report.windows.nominal_train.render(),
            "valid": split_report.windows.nominal_valid.render(),
            "test": split_report.windows.nominal_test.render(),
            "effective_train": split_report.windows.effective_train.render(),
            "effective_valid": split_report.windows.effective_valid.render(),
            "effective_test": split_report.windows.effective_test.render(),
            "purge_days": split_report.windows.purge_days,
            "embargo_calendar_days": split_report.windows.embargo_calendar_days,
            "margin_days": split_report.windows.margin_days,
            "per_label_horizons": dict(split_report.per_label_horizons),
            "n_folds": len(split_report.folds),
        }
        test_window = {
            "nominal": split_report.windows.nominal_test.render(),
            "effective": split_report.windows.effective_test.render(),
        }
        per_label = split_report.per_label

    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RunMetadata(
        run_id=identifier,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        config_snapshot=dict(config),
        split_definition=split_definition or {"definition": NOT_MEASURED},
        test_window=test_window or {"nominal": NOT_MEASURED, "effective": NOT_MEASURED},
        calibrator_type=calibrator_type,
        calibration_fold=calibration_fold,
        n_calibration_rows=n_calibration_rows,
        n_calibration_positives=n_calibration_positives,
        seed=seed,
        code_version=code_version,
        environment=dict(environment or {"environment": NOT_MEASURED}),
        regime_note=regime_note,
        per_label_split=per_label,
        headline_label=headline_label,
    )


# --------------------------------------------------------------------------- #
# Gates and falsification
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GateRow:
    """One acceptance criterion, target against achieved."""

    name: str
    target: str
    achieved: str
    passed: bool | None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "achieved": self.achieved,
            "passed": self.passed,
            "note": self.note,
        }


def _format(value: float, places: int = 4) -> str:
    """Format a metric, spelling out NaN rather than printing 'nan'."""
    if value is None or not np.isfinite(value):
        return "undefined"
    return f"{value:.{places}f}"


def _format_ci(interval: BootstrapCI | None) -> str:
    if interval is None:
        return "not computed"
    if not np.isfinite(interval.low) or not np.isfinite(interval.high):
        return "not computed"
    return f"[{interval.low:.4f}, {interval.high:.4f}]"


def gate_table(
    report: ClassificationReport,
    *,
    pr_auc_difference: BootstrapCI | None = None,
    backtest: QuantileBacktestResult | None = None,
    stability: RollingStabilityResult | None = None,
) -> list[GateRow]:
    """Build the target-versus-achieved table for section 9's acceptance gates.

    Args:
        report: The headline path's metrics on the test slice.
        pr_auc_difference: Bootstrapped ``PR-AUC(fused) - PR-AUC(structured_only)``.
            This is the project's central claim; without it the fusion gate is
            reported as ``not computed`` rather than as passed.
        backtest: Quantile backtest, for the economic-significance row.
        stability: Rolling stability, for the walk-forward row.

    Returns:
        One :class:`GateRow` per gate, in the order the document lists them.
    """
    rows: list[GateRow] = [
        GateRow(
            name="headline_auc",
            target=f"> {GATES['auc']:.2f}",
            achieved=_format(report.auc),
            passed=bool(np.isfinite(report.auc) and report.auc > GATES["auc"]),
            note=f"label={report.label}, n_positives={report.n_positives}",
        ),
        GateRow(
            name="headline_ks",
            target=f"> {GATES['ks']:.2f}",
            achieved=_format(report.ks),
            passed=bool(np.isfinite(report.ks) and report.ks > GATES["ks"]),
            note=f"direction={report.ks_direction}",
        ),
    ]

    if pr_auc_difference is not None:
        significant = bool(
            np.isfinite(pr_auc_difference.low)
            and np.isfinite(pr_auc_difference.high)
            and pr_auc_difference.low > 0.0
        )
        rows.append(
            GateRow(
                name="fusion_gain_pr_auc",
                target="fused PR-AUC > structured_only, CI excludes zero",
                achieved=(
                    f"delta={_format(pr_auc_difference.estimate)} {_format_ci(pr_auc_difference)}"
                ),
                passed=significant,
                note="the project's central claim",
            )
        )
    else:
        rows.append(
            GateRow(
                name="fusion_gain_pr_auc",
                target="fused PR-AUC > structured_only, CI excludes zero",
                achieved="not computed",
                passed=None,
                note="no paired bootstrap was supplied; this gate cannot be reported as passed",
            )
        )

    if report.calibration is not None:
        passes = report.calibration.passes()
        rows.append(
            GateRow(
                name="calibration_ece",
                target=f"< {GATES['ece']:.2f}",
                achieved=_format(report.calibration.ece),
                passed=passes["ece_below_gate"],
                note=f"tail ECE {_format(report.calibration.tail_ece)} on the top decile",
            )
        )
        rows.append(
            GateRow(
                name="brier_beats_base_rate",
                target="Brier skill > 0",
                achieved=_format(report.calibration.brier_skill),
                passed=passes["brier_beats_base_rate"],
                note=f"base rate {_format(report.calibration.base_rate)}",
            )
        )

    if backtest is not None:
        for name, passed in backtest.passes().items():
            target = {
                "spread_significant": "long-short spread |t| > 1.96 (Newey-West)",
                "high_risk_underperforms": "high-risk quantile return < low-risk",
                "quantiles_monotonic": "quantile returns monotone (<=1 tail reversal)",
                "sharpe_above_target": "|Sharpe| > 1.0 (target, not a gate)",
            }[name]
            achieved = (
                _format(backtest.long_short_stats.t_stat)
                if name == "spread_significant" and backtest.long_short_stats
                else "see backtest table"
            )
            rows.append(
                GateRow(
                    name=name,
                    target=target,
                    achieved=achieved,
                    passed=passed,
                    note="aspiration, not a gate" if name == "sharpe_above_target" else "",
                )
            )

    if stability is not None:
        for name, passed in stability.passes().items():
            rows.append(
                GateRow(
                    name=f"stability_{name}",
                    target="windows' AUC intervals above random, no systematic decline",
                    achieved=f"{stability.coverage.get('n_usable', 0)}/"
                    f"{stability.coverage.get('n_windows', 0)} usable windows",
                    passed=passed,
                    note="the '<10% swing' gate from the notes is unreachable and is not used",
                )
            )

    return rows


@dataclass(slots=True)
class FalsificationCheck:
    """One refutation condition from section 10 of the evaluation document."""

    code: str
    condition: str
    verdict: str
    triggered: bool | None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "condition": self.condition,
            "verdict": self.verdict,
            "triggered": self.triggered,
            "detail": self.detail,
        }


#: The eight conditions, verbatim in intent from section 10. Kept here so that the
#: report always lists all of them, including the ones it cannot decide.
FALSIFICATION_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("F1", "fused PR-AUC overlaps structured_only; no measurable text increment"),
    ("F2", "text-shuffled placebo does not degrade the metrics"),
    ("F3", "TF-IDF text baseline matches the fine-tuned model"),
    ("F4", "IC confidence interval crosses zero"),
    ("F5", "calibrated and uncalibrated ECE barely differ and both exceed the gate"),
    ("F6", "increment disappears once market regime variables are added"),
    ("F7", "later walk-forward folds are systematically worse than early ones"),
    ("F8", "all metrics beat baselines but the positive count is under 20"),
)


def falsification_table(
    *,
    headline: ClassificationReport | None = None,
    pr_auc_difference: BootstrapCI | None = None,
    text_lora: ClassificationReport | None = None,
    text_tfidf: ClassificationReport | None = None,
    placebo: ClassificationReport | None = None,
    ic_interval: BootstrapCI | None = None,
    uncalibrated_ece: float | None = None,
    stability: RollingStabilityResult | None = None,
) -> list[FalsificationCheck]:
    """Evaluate the refutation conditions that the supplied inputs can decide.

    Anything undecidable is reported with ``triggered=None`` and a reason. That is
    the point of the function: a blank cell in a falsification table reads as "fine",
    and the whole table exists because "fine" is the conclusion most worth doubting.

    Args:
        headline: The fused model's headline metrics.
        pr_auc_difference: Bootstrapped fused-minus-structured PR-AUC difference.
        text_lora: Text-only fine-tuned path, for F3.
        text_tfidf: TF-IDF path, for F3.
        placebo: The text-shuffled run, for F2.
        ic_interval: Bootstrapped IC interval, for F4.
        uncalibrated_ece: ECE before calibration, for F5.
        stability: Walk-forward results, for F7.

    Returns:
        One :class:`FalsificationCheck` per condition, in order.
    """
    checks: list[FalsificationCheck] = []

    # F1 — the central claim.
    if pr_auc_difference is None:
        verdict, triggered, detail = "not_evaluated", None, "no paired bootstrap supplied"
    elif not (np.isfinite(pr_auc_difference.low) and np.isfinite(pr_auc_difference.high)):
        verdict, triggered, detail = (
            "not_evaluated",
            None,
            "the bootstrap produced no usable interval",
        )
    elif pr_auc_difference.low > 0.0:
        verdict, triggered, detail = (
            "not_triggered",
            False,
            f"CI {_format_ci(pr_auc_difference)} excludes zero",
        )
    elif pr_auc_difference.high < 0.0:
        verdict, triggered, detail = (
            "triggered",
            True,
            (
                f"CI {_format_ci(pr_auc_difference)} is entirely below zero: the fusion is "
                "worse than structured-only"
            ),
        )
    else:
        verdict, triggered, detail = (
            "triggered",
            True,
            (f"CI {_format_ci(pr_auc_difference)} includes zero: no demonstrated increment"),
        )
    checks.append(
        FalsificationCheck("F1", FALSIFICATION_CONDITIONS[0][1], verdict, triggered, detail)
    )

    # F2 — placebo.
    if placebo is None or headline is None:
        checks.append(
            FalsificationCheck(
                "F2", FALSIFICATION_CONDITIONS[1][1], "not_evaluated", None, "no placebo run"
            )
        )
    else:
        drop = headline.auc - placebo.auc
        if not np.isfinite(drop):
            checks.append(
                FalsificationCheck(
                    "F2", FALSIFICATION_CONDITIONS[1][1], "not_evaluated", None, "undefined AUC"
                )
            )
        elif drop <= 0:
            checks.append(
                FalsificationCheck(
                    "F2",
                    FALSIFICATION_CONDITIONS[1][1],
                    "triggered",
                    True,
                    f"AUC changed by {drop:+.4f} under shuffled text; the model is not using "
                    "text content, so every positive result is void",
                )
            )
        else:
            checks.append(
                FalsificationCheck(
                    "F2",
                    FALSIFICATION_CONDITIONS[1][1],
                    "not_triggered",
                    False,
                    f"AUC fell {drop:.4f} under shuffled text; significance not assessed here",
                )
            )

    # F3 — does the LLM beat a bag of words?
    if text_lora is None or text_tfidf is None:
        checks.append(
            FalsificationCheck(
                "F3",
                FALSIFICATION_CONDITIONS[2][1],
                "not_evaluated",
                None,
                "one of the two text paths is missing",
            )
        )
    else:
        gap = text_lora.auc - text_tfidf.auc
        if not np.isfinite(gap):
            checks.append(
                FalsificationCheck(
                    "F3", FALSIFICATION_CONDITIONS[2][1], "not_evaluated", None, "undefined AUC"
                )
            )
        else:
            checks.append(
                FalsificationCheck(
                    "F3",
                    FALSIFICATION_CONDITIONS[2][1],
                    "triggered" if gap <= 0 else "not_triggered",
                    bool(gap <= 0),
                    f"LoRA AUC - TF-IDF AUC = {gap:+.4f}; a small positive gap at this "
                    "sample size is itself weak evidence",
                )
            )

    # F4 — cross-sectional ranking ability.
    if ic_interval is None:
        checks.append(
            FalsificationCheck(
                "F4",
                FALSIFICATION_CONDITIONS[3][1],
                "not_evaluated",
                None,
                "no bootstrap computed for IC",
            )
        )
    elif np.isfinite(ic_interval.estimate) and ic_interval.estimate < 0:
        checks.append(
            FalsificationCheck(
                "F4",
                FALSIFICATION_CONDITIONS[3][1],
                "triggered",
                True,
                f"IC {_format(ic_interval.estimate)} is negative; the score inverts the ordering",
            )
        )
    elif np.isfinite(ic_interval.low) and ic_interval.low > 0.0:
        checks.append(
            FalsificationCheck(
                "F4",
                FALSIFICATION_CONDITIONS[3][1],
                "not_triggered",
                False,
                f"CI {_format_ci(ic_interval)} excludes zero",
            )
        )
    else:
        checks.append(
            FalsificationCheck(
                "F4",
                FALSIFICATION_CONDITIONS[3][1],
                "triggered",
                True,
                f"IC CI {_format_ci(ic_interval)} includes zero: high AUC would then rest on "
                "a few extreme names rather than on usable ordering",
            )
        )

    # F5 — is the calibration real?
    if headline is None or headline.calibration is None or uncalibrated_ece is None:
        checks.append(
            FalsificationCheck(
                "F5",
                FALSIFICATION_CONDITIONS[4][1],
                "not_evaluated",
                None,
                "no uncalibrated ECE recorded",
            )
        )
    else:
        calibrated = headline.calibration.ece
        both_above = bool(
            np.isfinite(calibrated)
            and np.isfinite(uncalibrated_ece)
            and calibrated > GATES["ece"]
            and uncalibrated_ece > GATES["ece"]
        )
        detail = f"ECE before {_format(uncalibrated_ece)} -> after {_format(calibrated)}" + (
            f"; both exceed {GATES['ece']:.2f}" if both_above else ""
        )
        checks.append(
            FalsificationCheck(
                "F5",
                FALSIFICATION_CONDITIONS[4][1],
                "triggered" if both_above else "not_triggered",
                both_above,
                detail,
            )
        )

    # F6 — needs an incremental model that includes regime variables, which this
    # report does not contain. Reported rather than guessed.
    checks.append(
        FalsificationCheck(
            "F6",
            FALSIFICATION_CONDITIONS[5][1],
            "not_evaluated",
            None,
            "requires refitting with VIX and index drawdown as explicit features; "
            "that ablation is not part of this report",
        )
    )

    # F7 — trend across walk-forward folds.
    if stability is None or stability.table.empty:
        checks.append(
            FalsificationCheck(
                "F7", FALSIFICATION_CONDITIONS[6][1], "not_evaluated", None, "no walk-forward table"
            )
        )
    else:
        trend = stability.trends.get("auc")
        if trend is None or trend.n_points < 3:
            checks.append(
                FalsificationCheck(
                    "F7",
                    FALSIFICATION_CONDITIONS[6][1],
                    "not_evaluated",
                    None,
                    f"fewer than three usable windows ({trend.n_points if trend else 0})",
                )
            )
        else:
            declining = trend.direction == "down" and trend.delta < 0
            checks.append(
                FalsificationCheck(
                    "F7",
                    FALSIFICATION_CONDITIONS[6][1],
                    "triggered" if declining else "not_triggered",
                    declining,
                    f"Theil-Sen slope {trend.slope_per_period:+.4f} per window over "
                    f"{trend.n_points} windows; first {trend.first_value:.4f} to last {trend.last_value:.4f}",
                )
            )

    # F8 — statistically undecidable positive counts.
    if headline is None:
        checks.append(
            FalsificationCheck(
                "F8", FALSIFICATION_CONDITIONS[7][1], "not_evaluated", None, "no headline metrics"
            )
        )
    else:
        too_few = headline.n_positives < 20
        checks.append(
            FalsificationCheck(
                "F8",
                FALSIFICATION_CONDITIONS[7][1],
                "triggered" if too_few else "not_triggered",
                too_few,
                f"{headline.n_positives} positives in the test slice"
                + ("; report as insufficient evidence, not as a pass" if too_few else ""),
            )
        )

    return checks


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EvaluationReport:
    """One assembled evaluation report."""

    metadata: RunMetadata
    headline: ClassificationReport | None = None
    comparison: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    gates: list[GateRow] = field(default_factory=list)
    falsification: list[FalsificationCheck] = field(default_factory=list)
    backtest: QuantileBacktestResult | None = None
    #: Why section 5 is absent, when it is absent. The backtest refuses to run below a
    #: minimum cross-sectional breadth, and that refusal is a decision the reader is owed:
    #: omitting the section left the heading list running 4, 6, 7 ..., which reads as a
    #: dropped section rather than a deliberate one.
    backtest_note: str = ""
    stability: RollingStabilityResult | None = None
    drift: list[DriftReport] = field(default_factory=list)
    historical_stress: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    synthetic_stress: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    ablation: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    rolling_folds: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.as_dict(),
            "headline": self.headline.as_dict() if self.headline else None,
            "comparison": self.comparison.to_dict(orient="records")
            if not self.comparison.empty
            else [],
            "gates": [row.as_dict() for row in self.gates],
            "falsification": [check.as_dict() for check in self.falsification],
            "backtest": self.backtest.as_dict() if self.backtest else None,
            "backtest_note": self.backtest_note,
            "stability": self.stability.as_dict() if self.stability else None,
            "drift": [report.as_dict() for report in self.drift],
            "historical_stress": (
                self.historical_stress.to_dict(orient="records")
                if not self.historical_stress.empty
                else []
            ),
            "synthetic_stress": (
                self.synthetic_stress.to_dict(orient="records")
                if not self.synthetic_stress.empty
                else []
            ),
            "ablation": self.ablation.to_dict(orient="records") if not self.ablation.empty else [],
            "rolling_folds": (
                self.rolling_folds.to_dict(orient="records") if not self.rolling_folds.empty else []
            ),
            "caveats": list(self.caveats),
        }

    def to_markdown(self) -> str:
        """Render the report as Markdown.

        Section order follows the evaluation document, and every metric table carries
        its class counts. A table of metrics without counts is the specific artefact
        this project is trying not to produce.
        """
        lines: list[str] = []
        meta = self.metadata
        lines.append(f"# {PROJECT_DISPLAY_NAME} evaluation report")
        lines.append("")
        lines.append(f"_{PROJECT_TAGLINE}_")
        lines.append("")
        lines.append(f"- run id: `{meta.run_id}`")
        lines.append(f"- created (UTC): `{meta.created_utc}`")
        lines.append(f"- code version: `{meta.code_version}`")
        lines.append(f"- seed: `{meta.seed}` (n_seeds={meta.n_seeds})")
        lines.append(f"- headline label: `{meta.headline_label}`")
        lines.append(f"- calibrator: `{meta.calibrator_type}` fitted on `{meta.calibration_fold}`")
        lines.append(
            f"- calibration rows: {meta.n_calibration_rows} "
            f"({meta.n_calibration_positives} positives)"
        )
        lines.append(
            f"- per-label split: {meta.per_label_split} (cross-label windows differ when true)"
        )
        lines.append("")

        lines.append("## 1. Split actually used")
        lines.append("")
        lines.append("| block | window |")
        lines.append("| --- | --- |")
        for key, value in meta.split_definition.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append(f"- test window: `{meta.test_window.get('effective', NOT_MEASURED)}`")
        lines.append(f"- regime note: {meta.regime_note}")
        lines.append("")
        lines.append(
            "> Effective windows, not nominal ones. A report that prints only nominal "
            "windows overstates the data it used."
        )
        lines.append("")

        if self.headline is not None:
            lines.append("## 2. Headline metrics")
            lines.append("")
            lines.append(
                f"n_rows={self.headline.n_rows}, n_positives={self.headline.n_positives}, "
                f"base_rate={_format(self.headline.base_rate)}"
            )
            lines.append("")
            lines.append("| metric | value |")
            lines.append("| --- | --- |")
            for key in ("auc", "ks", "pr_auc", "pr_auc_lift", "capture_top5", "accuracy_ratio"):
                lines.append(f"| {key} | {_format(getattr(self.headline, key))} |")
            lines.append(f"| ks_direction | {self.headline.ks_direction} |")
            lines.append(
                f"| monotonic | {self.headline.monotonic} ({self.headline.n_reversals} reversals) |"
            )
            lines.append("")
            if self.headline.calibration is not None:
                calib = self.headline.calibration
                lines.append("### Calibration")
                lines.append("")
                lines.append("| metric | value |")
                lines.append("| --- | --- |")
                lines.append(f"| ECE | {_format(calib.ece)} |")
                lines.append(f"| MCE | {_format(calib.mce)} |")
                lines.append(f"| Brier | {_format(calib.brier)} |")
                lines.append(f"| Brier skill | {_format(calib.brier_skill)} |")
                lines.append(
                    f"| tail ECE (top decile, n={calib.tail_n}) | {_format(calib.tail_ece)} |"
                )
                lines.append("")

        if not self.comparison.empty:
            lines.append("## 3. Path comparison")
            lines.append("")
            lines.append(
                "Every performance claim needs structured-only, text-only and fused "
                "side by side, plus a no-model baseline."
            )
            lines.append("")
            lines.append(_frame_to_markdown(self.comparison))
            lines.append("")

        if self.gates:
            lines.append("## 4. Acceptance gates: target vs achieved")
            lines.append("")
            lines.append("| gate | target | achieved | result | note |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in self.gates:
                result = (
                    "not evaluated" if row.passed is None else ("pass" if row.passed else "FAIL")
                )
                lines.append(
                    f"| {row.name} | {row.target} | {row.achieved} | {result} | {row.note} |"
                )
            lines.append("")

        if self.backtest is not None:
            lines.append("## 5. Quantile backtest (cross-sectional, per date)")
            lines.append("")
            lines.append(
                f"quantiles={self.backtest.n_quantiles}, dates={self.backtest.n_dates}, "
                f"rows={self.backtest.n_rows}, "
                f"dates skipped for insufficient breadth={self.backtest.insufficient_dates}"
            )
            lines.append("")
            lines.append("| quantile (0 = highest risk) | mean forward return |")
            lines.append("| --- | --- |")
            for bucket, value in self.backtest.mean_by_quantile.items():
                lines.append(f"| {bucket} | {_format(float(value))} |")
            lines.append("")
            if self.backtest.long_short_stats is not None:
                stats = self.backtest.long_short_stats
                lines.append(
                    f"Low-minus-high spread: mean {_format(stats.mean_period_return)}, "
                    f"Sharpe {_format(stats.sharpe)}, Sortino {_format(stats.sortino)}, "
                    f"max drawdown {_format(stats.max_drawdown)}, Calmar {_format(stats.calmar)}, "
                    f"Newey-West t {_format(stats.t_stat)}"
                )
                lines.append("")
                lines.append(
                    "> The Sharpe figure is computed on overlapping forward windows and is "
                    "inflated by roughly sqrt(horizon). Quote the Newey-West t-statistic."
                )
            lines.append("")
        elif self.backtest_note:
            # Keeps the section numbering contiguous and states the decision. "Not
            # applicable" is a result; a missing section is indistinguishable from an
            # omission, and the reader has no way to tell which they are looking at.
            lines.append("## 5. Quantile backtest (cross-sectional, per date)")
            lines.append("")
            lines.append(f"Not run. {self.backtest_note}")
            lines.append("")

        if self.stability is not None and not self.stability.table.empty:
            lines.append("## 6. Walk-forward / rolling stability")
            lines.append("")
            if not self.rolling_folds.empty:
                # Printed rather than left in the JSON. These are the folds the
                # walk-forward section is named after, and a reader who can only see the
                # rolling windows cannot tell whether the two agree.
                lines.append("### Walk-forward folds")
                lines.append("")
                lines.append(
                    "Effective windows after purging. `nominal_*` is what the configuration "
                    "asked for, before the purge margin moved each block's end back."
                )
                lines.append("")
                lines.append(_frame_to_markdown(self.rolling_folds))
                lines.append("")
            # `coverage` counts windows whose *label* is observable, which is not the same
            # set as the windows the table below can actually score: `rolling_stability`
            # requires a label *and* a score, and the score exists only on the evaluation
            # block. Printed unlabelled, "n_usable: 14" sitting above twelve `empty` rows
            # reads as a contradiction rather than as two different denominators.
            lines.append("Label coverage per window, before a score is required:")
            lines.append("")
            for key, value in self.stability.coverage.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
            lines.append(_frame_to_markdown(self.stability.table))
            lines.append("")
            table = self.stability.table
            if "reason" in table.columns:
                scored = int((table["reason"] == "ok").sum())
                lines.append(
                    f"Windows carrying both a label and a score: {scored} of {len(table)}. "
                    "The remainder are empty because the model is scored only on the "
                    "evaluation block — a window outside it has labels but no prediction, "
                    "so it cannot contribute a metric. This bounds what the table above can "
                    "say about stability over time."
                )
                lines.append("")
            if self.stability.trends:
                lines.append("| metric | Theil-Sen slope | first | last | direction |")
                lines.append("| --- | --- | --- | --- | --- |")
                for name, summary in self.stability.trends.items():
                    lines.append(
                        f"| {name} | {_format(summary.slope_per_period)} | "
                        f"{_format(summary.first_value)} | {_format(summary.last_value)} | "
                        f"{summary.direction} |"
                    )
                lines.append("")

        if self.drift:
            lines.append("## 7. Drift")
            lines.append("")
            lines.append("| baseline | comparison | CSI | unstable features | max PSI |")
            lines.append("| --- | --- | --- | --- | --- |")
            for report in self.drift:
                payload = report.as_dict()
                lines.append(
                    f"| {payload['baseline']} | {payload['comparison']} | "
                    f"{_format(payload['csi'])} | {payload['n_unstable_features']} | "
                    f"{_format(payload['max_psi'])} |"
                )
            lines.append("")
            # A column of "undefined" reads as a broken metric unless the reason is stated.
            # CSI needs a score in *both* windows, and the pipeline only ever scores the
            # test block — the model is fitted on train, so a train-window score would be
            # in-sample and not comparable to the test-window score it is differenced
            # against. Feature PSI is therefore the drift measure that applies here.
            if all(not np.isfinite(report.csi) for report in self.drift):
                lines.append(
                    "CSI is undefined for every row above, by construction rather than by "
                    "failure: the score column exists only on the test block, and CSI needs "
                    "a score in both windows. A train-window score would be in-sample and so "
                    "not comparable. Read the feature PSI columns instead; they are computed "
                    "on features that exist in both windows."
                )
                lines.append("")

        if not self.historical_stress.empty:
            lines.append("## 8. Historical stress periods")
            lines.append("")
            lines.append(_frame_to_markdown(self.historical_stress))
            lines.append("")
        if not self.synthetic_stress.empty:
            lines.append("### Synthetic stress (robustness probes, not empirical evidence)")
            lines.append("")
            lines.append(_frame_to_markdown(self.synthetic_stress))
            lines.append("")

        if not self.ablation.empty:
            lines.append("## 9. Ablation")
            lines.append("")
            lines.append(_frame_to_markdown(self.ablation))
            lines.append("")

        if self.falsification:
            lines.append("## 10. Falsification conditions")
            lines.append("")
            lines.append("| code | condition | verdict | detail |")
            lines.append("| --- | --- | --- | --- |")
            for check in self.falsification:
                lines.append(
                    f"| {check.code} | {check.condition} | {check.verdict} | {check.detail} |"
                )
            lines.append("")
            lines.append(
                "> `not_evaluated` means the inputs needed to decide were not available. "
                "It does not mean the condition was checked and cleared."
            )
            lines.append("")

        if self.caveats:
            lines.append("## 11. Caveats")
            lines.append("")
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
            lines.append("")

        return "\n".join(lines)

    def write(self, directory: Path | str) -> dict[str, str]:
        """Write ``report.md`` and ``report.json`` into ``directory``.

        Validation runs first: a report whose metadata is incomplete is not written at
        all, so an incomplete report cannot be archived and later mistaken for a
        complete one.

        Returns:
            The paths written, keyed by artefact name.

        Raises:
            ReportValidationError: If :meth:`RunMetadata.validate` fails.
        """
        self.metadata.validate()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        markdown_path = target / "report.md"
        json_path = target / "report.json"
        config_path = target / "config_snapshot.json"
        metadata_path = target / "run_metadata.json"

        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        json_path.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(
                dict(self.metadata.config_snapshot),
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(self.metadata.as_dict(), indent=2, ensure_ascii=False, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        logger.info("wrote report to %s", target)
        return {
            "report_markdown": str(markdown_path),
            "report_json": str(json_path),
            "config_snapshot": str(config_path),
            "run_metadata": str(metadata_path),
        }


def _json_default(value: Any) -> Any:
    """Serialise the types that escape pandas and numpy into JSON.

    Without this, ``json.dumps`` fails on a numpy bool or a Timestamp deep inside the
    metrics dict — after the run, at the point where the result is being saved.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def _frame_to_markdown(frame: pd.DataFrame, *, max_rows: int = 60) -> str:
    """Render a frame as a Markdown table without requiring ``tabulate``.

    Long frames are truncated with an explicit note rather than silently: a table that
    stops mid-way with no marker is read as the whole table.
    """
    if frame.empty:
        return "_(empty)_"
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: _format(float(value)))
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    if len(frame) > max_rows:
        body.append(
            f"| _({len(frame) - max_rows} more rows not shown)_ |"
            + " |" * (len(display.columns) - 1)
        )
    return "\n".join([header, separator, *body])


# --------------------------------------------------------------------------- #
# Card rendering
# --------------------------------------------------------------------------- #

#: Matches ``{{ identifier }}`` but deliberately not ``{{...}}``. The card templates
#: use ``{{...}}`` as a literal marker meaning "an editor should fill this table row
#: in"; those must survive rendering untouched, and because ``...`` is not an
#: identifier the pattern simply does not match them.
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def template_placeholders(template_text: str) -> list[str]:
    """Identifiers referenced by ``{{ identifier }}`` in a template, in order.

    Returns:
        Unique names, sorted, excluding the ``{{...}}`` editorial markers.
    """
    return sorted(set(_PLACEHOLDER_RE.findall(template_text)))


def render_template(template_text: str, values: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Substitute ``{{ identifier }}`` placeholders, defaulting the rest.

    Every identifier the template mentions is replaced, falling back to
    :data:`NOT_MEASURED`. Unfilled placeholders are common and acceptable — an
    unmeasured stress result should say so — but they must be *counted*, because a
    card that reads "not measured" everywhere is a card that has not been completed.

    Args:
        template_text: Raw template.
        values: Known identifiers.

    Returns:
        ``(rendered_text, unfilled_names)``.
    """
    names = template_placeholders(template_text)
    unfilled = [name for name in names if name not in values or values[name] is None]

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        value = values.get(name)
        return NOT_MEASURED if value is None else str(value)

    rendered = _PLACEHOLDER_RE.sub(substitute, template_text)
    if unfilled:
        logger.warning(
            "%d placeholder(s) left as '%s': %s",
            len(unfilled),
            NOT_MEASURED,
            ", ".join(unfilled[:20]),
        )
    return rendered, unfilled


def render_model_card(
    values: Mapping[str, Any],
    *,
    template_path: Path | str | None = None,
) -> tuple[str, list[str]]:
    """Render the model card from ``templates/model_card.md``.

    Returns:
        ``(markdown, unfilled_placeholder_names)``.

    Raises:
        FileNotFoundError: If the template is absent.
    """
    path = Path(template_path) if template_path else Path("templates/model_card.md")
    if not path.is_file():
        raise FileNotFoundError(
            f"model card template not found at {path}; run from the repository root or "
            "pass template_path explicitly"
        )
    defaults = {
        "repo_url": GITHUB_URL,
        "model_repo": HF_MODEL_ID,
        "data_version": DATA_SCHEMA_VERSION,
    }
    merged = {**defaults, **values}
    return render_template(path.read_text(encoding="utf-8"), merged)


def render_dataset_card(
    values: Mapping[str, Any],
    *,
    template_path: Path | str | None = None,
) -> tuple[str, list[str]]:
    """Render the dataset card from ``templates/dataset_card.md``."""
    path = Path(template_path) if template_path else Path("templates/dataset_card.md")
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset card template not found at {path}; run from the repository root or "
            "pass template_path explicitly"
        )
    defaults = {
        "repo_url": GITHUB_URL,
        "dataset_repo": HF_DATASET_ID,
        "data_version": DATA_SCHEMA_VERSION,
    }
    merged = {**defaults, **values}
    return render_template(path.read_text(encoding="utf-8"), merged)


def describe_regime(
    prices: pd.DataFrame,
    window_start: str | date | pd.Timestamp,
    window_end: str | date | pd.Timestamp,
    *,
    date_column: str = "date",
    close_column: str = "close",
    returns_column: str | None = None,
) -> str:
    """Summarise the market regime of an evaluation window in one line.

    The evaluation document requires the test window's regime characteristics
    alongside any headline number, because a single number cannot be interpreted
    without knowing whether the window was calm or violent. This is a deliberately
    small description — realised volatility, worst drawdown, cumulative return — not a
    regime-classification model.

    Returns:
        A one-line description. Returns a ``not measured`` note rather than raising
        when the price frame lacks the needed columns, because a missing regime note
        should degrade the report, not abort it.
    """
    for column in (date_column, close_column):
        if column not in prices.columns:
            return f"{NOT_MEASURED}: price frame has no {column!r} column"
    dates = pd.to_datetime(prices[date_column])
    subset = prices.loc[(dates >= pd.Timestamp(window_start)) & (dates <= pd.Timestamp(window_end))]
    if subset.empty:
        return f"{NOT_MEASURED}: no price rows in the window"

    series = pd.to_numeric(subset[close_column], errors="coerce").dropna()
    if series.size < 3:
        return f"{NOT_MEASURED}: too few price observations"

    returns = (
        pd.to_numeric(subset[returns_column], errors="coerce").dropna()
        if returns_column and returns_column in subset.columns
        else series.pct_change().dropna()
    )
    if returns.empty:
        return f"{NOT_MEASURED}: no usable returns"

    annualised_vol = float(returns.std(ddof=1) * np.sqrt(252))
    equity = (1.0 + returns).cumprod()
    drawdown = float((equity / equity.cummax() - 1.0).min())
    total = float(equity.iloc[-1] - 1.0)
    return (
        f"annualised realised vol {annualised_vol:.1%}, "
        f"max drawdown {drawdown:.1%}, cumulative return {total:+.1%} "
        f"over {subset[date_column].min()} .. {subset[date_column].max()}"
    )


def default_caveats(
    *,
    contains_synthetic_data: bool = True,
    real_data_evaluation: bool = False,
) -> list[str]:
    """The standing limitations that belong in every report this project emits.

    These are not hedges. Each one is a statement that would change how a number
    should be read, and each follows from the design rather than having been
    discovered afterwards.
    """
    caveats = [
        "All metric values are targets-versus-achieved, not claims of attained performance.",
        "Panel rows are not independent; no i.i.d. p-value is reported anywhere in this project.",
        "The Sharpe ratio is computed on overlapping forward windows and is inflated; the "
        "Newey-West t-statistic is the number to quote.",
        "Attention weights are diagnostics, not evidence. Evidence citations are verified by "
        "verbatim substring match against the source documents.",
        "A '<10% rolling-metric fluctuation' gate is not used: at this base rate the "
        "sampling standard deviation of a single-fold AUC already exceeds it, making the "
        "gate unreachable rather than strict.",
    ]
    if contains_synthetic_data:
        caveats.append(
            "The panel contains synthetic rows. Any positive result on synthetic data "
            "demonstrates that the pipeline is wired correctly; it is exactly zero evidence "
            "about real markets, because the text-versus-structured information split is "
            "assumed into the generator."
        )
    if not real_data_evaluation:
        caveats.append(
            "No real-data evaluation has been performed. The EDGAR client and the price and "
            "news adapters have not been validated against live endpoints."
        )
    return caveats


__all__ = [
    "FALSIFICATION_CONDITIONS",
    "NOT_AVAILABLE",
    "NOT_MEASURED",
    "REQUIRED_METADATA_KEYS",
    "EvaluationReport",
    "FalsificationCheck",
    "GateRow",
    "ReportValidationError",
    "RunMetadata",
    "build_run_metadata",
    "default_caveats",
    "describe_regime",
    "falsification_table",
    "gate_table",
    "render_dataset_card",
    "render_model_card",
    "render_template",
    "template_placeholders",
]

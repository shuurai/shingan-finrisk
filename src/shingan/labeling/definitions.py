"""Risk label definitions.

A risk label is a statement about the *future* that only becomes decidable once
the future has happened. This module fixes the three definitions in code, with
their horizons and thresholds, and states plainly what each one does and does not
mean.

The definitions, from ``docs/03-labeling.md``:

===========================  ==========  ==================================================
Label                        Horizon     Positive when
===========================  ==========  ==================================================
``default_risk``             365 days    a credit rating is downgraded by >= 2 notches, or
                                         bankruptcy / default proceedings begin
``fraud_risk``               730 days    a restatement is filed, an enforcement action is
                                         brought, or a non-standard audit opinion is issued
``tail_risk``                30 trading  the running peak-to-trough drawdown falls below
                             days        -30%
===========================  ==========  ==================================================

Three things this module refuses to do, each for a substantive reason:

**No "future return sign" label.** Predicting whether the price goes up or down is
an alpha claim with a signal-to-noise ratio near zero at daily frequency, and it is
not risk. A model trained on it cannot be evaluated with the metrics that matter
here, and it invites the reader to think the tool is a trading system.

**No label without an observability decision.** Every label carries a mask. A row
whose horizon extends past the end of the data has not "not defaulted" — it is
unknown, and treating it as a negative manufactures a false negative. This is the
single most common way a backtest overstates a risk model.

**No horizon drift.** The horizon is part of the definition. Changing 365 to 180
creates a different label, and results from before and after are not comparable, so
the horizon is written into every row rather than read from config at evaluation
time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from shingan.config import LabelConfig
from shingan.data.schema import RiskLabel
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Source-of-record identifiers written into ``source_of_record_<label>``. These
#: name *where the label fact came from*, which matters when the answer is a proxy:
#: a proxy must be visible in the data, not buried in a README.
SOURCE_PROXY_EVENT_TABLE = "proxy_event_table"
SOURCE_PRICE_PANEL = "price_panel"
SOURCE_RATING_ACTION = "rating_action_table"
SOURCE_SEC_RESTATEMENT = "sec_item_4.02_8k"
SOURCE_SEC_ENFORCEMENT = "sec_enforcement_action"
SOURCE_AUDIT_OPINION = "audit_opinion_filing"

#: Which event kinds can justify each label. Used by the label builders to reject a
#: event that is relevant to one label being silently applied to another.
EVENT_KINDS: dict[RiskLabel, tuple[str, ...]] = {
    RiskLabel.DEFAULT_RISK: ("rating_downgrade", "bankruptcy", "payment_default"),
    RiskLabel.FRAUD_RISK: ("restatement", "enforcement_action", "adverse_audit_opinion"),
    RiskLabel.TAIL_RISK: ("drawdown_breach",),
}


@dataclass(slots=True)
class LabelDefinition:
    """Everything needed to decide one label, in one object."""

    label: RiskLabel
    horizon_days: int
    event_kinds: tuple[str, ...]
    description: str
    source_of_record: str
    calendar_horizon_days: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def horizon_label(self) -> str:
        """Human-readable horizon, e.g. ``'365 calendar days'``."""
        if self.label is RiskLabel.TAIL_RISK:
            return f"{self.horizon_days} trading days"
        return f"{self.horizon_days} calendar days"

    def window_end(self, as_of: date) -> date:
        """Last date on which an event would still fall inside this label's window.

        Args:
            as_of: The prediction date.

        Returns:
            ``as_of + horizon`` in calendar days. For ``tail_risk`` this uses the
            trading-day horizon converted to calendar days, which is the right
            resolution for deciding whether a window is observable; the exact
            trading-day arithmetic is done by the drawdown computation itself.
        """
        return as_of + timedelta(days=self.calendar_horizon_days)


def build_label_definitions(config: LabelConfig) -> dict[RiskLabel, LabelDefinition]:
    """Materialise the label definitions from configuration.

    Args:
        config: Label configuration supplying horizons and thresholds.

    Returns:
        Mapping from label to its definition, containing only the configured
        ``targets`` — an out-of-scope label is absent rather than present and
        disabled, so that downstream code cannot accidentally evaluate it.

    Raises:
        ValueError: If ``config.targets`` is empty.
    """
    if not config.targets:
        raise ValueError("labels.targets is empty; at least one label must be configured")

    definitions: dict[RiskLabel, LabelDefinition] = {}
    for raw_label in config.targets:
        label = RiskLabel(raw_label)
        horizon = config.horizon_days(label)
        calendar_horizon = config.calendar_horizon_days(label)

        if label is RiskLabel.DEFAULT_RISK:
            definitions[label] = LabelDefinition(
                label=label,
                horizon_days=horizon,
                event_kinds=EVENT_KINDS[label],
                description=(
                    f"credit rating downgraded by >= {config.default_risk_downgrade_notches} "
                    f"notches, or bankruptcy / default proceedings, within {horizon} days"
                ),
                source_of_record=SOURCE_PROXY_EVENT_TABLE,
                calendar_horizon_days=calendar_horizon,
                parameters={
                    "downgrade_notches": config.default_risk_downgrade_notches,
                    "use_bankruptcy": config.default_risk_use_bankruptcy,
                },
            )
        elif label is RiskLabel.FRAUD_RISK:
            definitions[label] = LabelDefinition(
                label=label,
                horizon_days=horizon,
                event_kinds=EVENT_KINDS[label],
                description=(
                    "financial restatement, SEC enforcement action, or non-standard "
                    f"audit opinion within {horizon} days"
                ),
                source_of_record=SOURCE_PROXY_EVENT_TABLE,
                calendar_horizon_days=calendar_horizon,
                parameters={},
            )
        elif label is RiskLabel.TAIL_RISK:
            definitions[label] = LabelDefinition(
                label=label,
                horizon_days=horizon,
                event_kinds=EVENT_KINDS[label],
                description=(
                    f"peak-to-trough drawdown worse than "
                    f"{config.tail_risk_drawdown_threshold:.0%} over {horizon} trading days"
                ),
                source_of_record=SOURCE_PRICE_PANEL,
                calendar_horizon_days=calendar_horizon,
                parameters={"drawdown_threshold": config.tail_risk_drawdown_threshold},
            )
        else:  # pragma: no cover - RiskLabel is exhaustive
            raise ValueError(f"no definition implemented for {label!r}")

    logger.debug("materialised %d label definitions: %s", len(definitions), list(definitions))
    return definitions


def is_positive(
    definition: LabelDefinition,
    *,
    event_kind: str,
    event_severity: float | None,
    event_date: date | None,
    as_of: date,
) -> bool:
    """Decide whether one event makes one label positive for one observation.

    The event must (a) be of a kind that can justify this label, (b) fall strictly
    after ``as_of``, and (c) for ``default_risk``, meet the notch threshold.

    The requirement that the event be strictly after ``as_of`` is what stops an
    already-public event from being used as a prediction target.

    Args:
        definition: The label being decided.
        event_kind: One of the kinds in :data:`EVENT_KINDS`.
        event_severity: Notch count for a downgrade, unused for other kinds.
        event_date: Date the event became public, or None when there was no event.
        as_of: The prediction date.

    Returns:
        True when this event makes this label positive for this row.

    Raises:
        ValueError: If the event kind cannot justify this label. Silently ignoring a
            mismatched kind would let a restatement count towards ``default_risk``.
    """
    if event_kind not in definition.event_kinds:
        raise ValueError(
            f"event kind {event_kind!r} cannot justify label {definition.label}; "
            f"allowed kinds are {definition.event_kinds}"
        )
    if event_date is None:
        return False
    if event_date <= as_of:
        return False
    if event_date > definition.window_end(as_of):
        return False

    # At this point the event is of an eligible kind and lies inside the window
    # ``(as_of, as_of + horizon]``. What remains is the label-specific severity
    # test. Each branch below returns explicitly, and an unhandled label raises
    # rather than falling out of the bottom.
    #
    # That last point is not stylistic. An earlier version of this function ended
    # with ``return False`` after handling ``default_risk`` and ``tail_risk``, so
    # ``fraud_risk`` was checked for eligibility and then silently rejected:
    # every restatement, enforcement action and adverse audit opinion in the
    # event table produced zero positives, and the label was all-negative on a
    # dataset that expressly contained 800+ of them. A bare trailing ``return
    # False`` turns "I forgot to implement this label" into "this label never
    # happens", which is the one failure mode a label builder must not have.
    if definition.label is RiskLabel.DEFAULT_RISK:
        if event_kind == "rating_downgrade":
            if event_severity is None:
                return False
            required = float(definition.parameters.get("downgrade_notches", 2))
            return event_severity >= required
        if event_kind in {"bankruptcy", "payment_default"}:
            return bool(definition.parameters.get("use_bankruptcy", True))
        raise ValueError(  # pragma: no cover - blocked by the event_kinds check above
            f"{event_kind!r} is listed for default_risk but has no severity rule"
        )

    if definition.label is RiskLabel.FRAUD_RISK:
        # Eligibility is the whole test: a restatement, an enforcement action and
        # an adverse audit opinion are each independently sufficient, and none of
        # them has a magnitude below which it stops counting.
        return True

    if definition.label is RiskLabel.TAIL_RISK:
        if event_kind != "drawdown_breach" or event_severity is None:
            return False
        threshold = float(definition.parameters.get("drawdown_threshold", -0.30))
        return event_severity <= threshold

    raise ValueError(  # pragma: no cover - RiskLabel is exhaustive today
        f"no severity rule implemented for label {definition.label!r}; add one rather "
        f"than letting the label fall through to False"
    )


def is_observable(definition: LabelDefinition, as_of: date, data_end: date) -> bool:
    """Whether ``as_of``'s label window has closed before ``data_end``.

    Args:
        definition: The label being decided.
        as_of: The prediction date.
        data_end: Last date for which data exists.

    Returns:
        True when the whole window lies inside the sample. When False, the row's
        label must be masked, never set to zero.
    """
    return definition.window_end(as_of) <= data_end


def label_summary_table(definitions: dict[RiskLabel, LabelDefinition]) -> pd.DataFrame:
    """A frame describing the configured labels, for the report.

    Returns:
        One row per label with its horizon, threshold parameters and source of record.
    """
    rows: list[dict[str, Any]] = []
    for label, definition in definitions.items():
        rows.append(
            {
                "label": str(label),
                "horizon": definition.horizon_label(),
                "event_kinds": ", ".join(definition.event_kinds),
                "source_of_record": definition.source_of_record,
                "definition": definition.description,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "EVENT_KINDS",
    "SOURCE_AUDIT_OPINION",
    "SOURCE_PRICE_PANEL",
    "SOURCE_PROXY_EVENT_TABLE",
    "SOURCE_RATING_ACTION",
    "SOURCE_SEC_ENFORCEMENT",
    "SOURCE_SEC_RESTATEMENT",
    "LabelDefinition",
    "build_label_definitions",
    "is_observable",
    "is_positive",
    "label_summary_table",
]

"""Risk labelling.

The definitions live in :mod:`shingan.labeling.definitions`, the machinery that
applies them in :mod:`shingan.labeling.builders`. Read the module docstring of
``builders`` before changing anything here: the forward-looking computations are
correct by construction and easy to break by accident.
"""

from shingan.labeling.builders import (
    add_forward_targets,
    apply_risk_labels,
    forward_max_drawdown,
    forward_realized_volatility,
    forward_return,
    label_rates,
    sample_weights_for_label,
)
from shingan.labeling.definitions import (
    EVENT_KINDS,
    LabelDefinition,
    build_label_definitions,
    is_observable,
    is_positive,
    label_summary_table,
)

__all__ = [
    "EVENT_KINDS",
    "LabelDefinition",
    "add_forward_targets",
    "apply_risk_labels",
    "build_label_definitions",
    "forward_max_drawdown",
    "forward_realized_volatility",
    "forward_return",
    "is_observable",
    "is_positive",
    "label_rates",
    "label_summary_table",
    "sample_weights_for_label",
]

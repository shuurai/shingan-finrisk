"""Data acquisition, assembly and the offline synthetic panel.

Layout of responsibilities:

* :mod:`shingan.data.schema` — the vocabulary: risk taxonomy and the panel data
  dictionary. Depends on nothing else in this package.
* :mod:`shingan.data.synthetic` — the deterministic offline generator. Everything
  the demo and the test suite need, with no network.
* :mod:`shingan.data.edgar`, :mod:`shingan.data.news`, :mod:`shingan.data.prices`
  — thin adapters for the real sources. They implement request construction,
  pagination, rate limiting and field mapping but have **not** been validated
  against live endpoints. Treat them as interfaces, not as working data sources.
* :mod:`shingan.data.builder` — the as-of join that turns the above into one
  processed panel with features, labels and masks.

Import graph note
-----------------
This package deliberately does **not** import :mod:`shingan.data.builder`. The
builder depends on :mod:`shingan.leakage`, which depends on
:mod:`shingan.data.schema` — so an eager ``from shingan.data.builder import ...``
here would close a cycle: importing ``shingan.data.schema`` would run this file,
which would import the builder, which would ask for a ``shingan.leakage`` whose
own import of ``shingan.data.schema`` is still in progress.

The cycle is broken by keeping this module free of the builder. Import it by its
full path instead::

    from shingan.data.builder import build_panel

Everything re-exported below depends only on modules that are already complete by
the time this file finishes executing.
"""

from shingan.data.schema import (
    PANEL_COLUMNS,
    RiskAssessment,
    RiskLabel,
    SampleRecord,
    Severity,
    read_jsonl,
    write_jsonl,
)
from shingan.data.synthetic import SyntheticDataset, generate_synthetic_dataset

__all__ = [
    "PANEL_COLUMNS",
    "RiskAssessment",
    "RiskLabel",
    "SampleRecord",
    "Severity",
    "SyntheticDataset",
    "generate_synthetic_dataset",
    "read_jsonl",
    "write_jsonl",
]

"""Feature engineering, split by data modality.

Three modules, deliberately separate because they fail differently and are
validated by different tests:

* :mod:`shingan.features.technical` — price and volume derived. Validated by a
  look-ahead test that recomputes on truncated history.
* :mod:`shingan.features.ratios` — financial statement derived. Validated by
  hand-computed ratios on a fixture filing.
* :mod:`shingan.features.text` — count-based text statistics. Validated against
  fixed strings with known term counts.

Everything here produces the ``ratios``, ``technical`` and ``text_counts`` column
groups defined in :mod:`shingan.data.schema`.
"""

from shingan.features.ratios import CANONICAL_FUNDAMENTALS, compute_ratios
from shingan.features.technical import compute_technical_features
from shingan.features.text import build_text_features, document_features, lexicon_summary

__all__ = [
    "CANONICAL_FUNDAMENTALS",
    "build_text_features",
    "compute_ratios",
    "compute_technical_features",
    "document_features",
    "lexicon_summary",
]

"""Shingan — evidence-grounded financial risk modelling.

This package is deliberately cheap to import: the top-level namespace exposes
only the version, and every heavy dependency (torch, transformers, peft, trl)
is imported lazily inside the functions that need it. That keeps
``import shingan`` fast in tests and lets the CPU-only install run the whole
offline pipeline without a GPU.

Nothing in this package is investment advice. It is research software for
predicting *risk* — the probability of adverse events such as a credit rating
downgrade, a financial restatement or an extreme drawdown — and it is not a
trading system.
"""

from shingan.__about__ import __version__

__all__ = ["__version__"]

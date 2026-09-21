"""Scalar coercion helpers for reading values out of DataFrame rows.

Why this module exists
----------------------
``DataFrame.itertuples`` and ``DataFrame.at`` are typed by pandas-stubs as
returning a union wide enough to include ``bytes``, ``timedelta`` and complex
numbers — a union that neither :class:`pandas.Timestamp` nor :class:`float`
accepts. The runtime values are always the column's real dtype, so the code is
correct; only the static type is unusable.

The coercion has to happen somewhere. The three options are:

1. ``# type: ignore[arg-type]`` at each call site — the ignore is invisible in
   review and cannot carry a reason.
2. Re-deriving the dtype at each call site — a second copy of knowledge that is
   already expressed by the column, free to drift.
3. One documented helper per coercion — this module.

Option 3 is taken. The ``Any`` in the signatures is deliberate and honest: the
caller has already established the column's dtype, and these functions exist only
to stop the type checker from making the caller re-assert it.

These are *not* general-purpose parsers. They do not validate, they do not handle
missing values specially, and they should not be used for anything other than
narrowing a value that came from a column whose dtype is already known.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["row_float", "row_timestamp"]


def row_timestamp(value: Any) -> pd.Timestamp:
    """Narrow a value read from a datetime64 column into a :class:`pandas.Timestamp`.

    Args:
        value: A value from a ``datetime64`` column, typically via ``itertuples``
            or ``.at``.

    Returns:
        The same instant as a Timestamp. NaT passes through as NaT.
    """
    return pd.Timestamp(value)


def row_float(value: Any) -> float:
    """Narrow a value read from a numeric column into a float.

    Args:
        value: A value from a float column, typically via ``itertuples``.

    Returns:
        The value as a Python float.
    """
    return float(value)

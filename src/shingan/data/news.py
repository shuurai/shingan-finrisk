"""News adapter and de-duplication.

**Status: thin adapter, not validated against a real FNSPID snapshot.** The expected
schema below comes from the dataset's published description, not from a file this
project has read.

Two things in here are more than plumbing:

**De-duplication, because syndication inflates counts.** A single piece of news is
republished by dozens of outlets within minutes. Counting rows rather than distinct
events turns one event into twenty, which turns the news-volume features into a
measure of wire-service behaviour rather than of anything about the company.

**Sentiment is a feature, never a label.** The score shipped with a news dataset comes
from an unknown model, is not calibrated against anything, and must not be used as a
target. The documentation says so; this module repeats it because it is the kind of
shortcut that is easy to take at 2am.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from shingan.frame_utils import row_timestamp
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_NEWS_COLUMNS: tuple[str, ...] = ("ticker", "published", "title")

#: Common column spellings across news datasets. Mapped onto the canonical names so
#: that a vendor renaming ``Date`` to ``publish_date`` does not require a code change.
_COLUMN_ALIASES: dict[str, str] = {
    "date": "published",
    "datetime": "published",
    "publish_date": "published",
    "article_date": "published",
    "headline": "title",
    "article_title": "title",
    "stock_symbol": "ticker",
    "symbol": "ticker",
    "article": "body",
    "content": "body",
    "text": "body",
    "sentiment_score": "sentiment",
    "score": "sentiment",
}

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^a-z0-9 ]+")


def canonicalise_news(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases and add the optional canonical ones.

    Args:
        frame: Raw news frame.

    Returns:
        A frame with at least ``ticker``, ``published`` and ``title``, plus
        ``body``, ``sentiment`` and ``source`` where available (NaN otherwise).

    Raises:
        KeyError: If a required column cannot be resolved from the aliases.
        ValueError: If the frame has no rows.
    """
    if frame.empty:
        raise ValueError("news frame is empty")

    renamed = {
        column: _COLUMN_ALIASES[column.lower()]
        for column in frame.columns
        if column.lower() in _COLUMN_ALIASES and column.lower() != _COLUMN_ALIASES[column.lower()]
    }
    result = frame.rename(columns=renamed).copy()

    missing = [column for column in REQUIRED_NEWS_COLUMNS if column not in result.columns]
    if missing:
        raise KeyError(
            f"news frame is missing required columns: {missing}. Present: "
            f"{sorted(result.columns)}. Add a mapping in _COLUMN_ALIASES if this source "
            "uses different names."
        )

    for column in ("body", "sentiment", "source"):
        if column not in result.columns:
            result[column] = np.nan
    result["published"] = pd.to_datetime(result["published"], errors="coerce", utc=True)
    result = result.loc[result["published"].notna()].copy()
    result["published"] = result["published"].dt.tz_localize(None).dt.normalize()
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    result["title"] = result["title"].astype(str)
    return result.reset_index(drop=True)


def normalise_text(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace, for exact comparison."""
    lowered = _WHITESPACE_RE.sub(" ", str(text).lower())
    return _PUNCTUATION_RE.sub(" ", lowered).strip()


def shingle_set(text: str, width: int = 8) -> set[str]:
    """Character shingles of a string, used for approximate duplicate detection."""
    normalised = normalise_text(text).replace(" ", "")
    if len(normalised) <= width:
        return {normalised} if normalised else set()
    return {normalised[index : index + width] for index in range(len(normalised) - width + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard similarity of two shingle sets; 0.0 when either is empty."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


def deduplicate_news(
    frame: pd.DataFrame,
    *,
    similarity: float = 0.9,
    prefix_chars: int = 256,
    date_tolerance_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse syndicated copies, keeping the earliest publication of each cluster.

    Two passes, matching the documented policy:

    1. Exact match on the normalised title.
    2. Approximate match, by comparing the first ``prefix_chars`` characters of the
       body with a Jaccard shingle similarity, restricted to articles for the same
       ticker published within ``date_tolerance_days`` of each other. The restriction
       is what makes this tractable: a global pairwise comparison over a news corpus
       is quadratic and unnecessary, since a reprint always follows its original
       closely in time.

    Args:
        frame: Canonicalised news frame.
        similarity: Jaccard threshold above which two articles are the same event.
        prefix_chars: Characters of the body compared in pass two.
        date_tolerance_days: Window within which approximate duplicates may fall.

    Returns:
        A ``(deduplicated, log)`` pair. The log has one row per dropped article with
        the reason and the identifier of the article that was kept, so a reviewer can
        always answer "why is this article not in the model".
    """
    if frame.empty:
        return frame, pd.DataFrame(columns=["ticker", "published", "title", "reason", "kept_title"])

    working = frame.sort_values(["ticker", "published"]).reset_index(drop=True)
    working["_normalised_title"] = working["title"].map(normalise_text)

    keep = np.ones(len(working), dtype=bool)
    log_rows: list[dict[str, object]] = []
    cluster_ids = np.full(len(working), "", dtype=object)

    for ticker, positions in working.groupby("ticker", sort=False, observed=True).indices.items():
        rows = np.asarray(positions)
        seen_titles: dict[str, int] = {}
        cluster_shingles: list[tuple[int, set[str], pd.Timestamp]] = []

        for index in rows:
            title_key = str(working.at[index, "_normalised_title"])
            if title_key in seen_titles:
                keep[index] = False
                kept_at = seen_titles[title_key]
                cluster_ids[index] = cluster_ids[kept_at]
                log_rows.append(
                    {
                        "ticker": ticker,
                        "published": working.at[index, "published"],
                        "title": working.at[index, "title"],
                        "reason": "exact_title_match",
                        "kept_title": working.at[kept_at, "title"],
                    }
                )
                continue

            # The body may be missing (a headline-only source ships no article
            # text), and ``pd.NA or ""`` raises -- its truthiness is ambiguous by
            # design. Test for missingness explicitly instead.
            body = working.at[index, "body"]
            prefix = "" if pd.isna(body) else str(body)
            prefix = prefix[:prefix_chars]
            current = shingle_set(prefix) if prefix else set()
            # Read the timestamp once and coerce it. `DataFrame.at` is typed as a
            # wide union that does not include Timestamp, and subtracting a union
            # from a Timestamp is not a legal expression as far as the type checker
            # is concerned, even though the column is datetime64 at runtime.
            published_at = row_timestamp(working.at[index, "published"])
            duplicate_of: int | None = None
            if current:
                for candidate, candidate_shingles, candidate_date in cluster_shingles:
                    if abs((published_at - candidate_date).days) > date_tolerance_days:
                        continue
                    if jaccard(current, candidate_shingles) >= similarity:
                        duplicate_of = candidate
                        break

            if duplicate_of is not None:
                keep[index] = False
                cluster_ids[index] = cluster_ids[duplicate_of]
                log_rows.append(
                    {
                        "ticker": ticker,
                        "published": published_at,
                        "title": working.at[index, "title"],
                        "reason": "near_duplicate_body",
                        "kept_title": working.at[duplicate_of, "title"],
                    }
                )
                continue

            seen_titles[title_key] = index
            cluster_ids[index] = f"{ticker}-{index}"
            if current:
                cluster_shingles.append((index, current, published_at))

    working["cluster_id"] = cluster_ids
    dropped = int((~keep).sum())
    if dropped:
        logger.info(
            "de-duplicated %d of %d news rows (%.1f%%); full log written alongside",
            dropped,
            len(working),
            dropped / len(working) * 100,
        )
    log = pd.DataFrame.from_records(log_rows) if log_rows else pd.DataFrame(
        columns=["ticker", "published", "title", "reason", "kept_title"]
    )
    return working.loc[keep].drop(columns=["_normalised_title"]).reset_index(drop=True), log


def align_news_to_trading_day(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    close_hour_utc: int = 21,
) -> pd.DataFrame:
    """Map each article onto the trading day on which it becomes actionable.

    An article published after the US close (21:00 UTC covers both 16:00 ET in winter
    and 17:00 ET in summer, to the hour) cannot inform that day's decision, so it is
    attributed to the next trading day. The same applies to anything published on a
    non-trading day.

    Args:
        frame: Canonicalised news frame.
        calendar: The trading calendar to snap onto, ascending.
        close_hour_utc: Hour, UTC, after which publication rolls to the next session.

    Returns:
        The frame with ``published`` replaced by the attributable trading day, and a
        ``published_at`` column retaining the original timestamp.

    Raises:
        ValueError: If the calendar is empty.
    """
    if len(calendar) == 0:
        raise ValueError("trading calendar is empty")
    if frame.empty:
        return frame.assign(published_at=pd.Series(dtype="datetime64[ns]"))

    result = frame.copy()
    result["published_at"] = pd.to_datetime(result["published"])
    timed = result["published_at"] + pd.Timedelta(hours=close_hour_utc)
    days = timed.dt.normalize()
    positions = calendar.searchsorted(days.to_numpy(), side="left")
    positions = np.clip(positions, 0, len(calendar) - 1)
    result["published"] = calendar.take(positions)
    return result


def load_news(
    path: Path | str,
    *,
    calendar: pd.DatetimeIndex | None = None,
    similarity: float = 0.9,
    prefix_chars: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, canonicalise, de-duplicate and optionally align a news snapshot.

    Args:
        path: CSV or parquet file. The format is chosen by suffix.
        calendar: Trading calendar to align onto. Skipped when None.
        similarity: Jaccard threshold for near-duplicate detection.
        prefix_chars: Body characters compared for near-duplicates.

    Returns:
        A ``(news, dedup_log)`` pair.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the suffix is not supported.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"news snapshot not found: {source}")

    if source.suffix.lower() == ".parquet":
        raw = pd.read_parquet(source)
    elif source.suffix.lower() in {".csv", ".tsv"}:
        raw = pd.read_csv(source, sep="\t" if source.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError(f"unsupported news format {source.suffix!r}; expected .csv, .tsv or .parquet")

    logger.warning(
        "news adapter is unvalidated: column mapping and the sentiment column are "
        "assumptions about %s, not verified facts",
        source.name,
    )
    canonical = canonicalise_news(raw)
    deduplicated, log = deduplicate_news(
        canonical, similarity=similarity, prefix_chars=prefix_chars
    )
    if calendar is not None:
        deduplicated = align_news_to_trading_day(deduplicated, calendar)
    return deduplicated, log


__all__ = [
    "REQUIRED_NEWS_COLUMNS",
    "align_news_to_trading_day",
    "canonicalise_news",
    "deduplicate_news",
    "jaccard",
    "load_news",
    "normalise_text",
    "shingle_set",
]

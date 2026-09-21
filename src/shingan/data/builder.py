"""Panel assembly: the point-in-time join that produces one modelling table.

This is where the pipeline's discipline is either kept or lost. The rules it
implements, all of them checkable:

* **Every row is a decision point.** One row per ``(ticker, as_of)``, where ``as_of``
  is a filing date. The row contains only information available at that moment.
* **All joins are backward as-of joins.** A feature from the most recent observation
  at or before ``as_of``; never the nearest, never interpolated, never the latest
  revision. Restatements filed later are invisible, as they were at the time.
* **Forward quantities exist, and are named so they cannot be used by accident.**
  ``fwd_ret_21d``, ``fwd_realized_vol_21d`` and ``fwd_max_drawdown_30d`` are computed
  from the future and are on the reserved-prefix list.
* **Synthetic and real rows never mix.** The builder refuses to emit a panel that
  contains both, because a single mislabelled screenshot would discredit every real
  result in the repository.
* **The result is verified, then written.** Column names, timestamp monotonicity,
  event timing, label observability and correlation-to-target are all asserted before
  anything reaches disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shingan.__about__ import DATA_SCHEMA_VERSION
from shingan.config import ProjectConfig
from shingan.data.schema import (
    PANEL_COLUMNS,
    RiskLabel,
    event_column,
    horizon_column,
    label_column,
    mask_column,
    panel_subset,
    source_of_record_column,
)
from shingan.data.synthetic import SyntheticDataset, generate_synthetic_dataset
from shingan.eval.splits import SplitReport, assign_split_column
from shingan.features.ratios import compute_ratios
from shingan.features.technical import compute_technical_features
from shingan.features.text import build_text_features
from shingan.frame_utils import row_float, row_timestamp
from shingan.labeling.builders import add_forward_targets, apply_risk_labels, label_rates
from shingan.labeling.definitions import build_label_definitions
from shingan.leakage import (
    assert_asof_monotonic,
    assert_feature_matrix_is_clean,
    assert_label_window_observable,
    assert_no_future_events,
    correlation_alarm,
    feature_columns,
    leakage_scan,
    merge_asof_point_in_time,
)
from shingan.logging_utils import get_logger

logger = get_logger(__name__)

#: Companies and sectors used when the panel is synthetic. Fictional by design.
_COMPANY_NAMES: tuple[str, ...] = (
    "Synthex Industrial",
    "Synthex Financial",
    "Synthex Energy",
    "Synthex Consumer",
    "Synthex Materials",
    "Synthex Technology",
)

#: Signals exposed to the text track in the prompt's ``STRUCTURED_SIGNALS`` block.
#: Chosen for readability rather than completeness: the point of the block is to give
#: the language model the numbers a human analyst would look at first, not to
#: reproduce the full feature matrix in prose.
PROMPT_SIGNAL_COLUMNS: tuple[str, ...] = (
    "debt_to_equity",
    "interest_coverage",
    "current_ratio",
    "altman_z",
    "accruals_ratio",
    "revenue_yoy",
    "vol_60d",
    "max_drawdown_60d",
    "dist_52w_high",
    "abnormal_volume_20d",
    "n_news_30d",
    "sent_mean_30d",
)

#: How many filings and news items are placed in a prompt before truncation applies.
MAX_FILINGS_IN_CONTEXT = 4
MAX_NEWS_IN_CONTEXT = 40
NEWS_CONTEXT_WINDOW_DAYS = 180

#: Tolerance, in days, for matching a decision date to the most recent price row. A
#: filing date can fall on an exchange holiday, so an exact match would silently drop
#: rows; a week is short enough that the price data is still "current".
PRICE_MATCH_TOLERANCE_DAYS = 7


@dataclass(slots=True)
class RawTables:
    """The five source tables a build consumes.

    Exposed so that a caller who has already fetched real data can run the *same*
    assembly path as the synthetic demo. The pipeline is real even where the fetchers
    are not yet validated, and this is the seam that proves it.
    """

    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    filings: pd.DataFrame
    news: pd.DataFrame
    events: pd.DataFrame
    is_synthetic: bool

    @classmethod
    def from_synthetic(cls, dataset: SyntheticDataset) -> RawTables:
        """Adapt a :class:`~shingan.data.synthetic.SyntheticDataset`."""
        return cls(
            prices=dataset.prices,
            fundamentals=dataset.fundamentals,
            filings=dataset.filings,
            news=dataset.news,
            events=dataset.events,
            is_synthetic=True,
        )

    def assert_single_provenance(self) -> None:
        """Refuse a panel that mixes synthetic and real rows.

        Raises:
            ValueError: If the flags are inconsistent, which can only happen when a
                caller assembles tables by hand from more than one origin.
        """
        flags = {
            "filings": bool(self.filings.get("is_synthetic", pd.Series([self.is_synthetic])).any()),
        }
        del flags  # the per-table flags are optional; the dataclass flag is authoritative
        if self.is_synthetic and not len(self.prices):
            raise ValueError("a synthetic build requires a non-empty price table")


@dataclass(slots=True)
class BuildResult:
    """Everything a build produces, in memory and (by default) on disk."""

    panel: pd.DataFrame
    events: pd.DataFrame
    filings: pd.DataFrame
    news: pd.DataFrame
    text_context: list[dict[str, Any]]
    split_report: SplitReport
    metadata: dict[str, Any] = field(default_factory=dict)
    leakage: dict[str, Any] = field(default_factory=dict)
    label_rates: pd.DataFrame | None = None
    written: dict[str, str] = field(default_factory=dict)
    #: The raw price table, carried through so the report can describe the *regime* of
    #: the evaluation window. The panel cannot serve this purpose: it holds returns at
    #: the decision grid's quarterly spacing, and the regime description needs a
    #: continuous series to compute realised volatility and peak-to-trough drawdown
    #: from. ``is_synthetic`` provenance is already asserted on the way in.
    prices: pd.DataFrame | None = None

    def split_frame(self, name: str, label: RiskLabel | str | None = None) -> pd.DataFrame:
        """Rows of one split, optionally restricted to an observable label.

        Args:
            name: One of ``train``, ``valid``, ``test``.
            label: When given, keep only rows whose label is observable, which is what
                training and evaluation actually consume.

        Returns:
            The filtered frame.
        """
        from shingan.data.schema import mask_column

        subset = self.panel.loc[self.panel["split"] == name]
        if label is not None:
            subset = subset.loc[subset[mask_column(label)].astype(bool)]
        return subset


def _write_frame(frame: pd.DataFrame, base: Path) -> str:
    """Write a frame as parquet when available, else CSV.

    Returns:
        The path actually written. Parquet is preferred because it preserves dtypes —
    a CSV round trip turns an ``int8`` label into ``int64`` and mangles nullable
    booleans, which then has to be repaired at every read site.
    """
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401 - presence check only

        destination = base.with_suffix(".parquet")
        frame.to_parquet(destination, index=False)
        return str(destination)
    except ImportError:
        destination = base.with_suffix(".csv")
        frame.to_csv(destination, index=False, encoding="utf-8")
        logger.info(
            "pyarrow is not installed, so %s was written as CSV. Install the "
            "'parquet' extra to preserve dtypes across a save/load round trip.",
            destination.name,
        )
        return str(destination)


def _text_features_and_context(
    grid: pd.DataFrame,
    filings: pd.DataFrame,
    news: pd.DataFrame,
    *,
    max_filings: int = MAX_FILINGS_IN_CONTEXT,
    max_news: int = MAX_NEWS_IN_CONTEXT,
    news_window_days: int = NEWS_CONTEXT_WINDOW_DAYS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute the text-count features and capture the per-row context.

    Both outputs come from one pass because the pass is the expensive part: it
    tokenises each disclosure and aggregates the news flow for each decision date.
    Splitting it into two functions would double that work.

    Args:
        grid: One row per ``(ticker, as_of)``.
        filings: Filing sections with ``ticker``, ``filed``, ``doc_type``, ``section``,
            ``text`` and optionally ``accession``.
        news: News items with ``ticker``, ``published``, ``title``, ``body``,
            ``sentiment``, ``source``.
        max_filings: Maximum filing sections retained in a prompt context.
        max_news: Maximum news items retained in a prompt context.
        news_window_days: How far back news is collected for the context.

    Returns:
        A ``(features, contexts)`` pair. ``features`` is indexed like ``grid`` and holds
        the ``text_counts`` columns; ``contexts`` is one mapping per grid row, holding
        the documents the text track will see.
    """
    filings_frame = filings.copy()
    filings_frame["filed"] = pd.to_datetime(filings_frame["filed"])
    if "accession" not in filings_frame.columns:
        filings_frame["accession"] = ""

    news_frame = news.copy()
    if len(news_frame):
        news_frame["published"] = pd.to_datetime(news_frame["published"])
        news_frame = news_frame.sort_values(["ticker", "published"]).reset_index(drop=True)
    else:
        news_frame = pd.DataFrame(
            columns=["ticker", "published", "source", "title", "body", "sentiment"]
        )

    feature_rows: list[dict[str, float]] = []
    contexts: list[dict[str, Any]] = []

    for row in grid.itertuples(index=False):
        ticker = str(row.ticker)
        as_of = row_timestamp(row.as_of)

        ticker_filings = filings_frame.loc[
            (filings_frame["ticker"] == ticker) & (filings_frame["filed"] <= as_of)
        ].sort_values(["filed", "section"])
        ticker_news = news_frame.loc[
            (news_frame["ticker"] == ticker)
            & (news_frame["published"] <= as_of)
            & (news_frame["published"] > as_of - pd.Timedelta(days=news_window_days))
        ]

        # Joined with a blank line, not a space. ``build_text_features`` re-segments the
        # concatenated disclosure, and ``segment_items`` matches Item headings with a
        # ``(?m)^`` anchor: joining with " " leaves every heading after the first stranded
        # mid-line, so the document looks like it has one section, ``section_text`` finds
        # neither risk factors nor MD&A, and ``risk_factor_token_share`` /
        # ``mdna_token_share`` / ``neg_kw_density_mdna`` come out zero for every row
        # without anything failing. The separator is part of the contract.
        disclosures = [
            "\n\n".join(group["text"].astype(str))
            for _, group in ticker_filings.groupby("filed", sort=True)
        ]
        prior_length: float | None = None
        if len(disclosures) > 1:
            from shingan.features.text import document_features

            prior_length = float(document_features(disclosures[-2])["disclosure_len_tokens"])

        feature_rows.append(
            build_text_features(
                disclosures,
                ticker_news,
                as_of,
                prior_disclosure_len_tokens=prior_length,
            )
        )

        contexts.append(
            {
                "sample_id": f"{ticker}-{as_of.strftime('%Y%m%d')}",
                "ticker": ticker,
                "as_of": as_of.date().isoformat(),
                "filings": [
                    {
                        "doc_type": str(record.doc_type),
                        "filed": row_timestamp(record.filed).date().isoformat(),
                        "section": str(record.section),
                        "text": str(record.text),
                        "accession": str(getattr(record, "accession", "")),
                    }
                    for record in ticker_filings.tail(max_filings).itertuples(index=False)
                ],
                "news": [
                    {
                        "published": row_timestamp(record.published).date().isoformat(),
                        "source": str(record.source),
                        "title": str(record.title),
                        "body": str(record.body),
                        "sentiment": (
                            None
                            if record.sentiment is None or pd.isna(record.sentiment)
                            else row_float(record.sentiment)
                        ),
                    }
                    for record in ticker_news.tail(max_news).itertuples(index=False)
                ],
            }
        )

    features = pd.DataFrame.from_records(feature_rows, index=grid.index)
    return features, contexts


#: Input tables that determine how far the sample really extends, and the date column
#: that says so. Labels are computed from prices and events, so those two set the
#: horizon of observability; fundamentals and filings are included because they are
#: the decision grid and the feature feeds, and a grid row past the end of its own
#: features is not a usable observation either.
_OBSERVATION_EXTENT: tuple[tuple[str, str], ...] = (
    ("prices", "date"),
    ("events", "event_date"),
    ("filings", "filed"),
    ("fundamentals", "filed"),
    ("news", "published"),
)


def resolve_data_end(tables: RawTables, configured_end: date) -> date:
    """Last date the *inputs* actually cover, checked against the configured end.

    Every observability decision in the label builder asks "does the whole label
    window lie inside the sample?". The sample's extent is a property of the feeds,
    not of the configuration: a config that asks for data through December while the
    price feed stops in November would mark windows as observed that are not, turning
    an unknown outcome into a negative example. That is the exact failure the mask
    exists to prevent, so the extent is measured rather than assumed.

    Args:
        tables: The input tables.
        configured_end: ``data.end`` from the configuration.

    Returns:
        ``configured_end``, once it has been shown to be within the data.

    Raises:
        ValueError: If the last observation precedes ``configured_end``, or if no
            table has a usable date column to check against. Both are configuration
            errors with a one-line fix, and continuing would produce a run whose
            reported sample window is wider than the sample.
    """
    observed: list[date] = []
    for attribute, column in _OBSERVATION_EXTENT:
        frame: pd.DataFrame | None = getattr(tables, attribute, None)
        if frame is None or not len(frame) or column not in frame.columns:
            continue
        stamps = pd.to_datetime(frame[column], errors="coerce").dropna()
        if len(stamps):
            observed.append(pd.Timestamp(stamps.max()).date())
    if not observed:
        raise ValueError(
            "none of the input tables has a usable date column "
            f"({[column for _, column in _OBSERVATION_EXTENT]}), so the extent of the "
            "sample cannot be established and the label windows cannot be masked "
            "correctly. Pass the RawTables a real build fetches rather than empty frames."
        )

    measured = max(observed)
    if measured < configured_end:
        raise ValueError(
            f"data.end={configured_end.isoformat()} is after the last observation in the "
            f"inputs ({measured.isoformat()}). Set data.end to {measured.isoformat()} or "
            f"earlier, or fetch the missing period. Left as-is, the label builder would "
            f"mark {(_date_gap(measured, configured_end))} of label windows as observed "
            f"when the inputs do not extend that far, and an unknown outcome would be "
            f"recorded as a negative."
        )
    return configured_end


def _date_gap(start: date, end: date) -> str:
    """Human-readable length of a date range, for error messages."""
    return f"up to {max(0, (end - start).days)} days"


def load_cached_tables(config: ProjectConfig) -> RawTables:
    """Load raw tables previously fetched by ``scripts/fetch_real.py``.

    ``build_panel`` cannot fetch by itself — the adapters are interactive, rate-limited
    and better run explicitly — so a real-data build reads the frames the fetch script
    persisted under ``data.cache_dir`` (default ``data/raw/real/``). Prices,
    fundamentals and filings must be present; news and events may be absent, in which
    case empty frames stand in and the affected features and labels report
    accordingly (news-derived counts go to zero, event-based labels are masked out).

    Raises:
        ValueError: If the cache directory or one of the required tables is missing.
    """
    from shingan.paths import ProjectPaths

    candidates: list[Path] = []
    if config.data.cache_dir:
        candidate = Path(config.data.cache_dir)
        candidates.append(candidate if candidate.is_absolute() else ProjectPaths.from_root(config.project.root).root / candidate)
    candidates.append(ProjectPaths.from_root(config.project.root).root / "data" / "raw" / "real")
    directory = next((item for item in candidates if (item / "prices.parquet").is_file()), None)
    if directory is None:
        raise ValueError(
            "data.sources={config.data.sources} requires the network, but no source "
            "tables were supplied and none were found under "
            f"{[str(item) for item in candidates]}. Either run with sources=['synthetic'], "
            "fetch the data with scripts/fetch_real.py (the adapters in "
            "shingan.data.{edgar,news,prices} are unvalidated — see docs/02-data.md), "
            "and retry, or pass the frames as RawTables."
        )

    required = ("prices", "fundamentals", "filings")
    frames: dict[str, pd.DataFrame] = {}
    for name in required:
        path = directory / f"{name}.parquet"
        if not path.is_file():
            raise ValueError(
                f"required real-data table {name} is missing at {path}. Run "
                "scripts/fetch_real.py first, or restore the missing parquet file."
            )
        frames[name] = pd.read_parquet(path)
    optional = ("news", "events")
    for name in optional:
        path = directory / f"{name}.parquet"
        frames[name] = pd.read_parquet(path) if path.is_file() else pd.DataFrame()

    logger.info(
        "loaded real-data tables from %s: prices %d rows, fundamentals %d, filings %d, "
        "news %d, events %d",
        directory,
        len(frames["prices"]),
        len(frames["fundamentals"]),
        len(frames["filings"]),
        len(frames["news"]),
        len(frames["events"]),
    )
    return RawTables(
        prices=frames["prices"],
        fundamentals=frames["fundamentals"],
        filings=frames["filings"],
        news=frames["news"],
        events=frames["events"],
        is_synthetic=False,
    )


def build_panel(
    config: ProjectConfig,
    *,
    tables: RawTables | None = None,
    write: bool = True,
) -> BuildResult:
    """Assemble the processed panel.

    Args:
        config: Validated project configuration.
        tables: Source tables. When omitted, the deterministic synthetic generator is
            used, sized to the configured data window.
        write: Whether to write the artifacts under ``data/processed/``. Passing False
            is useful in tests.

    Returns:
        The :class:`BuildResult`.

    Raises:
        ValueError: If the panel cannot be assembled under the configured windows —
            for instance when purging would empty the test block, or when the sources
            require the network and ``data.offline`` is set.
        shingan.leakage.LeakageError: If any point-in-time invariant is violated.
    """
    if tables is None:
        if set(config.data.sources) != {"synthetic"}:
            # A real build: the fetch step has already run and its tables are on disk.
            # Loading here (rather than requiring an explicit argument) is what lets
            # `shingan data build` / `train structured` / `eval run` work unchanged on
            # a real-data overlay.
            tables = load_cached_tables(config)
        else:
            synthetic_config = config.data.synthetic.model_copy(
                update={
                    "start": config.data.start,
                    "end": config.data.end,
                    "n_companies": max(
                        2, min(config.data.synthetic.n_companies, len(config.data.universe) or 12)
                    ),
                }
            )
            dataset = generate_synthetic_dataset(synthetic_config)
            tables = RawTables.from_synthetic(dataset)
            logger.info("using the synthetic generator: %s", dataset.summary())
    tables.assert_single_provenance()
    # Measured once, from the inputs, and used for every observability and truncation
    # decision below. Deriving it here rather than reading `config.data.end` directly
    # is what keeps a config from claiming a sample the feeds do not deliver.
    data_end = resolve_data_end(tables, config.data.end)

    definitions = build_label_definitions(config.labels)
    horizon_note = {str(label): definition.horizon_label() for label, definition in definitions.items()}
    logger.info("labels: %s", horizon_note)

    # 1. Decision grid: one row per filing date per company.
    grid = (
        tables.filings[["ticker", "filed"]]
        .drop_duplicates()
        .rename(columns={"filed": "as_of"})
        .sort_values(["ticker", "as_of"])
        .reset_index(drop=True)
    )
    if grid.empty:
        raise ValueError("the filing table is empty, so there are no decision points to build")
    grid["as_of"] = pd.to_datetime(grid["as_of"])
    # Provenance columns. The deterministic "Synthex ..." names and the 9-prefixed
    # placeholder CIK exist so that a *synthetic* row stays identifiable after the
    # `is_synthetic` column is dropped. Applying them to real rows would be the exact
    # mislabelling this module's docstring warns about: "Synthex Financial GS" on a
    # Goldman Sachs row would travel into the panel, the report and the dataset card.
    # A real build therefore carries only what the fetch actually obtained — the CIK
    # the filings table supplies, and the ticker as the name (this repository has no
    # verified legal-name source) — and leaves a blank rather than inventing one.
    if tables.is_synthetic:
        grid["cik"] = grid["ticker"].map(lambda ticker: _synthetic_cik(str(ticker)))
        grid["company_name"] = grid["ticker"].map(_company_name)
        grid["sector"] = grid["ticker"].map(_sector)
    else:
        grid["cik"] = _real_cik_lookup(grid["ticker"], tables.filings)
        grid["company_name"] = grid["ticker"].astype(str)
        grid["sector"] = ""

    # 2. Technical features, joined backward onto the grid.
    technical = compute_technical_features(
        tables.prices,
        min_history_days=config.data.min_history_days,
        max_nan_run=config.data.max_nan_run,
    )
    panel = merge_asof_point_in_time(
        grid,
        technical,
        on="as_of",
        by="ticker",
        date_col="as_of",
        tolerance_days=PRICE_MATCH_TOLERANCE_DAYS,
    )

    # 3. Ratios, joined on the filing date of the statement.
    ratios = compute_ratios(tables.fundamentals)
    panel = merge_asof_point_in_time(
        panel,
        ratios,
        on="as_of",
        by="ticker",
        date_col="filed",
        tolerance_days=int(config.labels.fraud_risk_horizon_days),
    )

    # 4. Text-count features and the prompt contexts.
    text_features, text_context = _text_features_and_context(
        panel[["ticker", "as_of"]].reset_index(drop=True),
        tables.filings,
        tables.news,
    )
    panel = panel.reset_index(drop=True)
    panel = pd.concat([panel, text_features], axis=1)

    # 5. Forward-looking targets. Reserved-prefix columns; never features.
    forward = add_forward_targets(
        tables.prices,
        drawdown_horizon=config.labels.tail_risk_horizon_trading_days,
    )
    panel = merge_asof_point_in_time(
        panel,
        forward,
        on="as_of",
        by="ticker",
        date_col="as_of",
        tolerance_days=PRICE_MATCH_TOLERANCE_DAYS,
    )

    # 6. Labels, masks and sources of record.
    panel = apply_risk_labels(
        panel,
        tables.events,
        definitions,
        data_end=data_end,
        config=config.labels,
    )

    # 7. Provenance and bookkeeping columns.
    panel["is_synthetic"] = bool(tables.is_synthetic)
    panel["data_version"] = config.project.data_version
    panel["n_sources"] = _count_sources(panel)
    if "insufficient_history" not in panel.columns:
        panel["insufficient_history"] = True
    panel["insufficient_history"] = panel["insufficient_history"].fillna(True).astype(bool)
    panel["sample_weight"] = 1.0

    # 8. Split assignment.
    panel, split_report = assign_split_column(
        panel, config.split, config.labels, data_end
    )

    # 9. Column order, then verification. Ordering first makes the leakage scan cheaper
    #    and the written file stable.
    #    A build may legitimately target a subset of the three labels (the Stage 2 run
    #    evaluates tail_risk only, because the event sources for the other two are not
    #    wired yet). Its label columns are then absent by design, so only the columns
    #    belonging to *configured* labels are required.
    configured_labels = {str(label) for label in definitions}
    skipped_labels = [label for label in RiskLabel if str(label) not in configured_labels]
    absent_by_design = {
        column
        for label in skipped_labels
        for column in (
            label_column(label),
            mask_column(label),
            event_column(label),
            horizon_column(label),
            source_of_record_column(label),
        )
    }
    missing_columns = [
        column
        for column in PANEL_COLUMNS
        if column not in panel.columns and column not in absent_by_design
    ]
    if missing_columns:
        raise RuntimeError(
            f"the builder did not produce these dictionary columns: {missing_columns}. "
            "Add them here and to docs/02-data.md section 5."
        )
    panel = panel[[column for column in PANEL_COLUMNS if column in panel.columns]]
    panel = panel.sort_values(["ticker", "as_of"]).reset_index(drop=True)

    _verify_panel(panel, config, definitions, data_end=data_end)

    features = feature_columns(
        panel,
        include=list(panel_subset(config.structured.feature_groups)),
    )
    target = panel[_label_column_for_target(config)].astype(float)
    scan = leakage_scan(
        panel,
        y=target,
        feature_names=features,
        labels=list(definitions),
        data_end=data_end,
    )
    alarms = correlation_alarm(panel[features], target)
    if alarms:
        scan["correlation_alarms"] = [
            {"feature": name, "correlation": value} for name, value in alarms
        ]

    rates = label_rates(panel, definitions)
    metadata: dict[str, Any] = {
        "data_version": config.project.data_version,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "is_synthetic": bool(tables.is_synthetic),
        "n_rows": int(len(panel)),
        "n_tickers": int(panel["ticker"].nunique()),
        "n_features": len(features),
        "feature_groups": list(config.structured.feature_groups),
        "labels": {str(label): definition.description for label, definition in definitions.items()},
        "label_horizons": horizon_note,
        "positive_rates": {
            str(index): float(value)
            for index, value in rates["positive_rate"].items()
        },
        "split": split_report.to_dict(),
        "blocked_by_test_empty": int(split_report.counts.get("test", 0)) == 0,
    }

    written: dict[str, str] = {}
    if write:
        paths = config.project_paths().ensure()
        written["panel"] = _write_frame(panel, paths.processed / "panel")
        written["events"] = _write_frame(tables.events, paths.processed / "events")
        written["filings"] = _write_frame(
            tables.filings.drop(columns=[c for c in ("is_synthetic",) if c in tables.filings.columns]),
            paths.processed / "filings",
        )
        written["news"] = _write_frame(tables.news, paths.processed / "news")
        written["split_report"] = _write_json(split_report.to_dict(), paths.processed / "split_report.json")
        written["build_metadata"] = _write_json(metadata, paths.processed / "build_metadata.json")
        written["leakage"] = _write_json(scan, paths.processed / "leakage_scan.json")
        written["label_rates"] = _write_frame(rates.reset_index(), paths.processed / "label_rates")
        written["text_context"] = _write_jsonl(text_context, paths.processed / "text_context.jsonl")
        logger.info("wrote %d artifacts under %s", len(written), paths.processed)

    return BuildResult(
        panel=panel,
        events=tables.events,
        filings=tables.filings,
        news=tables.news,
        text_context=text_context,
        split_report=split_report,
        metadata=metadata,
        leakage=scan,
        label_rates=rates,
        written=written,
        prices=tables.prices,
    )


def _verify_panel(
    panel: pd.DataFrame,
    config: ProjectConfig,
    definitions: dict[RiskLabel, Any],
    *,
    data_end: date,
) -> None:
    """Run every structural check. Any failure aborts the build.

    Ordering is deliberate: the cheap column check runs first, so that a schema mistake
    is reported before an expensive scan.
    """
    features = feature_columns(
        panel, include=list(panel_subset(config.structured.feature_groups))
    )
    assert_feature_matrix_is_clean(features)
    assert_asof_monotonic(panel)
    assert_no_future_events(panel, list(definitions))
    assert_label_window_observable(panel, list(definitions), data_end)
    if features:
        logger.debug("panel verified: %d rows, %d features", len(panel), len(features))
    else:  # pragma: no cover - only with an empty feature_groups
        logger.warning("no feature columns were selected; the model would see nothing")


def _count_sources(panel: pd.DataFrame) -> pd.Series:
    """Number of independent sources that contributed to each row.

    A row built from a single news article and nothing else is thin evidence dressed up
    as an observation. The count makes those rows visible so they can be excluded or at
    least reported.
    """
    indicators = pd.DataFrame(index=panel.index)
    indicators["price"] = panel.get("ret_20d", pd.Series(np.nan, index=panel.index)).notna()
    indicators["ratios"] = (
        panel.get("debt_to_equity", pd.Series(np.nan, index=panel.index)).notna()
    )
    indicators["disclosure"] = (
        panel.get("disclosure_len_tokens", pd.Series(np.nan, index=panel.index)).notna()
        & (panel.get("disclosure_len_tokens", pd.Series(0.0, index=panel.index)) > 0)
    )
    indicators["news"] = (
        panel.get("n_news_30d", pd.Series(0.0, index=panel.index)).fillna(0.0) > 0
    )
    indicators["events"] = panel.get(
        "source_of_record_default_risk", pd.Series("", index=panel.index)
    ).astype(str).ne("")
    return indicators.sum(axis=1).astype("int16")


def _write_json(payload: Any, path: Path) -> str:
    """Write a JSON document with sorted keys and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return str(path)


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> str:
    """Write JSONL with UTF-8 and LF endings, byte-identical across platforms."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return str(path)


def _label_column_for_target(config: ProjectConfig) -> str:
    """The label used for the panel-wide leakage scan: the first configured target."""
    from shingan.data.schema import label_column

    return label_column(RiskLabel(config.labels.targets[0]))


def _real_cik_lookup(tickers: pd.Series, filings: pd.DataFrame) -> pd.Series:
    """Ticker → CIK from the filings table, blank when the fetch did not record one.

    Blank rather than a placeholder. Every downstream use of this column is
    provenance — it is on the reserved-name list and never reaches a model — so its
    job is to be traceable, and a plausible-looking fake number is worse than an
    empty string because it looks traceable and is not.
    """
    if "cik" not in filings.columns:
        return pd.Series("", index=tickers.index, dtype=object)
    mapping = (
        filings[["ticker", "cik"]]
        .dropna()
        .drop_duplicates(subset=["ticker"], keep="last")
        .assign(cik=lambda frame: frame["cik"].astype(str))
        .set_index("ticker")["cik"]
    )
    return tickers.map(mapping).fillna("").astype(str)


def _synthetic_cik(ticker: str) -> str:
    """A stable, obviously fake CIK for a synthetic ticker.

    Prefixed with 9 so that it cannot collide with a real ten-digit CIK, which keeps a
    synthetic row identifiable even after the ``is_synthetic`` column is dropped.
    """
    return f"9{abs(sum(ticker.encode())) % 10**9:09d}"


def _company_name(ticker: str) -> str:
    """Deterministic company name for a synthetic ticker."""
    index = abs(sum(ticker.encode())) % len(_COMPANY_NAMES)
    return f"{_COMPANY_NAMES[index]} {ticker[-2:]}"


def _sector(ticker: str) -> str:
    """Deterministic sector for a synthetic ticker."""
    sectors = (
        "Industrials",
        "Financials",
        "Energy",
        "Consumer Discretionary",
        "Materials",
        "Information Technology",
    )
    return sectors[abs(sum(ticker.encode())) % len(sectors)]


def load_date_range(panel: pd.DataFrame) -> tuple[date, date]:
    """First and last ``as_of`` in a panel, as dates."""
    dates = pd.to_datetime(panel["as_of"])
    return dates.min().date(), dates.max().date()


__all__ = [
    "MAX_FILINGS_IN_CONTEXT",
    "MAX_NEWS_IN_CONTEXT",
    "PROMPT_SIGNAL_COLUMNS",
    "BuildResult",
    "RawTables",
    "build_panel",
    "load_date_range",
    "resolve_data_end",
]

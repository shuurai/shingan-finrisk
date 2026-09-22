"""Stage 2 deliverable: data acquisition and cleaning loss accounting.

``docs/07-roadmap.md`` Stage 2 asks for four things this script measures, and the
acceptance table asks for two more:

* EDGAR request success rate — measured as coverage per ticker, because the fetch
  script does not retain a per-request log and inventing one here would be worse than
  reporting the coverage the artifacts actually show.
* News coverage — the FNSPID snapshot is not wired, so this is 0 by design and is
  stated rather than omitted.
* Price gap rate — from :func:`shingan.data.prices.price_quality_report`, aggregated.
* ``label_mask`` false ratio, per label.
* Every positive traced to its ``source_of_record`` (the "labels can be checked"
  acceptance condition).
* Feature coverage, so a feature the pipeline never managed to supply is visible
  rather than showing up as a model that mysteriously ignores an input.

Read-only with respect to the source tables. Writes ``data_coverage.md`` and
``data_coverage.json`` under the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shingan.config import ProjectConfig, load_config  # noqa: E402
from shingan.data.prices import price_quality_report  # noqa: E402
from shingan.data.schema import (  # noqa: E402
    RiskLabel,
    label_column,
    mask_column,
    source_of_record_column,
)
from shingan.pipeline import select_feature_columns  # noqa: E402

DEFAULT_OVERLAY = ROOT / "configs" / "data" / "stage2_real.yaml"


def load_stack(overlay: Path) -> ProjectConfig:
    return load_config(
        ROOT / "configs" / "default.yaml",
        [overlay, ROOT / "configs" / "eval" / "default.yaml"],
        root=ROOT,
    )


def raw_dir(config: ProjectConfig) -> Path:
    if config.data.cache_dir:
        candidate = Path(config.data.cache_dir)
        return candidate if candidate.is_absolute() else ROOT / candidate
    return ROOT / "data" / "raw" / "real"


def acquisition_section(config: ProjectConfig, raw: Path) -> tuple[list[str], dict[str, Any]]:
    """What was asked for versus what arrived, per table and per ticker."""
    universe = [str(item) for item in config.data.universe]
    lines: list[str] = []
    detail: dict[str, Any] = {"universe_requested": len(universe), "tickers": {}}

    tables: dict[str, pd.DataFrame] = {}
    for name in ("prices", "fundamentals", "filings", "news", "events"):
        path = raw / f"{name}.parquet"
        tables[name] = pd.read_parquet(path) if path.is_file() else pd.DataFrame()

    lines.append("| table | rows | tickers | tickers vs universe | notes |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    notes = {
        "prices": "yfinance daily adjusted OHLCV",
        "fundamentals": "EDGAR inline XBRL companyfacts, every filed revision kept",
        "filings": "EDGAR submissions index: dates only, section='unavailable'",
        "news": "FNSPID snapshot not wired — 0 by design",
        "events": "rating / enforcement sources not wired — 0 by design",
    }
    for name, frame in tables.items():
        tickers = int(frame["ticker"].nunique()) if "ticker" in frame.columns and len(frame) else 0
        lines.append(
            f"| {name} | {len(frame)} | {tickers} | {tickers}/{len(universe)} | {notes[name]} |"
        )
        if "ticker" in frame.columns and len(frame):
            for ticker, count in frame.groupby("ticker").size().items():
                detail["tickers"].setdefault(str(ticker), {})[name] = int(count)

    doc_cache = raw / "docs"
    docs = len(list(doc_cache.glob("*.txt"))) if doc_cache.is_dir() else 0
    lines.append("")
    lines.append(f"- Filing documents cached under `{doc_cache.name}/`: **{docs}**.")
    if docs == 0 and len(tables["filings"]):
        lines.append(
            "- Filing **text** was not obtained. `www.sec.gov/Archives` returns 403 "
            '"Your Request Originates from an Undeclared Automated Tool" for every '
            "declared User-Agent tested from this network, including the browser UA and "
            "four compliant declaration shapes; `data.sec.gov` is unaffected. The "
            "decision grid therefore comes from the submissions index and every "
            "text-derived feature is zero."
        )
    detail["filing_documents_cached"] = docs

    missing = sorted(set(universe) - set(detail["tickers"]))
    detail["tickers_with_no_table_data"] = missing
    if missing:
        lines.append(f"- No data at all for: {', '.join(missing)}.")
    for ticker, counts in sorted(detail["tickers"].items()):
        if "prices" not in counts:
            lines.append(f"- `{ticker}` has EDGAR data but **no price series**.")
    return lines, detail


def price_section(prices: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    """Per-ticker price coverage, aggregated and worst-first."""
    if prices.empty:
        return ["- No price table.", ""], {"present": False}
    quality = price_quality_report(prices)
    worst = quality.sort_values("close_missing_frac", ascending=False).head(8)
    lines = ["| ticker | rows | first | last | close missing | longest gap | contaminated |"]
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: |")
    for ticker, row in worst.iterrows():
        lines.append(
            f"| {ticker} | {int(row['rows'])} | {row['first_date']} | {row['last_date']} | "
            f"{row['close_missing_frac']:.4f} | {int(row['longest_missing_run'])} | "
            f"{int(row['contaminated_rows'])} |"
        )
    detail = {
        "present": True,
        "n_tickers": len(quality),
        "total_rows": int(quality["rows"].sum()),
        "mean_close_missing_frac": float(quality["close_missing_frac"].mean()),
        "max_close_missing_frac": float(quality["close_missing_frac"].max()),
        "tickers_with_contaminated_rows": int((quality["contaminated_rows"] > 0).sum()),
    }
    return lines, detail


def panel_section(
    panel: pd.DataFrame, config: ProjectConfig, definitions: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Label observability, positives with their source of record, and feature coverage."""
    lines: list[str] = []
    detail: dict[str, Any] = {"n_rows": len(panel), "n_tickers": int(panel["ticker"].nunique())}

    lines.append("| label | rows | observable | masked out | masked frac | positives | rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    labels_detail: dict[str, Any] = {}
    for label in definitions:
        mask = panel[mask_column(label)].astype(bool)
        values = pd.to_numeric(panel[label_column(label)], errors="coerce")
        positives = int(values[mask].fillna(0).sum())
        observable = int(mask.sum())
        labels_detail[label] = {
            "rows": len(panel),
            "observable": observable,
            "masked_out": int((~mask).sum()),
            "masked_frac": float((~mask).mean()),
            "positives": positives,
            "positive_rate": float(positives / observable) if observable else float("nan"),
        }
        lines.append(
            f"| {label} | {len(panel)} | {observable} | {int((~mask).sum())} | "
            f"{(~mask).mean():.4f} | {positives} | "
            f"{(positives / observable if observable else float('nan')):.4f} |"
        )
    detail["labels"] = labels_detail

    lines.append("")
    if config.labels.targets:
        label = str(config.labels.targets[0])
        mask = panel[mask_column(label)].astype(bool)
        values = pd.to_numeric(panel[label_column(label)], errors="coerce").fillna(0)
        positives = panel.loc[mask & (values > 0)]
        source_column = source_of_record_column(label)
        lines.append(
            f"**Every `{label}` positive, with its source of record** "
            f"(`docs/07-roadmap.md` Stage 2 acceptance: *labels can be checked*):"
        )
        lines.append("")
        lines.append("| ticker | as_of | event date | source of record |")
        lines.append("| --- | --- | --- | --- |")
        for record in positives.sort_values(["as_of", "ticker"]).itertuples(index=False):
            event = getattr(record, f"event_date_{label}", None)
            # A `tail_risk` event is the drawdown trough inside the forward window, not
            # a dated corporate action, so the column is legitimately empty here. Print
            # a dash rather than pandas' "NaT", which reads as a broken join.
            event_text = "—" if event is None or pd.isna(event) else str(pd.Timestamp(event).date())
            lines.append(
                f"| {record.ticker} | {pd.Timestamp(record.as_of).date()} | "
                f"{event_text} | {getattr(record, source_column, '')} |"
            )
        untraceable = int(
            (
                positives[source_column].astype(str).str.strip() == ""
            ).sum()
            if source_column in positives.columns
            else len(positives)
        )
        detail["positives_without_source_of_record"] = untraceable
        lines.append("")
        lines.append(f"- Positives with no `source_of_record`: **{untraceable}**.")

    lines.append("")
    features = select_feature_columns(panel, config)
    coverage = panel[features].apply(lambda column: pd.to_numeric(column, errors="coerce").notna().mean())
    absent = sorted(coverage[coverage == 0].index.tolist())
    detail["n_features"] = len(features)
    detail["features_with_zero_coverage"] = absent
    lines.append(f"- Feature columns: **{len(features)}**; non-null fraction below.")
    lines.append("")
    lines.append("| feature | non-null frac |")
    lines.append("| --- | ---: |")
    for name in sorted(coverage.index, key=lambda item: (coverage[item], item)):
        lines.append(f"| {name} | {coverage[name]:.4f} |")
    if absent:
        lines.append("")
        lines.append(
            f"- **{len(absent)} feature(s) have zero coverage and are dropped by the model "
            f"at fit time**: {', '.join(absent)}. They are configured in "
            "`structured.feature_groups` but nothing in the data layer supplies them "
            "(`vix_*` and `credit_spread_chg_20d` need external macro series; "
            "`beta_252d` needs an index benchmark; the `sent_*`/`neg_kw_*` columns need "
            "news or filing text)."
        )
    return lines, detail


def split_section(panel: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    counts = panel["split"].value_counts().to_dict()
    lines = ["| split | rows |", "| --- | ---: |"]
    for name in ("train", "valid", "test", "purged", "excluded"):
        lines.append(f"| {name} | {int(counts.get(name, 0))} |")
    by_split = {
        name: {
            "rows": int((panel["split"] == name).sum()),
            "tickers": int(panel.loc[panel["split"] == name, "ticker"].nunique()),
        }
        for name in ("train", "valid", "test")
    }
    lines.append("")
    lines.append(
        "distinct companies per split: "
        + ", ".join(f"{name} {item['tickers']}" for name, item in by_split.items())
    )
    return lines, {"counts": {k: int(v) for k, v in counts.items()}, "per_split": by_split}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts" / "stage2")
    args = parser.parse_args()

    config = load_stack(args.data_config)
    raw = raw_dir(config)
    processed = ROOT / "data" / "processed"

    panel_path = processed / "panel.parquet"
    if not panel_path.is_file():
        raise SystemExit(
            f"no panel at {panel_path}; run the build first "
            "(python -m shingan data build --data-config configs/data/stage2_real.yaml)"
        )
    panel = pd.read_parquet(panel_path)
    definitions = [str(label) for label in config.labels.targets]
    unknown = [name for name in definitions if name not in {str(item) for item in RiskLabel}]
    if unknown:
        raise SystemExit(f"configured targets {unknown} are not RiskLabel members")

    sections: dict[str, tuple[list[str], dict[str, Any]]] = {}
    sections["1. Acquisition: requested vs arrived"] = acquisition_section(config, raw)
    sections["2. Price coverage"] = price_section(
        pd.read_parquet(raw / "prices.parquet") if (raw / "prices.parquet").is_file() else pd.DataFrame()
    )
    sections["3. Labels and feature coverage"] = panel_section(panel, config, definitions)
    sections["4. Split" ] = split_section(panel)

    lines = [
        "# Stage 2 — data acquisition and cleaning loss accounting",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} by "
        "`scripts/stage2_coverage.py`. `docs/07-roadmap.md` Stage 2 requires these "
        "losses to be measured rather than assumed.",
        "",
        f"- Universe configured: {len(config.data.universe)} names",
        f"- Data window: {config.data.start} .. {config.data.end}",
        f"- Sources: {', '.join(config.data.sources)}",
        f"- Targets evaluated: {', '.join(definitions)}",
        f"- Raw tables read from: `{raw}`",
        "",
    ]
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "universe_configured": [str(item) for item in config.data.universe],
        "sources": [str(item) for item in config.data.sources],
        "targets": definitions,
        "data_window": [str(config.data.start), str(config.data.end)],
    }
    for title, (body, detail) in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(body)
        lines.append("")
        payload[title] = detail

    args.run_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.run_dir / "data_coverage.md"
    json_path = args.run_dir / "data_coverage.json"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {markdown_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

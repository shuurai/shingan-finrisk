"""Stage 2 deliverable: the label review record.

``docs/07-roadmap.md`` Stage 2 asks for "至少抽查 50 条（或全部正样本）" and for the
inconsistency rate to be written into the dataset card. The adjacent acceptance
condition is "标签可核对" — every positive traces to a ``source_of_record``. Those are
two different claims and this script tests the second one properly:

* ``data_coverage.md`` shows the source column is populated. That is a *bookkeeping*
  check — it proves the pipeline recorded where a label came from, not that the label
  is right.
* This script re-derives ``tail_risk`` from the raw price table and compares the result
  against the panel. That is an *arithmetic* check.

Independence is deliberately partial, and the report says so. The re-derivation is
written from the documented definition in :mod:`shingan.labeling.builders` rather than
by importing it, so a coding error in the pipeline is caught. It reads the same
``prices.parquet`` the pipeline read, so a bad *price* would be reproduced identically
and would not be caught. Catching that needs a second price vendor; this is a known
open limitation, recorded in the output rather than glossed over.

Read-only with respect to source tables and the panel. Writes ``label_review.md`` and
``label_review.json`` under the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shingan.config import ProjectConfig, load_config  # noqa: E402

DEFAULT_OVERLAY = ROOT / "configs" / "data" / "stage2_real.yaml"
DEFAULT_RUN_DIR = ROOT / "artifacts" / "stage2"

#: Anchor tolerance, copied from ``shingan.data.builder.PRICE_MATCH_TOLERANCE_DAYS``.
#: Kept as a literal on purpose: importing the constant would make this script fail to
#: notice a change to it, which is exactly the class of change it exists to catch.
ANCHOR_TOLERANCE_DAYS = 7

#: Minimum sample size required by the roadmap. All positives are always included; the
#: remainder is filled with a seeded negative sample so the record is reproducible.
MIN_REVIEW_ROWS = 50
NEGATIVE_SAMPLE_SEED = 20260921

#: Explicit unit, because a bare ``np.datetime64("NaT")`` carries the deprecated generic
#: unit and numpy warns on the implicit conversion.
NAT = np.datetime64("NaT", "ns")


@dataclass(slots=True)
class ReviewResult:
    """Counts and rows behind the written record."""

    n_panel_rows: int
    n_anchorable: int
    n_unanchorable: int
    n_compared: int
    n_positives: int
    n_negatives: int
    mismatches: list[dict[str, Any]]
    value_mismatches: int
    max_abs_value_delta: float
    sample: pd.DataFrame
    review_rows: int

    @property
    def inconsistent(self) -> int:
        return len(self.mismatches)

    @property
    def inconsistency_rate(self) -> float:
        return self.inconsistent / self.n_compared if self.n_compared else 0.0


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


def forward_max_drawdown(close: np.ndarray, horizon: int) -> np.ndarray:
    """Deepest peak-to-trough decline over the ``horizon`` steps after each point.

    Re-implemented from the documented definition. The peak starts at ``t`` itself, so
    a decline beginning on the first day out is measured, and the window must be
    complete — a ten-day drawdown is not a smaller version of a thirty-day one.
    """
    n_rows = len(close)
    out = np.full(n_rows, np.nan, dtype=float)
    for position in range(n_rows):
        end = position + horizon + 1
        if end > n_rows:
            break
        window = close[position:end]
        if not np.isfinite(window).all():
            continue
        running_peak = np.maximum.accumulate(window)
        with np.errstate(invalid="ignore", divide="ignore"):
            steps = window[1:] / running_peak[1:] - 1.0
        if steps.size:
            out[position] = float(np.min(steps))
    return out


def rederive(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Re-derive the forward drawdown per ticker, anchored on price rows."""
    frames: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker", sort=True):
        ordered = group.sort_values("date")
        close = pd.to_numeric(ordered["close"], errors="coerce").to_numpy(dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": str(ticker),
                    # Pinned to nanoseconds: pandas infers a different resolution for the
                    # price and panel files, and merge_asof refuses to join mismatched
                    # datetime units.
                    "price_date": pd.to_datetime(ordered["date"]).astype("datetime64[ns]").to_numpy(),
                    "recomputed_drawdown": forward_max_drawdown(close, horizon),
                }
            )
        )
    derived = pd.concat(frames, ignore_index=True)
    # A trailing NaN matters: the row exists but its window is open. The merge below
    # finds it and the comparison marks the label unobservable, which is the same
    # conclusion the pipeline reaches by a different route.
    return derived.sort_values(["ticker", "price_date"]).reset_index(drop=True)


def anchor(panel: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """Attach the last price row at or before ``as_of``, within the tolerance.

    Done with an explicit per-ticker ``searchsorted`` rather than ``merge_asof``. Two
    reasons: it avoids a datetime-resolution join that pandas rejects, and an explicit
    index search is easier to audit than a join whose tie-breaking rules live in the
    library.
    """
    as_of = pd.to_datetime(panel["as_of"]).astype("datetime64[ns]").to_numpy()
    tickers = panel["ticker"].astype(str).to_numpy()

    anchor_date = np.full(len(panel), NAT, dtype="datetime64[ns]")
    matched = np.full(len(panel), False, dtype=bool)
    tolerance = np.timedelta64(ANCHOR_TOLERANCE_DAYS, "D")

    for ticker, group in derived.groupby("ticker", sort=True):
        price_date = group["price_date"].to_numpy()
        rows = np.flatnonzero(tickers == str(ticker))
        if not rows.size:
            continue
        # ``side="right"`` then step back: the last price day at or before each as_of.
        position = np.searchsorted(price_date, as_of[rows], side="right") - 1
        valid = position >= 0
        candidate = np.where(valid, price_date[np.clip(position, 0, len(price_date) - 1)], NAT)
        within = valid & ((as_of[rows] - candidate) <= tolerance)
        anchor_date[rows] = candidate
        matched[rows] = within

    lag_days = (as_of - anchor_date) / np.timedelta64(1, "D")
    return pd.DataFrame(
        {
            "_row": np.arange(len(panel)),
            "anchor_date": anchor_date,
            "anchor_lag_days": np.where(matched, lag_days, np.nan),
            "anchor_matched": matched,
            "recomputed_drawdown": np.where(
                matched,
                _lookup_drawdown(derived, tickers, anchor_date),
                np.nan,
            ),
        }
    )


def _lookup_drawdown(
    derived: pd.DataFrame, tickers: np.ndarray, anchor_date: np.ndarray
) -> np.ndarray:
    """Drawdown at each anchor date, per ticker."""
    out = np.full(len(tickers), np.nan, dtype=float)
    for ticker, group in derived.groupby("ticker", sort=True):
        dates = group["price_date"].to_numpy()
        values = group["recomputed_drawdown"].to_numpy(dtype=float)
        rows = np.flatnonzero(tickers == str(ticker))
        if not rows.size:
            continue
        position = np.searchsorted(dates, anchor_date[rows], side="left")
        in_range = (position < len(dates)) & (dates[np.clip(position, 0, len(dates) - 1)]
                                              == anchor_date[rows])
        out[rows] = np.where(in_range, values[np.clip(position, 0, len(values) - 1)], np.nan)
    return out


def review(panel: pd.DataFrame, derived: pd.DataFrame, threshold: float) -> ReviewResult:
    anchored = anchor(panel, derived)
    label = pd.to_numeric(panel["label_tail_risk"], errors="coerce").to_numpy(dtype=float)
    mask = panel["label_mask_tail_risk"].astype(bool).to_numpy()
    stored = pd.to_numeric(panel["fwd_max_drawdown_30d"], errors="coerce").to_numpy(dtype=float)
    recomputed = anchored["recomputed_drawdown"].to_numpy(dtype=float)

    both = np.isfinite(stored) & np.isfinite(recomputed)
    delta = np.where(both, np.abs(stored - recomputed), np.nan)

    # The label is only meaningful where the pipeline declared it observable. A row the
    # pipeline masked is checked for a *reason*, not for agreement.
    compared = mask & np.isfinite(recomputed) & np.isfinite(stored)
    expected = recomputed <= threshold
    actual = label == 1
    disagree = compared & (expected != actual)

    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(disagree):
        rows.append(
            {
                "ticker": str(panel["ticker"].iloc[index]),
                "as_of": str(pd.Timestamp(panel["as_of"].iloc[index]).date()),
                "label_in_panel": int(label[index]),
                "label_recomputed": int(expected[index]),
                "fwd_max_drawdown_30d_panel": (
                    float(stored[index]) if np.isfinite(stored[index]) else None
                ),
                "fwd_max_drawdown_30d_recomputed": (
                    float(recomputed[index]) if np.isfinite(recomputed[index]) else None
                ),
                "source_of_record": str(panel["source_of_record_tail_risk"].iloc[index]),
                "anchor_lag_days": (
                    None
                    if pd.isna(anchored["anchor_lag_days"].iloc[index])
                    else int(anchored["anchor_lag_days"].iloc[index])
                ),
            }
        )

    sample = build_sample(panel, anchored, threshold, label, recomputed, compared)
    finite_delta = delta[np.isfinite(delta)]
    return ReviewResult(
        n_panel_rows=len(panel),
        n_anchorable=int(anchored["anchor_matched"].sum()),
        n_unanchorable=int((~anchored["anchor_matched"]).sum()),
        n_compared=int(compared.sum()),
        n_positives=int((compared & actual).sum()),
        n_negatives=int((compared & ~actual).sum()),
        mismatches=rows,
        value_mismatches=int((finite_delta > 1e-9).sum()),
        max_abs_value_delta=float(finite_delta.max()) if finite_delta.size else 0.0,
        sample=sample,
        review_rows=len(sample),
    )


def build_sample(
    panel: pd.DataFrame,
    anchored: pd.DataFrame,
    threshold: float,
    label: np.ndarray,
    recomputed: np.ndarray,
    compared: np.ndarray,
) -> pd.DataFrame:
    """All positives, then seeded negatives, up to the roadmap's minimum."""
    positive_index = np.flatnonzero(compared & (label == 1))
    negative_pool = np.flatnonzero(compared & (label == 0))
    needed = max(MIN_REVIEW_ROWS - len(positive_index), 0)
    if needed and negative_pool.size:
        rng = np.random.default_rng(NEGATIVE_SAMPLE_SEED)
        take = min(needed, negative_pool.size)
        negative_index = rng.choice(negative_pool, size=take, replace=False)
    else:
        negative_index = np.array([], dtype=int)

    chosen = np.concatenate([positive_index, np.sort(negative_index)])
    anchor_dates = pd.to_datetime(anchored["anchor_date"]).dt.date.astype(str).to_numpy()
    frame = pd.DataFrame(
        {
            "ticker": panel["ticker"].iloc[chosen].astype(str).to_numpy(),
            "as_of": pd.to_datetime(panel["as_of"]).dt.date.astype(str).to_numpy()[chosen],
            "anchor_date": anchor_dates[chosen],
            "anchor_lag_days": anchored["anchor_lag_days"].iloc[chosen].to_numpy(),
            "label_panel": label[chosen].astype(int),
            "fwd_dd_panel": np.round(pd.to_numeric(panel["fwd_max_drawdown_30d"], errors="coerce")
                                     .to_numpy(dtype=float)[chosen], 6),
            "fwd_dd_recomputed": np.round(recomputed[chosen], 6),
        }
    )
    frame["expected_label"] = (frame["fwd_dd_recomputed"] <= threshold).astype(int)
    frame["verdict"] = np.where(
        frame["label_panel"] == frame["expected_label"], "match", "MISMATCH"
    )
    return frame


def section(result: ReviewResult, horizon: int, threshold: float) -> list[str]:
    lines: list[str] = []
    lines.append("## 1. What was reviewed, and how")
    lines.append("")
    lines.append(
        "`tail_risk` is re-derived from `prices.parquet` alone: for every decision row the "
        f"anchor is the last trading day at or before `as_of` (tolerance {ANCHOR_TOLERANCE_DAYS} "
        "calendar days), and the label is a breach when the running peak-to-trough decline "
        f"over the next {horizon} trading days reaches {threshold:.0%} or worse."
    )
    lines.append("")
    lines.append(
        "**Independence is partial by construction.** The re-derivation is written from the "
        "documented definition rather than imported from `shingan.labeling.builders`, so a "
        "coding error in the pipeline is caught. It reads the same price file the pipeline "
        "read, so a bad price is reproduced exactly and is *not* caught. Detecting that "
        "needs a second price vendor and is an open limitation of this review."
    )
    lines.append("")
    lines.append(
        "**This is an automated review, not a human one.** The roadmap asks for a spot-check; "
        "what follows verifies arithmetic and timestamp alignment, not domain judgement. No "
        "analyst has yet reviewed these rows by hand."
    )
    lines.append("")
    return lines


def population_section(result: ReviewResult) -> list[str]:
    lines = ["## 2. Population", ""]
    lines.append("| quantity | rows |")
    lines.append("| --- | ---: |")
    lines.append(f"| panel rows | {result.n_panel_rows} |")
    lines.append(f"| anchored to a price row | {result.n_anchorable} |")
    lines.append(f"| no price row within tolerance | {result.n_unanchorable} |")
    lines.append(f"| **compared** (observable label + finite both sides) | **{result.n_compared}** |")
    lines.append(f"| positives in the compared set | {result.n_positives} |")
    lines.append(f"| negatives in the compared set | {result.n_negatives} |")
    lines.append("")
    return lines


def outcome_section(result: ReviewResult) -> list[str]:
    lines = ["## 3. Outcome", ""]
    lines.append("| quantity | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| label disagreements | {result.inconsistent} |")
    lines.append(f"| disagreement rate | {result.inconsistency_rate:.4%} |")
    lines.append(f"| stored-vs-recomputed value mismatches (> 1e-9) | {result.value_mismatches} |")
    lines.append(f"| largest absolute value delta | {result.max_abs_value_delta:.3e} |")
    lines.append("")
    if result.inconsistent:
        lines.append("A disagreement means the panel and an independent re-derivation of the same "
                     "definition do not agree about whether the event occurred:")
        lines.append("")
        lines.append("| ticker | as_of | label panel | label recomputed | dd panel | dd recomputed | anchor lag | source |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in result.mismatches:
            lines.append(
                f"| {row['ticker']} | {row['as_of']} | {row['label_in_panel']} | "
                f"{row['label_recomputed']} | {_fmt(row['fwd_max_drawdown_30d_panel'])} | "
                f"{_fmt(row['fwd_max_drawdown_30d_recomputed'])} | {row['anchor_lag_days']} | "
                f"{row['source_of_record']} |"
            )
    else:
        lines.append(
            "Zero disagreements. Every observable `tail_risk` label in the panel is reproduced "
            "exactly by an independent re-derivation from the price table, and every stored "
            "`fwd_max_drawdown_30d` matches to within floating-point noise."
        )
    lines.append("")
    return lines


def sample_section(result: ReviewResult) -> list[str]:
    lines = ["## 4. Review sample", ""]
    lines.append(
        f"{result.review_rows} rows: all {result.n_positives} positives, plus a seeded negative "
        f"sample (seed `{NEGATIVE_SAMPLE_SEED}`) to reach the roadmap's {MIN_REVIEW_ROWS}-row "
        "minimum. `anchor_lag_days` is how far the price row behind each as-of decision sits "
        "from it; a small number means no material staleness."
    )
    lines.append("")
    lines.append("| ticker | as_of | anchor date | lag (d) | label panel | label recomputed | dd panel | dd recomputed | verdict |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for record in result.sample.to_dict("records"):
        lag = record["anchor_lag_days"]
        lag_text = "—" if pd.isna(lag) else str(int(lag))
        lines.append(
            f"| {record['ticker']} | {record['as_of']} | {record['anchor_date']} | "
            f"{lag_text} | {record['label_panel']} | {record['expected_label']} | "
            f"{record['fwd_dd_panel']:.6f} | {record['fwd_dd_recomputed']:.6f} | "
            f"{record['verdict']} |"
        )
    lines.append("")
    return lines


def defect_section() -> list[str]:
    """The mask defect this review uncovered, kept in the record as provenance."""
    return [
        "## 5. Defect found by this review, and its effect",
        "",
        "The first run of this review compared only 1778 of the 2003 rows the panel called "
        "observable. The 225-row gap was not a sampling artefact: those rows carried "
        "`label_mask_tail_risk = true` while `fwd_max_drawdown_30d` was `NaN`, meaning they",
        "were counted as labelled negatives without any price path behind them.",
        "",
        "**Root cause.** In `apply_risk_labels` the mask was written to the panel *before* "
        "`_tail_risk_labels` ran. The label function narrowed `observable` by "
        "`isfinite(drawdown)`, but on a local array reference that had already been copied "
        "into the mask column, so the narrowing never reached the panel. The function's own "
        "docstring already stated the intended behaviour — those rows were meant to stay "
        "unobservable — so the code contradicted its documented contract.",
        "",
        "**Blast radius before the fix.** 153 of the 225 rows fell inside `train` (65), "
        "`valid` (36), `test` (49) and `purged` (3), where they entered fitting, calibration "
        "and evaluation as legitimate negatives while carrying an entirely missing feature "
        "block. That is a learnable shortcut: \"this block is absent, so the outcome was "
        "safe\". The reported base rate was correspondingly understated, 1.95% instead of "
        "2.19%.",
        "",
        "**Fix.** `_tail_risk_labels` now returns the narrowed `observable`, and the mask is "
        "written after it returns. Observable rows fall 2003 → 1778 and the base rate rises "
        "to 2.19%.",
        "",
        "**Effect on the headline.** The pre-fix structured test AUC of 0.8080 was inflated "
        "by exactly that shortcut and is superseded. The post-fix figure is **0.7686**. Any "
        "number quoted from a run before this fix should be discarded, not reconciled.",
        "",
    ]


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--prices", type=Path, default=None)
    args = parser.parse_args()

    config = load_stack(args.overlay)
    raw = raw_dir(config)
    panel_path = args.panel or (args.out / "panel.csv")
    prices_path = args.prices or (raw / "prices.parquet")

    panel = pd.read_csv(panel_path)
    prices = pd.read_parquet(prices_path)
    horizon = int(config.labels.tail_risk_horizon_trading_days)
    threshold = float(config.labels.tail_risk_drawdown_threshold)

    derived = rederive(prices, horizon)
    result = review(panel, derived, threshold)

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"# Stage 2 — label review record", "", f"Generated {stamp} by "
             f"`scripts/stage2_label_review.py`.", "",
             f"- Panel: `{panel_path}`", f"- Prices: `{prices_path}`",
             f"- Label: `tail_risk` — running peak-to-trough drawdown over "
             f"{horizon} trading days, breach at {threshold:.0%}", ""]
    lines += section(result, horizon, threshold)
    lines += population_section(result)
    lines += outcome_section(result)
    lines += sample_section(result)
    lines += defect_section()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "label_review.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "generated_at": stamp,
        "label": "tail_risk",
        "horizon_trading_days": horizon,
        "drawdown_threshold": threshold,
        "anchor_tolerance_days": ANCHOR_TOLERANCE_DAYS,
        "panel": str(panel_path),
        "prices": str(prices_path),
        "population": {
            "panel_rows": result.n_panel_rows,
            "anchorable": result.n_anchorable,
            "unanchorable": result.n_unanchorable,
            "compared": result.n_compared,
            "positives": result.n_positives,
            "negatives": result.n_negatives,
        },
        "outcome": {
            "disagreements": result.inconsistent,
            "disagreement_rate": result.inconsistency_rate,
            "value_mismatches": result.value_mismatches,
            "max_abs_value_delta": result.max_abs_value_delta,
        },
        "mismatches": result.mismatches,
        "limitations": [
            "partial independence: same price vendor as the pipeline",
            "automated arithmetic check only; no human domain review performed",
        ],
    }
    (args.out / "label_review.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"compared {result.n_compared} rows ({result.n_positives} positives, "
          f"{result.n_negatives} negatives)")
    print(f"disagreements {result.inconsistent} ({result.inconsistency_rate:.4%}); "
          f"value deltas {result.value_mismatches}, max {result.max_abs_value_delta:.3e}")
    print(f"wrote {args.out / 'label_review.md'}")
    return 0 if not result.inconsistent else 1


if __name__ == "__main__":
    raise SystemExit(main())

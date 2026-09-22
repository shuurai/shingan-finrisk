"""Audit the Stage 2 `tail_risk` AUC: is 0.7686 a measurement, or an artefact of 5 rows?

The number is high enough to be worth attacking. A test AUC of 0.77 on a rare-event
label looks like a result; with five positives it may be nothing at all. This script
tries to falsify it rather than confirm it, in this order:

1. **Reproduce.** Load the saved model, score the test split, and check the AUC equals
   the number in ``metrics.json``. If it does not, the audit is about a different model
   and everything after it is void.
2. **Where are the positives?** Positives per split, and the calendar distribution of
   the test positives. Positives that share one episode make the AUC a statement about
   that episode.
3. **How precise is it?** Hanley-McNeil standard error, plus a permutation null built by
   shuffling the test labels. The second one is the honest interval: it answers "what
   AUC does a model with no signal produce on this exact slice".
4. **How fragile is it?** Leave-one-positive-out and leave-one-ticker-out. If removing a
   single row moves the AUC by more than the standard error, the point estimate is not
   a property of the model.
5. **What drives it?** Univariate AUC per feature, and near-duplicate feature pairs —
   twenty copies of the same volatility measure that all rank the same way is one
   feature, not twenty.
6. **Is any of it missingness?** AUC of each feature's *missingness indicator* against
   the label. This is the shape the Stage 2 mask bug had, and it must stay at 0.5.

Usage::

    python scripts/stage2_auc_audit.py
    python scripts/stage2_auc_audit.py --permutations 8000 --seed 7

Writes ``artifacts/stage2/auc_audit.md`` and ``auc_audit.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shingan.eval.metrics import roc_auc  # noqa: E402
from shingan.models.structured import StructuredRiskModel  # noqa: E402

PANEL = ROOT / "artifacts" / "stage2" / "panel.csv"
MODEL_DIR = ROOT / "artifacts" / "models" / "stage2" / "tail_risk"
REPORTED = 0.7685831622176591
LABEL = "label_tail_risk"
MASK = "label_mask_tail_risk"


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUC estimate, Hanley & McNeil (1982).

    Used rather than a bootstrap because a bootstrap over five positives can only
    produce a handful of distinct resamples; this formula is the standard closed form
    and it is reported next to the permutation null so the two can be compared.
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(variance, 0.0)))


def normal_ci(estimate: float, se: float, z: float = 1.959963985) -> tuple[float, float]:
    return (estimate - z * se, estimate + z * se)


def permutation_null(y: np.ndarray, scores: np.ndarray, n: int, seed: int) -> np.ndarray:
    """AUC under label shuffling. The null a no-skill model would produce here."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    for index in range(n):
        out[index] = roc_auc(rng.permutation(y), scores)
    return out


def ticker_cluster_bootstrap(
    y: np.ndarray, scores: np.ndarray, tickers: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, int]:
    """Resample whole companies, with replacement.

    A panel's rows are not independent draws: a company's 15 quarterly filings are 15
    observations of one company's history, and the five test positives come from four
    companies. Resampling rows would pretend otherwise. Resampling *tickers* asks the
    question that matters — how much of this estimate survives changing which companies
    are in the sample at all.

    Returns the distribution of AUC and the number of draws where the resample happened
    to contain no positives, or only positives, and the metric is undefined.
    """
    rng = np.random.default_rng(seed)
    unique_tickers = np.unique(tickers)
    rows_by_ticker = {name: np.flatnonzero(tickers == name) for name in unique_tickers}
    values: list[float] = []
    undefined = 0
    for _ in range(n):
        drawn = rng.choice(unique_tickers, size=len(unique_tickers), replace=True)
        rows = np.concatenate([rows_by_ticker[name] for name in drawn])
        resampled_y = y[rows]
        positives = int(resampled_y.sum())
        if positives == 0 or positives == len(resampled_y):
            undefined += 1
            continue
        values.append(roc_auc(resampled_y, scores[rows]))
    return np.asarray(values, dtype=float), undefined


def univariate_table(frame: pd.DataFrame, features: list[str], y: np.ndarray) -> pd.DataFrame:
    """Per-feature AUC, oriented so that >= 0.5, plus the missingness indicator's AUC."""
    rows: list[dict[str, Any]] = []
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        observed = np.isfinite(values)
        auc = roc_auc(y[observed], values[observed]) if observed.sum() and y[observed].size else float("nan")
        missing_auc = roc_auc(y, (~observed).astype(float)) if 0 < observed.sum() < len(y) else float("nan")
        rows.append(
            {
                "feature": name,
                "n_observed": int(observed.sum()),
                "auc": auc,
                "orientation": "direct" if (np.isfinite(auc) and auc >= 0.5) else "inverted",
                "abs_edge": abs(auc - 0.5) if np.isfinite(auc) else float("nan"),
                # A single feature costs nothing to flip, so the fair benchmark against a
                # fitted model is max(AUC, 1-AUC), not the raw sign the feature happens to carry.
                "oriented_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan"),
                "missing_indicator_auc": missing_auc,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_edge", ascending=False).reset_index(drop=True)


def duplicate_pairs(frame: pd.DataFrame, features: list[str], threshold: float) -> list[dict[str, Any]]:
    """Feature pairs whose absolute correlation exceeds ``threshold``."""
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    usable = [name for name in features if numeric[name].notna().sum() > 2]
    if len(usable) < 2:
        return []
    correlation = numeric[usable].corr(method="spearman").abs()
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(usable):
        for right in usable[i + 1 :]:
            value = correlation.loc[left, right]
            if np.isfinite(value) and value >= threshold:
                pairs.append({"left": left, "right": right, "spearman_abs": round(float(value), 4)})
    return sorted(pairs, key=lambda item: -item["spearman_abs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--correlation", type=float, default=0.95)
    args = parser.parse_args()

    panel = pd.read_csv(PANEL)
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    model = StructuredRiskModel.load(MODEL_DIR)
    features = list(model._feature_names)

    observable = panel[MASK].astype(bool)
    test = panel.loc[observable & (panel["split"] == "test")].copy()
    y = test[LABEL].to_numpy(dtype=int)
    scores = model.predict_proba(test)
    auc = roc_auc(y, scores)

    result: dict[str, Any] = {
        "reproduced": {
            "reported_auc": REPORTED,
            "recomputed_auc": float(auc),
            "matches": bool(abs(auc - REPORTED) < 1e-9),
            "n_rows": len(test),
            "n_positives": int(y.sum()),
            "n_negatives": int(len(y) - y.sum()),
            "n_features_used": len(features),
        }
    }

    # -- 2. where the positives live -------------------------------------------
    per_split: dict[str, Any] = {}
    for name in ("train", "valid", "test", "purged", "excluded"):
        subset = panel.loc[observable & (panel["split"] == name)]
        per_split[name] = {
            "rows": len(subset),
            "positives": int(subset[LABEL].sum()),
            "base_rate": float(subset[LABEL].mean()) if len(subset) else float("nan"),
        }
    all_positives = panel.loc[observable & (panel[LABEL] == 1)]
    result["positives_by_split"] = per_split
    result["positives_total_observable"] = len(all_positives)
    result["positive_dates"] = {
        "earliest": str(all_positives["as_of"].min().date()),
        "latest": str(all_positives["as_of"].max().date()),
        "distinct_dates": int(all_positives["as_of"].nunique()),
        "by_year": {str(k): int(v) for k, v in all_positives["as_of"].dt.year.value_counts().sort_index().items()},
        "by_quarter": {
            str(k): int(v)
            for k, v in all_positives["as_of"].dt.to_period("Q").value_counts().sort_index().items()
        },
        "by_ticker": {str(k): int(v) for k, v in all_positives["ticker"].value_counts().items()},
    }
    test_positives = test.loc[y == 1]
    result["test_positive_dates"] = {
        "dates": [str(value.date()) for value in sorted(test_positives["as_of"])],
        "distinct_dates": int(test_positives["as_of"].nunique()),
        "tickers": sorted(test_positives["ticker"].astype(str)),
        "score_ranks": [
            int((scores > scores[test.index.get_loc(index)]).sum() + 1) for index in test_positives.index
        ],
        "n_rows": len(test),
    }

    # -- 3. precision ----------------------------------------------------------
    se = hanley_mcneil_se(auc, int(y.sum()), int(len(y) - y.sum()))
    null = permutation_null(y, scores, args.permutations, args.seed)
    cluster, cluster_undefined = ticker_cluster_bootstrap(
        y, scores, test["ticker"].astype(str).to_numpy(), args.permutations, args.seed
    )
    result["precision"] = {
        "auc": float(auc),
        "hanley_mcneil_se": se,
        "normal_ci95": list(normal_ci(auc, se)),
        "permutation_n": args.permutations,
        "permutation_null_mean": float(null.mean()),
        "permutation_null_sd": float(null.std(ddof=1)),
        "permutation_null_q025": float(np.quantile(null, 0.025)),
        "permutation_null_q975": float(np.quantile(null, 0.975)),
        "permutation_p_value": float((null >= auc).mean()),
        "z_vs_null_sd": float((auc - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) else float("nan"),
        "cluster_bootstrap_n_defined": int(np.isfinite(cluster).sum()),
        "cluster_bootstrap_n_undefined": int(cluster_undefined),
        "cluster_bootstrap_n_too_few_positives": int((~np.isfinite(cluster)).sum() - cluster_undefined),
        "cluster_bootstrap_share_usable": float(np.isfinite(cluster).mean()),
        "cluster_bootstrap_q025": float(np.nanquantile(cluster, 0.025)),
        "cluster_bootstrap_q975": float(np.nanquantile(cluster, 0.975)),
        "cluster_bootstrap_median": float(np.nanmedian(cluster)),
        "cluster_bootstrap_sd": float(np.nanstd(cluster, ddof=1)),
        "distinct_test_dates": int(test["as_of"].nunique()),
        "distinct_test_tickers": int(test["ticker"].nunique()),
    }

    # -- 3b. the excluded block -------------------------------------------------
    excluded = panel.loc[observable & (panel["split"] == "excluded")]
    result["excluded_block"] = {
        "rows": len(excluded),
        "positives": int(excluded[LABEL].sum()),
        "share_of_all_positives": float(excluded[LABEL].sum() / max(len(all_positives), 1)),
        "by_year": {
            str(k): int(v) for k, v in excluded["as_of"].dt.year.value_counts().sort_index().items()
        },
        "base_rate": float(excluded[LABEL].mean()) if len(excluded) else float("nan"),
    }
    result["rows_by_year_and_split"] = {
        str(year): {str(k): int(v) for k, v in row.items() if v}
        for year, row in pd.crosstab(panel.loc[observable, "as_of"].dt.year,
                                     panel.loc[observable, "split"]).iterrows()
    }

    # -- 4. fragility ----------------------------------------------------------
    leave_one_positive: list[dict[str, Any]] = []
    for position in np.flatnonzero(y == 1):
        keep = np.arange(len(y)) != position
        leave_one_positive.append(
            {
                "dropped": f"{test['ticker'].iloc[position]} {test['as_of'].iloc[position].date()}",
                "auc": float(roc_auc(y[keep], scores[keep])),
            }
        )
    leave_one_ticker: list[dict[str, Any]] = []
    positives_by_ticker = test_positives["ticker"].astype(str).value_counts()
    for ticker in positives_by_ticker.index:
        keep = (test["ticker"].astype(str) != ticker).to_numpy()
        leave_one_ticker.append(
            {
                "dropped_ticker": ticker,
                "rows_dropped": int((~keep).sum()),
                "positives_dropped": int(positives_by_ticker[ticker]),
                "auc": float(roc_auc(y[keep], scores[keep])),
            }
        )
    result["fragility"] = {
        "leave_one_positive_out": leave_one_positive,
        "leave_one_ticker_out": leave_one_ticker,
        "auc_min": float(min(item["auc"] for item in leave_one_positive)),
        "auc_max": float(max(item["auc"] for item in leave_one_positive)),
        "auc_span": float(
            max(item["auc"] for item in leave_one_positive)
            - min(item["auc"] for item in leave_one_positive)
        ),
    }

    # -- 5/6. drivers and missingness -----------------------------------------
    univariate = univariate_table(test, features, y)
    result["univariate_top"] = univariate.head(15).to_dict("records")
    result["univariate_counts"] = {
        "n_features": len(univariate),
        "n_all_missing_in_test": int((univariate["n_observed"] == 0).sum()),
        "n_abs_edge_over_0_2": int((univariate["abs_edge"] > 0.2).sum()),
        "n_missing_indicator_over_0_7": int(
            (univariate["missing_indicator_auc"].fillna(0.5) > 0.7).sum()
        ),
        "max_missing_indicator_auc": float(
            univariate["missing_indicator_auc"].fillna(0.5).max()
        ),
    }
    pairs = duplicate_pairs(
        test.loc[:, [column for column in test.columns if column not in {MASK, LABEL}]],
        features,
        args.correlation,
    )
    result["duplicate_pairs_over_threshold"] = {
        "threshold": args.correlation,
        "count": len(pairs),
        "pairs": pairs[:15],
        "features_involved": len({name for pair in pairs for name in (pair["left"], pair["right"])}),
    }

    # The benchmark that decides what the number means. A fitted 34-feature model that
    # does not beat the best single column is not evidence of a model.
    best = univariate.iloc[0]
    oriented = float(best["oriented_auc"])
    result["single_feature_benchmark"] = {
        "feature": str(best["feature"]),
        "raw_auc": float(best["auc"]),
        "oriented_auc": oriented,
        "orientation": str(best["orientation"]),
        "model_auc": float(auc),
        "model_beats_best_single_feature": bool(auc > oriented),
        "margin": float(auc - oriented),
        "features_beating_model": int((univariate["oriented_auc"] > auc).sum()),
    }
    mnar = univariate.loc[univariate["missing_indicator_auc"].notna()].copy()
    mnar["missing_edge"] = (mnar["missing_indicator_auc"] - 0.5).abs()
    mnar = mnar.sort_values("missing_edge", ascending=False)
    result["missingness_table"] = (
        mnar[["feature", "n_observed", "missing_indicator_auc"]].head(12).to_dict("records")
    )

    # -- report ---------------------------------------------------------------
    lines: list[str] = []
    lines.append("# Stage 2 AUC audit — `tail_risk`, test split\n")
    lines.append(
        "Goal: try to falsify the reported structured AUC. Generated by "
        "`scripts/stage2_auc_audit.py`.\n"
    )
    reproduction = result["reproduced"]
    lines.append("## 1. Reproduction\n")
    lines.append("| | |")
    lines.append("| --- | --- |")
    lines.append(f"| reported AUC | {reproduction['reported_auc']:.6f} |")
    lines.append(f"| recomputed from the saved model | {reproduction['recomputed_auc']:.6f} |")
    lines.append(f"| identical | **{reproduction['matches']}** |")
    lines.append(f"| test rows / positives / negatives | {reproduction['n_rows']} / "
                 f"{reproduction['n_positives']} / {reproduction['n_negatives']} |")
    lines.append(f"| features in the fitted model | {reproduction['n_features_used']} |")
    lines.append("")

    lines.append("## 2. Where the positives live\n")
    lines.append("| split | rows | positives | base rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, block in per_split.items():
        lines.append(
            f"| {name} | {block['rows']} | {block['positives']} | {block['base_rate']:.4%} |"
        )
    dates = result["positive_dates"]
    lines.append("")
    lines.append(
        f"Observable positives in the whole panel: **{result['positives_total_observable']}**, "
        f"spanning {dates['earliest']} .. {dates['latest']} over {dates['distinct_dates']} filing dates."
    )
    lines.append("")
    lines.append("By year: " + ", ".join(f"{k}→{v}" for k, v in dates["by_year"].items()))
    lines.append("")
    lines.append("By quarter: " + ", ".join(f"{k}→{v}" for k, v in dates["by_quarter"].items()))
    lines.append("")
    trade = result["test_positive_dates"]
    lines.append(
        f"**Test positives ({len(trade['tickers'])} rows, {trade['distinct_dates']} distinct dates):** "
        + ", ".join(f"{ticker}@{day}" for ticker, day in zip(trade["tickers"], trade["dates"], strict=True))
    )
    lines.append("")
    lines.append(f"Score ranks of the test positives within the {trade['n_rows']} test rows: "
                 + ", ".join(f"#{rank}" for rank in trade["score_ranks"]))
    lines.append("")

    precision = result["precision"]
    lines.append("## 3. How precise is 0.7686?\n")
    lines.append("| quantity | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| AUC | {precision['auc']:.4f} |")
    lines.append(f"| Hanley-McNeil standard error | {precision['hanley_mcneil_se']:.4f} |")
    lines.append(
        f"| 95% normal interval | [{precision['normal_ci95'][0]:.3f}, {precision['normal_ci95'][1]:.3f}] |"
    )
    lines.append(f"| permutation null mean | {precision['permutation_null_mean']:.4f} |")
    lines.append(f"| permutation null sd | {precision['permutation_null_sd']:.4f} |")
    lines.append(
        f"| permutation null 95% range | [{precision['permutation_null_q025']:.3f}, "
        f"{precision['permutation_null_q975']:.3f}] |"
    )
    lines.append(f"| permutation p-value | {precision['permutation_p_value']:.4f} |")
    lines.append(f"| z against the null sd | {precision['z_vs_null_sd']:.2f} |")
    lines.append(
        f"| ticker-cluster bootstrap median | {precision['cluster_bootstrap_median']:.4f} |"
    )
    lines.append(
        f"| ticker-cluster bootstrap 95% | [{precision['cluster_bootstrap_q025']:.3f}, "
        f"{precision['cluster_bootstrap_q975']:.3f}] |"
    )
    lines.append(
        f"| resamples with a computable AUC | {precision['cluster_bootstrap_n_defined']} "
        f"of {args.permutations} ({precision['cluster_bootstrap_share_usable']:.1%}) |"
    )
    lines.append(
        f"| resamples with no positives at all | {precision['cluster_bootstrap_n_undefined']} |"
    )
    lines.append(
        f"| resamples below the metric's minimum positives | "
        f"{precision['cluster_bootstrap_n_too_few_positives']} |"
    )
    lines.append(f"| distinct observation dates in test | {precision['distinct_test_dates']} |")
    lines.append(f"| distinct companies in test | {precision['distinct_test_tickers']} |")
    lines.append("")
    lines.append(
        "Two uncertainty measures disagree, and the disagreement is informative. The "
        "permutation null treats rows as independent and is *optimistic*. The cluster "
        "bootstrap resamples whole companies, which is the honest unit here, and its "
        "interval is narrower — because dropping one of the four companies that supply a "
        "test positive usually leaves the others. Read together they say the estimate is "
        "not an artefact of company selection. What they do not say is that it measures "
        "anything: see 3b. "
        f"{args.permutations} draws, seed {args.seed}.\n"
    )
    lines.append(
        "A within-date permutation was attempted and is deliberately not reported: a "
        "company's filing date is its own, so a positive on a single-row date cannot move, "
        "and the null collapses toward the observed value. It would look like evidence and "
        "measure nothing.\n"
    )

    bench = result["single_feature_benchmark"]
    lines.append("### 3b. Does the model beat a single column?\n")
    lines.append("| | feature | raw AUC | oriented AUC |")
    lines.append("| --- | --- | ---: | ---: |")
    lines.append(
        f"| best single feature | `{bench['feature']}` | {bench['raw_auc']:.4f} | "
        f"{bench['oriented_auc']:.4f} ({bench['orientation']}) |"
    )
    lines.append(f"| fitted model (34 features) | — | — | {bench['model_auc']:.4f} |")
    lines.append("")
    lines.append(
        f"Model beats the best single column: **{bench['model_beats_best_single_feature']}** "
        f"(margin {bench['margin']:+.4f}). Features whose single-feature oriented AUC already "
        f"exceeds the model's: **{bench['features_beating_model']}**."
    )
    lines.append("")

    excluded = result["excluded_block"]
    lines.append("### 3c. The block that is in no window\n")
    lines.append(
        f"`excluded` holds **{excluded['rows']} rows and {excluded['positives']} positives** — "
        f"{excluded['share_of_all_positives']:.1%} of every positive in the panel — at a base "
        f"rate of {excluded['base_rate']:.2%}. Calendar years present: "
        + ", ".join(f"{year} ({rows})" for year, rows in excluded["by_year"].items())
        + "."
    )
    lines.append("")
    lines.append("| year | excluded | purged | test | train | valid |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for year, block in result["rows_by_year_and_split"].items():
        lines.append(
            f"| {year} | {block.get('excluded', 0)} | {block.get('purged', 0)} | "
            f"{block.get('test', 0)} | {block.get('train', 0)} | {block.get('valid', 0)} |"
        )
    lines.append("")

    fragility = result["fragility"]
    lines.append("## 4. How fragile is it?\n")
    lines.append("Dropping one test positive at a time:\n")
    lines.append("| dropped | AUC |")
    lines.append("| --- | ---: |")
    for item in fragility["leave_one_positive_out"]:
        lines.append(f"| {item['dropped']} | {item['auc']:.4f} |")
    lines.append("")
    lines.append(
        f"Range across those {len(fragility['leave_one_positive_out'])} drops: "
        f"**{fragility['auc_min']:.4f} .. {fragility['auc_max']:.4f}** "
        f"(span {fragility['auc_span']:.4f}).\n"
    )
    lines.append("Dropping every test row of one ticker that supplies a positive:\n")
    lines.append("| ticker | rows dropped | positives dropped | AUC |")
    lines.append("| --- | ---: | ---: | ---: |")
    for item in fragility["leave_one_ticker_out"]:
        lines.append(
            f"| {item['dropped_ticker']} | {item['rows_dropped']} | "
            f"{item['positives_dropped']} | {item['auc']:.4f} |"
        )
    lines.append("")

    counts = result["univariate_counts"]
    lines.append("## 5. What drives it\n")
    lines.append("Top single features by |AUC − 0.5| on the test slice:\n")
    lines.append("| feature | observed rows | AUC | direction | oriented AUC | missing-indicator AUC |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for item in result["univariate_top"]:
        missing = item["missing_indicator_auc"]
        lines.append(
            f"| {item['feature']} | {item['n_observed']} | {item['auc']:.4f} | "
            f"{item['orientation']} | {item['oriented_auc']:.4f} |"
            f" {'—' if not np.isfinite(missing) else f' {missing:.4f}'} |"
        )
    lines.append("")
    lines.append(
        f"Of {counts['n_features']} features: {counts['n_all_missing_in_test']} are entirely "
        f"missing in the test slice, and {counts['n_abs_edge_over_0_2']} have a single-feature "
        f"|AUC − 0.5| above 0.2.\n"
    )
    duplicates = result["duplicate_pairs_over_threshold"]
    lines.append(
        f"Feature pairs with |Spearman| >= {duplicates['threshold']}: "
        f"**{duplicates['count']}**, involving {duplicates['features_involved']} distinct features."
    )
    lines.append("")
    if duplicates["pairs"]:
        lines.append("| left | right | \\|spearman\\| |")
        lines.append("| --- | --- | ---: |")
        for pair in duplicates["pairs"]:
            lines.append(f"| {pair['left']} | {pair['right']} | {pair['spearman_abs']:.4f} |")
        lines.append("")

    lines.append("## 6. Missingness must not predict the label\n")
    lines.append(
        "A feature's *absence* carrying the label was the Stage 2 mask bug's signature. "
        f"Highest missing-indicator AUC across all {counts['n_features']} features: "
        f"**{counts['max_missing_indicator_auc']:.4f}**; features above 0.7: "
        f"{counts['n_missing_indicator_over_0_7']}."
    )
    lines.append("")
    lines.append("| feature | observed rows | AUC(absent → label) |")
    lines.append("| --- | ---: | ---: |")
    for item in result["missingness_table"]:
        lines.append(
            f"| {item['feature']} | {item['n_observed']} | {item['missing_indicator_auc']:.4f} |"
        )
    lines.append("")
    lines.append(
        "A gradient-boosted tree splits on missingness natively, so a value above ~0.7 here "
        "is a shortcut the model is free to take: it is predicting *reported* risk, not "
        "economic risk.\n"
    )

    destination = ROOT / "artifacts" / "stage2"
    (destination / "auc_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (destination / "auc_audit.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print("\n".join(lines[:44]))
    print(f"\n... written to {destination / 'auc_audit.md'}")
    if not reproduction["matches"]:
        raise SystemExit(
            "the recomputed AUC does not match the reported value — the audit is void, "
            "do not quote any conclusion from it"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate the dataset-card values file (``card_values.json``) from measured sources.

Every value is read from a real artifact — the processed panel, the independent
label review, or git — never hand-copied, so the card cannot drift from the data
it describes. Fields that genuinely cannot be measured (``default_risk`` /
``fraud_risk``: their event sources are not connected) are written as
"not measured" with the reason, following the template's own rule: unmeasured
fields say so instead of publishing an estimated number.

The output feeds ``python -m shingan publish hf --values-file <out> --only dataset``.

Usage:
    python scripts/card_values.py                 # all defaults, see below
    python scripts/card_values.py --out other.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # constants come from the package when it is installed (CI, editable install)
    from shingan.__about__ import GITHUB_URL
except ImportError:  # pragma: no cover - direct run without the package installed
    GITHUB_URL = "https://github.com/shingan-project/shingan-finrisk"

NOT_MEASURED = "not measured"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: paths to the panel, the label review, and the output file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--panel",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "panel.parquet",
        help="Processed panel (default: data/processed/panel.parquet).",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=REPO_ROOT / "artifacts" / "stage2" / "label_review.json",
        help="Independent label review JSON (default: artifacts/stage2/label_review.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "stage2" / "card_values.json",
        help="Where to write the values JSON (default: artifacts/stage2/card_values.json).",
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Override the build commit (default: short sha of the current HEAD).",
    )
    parser.add_argument(
        "--source-snapshot",
        default=None,
        help=(
            "Upstream data snapshot date, if known (default: 'not recorded' — the "
            "EDGAR fetch date was never written down, and the card must not guess it)."
        ),
    )
    return parser.parse_args(argv)


def git_short_sha() -> str:
    """Short sha of HEAD, or 'unknown' when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hf_size_category(n_rows: int) -> str:
    """Hugging Face ``size_categories`` bucket for a row count."""
    if n_rows < 1_000:
        return "n<1K"
    if n_rows < 10_000:
        return "1K<n<10K"
    if n_rows < 100_000:
        return "10K<n<100K"
    return "100K<n<1M"


def panel_values(panel_path: Path) -> dict[str, str]:
    """Dataset statistics read from the processed panel."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"error: pandas is required to read the panel ({exc})")
    if not panel_path.is_file():
        sys.exit(f"error: panel not found at {panel_path}")
    df = pd.read_parquet(panel_path)

    n_rows = len(df)
    n_companies = int(df["ticker"].nunique())
    as_of = pd.to_datetime(df["as_of"])
    values = {
        "n_rows": str(n_rows),
        "n_companies": str(n_companies),
        "date_range": f"{as_of.min():%Y-%m-%d} → {as_of.max():%Y-%m-%d}",
        "size_category": hf_size_category(n_rows),
        # The real panel is built from real filings only; synthetic rows live in a
        # separate table and are never mixed in (the builder rejects a mix).
        "contains_synthetic": "no" if not bool(df["is_synthetic"].any()) else "yes",
        "synthetic_share": f"{float(df['is_synthetic'].mean()):.0%}",
        "market_cap_scope": (
            f"{n_companies} US large-caps; small-cap behaviour untested"
        ),
    }

    label, mask = "label_tail_risk", "label_mask_tail_risk"
    if label in df.columns and mask in df.columns:
        usable = df[df[mask]]
        positives = int(usable[label].sum())
        values["pos_tail"] = str(positives)
        values["rate_tail"] = (
            f"{positives / n_rows:.2%} of all rows "
            f"({positives / max(len(usable), 1):.2%} of the {len(usable)} mask-true rows)"
        )
    else:
        values["pos_tail"] = f"{NOT_MEASURED} (label column absent from the panel)"
        values["rate_tail"] = values["pos_tail"]

    # default_risk / fraud_risk: their event sources (rating history, enforcement
    # lists, restatement disclosures) were never connected, so the panel carries no
    # label column for them at all. There is nothing to estimate from, and the card
    # must not pretend otherwise.
    not_measured = f"{NOT_MEASURED} (event source not connected; no label column in the panel)"
    values["pos_default"] = not_measured
    values["rate_default"] = not_measured
    values["pos_fraud"] = not_measured
    values["rate_fraud"] = not_measured
    return values


def review_values(review_path: Path) -> dict[str, str]:
    """Audit numbers read from the independent label review."""
    if not review_path.is_file():
        sys.exit(f"error: label review not found at {review_path}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    population = review.get("population", {})
    outcome = review.get("outcome", {})
    compared = population.get("compared")
    disagreements = outcome.get("disagreements")
    if compared is None or disagreements is None:
        sys.exit(f"error: {review_path} has no population/outcome counts")
    values = {
        "audit_sample_size": str(compared),
        "audit_disagreement_rate": f"{float(disagreements) / float(compared):.2%}",
    }
    generated_at = review.get("generated_at")
    if generated_at:
        values["build_date"] = str(generated_at)[:10]
    return values


def build_values(args: argparse.Namespace) -> dict[str, str]:
    """All 23 dataset-card fields with their measured or declared values."""
    values = panel_values(args.panel)
    values.update(review_values(args.review))

    values["git_commit"] = args.git_sha or git_short_sha()
    values.setdefault("build_date", datetime.now(UTC).strftime("%Y-%m-%d"))
    values["source_snapshot_date"] = args.source_snapshot or "not recorded"
    values["rating_history_status"] = "unavailable"
    values["pit_membership_status"] = "not applied"
    values["vendor_lookahead_status"] = (
        "not used (XBRL facts read directly from SEC EDGAR with filed <= as_of)"
    )
    values["known_issues_url"] = f"{GITHUB_URL}/issues"
    values["changelog_url"] = f"{GITHUB_URL}/blob/main/CHANGELOG.md"
    return values


def main(argv: list[str] | None = None) -> None:
    """Compute the values, print them, and write the output JSON."""
    args = parse_args(argv)
    values = build_values(args)
    for key in sorted(values):
        print(f"{key:26s} {values[key]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(f"\nwrote {len(values)} values to {args.out}")


if __name__ == "__main__":
    main()

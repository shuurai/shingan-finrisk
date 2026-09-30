"""Report whether the universe expansion delivered what docs/11 section 8 asked for.

The acceptance criteria are stated as numbers in the decision document, so the point of
this script is that each one is *decided* against the built panel rather than asserted in
prose. Two rules from this repository's own history shape it:

* **A displayed value and a verdict must not be able to disagree.** Every verdict is
  derived from the same frame that is printed, and `passed=None` means "not measurable",
  never "failed" — a block with no positives has an unknown rate, not a bad one.
* **The test block must be reported per year.** Its positives are concentrated in one
  regime, and a single test figure hides that. The per-year table is printed whether or
  not the totals pass.

Usage::

    python scripts/audit_panel.py
    python scripts/audit_panel.py --panel data/processed/stage2_34names/panel.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Thresholds from docs/11-universe-expansion.md section 8. Quoted values are the
#: pre-expansion baseline, kept so the report shows the movement rather than only the bar.
THRESHOLDS = {
    "observable_rows": 10_000,
    "positives_total": 300,
    "train_positives": 60,
    "test_positives": 150,
}
BASELINE = {
    "observable_rows": 1_778,
    "positives_total": 39,
    "train_positives": 4,
    "test_positives": 32,
}


@dataclass(slots=True)
class Verdict:
    """One acceptance criterion, decided."""

    name: str
    value: int | None
    target: int
    baseline: int

    @property
    def passed(self) -> bool | None:
        """``None`` when the value could not be measured, which is not a failure."""
        if self.value is None:
            return None
        return self.value >= self.target

    def line(self) -> str:
        if self.value is None:
            state = "NOT MEASURED"
            shown = "n/a"
        else:
            state = "pass" if self.passed else "FAIL"
            shown = f"{self.value:,}"
        return (
            f"  [{state:>12s}] {self.name:22s} {shown:>9s} / {self.target:,} "
            f"(was {self.baseline:,})"
        )


def acceptance_verdicts(frame) -> list[Verdict]:
    """Decide docs/11 section 8 criteria against a built panel.

    ``frame`` needs ``split``, ``label_tail_risk`` and ``label_mask_tail_risk``. Kept
    separate from the printing so a test can pin the decision rule without a real panel.
    """
    observable = frame[frame["label_mask_tail_risk"] == 1]
    train = observable[observable["split"] == "train"]
    test = observable[observable["split"] == "test"]
    return [
        Verdict("observable rows", int(len(observable)), THRESHOLDS["observable_rows"], BASELINE["observable_rows"]),
        Verdict(
            "positives (all)",
            int(observable["label_tail_risk"].sum()),
            THRESHOLDS["positives_total"],
            BASELINE["positives_total"],
        ),
        Verdict(
            "train positives",
            int(train["label_tail_risk"].sum()),
            THRESHOLDS["train_positives"],
            BASELINE["train_positives"],
        ),
        Verdict(
            "test positives",
            int(test["label_tail_risk"].sum()),
            THRESHOLDS["test_positives"],
            BASELINE["test_positives"],
        ),
    ]


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "processed" / "panel.parquet")
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    if "is_synthetic" in frame.columns and bool(frame["is_synthetic"].any()):
        print(f"WARNING: {args.panel} holds synthetic rows; these numbers mean nothing about reality")
    print(f"panel: {args.panel}  ({len(frame):,} rows, {frame['ticker'].nunique()} tickers)")
    print(f"window: {frame['as_of'].min()} .. {frame['as_of'].max()}\n")

    print("split blocks (observable rows only):")
    observable = frame[frame["label_mask_tail_risk"] == 1]
    for split in ("train", "valid", "test", "purged", "excluded"):
        block = observable[observable["split"] == split]
        positives = int(block["label_tail_risk"].sum())
        rate = f"{positives / len(block):.3%}" if len(block) else "n/a"
        print(f"  {split:9s} {len(block):>7,} rows   {positives:>5,} positives   {rate:>8s}")

    print("\ntest positives by year (the block is regime-concentrated; this is why):")
    test = observable[observable["split"] == "test"].copy()
    if len(test):
        test["year"] = test["as_of"].astype(str).str.slice(0, 4)
        by_year = test.groupby("year")["label_tail_risk"].agg(["size", "sum"])
        for year, row in by_year.iterrows():
            share = row["sum"] / max(int(by_year["sum"].sum()), 1)
            print(f"  {year}  {int(row['size']):>6,} rows   {int(row['sum']):>4,} positives   {share:6.1%} of test positives")

    print("\nacceptance (docs/11 section 8):")
    for verdict in acceptance_verdicts(frame):
        print(verdict.line())
    print(
        "\nNot covered here: criterion 5 (the retrained LoRA arm must stop being a constant "
        "answer) needs an SFT run and lives in docs/09."
    )


if __name__ == "__main__":
    main()

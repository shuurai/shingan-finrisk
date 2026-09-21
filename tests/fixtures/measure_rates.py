"""Scratch harness: build the panel and print realised label base rates.

Not a test. Run directly to tune the synthetic generator's calibration constants
against the bands in ``shingan.data.synthetic.TARGET_BASE_RATES``:

    PYTHONPATH=src python tests/fixtures/measure_rates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shingan.config import default_config
from shingan.data.builder import build_panel
from shingan.data.synthetic import TARGET_BASE_RATES
from shingan.labeling.definitions import build_label_definitions

FRAUD_KINDS = {"restatement", "enforcement_action", "adverse_audit_opinion"}


def main() -> int:
    config = default_config()
    n_universe = len(config.data.universe)
    print("data window:", config.data.start, "->", config.data.end)
    print("rows_per_year:", config.data.synthetic.rows_per_year)
    print("universe:", n_universe, "tickers -> generator makes", min(12, n_universe or 12))

    result = build_panel(config, write=False)
    panel = result.panel
    events = result.events

    print("\n-- events that fed the panel --")
    print("count:", len(events), "| by kind:", events["event_kind"].value_counts().to_dict())
    print("by source:", events["source"].value_counts().to_dict())
    fraud = events.loc[events["event_kind"].isin(FRAUD_KINDS)]
    print(
        "fraud-kind events:",
        len(fraud),
        "| injected:",
        int((fraud["source"] == "synthetic_injected").sum()),
    )

    print("\n-- panel --")
    print("shape:", panel.shape)
    print("rows per ticker:", panel.groupby("ticker").size().to_dict())
    print("split sizes:", panel["split"].value_counts().to_dict())

    definitions = build_label_definitions(config.labels)
    print("\n-- realised base rates (observable rows only) --")
    print(f"{'label':<14}{'obs':>7}{'pos':>7}{'rate':>10}   {'target band':<20} verdict")
    failures = 0
    for label, _definition in definitions.items():
        name = str(label)
        column = f"label_{name}"
        mask = panel[f"label_mask_{name}"].to_numpy(dtype=bool)
        if column not in panel.columns:
            print(
                f"{name:<14}{'--':>7}{'--':>7}{'--':>10}   {TARGET_BASE_RATES[name]}   MISSING COLUMN"
            )
            failures += 1
            continue
        values = pd.to_numeric(panel[column], errors="coerce").fillna(0).to_numpy()
        observable = int(mask.sum())
        positives = int(values[mask].sum())
        rate = positives / observable if observable else float("nan")
        low, high = TARGET_BASE_RATES[name]
        ok = low <= rate <= high
        failures += 0 if ok else 1
        print(
            f"{name:<14}{observable:>7}{positives:>7}{rate:>10.4f}   "
            f"[{low:.3f}, {high:.3f}]{'':<4} {'in band' if ok else 'OUT OF BAND'}"
        )

    print("\n-- positives per split (observable) --")
    for label, _definition in definitions.items():
        name = str(label)
        row = []
        for split in ("train", "valid", "test", "purged"):
            subset = panel.loc[panel["split"] == split]
            observable = subset[f"label_mask_{name}"].to_numpy(dtype=bool)
            positives = int(
                pd.to_numeric(subset[f"label_{name}"], errors="coerce")
                .fillna(0)
                .to_numpy()[observable]
                .sum()
            )
            row.append(f"{split}={int(observable.sum())}/{positives}")
        print(f"{name:<14}" + "  ".join(row))

    print("\nfailures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

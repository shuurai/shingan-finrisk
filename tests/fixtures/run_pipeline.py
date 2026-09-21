"""End-to-end integration check: panel -> train -> calibrate -> evaluate.

Not a test (yet). It is the skeleton of the demo flow and the place to see how the
pipeline degrades when a label has too few positives to say anything about.

    PYTHONPATH=src python tests/fixtures/run_pipeline.py
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shingan.config import default_config
from shingan.data.builder import build_panel
from shingan.eval.metrics import evaluate_classification
from shingan.labeling.definitions import build_label_definitions
from shingan.models.structured import StructuredRiskModel

#: Columns that describe provenance or bookkeeping rather than the company's state.
#: Passing any of these to the model would be a leak or a constant.
NON_FEATURE_COLUMNS = frozenset(
    {
        "as_of",
        "cik",
        "company_name",
        "data_version",
        "insufficient_history",
        "is_synthetic",
        "n_sources",
        "sample_weight",
        "sector",
        "split",
        "ticker",
    }
)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Numeric columns that are neither labels, forward targets, nor bookkeeping."""
    selected: list[str] = []
    for column in panel.columns:
        if column.startswith(("label_", "fwd_", "event_", "source_of_record_", "horizon_days_")):
            continue
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(panel[column]) or pd.api.types.is_bool_dtype(
            panel[column]
        ):
            selected.append(column)
    return selected


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    config = default_config()
    result = build_panel(config, write=False)
    panel = result.panel
    definitions = build_label_definitions(config.labels)
    features = feature_columns(panel)

    print(
        f"panel {panel.shape} | {len(features)} features | splits {panel['split'].value_counts().to_dict()}"
    )
    windows = result.split_report.windows
    print(
        f"effective windows: train {windows.effective_train.render()} | "
        f"valid {windows.effective_valid.render()} | test {windows.effective_test.render()}"
    )
    print(f"walk-forward folds: {len(result.split_report.folds)}")
    print()

    n_failed = 0
    for label, _definition in definitions.items():
        name = str(label)
        mask_column = f"label_mask_{name}"
        label_column = f"label_{name}"
        print(f"=== {name} ===")

        subsets: dict[str, pd.DataFrame] = {}
        for split in ("train", "valid", "test"):
            rows = panel.loc[(panel["split"] == split) & panel[mask_column].astype(bool)]
            subsets[split] = rows
            positives = int(pd.to_numeric(rows[label_column], errors="coerce").fillna(0).sum())
            print(f"  {split:<6} rows={len(rows):>4} positives={positives:>3}")

        try:
            model = StructuredRiskModel(config.structured)
            model.fit(
                subsets["train"][features],
                subsets["train"][label_column].to_numpy(),
                subsets["valid"][features],
                subsets["valid"][label_column].to_numpy(),
            )
        except Exception as exc:
            n_failed += 1
            print(f"  FIT FAILED: {type(exc).__name__}: {exc}")
            print()
            if "-v" in sys.argv:
                traceback.print_exc()
            continue

        scores = model.predict_proba(subsets["test"][features])
        report = evaluate_classification(
            subsets["test"][label_column].to_numpy(),
            scores,
            path="structured",
            label=name,
            split="test",
        )
        calibration = report.calibration
        ece = calibration.ece if calibration else float("nan")
        print(
            f"  test: n={report.n_rows} pos={report.n_positives} base={report.base_rate:.4f} "
            f"auc={report.auc:.4f} ks={report.ks:.4f} pr_auc={report.pr_auc:.4f} ece={ece:.4f}"
        )
        print(f"  calibration: {model.calibration_report}")
        gates = report.gates()
        failing = [k for k, v in gates.items() if v is False]
        unknown = [k for k, v in gates.items() if v is None]
        print(f"  gates failing={failing} undecidable={unknown}")
        print()

    print("labels that could not be fitted:", n_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

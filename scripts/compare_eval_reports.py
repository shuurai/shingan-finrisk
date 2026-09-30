"""Put two evaluation reports side by side, and say which way each number moved.

An evaluation report answers "what did this run score". It does not answer "did the
change I just made help", which is the question a feature or window change is actually
asked. Reading two reports by eye is where a real movement gets confused with sampling
noise, so this script prints the difference and, before anything else, whether the two
runs are even comparable.

Four rules, each from something that already went wrong in this repository:

* **A delta across different splits describes two datasets, not an improvement.** The
  split definition is checked first and a mismatch is a refusal, not a footnote. The
  wide rebuild once moved `split.train.start` while the evaluation block stayed put; a
  comparison against the prior report would have looked like a clean gain.
* **A different positive count in the test block is a sample change.** It is printed
  next to every delta rather than left implicit, because at a few hundred positives a
  few rows of movement is worth more than several decimal places of PR-AUC.
* **Unmeasured stays unmeasured.** When a metric exists on one side only, the delta is
  reported as `not measured` with the reason. It is never rendered as zero, and never as
  a failure.
* **A year the metric module refuses to score is not scored here either.** `roc_auc`
  returns NaN below its own minimum class counts, and the report prints "not measured"
  on the strength of it. A comparison that applied a weaker rule would display a number
  for a fold the report calls undecidable, which is the shape this repository treats as
  a bug rather than a rounding difference.

Usage::

    python scripts/compare_eval_reports.py \\
        --baseline data/processed/wide_stage2_premarket/eval_20260930T092143Z.json \\
        --baseline-panel data/processed/wide_stage2_premarket/eval_panel.csv \\
        --candidate artifacts/eval/wide_stage2_mktctx/20260930T102842Z.json \\
        --candidate-panel artifacts/eval/wide_stage2_mktctx/panel.csv \\
        --focus-year 2020
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
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time convenience
    sys.path.insert(0, str(ROOT / "src"))

from shingan.eval.metrics import MIN_NEGATIVES, MIN_POSITIVES, roc_auc  # noqa: E402

#: Headline metrics where a movement is worth reading. `auc` first: it is the number the
#: report leads with, and the one most likely to be quoted without its interval.
HEADLINE_KEYS = ("auc", "ks", "pr_auc", "pr_auc_lift", "capture_top5", "ece", "brier_skill")

#: Fields that must agree for a delta to mean anything. `split_definition` decides which
#: rows are train/valid/test; `headline` carries the size of the sample they produced.
COMPARABILITY_KEYS = ("headline_label", "split_definition")

#: Columns the evaluation writes into its panel dump. Only the paths saved per row can be
#: broken out by year here; the text baseline is not persisted, which is why a per-year
#: text row can be absent even though the report has a text row overall.
SCORE_COLUMNS = {
    "structured": "score_structured_tail_risk",
    "fused": "score_fused_tail_risk",
}

LABEL_COLUMN = "label_tail_risk"
MASK_COLUMN = "label_mask_tail_risk"


def load_report(path: Path) -> dict[str, Any]:
    """Read one evaluation report JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _optional(value: Any) -> float | None:
    """``None`` unless ``value`` is a finite number.

    pandas turns ``None`` into ``nan`` inside a float column, and a ``nan`` AUC is a
    number to anything that reads the frame afterwards. Forcing the missing value back
    to a Python ``None`` in an object column is the only way "not measured" survives a
    round trip through a DataFrame.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _is_number(value: Any) -> bool:
    return _optional(value) is not None


def comparability(us: dict[str, Any], them: dict[str, Any]) -> list[str]:
    """Reasons the two reports may not be comparable. Empty list means they are.

    Differences are returned rather than raised so the caller can print them next to the
    deltas: a reader who is shown an incomparable pair and told so is better served than
    one whose script silently exits.
    """
    reasons: list[str] = []
    for key in COMPARABILITY_KEYS:
        if us.get(key) != them.get(key):
            reasons.append(f"`{key}` differs")
    ours = us.get("headline", {})
    theirs = them.get("headline", {})
    if ours.get("n_positives") != theirs.get("n_positives"):
        reasons.append(
            f"test positives differ ({theirs.get('n_positives')} -> {ours.get('n_positives')})"
        )
    if ours.get("n_rows") != theirs.get("n_rows"):
        reasons.append(f"test rows differ ({theirs.get('n_rows')} -> {ours.get('n_rows')})")
    return reasons


def scalar_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Before/after/delta for one flat dict of metrics.

    A key present on one side only yields ``delta=None``. Substituting 0.0 here is
    exactly how a missing metric becomes an apparent regression.
    """
    rows: list[dict[str, Any]] = []
    for key in keys:
        before = _optional(baseline.get(key))
        after = _optional(candidate.get(key))
        delta = None if before is None or after is None else after - before
        rows.append({"metric": key, "baseline": before, "candidate": after, "delta": delta})
    return rows


def per_path_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-path deltas, matched on the path name.

    Paths are matched rather than zipped by position: a report that gains or loses a path
    would otherwise compare `fused` against `text_baseline` and call it a movement.
    """
    before = {row["path"]: row for row in baseline.get("comparison", [])}
    after = {row["path"]: row for row in candidate.get("comparison", [])}
    out: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        left = before.get(path)
        right = after.get(path)
        row: dict[str, Any] = {
            "path": path,
            "present_baseline": left is not None,
            "present_candidate": right is not None,
        }
        for key in ("auc", "pr_auc", "ece", "brier_skill", "capture_top5"):
            before_value = None if left is None else _optional(left.get(key))
            after_value = None if right is None else _optional(right.get(key))
            row[f"{key}_baseline"] = before_value
            row[f"{key}_candidate"] = after_value
            row[f"{key}_delta"] = (
                None if before_value is None or after_value is None else after_value - before_value
            )
        out.append(row)
    return out


def gates_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Every gate, both verdicts, and whether the verdict changed.

    ``passed`` is three-valued in the report and stays three-valued here: a gate that is
    ``None`` on either side is `undecidable`, not `failed`.
    """
    before = {row["name"]: row for row in baseline.get("gates", [])}
    after = {row["name"]: row for row in candidate.get("gates", [])}
    out: list[dict[str, Any]] = []
    for name in sorted(set(before) | set(after)):
        left = before.get(name)
        right = after.get(name)
        left_passed = None if left is None else left.get("passed")
        right_passed = None if right is None else right.get("passed")
        if left_passed is None or right_passed is None:
            change = "undecidable"
        elif bool(left_passed) == bool(right_passed):
            change = "unchanged"
        else:
            change = "improved" if right_passed else "regressed"
        out.append(
            {
                "name": name,
                "baseline_passed": left_passed,
                "candidate_passed": right_passed,
                "baseline_achieved": None if left is None else left.get("achieved"),
                "candidate_achieved": None if right is None else right.get("achieved"),
                "target": (right or left or {}).get("target"),
                "change": change,
            }
        )
    return out


def falsification_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Every falsification condition, both verdicts, and whether it moved."""
    before = {row["code"]: row for row in baseline.get("falsification", [])}
    after = {row["code"]: row for row in candidate.get("falsification", [])}
    out: list[dict[str, Any]] = []
    for code in sorted(set(before) | set(after)):
        left = (before.get(code) or {}).get("verdict", "absent")
        right = (after.get(code) or {}).get("verdict", "absent")
        out.append(
            {
                "code": code,
                "condition": (after.get(code) or before.get(code) or {}).get("condition", ""),
                "baseline_verdict": left,
                "candidate_verdict": right,
                "changed": left != right,
                "candidate_detail": (after.get(code) or {}).get("detail", ""),
            }
        )
    return out


def fused_gain(report: dict[str, Any]) -> dict[str, Any]:
    """The fused-minus-structured PR-AUC, read out of the F1 falsification detail.

    Read from F1 rather than recomputed: F1 is the condition the project's central claim
    is decided on, and a second implementation of the same arithmetic is a second thing
    to disagree with the report.
    """
    for row in report.get("falsification", []):
        if row.get("code") == "F1":
            return {"verdict": row.get("verdict", "not_evaluated"), "detail": row.get("detail", "")}
    return {"verdict": "absent", "detail": "no F1 row in this report"}


def year_auc(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """AUC per calendar year on the evaluation block, plus rows and positives.

    A year needs a score to exist, and enough of each class for `roc_auc` to be defined
    at all. Years that fail are returned with ``auc=None`` and a reason naming the
    quantity that was missing, because "this year could not be scored" and "this year
    scored 0.5" are different statements and the difference is why the block is broken
    out by year in the first place.
    """
    block = frame.loc[frame[MASK_COLUMN] == 1].copy()
    block["year"] = pd.to_datetime(block["as_of"]).dt.year
    rows: list[dict[str, Any]] = []
    for year, group in block.groupby("year", sort=True):
        scores = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
        labels = group[LABEL_COLUMN].to_numpy(dtype=int)
        observed = np.isfinite(scores)
        positives = int(labels[observed].sum())
        negatives = int(observed.sum()) - positives
        if observed.sum() == 0:
            auc, reason = None, "no score"
        elif positives < MIN_POSITIVES:
            auc, reason = None, f"needs {MIN_POSITIVES} positives, has {positives}"
        elif negatives < MIN_NEGATIVES:
            auc, reason = None, f"needs {MIN_NEGATIVES} negatives, has {negatives}"
        else:
            auc, reason = _optional(roc_auc(labels[observed], scores[observed])), ""
        rows.append(
            {
                "year": int(year),
                "rows": len(group),
                "positives": int(labels.sum()),
                "base_rate": float(group[LABEL_COLUMN].mean()),
                "auc": auc,
                "reason": reason,
            }
        )
    frame_out = pd.DataFrame(rows)
    if not frame_out.empty:
        # Object dtype, or pandas coerces the `None` AUCs back to `nan` and every reader
        # downstream sees a number where the report says "not measured".
        frame_out["auc"] = pd.Series([_optional(value) for value in frame_out["auc"]], dtype=object)
    return frame_out


def year_comparison(
    baseline_panel: pd.DataFrame, candidate_panel: pd.DataFrame, path: str
) -> pd.DataFrame:
    """Per-year AUC before and after for one scoring path, with the delta.

    The canonical `rows`/`positives`/`base_rate` columns describe the year once, taking
    the candidate's view where it has one and falling back to the baseline's otherwise,
    so a year present in only one panel is still fully described rather than printed as
    a row of blanks.
    """
    column = SCORE_COLUMNS[path]
    left = year_auc(baseline_panel, column).rename(
        columns={name: f"{name}_baseline" for name in ("rows", "positives", "base_rate", "auc", "reason")}
    )
    right = year_auc(candidate_panel, column).rename(
        columns={name: f"{name}_candidate" for name in ("rows", "positives", "base_rate", "auc", "reason")}
    )
    frame = left.merge(right, on="year", how="outer").sort_values("year").reset_index(drop=True)

    for side in ("baseline", "candidate"):
        present = frame[f"rows_{side}"].notna()
        frame[f"reason_{side}"] = frame[f"reason_{side}"].where(present, "absent from panel")
        frame[f"auc_{side}"] = pd.Series(
            [_optional(value) for value in frame[f"auc_{side}"]], dtype=object
        )

    frame["auc_delta"] = pd.Series(
        [
            None if not (_is_number(before) and _is_number(after)) else float(after) - float(before)
            for before, after in zip(frame["auc_baseline"], frame["auc_candidate"], strict=True)
        ],
        dtype=object,
    )

    has_candidate = frame["rows_candidate"].notna()
    for name in ("rows", "positives", "base_rate"):
        frame[name] = frame[f"{name}_candidate"].where(has_candidate, frame[f"{name}_baseline"])
    return frame


def _number(value: Any, digits: int = 4) -> str:
    """A metric as text, with "not measured" for anything that is not a finite number.

    Three states, and only three: a number, `not measured`, or the string the report
    itself put there (`see backtest table` cannot be rendered as a float without
    inventing one). `nan` counts as not measured — pandas produces it from a genuinely
    absent value, so printing `n/a` next to it would be a distinction without a
    difference.
    """
    if isinstance(value, str):
        return value or "not measured"
    number = _optional(value)
    return "not measured" if number is None else f"{number:.{digits}f}"


def _signed(value: Any, digits: int = 4) -> str:
    number = _optional(value)
    if number is None:
        return "not measured"
    return f"{number:+.{digits}f}"


def _verdict_word(passed: Any) -> str:
    if passed is None:
        return "undecidable"
    return "pass" if passed else "FAIL"


def render(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_name: str,
    candidate_name: str,
    incomparable: list[str],
    headlines: list[dict[str, Any]],
    per_path: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    falsification: list[dict[str, Any]],
    year_tables: dict[str, pd.DataFrame],
    focus_year: int | None,
) -> str:
    """Render the whole comparison. Separate from the computation so it is testable."""
    lines: list[str] = []
    lines.append("# Evaluation comparison")
    lines.append("")
    lines.append(f"- baseline: `{baseline_name}` (run `{baseline.get('metadata', {}).get('run_id', '?')}`)")
    lines.append(f"- candidate: `{candidate_name}` (run `{candidate.get('metadata', {}).get('run_id', '?')}`)")
    lines.append("")

    lines.append("## 0. Are these two runs comparable?")
    lines.append("")
    if incomparable:
        lines.append("**No.** The differences below mean a delta describes two samples, not a change:")
        lines.append("")
        for reason in incomparable:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append(
            "Read the deltas as a description of two datasets. Do not quote any of them as "
            "the effect of the code change."
        )
    else:
        lines.append(
            "**Yes.** Headline label, split definition, test rows and test positives all agree, "
            "so a delta below is attributable to the change under test."
        )
    lines.append("")

    lines.append("## 1. Headline metrics")
    lines.append("")
    lines.append("| metric | baseline | candidate | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in headlines:
        lines.append(
            f"| {row['metric']} | {_number(row['baseline'])} | {_number(row['candidate'])} | "
            f"{_signed(row['delta'])} |"
        )
    lines.append("")

    lines.append("## 2. Per-path")
    lines.append("")
    lines.append(
        "| path | AUC before | AUC after | ΔAUC | PR-AUC before | PR-AUC after | ΔPR-AUC | capture@5 before | after |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in per_path:
        lines.append(
            f"| {row['path']} | {_number(row['auc_baseline'])} | {_number(row['auc_candidate'])} | "
            f"{_signed(row['auc_delta'])} | {_number(row['pr_auc_baseline'])} | "
            f"{_number(row['pr_auc_candidate'])} | {_signed(row['pr_auc_delta'])} | "
            f"{_number(row['capture_top5_baseline'])} | {_number(row['capture_top5_candidate'])} |"
        )
    lines.append("")

    lines.append("## 3. The central claim (F1: does fusion beat structured alone)")
    lines.append("")
    before_gain = fused_gain(baseline)
    after_gain = fused_gain(candidate)
    lines.append(f"- baseline: `{before_gain['verdict']}` — {before_gain['detail']}")
    lines.append(f"- candidate: `{after_gain['verdict']}` — {after_gain['detail']}")
    lines.append("")
    if before_gain["verdict"] == after_gain["verdict"]:
        lines.append(
            f"The verdict is unchanged (`{after_gain['verdict']}`). A change in the inputs did not "
            "move the answer the project is actually asking."
        )
    else:
        lines.append(
            f"The verdict moved from `{before_gain['verdict']}` to `{after_gain['verdict']}`. "
            "Check the interval, not the point estimate, before acting on it."
        )
    lines.append("")

    lines.append("## 4. Gates")
    lines.append("")
    lines.append("| gate | target | baseline | candidate | change |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in gates:
        lines.append(
            f"| {row['name']} | {row['target']} | {_verdict_word(row['baseline_passed'])} "
            f"({row['baseline_achieved']}) | {_verdict_word(row['candidate_passed'])} "
            f"({row['candidate_achieved']}) | {row['change']} |"
        )
    lines.append("")
    moved = [row for row in gates if row["change"] in {"improved", "regressed"}]
    undecidable = [row for row in gates if row["change"] == "undecidable"]
    lines.append(f"Verdicts that changed: **{len(moved)}**; undecidable on one side: {len(undecidable)}.")
    lines.append("")
    if undecidable:
        lines.append(
            "`undecidable` is not a failure. It means the inputs needed to decide the gate were "
            "absent on one side of the comparison."
        )
        lines.append("")

    lines.append("## 5. Falsification conditions")
    lines.append("")
    lines.append("| code | baseline | candidate | changed | detail |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in falsification:
        lines.append(
            f"| {row['code']} | {row['baseline_verdict']} | {row['candidate_verdict']} | "
            f"{'yes' if row['changed'] else 'no'} | {row['candidate_detail']} |"
        )
    lines.append("")

    lines.append("## 6. Per-year AUC")
    lines.append("")
    lines.append(
        "Computed from each run's own evaluation panel, not from the report: the report does not "
        "serialise the per-year table. Only paths whose score is persisted per row can be broken "
        "out here."
    )
    lines.append("")
    for path, table in year_tables.items():
        lines.append(f"### {path}")
        lines.append("")
        has_auc = pd.Series(
            [
                _is_number(before) or _is_number(after)
                for before, after in zip(table["auc_baseline"], table["auc_candidate"], strict=True)
            ]
        )
        scored = table.loc[has_auc]
        unscored = sorted(int(year) for year in table.loc[~has_auc, "year"])
        if unscored:
            lines.append(
                f"Years with no score in either run ({len(unscored)}): "
                + ", ".join(str(year) for year in unscored)
                + ". They carry a label but no prediction — the model is scored only on the "
                "evaluation block — so they cannot contribute a metric. This bounds what the "
                "table below says about stability over time."
            )
            lines.append("")
        lines.append("| year | rows | positives | base rate | AUC before | AUC after | ΔAUC | note |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for _, row in scored.iterrows():
            notes = []
            for side in ("baseline", "candidate"):
                reason = row.get(f"reason_{side}")
                if isinstance(reason, str) and reason:
                    notes.append(f"{side}: {reason}")
            lines.append(
                f"| {int(row['year'])} | {int(row['rows'])} | {int(row['positives'])} | "
                f"{float(row['base_rate']):.4%} | {_number(row['auc_baseline'])} | "
                f"{_number(row['auc_candidate'])} | {_signed(row['auc_delta'])} | "
                f"{'; '.join(notes)} |"
            )
        lines.append("")

    if focus_year is not None:
        lines.append(f"### Focus year: {focus_year}")
        lines.append("")
        found = False
        for path, table in year_tables.items():
            match = table.loc[table["year"] == focus_year]
            if match.empty:
                continue
            found = True
            row = match.iloc[0]
            lines.append(
                f"- {path}: AUC {_number(row['auc_baseline'])} -> {_number(row['auc_candidate'])} "
                f"({_signed(row['auc_delta'])}) on {int(row['rows'])} rows and "
                f"{int(row['positives'])} positives "
                f"({int(row['positives']) / max(int(row['rows']), 1):.1%} of the block)"
            )
        if not found:
            lines.append(f"- {focus_year} is absent from both panels; nothing to compare.")
        lines.append("")
        lines.append(
            "A year this weak is only worth acting on if the movement is larger than the year's own "
            "sampling noise. Compare ΔAUC against the base rate and positive count on the row above "
            "before treating it as a fix."
        )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="earlier report JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="later report JSON")
    parser.add_argument("--baseline-panel", type=Path, default=None, help="baseline eval panel dump")
    parser.add_argument("--candidate-panel", type=Path, default=None, help="candidate eval panel dump")
    parser.add_argument(
        "--focus-year",
        type=int,
        default=None,
        help="call out one calendar year explicitly (e.g. 2020, the test block's weak year)",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the markdown here")
    args = parser.parse_args(argv)

    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    incomparable = comparability(candidate, baseline)

    year_tables: dict[str, pd.DataFrame] = {}
    if args.baseline_panel and args.candidate_panel:
        baseline_panel = pd.read_csv(args.baseline_panel)
        candidate_panel = pd.read_csv(args.candidate_panel)
        for path in SCORE_COLUMNS:
            if path not in {row["path"] for row in candidate.get("comparison", [])}:
                continue
            year_tables[path] = year_comparison(baseline_panel, candidate_panel, path)

    report = render(
        baseline,
        candidate,
        baseline_name=str(args.baseline),
        candidate_name=str(args.candidate),
        incomparable=incomparable,
        headlines=scalar_deltas(baseline.get("headline", {}), candidate.get("headline", {}), HEADLINE_KEYS),
        per_path=per_path_table(baseline, candidate),
        gates=gates_diff(baseline, candidate),
        falsification=falsification_diff(baseline, candidate),
        year_tables=year_tables,
        focus_year=args.focus_year,
    )
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8", newline="\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

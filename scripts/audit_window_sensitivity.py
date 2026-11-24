#!/usr/bin/env python
"""Does varying ``max_seq_length`` actually change what the model is fed?

The 2026-10-14 window scan varied the *truncation ceiling* while holding the corpus
fixed. That corpus had already been built at a fixed character budget, so 4,096 / 6,144 /
8,192 truncated almost nothing: the three probes fed the model near-identical inputs.
The scan was then read as evidence that "the window is not the variable" and, from there,
that the machine or its driver must have changed. Neither inference was warranted.

This script decides the question **before** any GPU time is spent. It measures the real
token length of each record, applies trl's truncation rule (``slice(None, max_length)``,
no padding up), and reports how much of the input each candidate window would change.

It also prints the corpus's ``character_budget`` from its manifest next to the candidate
windows. That is the number that actually sets how much source text each prompt contains:
``max_seq_length`` decides it at *build* time via ``chars_budget_for_seq_length``, but at
*train* time it is only a ceiling. Varying the ceiling on an already-built corpus does not
rebuild the corpus.

Three-valued verdict, so an absent measurement never reads as a pass:

* ``True``  -- the windows produce materially different inputs; a scan can discriminate.
* ``False`` -- they do not; changing the ceiling changes nothing, so a scan cannot answer
  the question. Rebuild the corpus at a different budget instead.
* ``None``  -- not measured (no rows, no tokenizer, no windows).

Exit status is 0 only for ``True``. ``False`` and ``None`` both exit 1: a runbook must not
treat "we could not vary the input" as "the input does not matter".

Usage::

    python scripts/audit_window_sensitivity.py --sft-dir data/processed/sft_stage2_wide
    python scripts/audit_window_sensitivity.py \\
        --sft-dir data/processed/sft_stage2_wide --windows 4096,6144,8192 --sample 0
    python scripts/audit_window_sensitivity.py \\
        --sft-dir data/processed/sft --sft-dir data/processed/sft_stage2_wide \\
        --windows 4096,8192 --out data/raw/analysis/window_sensitivity.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time convenience
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_WINDOWS = (4096, 6144, 8192)
#: Below this relative spread in effective tokens, the windows cannot be told apart.
#: 5% is deliberately loose: a scan that shifts the input by 3% cannot attribute a
#: 30x change in step time to the window, and should say so rather than guess.
MIN_CONTRAST = 0.05


def effective_lengths(lengths: list[int], window: int) -> list[int]:
    """Apply trl's truncation rule to every row.

    trl 1.13 truncates with ``slice(None, max_length)``; it does not pad short rows up
    to the window. So a window larger than a row leaves that row *unchanged*.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return [min(length, window) for length in lengths]


def scan_contrast(
    per_window: dict[int, list[int]], min_contrast: float = MIN_CONTRAST
) -> tuple[bool | None, dict[str, Any]]:
    """Decide whether candidate windows materially differ, from effective lengths.

    Pure function: no tokenizer, no torch. ``per_window`` maps each window to the
    per-row effective token lengths *already truncated*. A missing or empty entry means
    the scan was not measured, which is a third state distinct from "no difference".
    """
    if not per_window or any(not lengths for lengths in per_window.values()):
        return None, {"reason": "no rows or no windows; the scan was not measured"}

    totals = {window: sum(lengths) for window, lengths in per_window.items()}
    biggest, smallest = max(totals.values()), min(totals.values())
    spread = (biggest - smallest) / biggest if biggest else 0.0

    # How many rows differ between the narrowest and the widest ceiling. The lists are
    # per-row, in the same order, so a positional comparison is the right one -- using
    # the lengths themselves as keys would silently merge duplicate lengths.
    narrow, wide = min(per_window), max(per_window)
    changed_rows = sum(
        1
        for left, right in zip(per_window[narrow], per_window[wide], strict=True)
        if left != right
    )
    rows = len(per_window[wide])

    detail: dict[str, Any] = {
        "totals": totals,
        "spread": spread,
        "min_contrast": min_contrast,
        "rows": rows,
        "rows_changed": changed_rows,
        "narrow": narrow,
        "wide": wide,
    }
    return spread >= min_contrast, detail


def manifest_budget(sft_dir: Path) -> dict[str, Any]:
    """Read the build-time knobs. Missing fields are ``None`` = not measured."""
    path = sft_dir / "manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "max_seq_length": data.get("max_seq_length"),
        "character_budget": data.get("character_budget"),
        "train_config": data.get("train_config"),
    }


def render_lengths(path: Path, tokenizer: Any, sample: int) -> list[int]:
    """Real token lengths of the rendered chat, using the production chat template."""
    lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
            lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            if sample and len(lengths) >= sample:
                break
    return lengths


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


def report_for(sft_dir: Path, windows: list[int], tokenizer: Any, sample: int) -> tuple[str, bool | None]:
    train = sft_dir / "train.jsonl"
    if not train.is_file():
        return f"## {sft_dir}\n\n**not measured**: {train} does not exist.\n", None
    lengths = render_lengths(train, tokenizer, sample)
    if not lengths:
        return f"## {sft_dir}\n\n**not measured**: {train} has no rows.\n", None

    per_window = {window: effective_lengths(lengths, window) for window in windows}
    verdict, detail = scan_contrast(per_window)
    budget = manifest_budget(sft_dir)
    raw = sorted(lengths)

    lines = [
        f"## {sft_dir}",
        "",
        f"- rows measured: **{len(raw)}**" + (f" (first {sample})" if sample else " (all)"),
        f"- built with `max_seq_length: {budget.get('max_seq_length', 'not recorded')}`, "
        f"`character_budget: {budget.get('character_budget', 'not recorded')}`"
        + (f", `train_config: {budget['train_config']}`" if budget.get("train_config") else ""),
        f"- raw rendered tokens: p50 **{percentile(raw, 0.50)}**, p90 {percentile(raw, 0.90)}, "
        f"p100 **{raw[-1]}**, mean {sum(raw) // len(raw)}",
        "",
        "| window | rows clipped | effective mean | effective p100 | effective total |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in windows:
        effective = sorted(per_window[window])
        clipped = sum(1 for length in lengths if length > window)
        lines.append(
            f"| {window} | {clipped} | {sum(effective) // len(effective)} | {effective[-1]} "
            f"| {detail['totals'][window]} |"
        )

    lines += ["", f"Spread across windows: **{detail['spread'] * 100:.2f}%** "
              f"(threshold {MIN_CONTRAST * 100:.0f}%)."]
    if verdict is True:
        lines += ["", "**VERDICT: informative.** The windows materially change the input; "
                  "a throughput scan across them can discriminate."]
    elif verdict is False:
        lines += [
            "",
            "**VERDICT: NOT informative.** The windows leave the input essentially unchanged — "
            "varying a truncation ceiling on an already-built corpus does not rebuild it. "
            "A scan across these windows cannot answer whether sequence length matters, "
            "and a constant result is evidence about the *experiment design*, not the machine.",
            "",
            "To actually vary the input, rebuild the corpus at a different budget:",
            "",
            "```bash",
            "python -m shingan data sft --data-config <data overlay> \\",
            "    --train-config <overlay with a different max_seq_length>",
            "```",
        ]
    else:
        lines += ["", "**VERDICT: not measured.** " + str(detail.get("reason", ""))]
    return "\n".join(lines) + "\n", verdict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-dir", type=Path, action="append", required=True,
                        help="SFT corpus directory; repeat to compare corpora built at different budgets")
    parser.add_argument("--windows", type=str, default=",".join(str(w) for w in DEFAULT_WINDOWS),
                        help="comma-separated candidate max_seq_length values")
    parser.add_argument("--sample", type=int, default=256, help="measure only the first N rows (0 = all)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="tokenizer to measure with")
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    windows = [int(item) for item in args.windows.split(",") if item.strip()]
    if not windows or any(window <= 0 for window in windows):
        raise SystemExit("--windows must be a non-empty list of positive integers")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except Exception as exc:  # a missing tokenizer is "not measured", not a crash
        print(f"VERDICT: not measured -- tokenizer unavailable: {type(exc).__name__}: {exc}")
        return 1

    reports: list[str] = []
    verdicts: list[bool | None] = []
    for sft_dir in args.sft_dir:
        text, verdict = report_for(sft_dir, windows, tokenizer, args.sample)
        reports.append(text)
        verdicts.append(verdict)
        print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(("\n".join(reports)).replace("\r\n", "\n").encode("utf-8"))

    if any(verdict is not True for verdict in verdicts):
        print("VERDICT: FAIL -- at least one corpus cannot support a window scan.")
        return 1
    print("VERDICT: pass -- all corpora show materially different inputs across the windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

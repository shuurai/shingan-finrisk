"""Does the built corpus actually fit the limit, and what would a different limit admit?

`lora.max_seq_length` has two effects and they are easy to confuse:

1. it caps the sequence at training time, and
2. via `chars_budget_for_seq_length` it sets the **character budget** the prompt's
   filing and news sections are trimmed to fit.

Effect 2 is the one that binds first. The manifest reports `prompts_truncated: 8,971`
of 9,140 for the wide corpus, which reads as "the prompts are too long" — they are not.
Every one of them is *short*: the sections were trimmed to fit a 5,982-character budget
that `max_seq_length` implies. The median rendered prompt uses less than half the token
limit. So "truncated" here means "trimmed by the budget", and the lever that would let
more source text through is the budget, not a longer sequence.

That distinction decides whether raising the limit is worth GPU hours, so it is measured
with the real tokenizer rather than argued from a characters-per-token constant. The
constant is a *planning* figure (see `shingan.prompts.CHARS_PER_TOKEN`); this script
reports what the corpus in front of it actually tokenises to.

Runs on CPU, no GPU and no model download if the tokenizer is cached.

Usage::

    python scripts/audit_prompt_tokens.py --sft-dir data/processed/sft_stage2_wide
    python scripts/audit_prompt_tokens.py --sft-dir data/processed/sft_stage2_wide \\
        --candidates 4096,6144,8192 --out data/raw/analysis/prompt_tokens.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time convenience
    sys.path.insert(0, str(ROOT / "src"))

from shingan.prompts import (  # noqa: E402
    CHARS_PER_TOKEN,
    chars_budget_for_seq_length,
)

#: Percentiles worth printing. The maximum matters most here: it is what decides whether a
#: limit is safe, and an average would hide the one prompt that overflows.
PERCENTILES = (50, 90, 95, 99, 100)


def iter_messages(path: Path) -> Iterable[str]:
    """The concatenated message text of every record in one JSONL file.

    The training file is `messages` shaped; a file in any other shape is reported rather
    than skipped, because silently measuring zero rows would read as "nothing overflows".
    """
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{number} has no `messages` list; cannot measure it")
            yield "".join(str(message.get("content", "")) for message in messages)


def token_lengths(texts: Iterable[str], tokenizer: Any) -> tuple[np.ndarray, np.ndarray]:
    """Token counts and character counts for each rendered prompt.

    Characters are returned alongside so the report can show the effective
    characters-per-token ratio of *this* corpus instead of quoting the planning constant.
    """
    tokens: list[int] = []
    characters: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True)["input_ids"]
        tokens.append(len(encoded))
        characters.append(len(text))
    return np.asarray(tokens, dtype=int), np.asarray(characters, dtype=int)


def summarise(lengths: np.ndarray, limit: int) -> dict[str, Any]:
    """How an observed token-length distribution sits against one sequence limit.

    `over` counts the prompts a trainer would cut. `headroom` is the distance from the
    longest prompt to the limit and is negative when nothing fits — the two are reported
    together so a pass cannot hide a corpus that only just fits.
    """
    if lengths.size == 0:
        return {"n": 0, "limit": limit, "over": None, "headroom": None, "percentiles": {}}
    longest = int(lengths.max())
    return {
        "n": int(lengths.size),
        "limit": int(limit),
        "over": int((lengths > limit).sum()),
        "headroom": int(limit - longest),
        "max": longest,
        "percentiles": {f"p{point}": int(np.percentile(lengths, point)) for point in PERCENTILES},
    }


#: Marginal characters per token to project newly admitted text with.
#:
#: The budget formula uses `CHARS_PER_TOKEN` (1.9), and projecting at the same constant is
#: circular: `budget = limit * c - overhead`, so spending the larger budget at ratio `c`
#: consumes exactly the extra tokens by construction and the answer is `limit` every time,
#: for every limit. These three come from elsewhere — the project's documented measured
#: range for the prompt body (1.84-2.05, median ~1.94) and the corpus's own densest real
#: sample — so the projection can disagree with the formula that produced the budget.
DENSE_RATIO = 1.84

#: The corpus aggregate, the planning constant and the densest measured end, in the order
#: the report prints them.
RATIO_LABELS: tuple[tuple[str, float], ...] = (
    ("corpus aggregate", 0.0),  # replaced by the measured value at run time
    ("planning constant", CHARS_PER_TOKEN),
    ("dense measured end", DENSE_RATIO),
)


def project_after_budget_raise(
    lengths: np.ndarray, from_limit: int, to_limit: int, marginal_ratios: dict[str, float]
) -> dict[str, Any]:
    """How the corpus's longest prompt would sit against a larger budget, per ratio.

    Reported as a range rather than a point, because the answer genuinely depends on how
    the newly admitted filing prose tokenises and that is not knowable before it is
    admitted. `fits` is the conjunction: a limit only counts as fitting when every
    plausible ratio fits.

    This answers "could more text overflow", not "how long will the prompts be".
    """
    baseline_budget = chars_budget_for_seq_length(from_limit)
    target_budget = chars_budget_for_seq_length(to_limit)
    growth_chars = max(0, target_budget - baseline_budget)
    longest = int(lengths.max()) if lengths.size else 0
    per_ratio: dict[str, dict[str, Any]] = {}
    for label, ratio in marginal_ratios.items():
        if not ratio or ratio <= 0:
            per_ratio[label] = {"projected_max_tokens": float("nan"), "fits": None, "slack_tokens": None}
            continue
        growth_tokens = growth_chars / ratio
        projected = longest + growth_tokens
        per_ratio[label] = {
            "ratio": float(ratio),
            "growth_tokens": float(growth_tokens),
            "projected_max_tokens": float(projected),
            "fits": bool(projected <= to_limit),
            "slack_tokens": float(to_limit - projected),
        }
    fits = [block["fits"] for block in per_ratio.values()]
    return {
        "from_limit": int(from_limit),
        "to_limit": int(to_limit),
        "baseline_budget": int(baseline_budget),
        "target_budget": int(target_budget),
        "growth_chars": int(growth_chars),
        "longest_tokens": longest,
        "per_ratio": per_ratio,
        "fits": all(value for value in fits) if all(value is not None for value in fits) else None,
        "worst_slack": min(
            (block["slack_tokens"] for block in per_ratio.values() if block["slack_tokens"] is not None),
            default=None,
        ),
    }


def effective_chars_per_token(lengths: np.ndarray, characters: np.ndarray) -> float:
    """Corpus-wide characters per token, from the sums rather than a mean of ratios.

    A mean of per-row ratios is dominated by the shortest prompts; the sums answer the
    question the budget cares about — how much text one token buys on average here.
    """
    total_tokens = int(lengths.sum())
    return float(characters.sum() / total_tokens) if total_tokens else float("nan")


def render(
    per_split: dict[str, dict[str, Any]],
    ratio: float,
    limit: int,
    projections: dict[int, dict[str, Any]],
    manifest: dict[str, Any] | None,
) -> str:
    """Render the audit. Separate from the measurement so it is testable."""
    lines: list[str] = []
    lines.append("# Prompt token audit")
    lines.append("")
    lines.append(
        "Measured with the real tokenizer on the built corpus. No GPU, no model load."
    )
    lines.append("")
    if manifest:
        lines.append(
            f"- manifest: `max_seq_length` {manifest.get('max_seq_length')}, "
            f"`character_budget` {manifest.get('character_budget')}, "
            f"`prompts_truncated` {manifest.get('prompts_truncated')}"
        )
        lines.append(f"- `train_config` that produced it: `{manifest.get('train_config')}`")
        lines.append("")
    lines.append(
        f"Effective characters per token over the corpus: **{ratio:.3f}** "
        f"(the planning constant in `shingan.prompts` is {CHARS_PER_TOKEN}). The aggregate "
        "is pulled up by short prompts, so it is not the ratio to project new prose with."
    )
    lines.append("")

    for split, block in per_split.items():
        lines.append(f"## {split}")
        lines.append("")
        if not block.get("n"):
            lines.append("No rows measured.")
            lines.append("")
            continue
        lines.append("| percentile | tokens |")
        lines.append("| --- | ---: |")
        for name, value in block["percentiles"].items():
            lines.append(f"| {name} | {value:,} |")
        lines.append("")
        over = block.get("over")
        headroom = block.get("headroom")
        verdict = "fits" if over == 0 and (headroom or 0) >= 0 else "**does not fit**"
        lines.append(
            f"Against the {limit:,}-token limit this corpus was built at: "
            f"{'not measured' if over is None else f'{over:,} prompt(s) over'}, "
            f"headroom {'not measured' if headroom is None else format(headroom, ',')} → {verdict}."
        )
        lines.append("")

    lines.append("## What a larger budget would admit")
    lines.append("")
    lines.append(
        "The answer depends on how the newly admitted filing prose tokenises, which is not "
        "knowable before it is admitted, so it is given as a range over three ratios. "
        "Projecting at the planning constant alone would be circular: "
        "`budget = limit * c - overhead`, so spending the larger budget at ratio `c` "
        "consumes exactly the extra tokens and the answer is the limit itself, for every "
        "limit, whichever way the corpus is built."
    )
    lines.append("")
    for block in projections.values():
        lines.append(
            f"**{block['from_limit']:,} → {block['to_limit']:,}**: budget "
            f"{block['baseline_budget']:,} → {block['target_budget']:,} characters "
            f"(+{block['growth_chars']:,}), against a longest current prompt of "
            f"{block['longest_tokens']:,} tokens."
        )
        lines.append("")
        lines.append("| marginal chars/token | ≈ extra tokens | projected longest | fits | slack |")
        lines.append("| ---: | ---: | ---: | --- | ---: |")
        for label, ratio_block in block["per_ratio"].items():
            if ratio_block.get("fits") is None:
                lines.append(f"| {label} | n/a | n/a | not measured | n/a |")
                continue
            lines.append(
                f"| {label} ({ratio_block['ratio']:.2f}) | {ratio_block['growth_tokens']:,.0f} | "
                f"{ratio_block['projected_max_tokens']:,.0f} | "
                f"{'yes' if ratio_block['fits'] else '**no**'} | "
                f"{ratio_block['slack_tokens']:,.0f} |"
            )
        lines.append("")
        verdict = {True: "fits under every ratio", False: "**overflows under at least one**"}.get(
            block["fits"], "not measured"
        )
        lines.append(f"Verdict: {verdict}.")
        lines.append("")
    lines.append(
        "The budget is `max_seq_length * CHARS_PER_TOKEN - PROMPT_OVERHEAD_CHARS`, so the "
        "limit is a content budget first and a sequence cap second. Raising it without "
        "rebuilding the corpus changes nothing: prompts are rendered once, at build time, "
        "and `data sft --train-config` is what lets the build see the value the trainer "
        "will use."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-dir", type=Path, required=True, help="directory with train.jsonl / valid.jsonl")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-14B",
        help="base model whose tokenizer is used; the real one, not an approximation",
    )
    parser.add_argument("--candidates", default="4096,6144,8192", help="comma-separated sequence limits")
    parser.add_argument("--baseline-limit", type=int, default=4096, help="limit the corpus was built at")
    parser.add_argument("--sample", type=int, default=0, help="measure only the first N rows (0 = all)")
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    candidates = [int(item) for item in str(args.candidates).split(",") if item.strip()]

    manifest_path = args.sft_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None

    per_split: dict[str, dict[str, Any]] = {}
    combined_lengths: list[np.ndarray] = []
    combined_chars: list[np.ndarray] = []
    for name in ("train", "valid"):
        path = args.sft_dir / f"{name}.jsonl"
        if not path.is_file():
            continue
        texts = iter_messages(path)
        if args.sample:
            texts = (text for index, text in enumerate(texts) if index < args.sample)
        lengths, characters = token_lengths(texts, tokenizer)
        combined_lengths.append(lengths)
        combined_chars.append(characters)
        # Reported against the limit the corpus was built at; every other limit is in the
        # comparison table below.
        per_split[name] = summarise(lengths, args.baseline_limit)
        print(f"{name}: {lengths.size:,} prompts measured")

    if not combined_lengths:
        print(f"no train.jsonl or valid.jsonl under {args.sft_dir}", file=sys.stderr)
        return 2

    lengths = np.concatenate(combined_lengths)
    characters = np.concatenate(combined_chars)
    ratio = effective_chars_per_token(lengths, characters)
    ratios = {label: (ratio if label == "corpus aggregate" else value) for label, value in RATIO_LABELS}
    projections = {
        limit: project_after_budget_raise(lengths, args.baseline_limit, limit, ratios)
        for limit in candidates
        if limit != args.baseline_limit
    }
    report = render(per_split, ratio, args.baseline_limit, projections, manifest)
    print()
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

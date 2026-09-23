#!/usr/bin/env python
"""Build the placebo SFT corpus: the same targets, randomly reassigned prompts.

ADR-0003 requires a placebo control before any "the fine-tuning learned something"
claim: retrain the adapter on a corpus where the text->label link is destroyed and
compare. The cheapest honest way to destroy the link is to **permute the user turns
within each split** (fixed seed). Everything else — the instruction contract, the
JSON target shape, the label distribution, the prompt-length distribution, the split
sizes — is preserved byte for byte, so a placebo adapter that still emits a constant
score can only mean the signal was never in the text.

What the permutation deliberately does NOT touch:

* the system turn (the instruction contract stays identical);
* the assistant targets (the label distribution per split is unchanged);
* the ``meta`` blocks (``sample_id`` keeps pointing at the row whose *target* is on
  that line — its text no longer matches, which is the point).

Usage:
    python scripts/placebo_corpus.py \
        --src data/processed/sft --dst data/processed/sft_placebo --seed 7

Train with:
    python -m shingan train lora --train-file data/processed/sft_placebo/train.jsonl \
        --eval-file data/processed/sft_placebo/valid.jsonl \
        --output-dir artifacts/lora-placebo

Scoring the placebo needs ``--no-verify-prompts``: the rebuilt prompts are *supposed*
to differ from the placebo file, and the run.json written next to the placebo adapter
records which corpus it actually read (hashes included) — that is the provenance
chain, not the byte-comparison gate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SPLITS = ("train", "valid")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("data/processed/sft"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed/sft_placebo"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def permute_user_turns(records: list[dict], rng: random.Random) -> int:
    """Reassign user turns among records in place. Returns the number of moved turns."""
    turns = [record["messages"][i]["content"] for record in records
             for i, message in enumerate(record["messages"]) if message["role"] == "user"]
    if len(turns) != len(records):
        sys.exit(f"error: expected exactly one user turn per record, found {len(turns)} "
                 f"turns across {len(records)} records")
    order = list(range(len(records)))
    while True:
        rng.shuffle(order)
        # A derangement is not required, but a permutation that leaves most records in
        # place would weaken the placebo; redraw rather than accept a near-identity.
        fixed = sum(1 for position, target in enumerate(order) if position == target)
        if fixed <= len(records) // 10:
            break
    for record, target in zip(records, order, strict=True):
        for message in record["messages"]:
            if message["role"] == "user":
                message["content"] = turns[target]
    return fixed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    args.dst.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        source = args.src / f"{split}.jsonl"
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        fixed = permute_user_turns(records, rng)
        out = args.dst / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split}: {len(records)} records, user turns permuted ({fixed} stayed in place)")
    print(f"placebo corpus written to {args.dst} (seed {args.seed})")


if __name__ == "__main__":
    main()

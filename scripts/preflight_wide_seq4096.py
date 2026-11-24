#!/usr/bin/env python
"""Preflight the wide-pool 4096 retrain: prove the run is what the docs claim.

Why this exists
---------------
The 2026-10-01 run spent 6.5 days and produced nothing because two cheap facts were
never checked before loading 28 GB of weights: the trainer scheduled no intermediate
checkpoint, and the corpus had been rebuilt at a budget the card could not hold
(docs/09 sections 25 and 26). Both are decidable from config files and two JSON files.
This script decides them in under a second, on CPU, without importing the model.

It answers four questions and nothing else:

  1. Is the archived corpus byte-for-byte the one its ARCHIVE_MANIFEST.json pins?
  2. Does the training overlay's character budget equal the corpus's recorded budget?
     (This is exactly what `eval lora`'s `CorpusScope` refuses on disagreement.)
  3. How many optimiser steps, warmup steps and recoverable checkpoints will the run
     have, and does `assert_run_is_resumable` accept it?
  4. Will the installed trl `SFTConfig` accept every argument we are about to pass?

Exit status is 0 only when all four are true. It never loads a model and never writes
to the real output directory.

    python scripts/preflight_wide_seq4096.py
    python scripts/preflight_wide_seq4096.py --train-config <overlay> --sft-dir <corpus>

A mutation test lives in tests/test_preflight_wide_seq4096.py: flip the budget, the
epoch count or a corpus byte and the verdict must change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shingan.config import load_config
from shingan.models.lora import (
    assert_run_is_resumable,
    assert_trainer_arguments_supported,
    build_sft_config_kwargs,
    resolve_warmup_steps,
    scheduled_checkpoints,
    total_optimizer_steps,
)
from shingan.prompts import chars_budget_for_seq_length

DEFAULT_OVERLAY = Path("configs/train/qlora_qwen3_14b_seq4096.yaml")
DEFAULT_CORPUS = Path("data/processed/sft_stage2_wide_seq4096")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_CORPUS)
    return parser.parse_args(argv)


def verify_corpus(corpus: Path) -> tuple[bool, list[str]]:
    """SHA-256 every file the archive manifest pins. A missing manifest is a failure."""
    lines: list[str] = []
    manifest = corpus / "ARCHIVE_MANIFEST.json"
    if not manifest.is_file():
        return False, [f"ARCHIVE_MANIFEST.json missing in {corpus}"]
    try:
        pinned = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, [f"ARCHIVE_MANIFEST.json unreadable: {type(exc).__name__}: {exc}"]
    ok = True
    for name, meta in pinned.get("files", {}).items():
        target = corpus / name
        if not target.is_file():
            ok = False
            lines.append(f"  {name:16s} MISSING")
            continue
        raw = target.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        match = digest == meta["sha256"] and len(raw) == meta["bytes"]
        ok &= match
        lines.append(f"  {name:16s} {'OK' if match else 'MISMATCH'}  {digest[:16]}...")
    return ok, lines


def verdict(train_config: Path, corpus: Path) -> tuple[bool, list[str]]:
    """The four checks, in the order a careful operator would ask them."""
    lines: list[str] = []
    ok = True

    corpus_ok, hash_lines = verify_corpus(corpus)
    lines.append("1. archived corpus integrity")
    lines.extend(hash_lines)
    ok &= corpus_ok

    config = load_config(overlays=[train_config])
    run_budget = chars_budget_for_seq_length(config.lora.max_seq_length)
    manifest = corpus / "manifest.json"
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        recorded = {}
        lines.append(f"  manifest.json unreadable: {type(exc).__name__}: {exc}")
    corpus_budget = recorded.get("character_budget")
    n_train = (recorded.get("splits") or {}).get("train")
    agrees = corpus_budget is not None and corpus_budget == run_budget
    lines.append("2. corpus scope (what eval lora refuses on)")
    lines.append(
        f"  overlay max_seq_length {config.lora.max_seq_length} -> run budget {run_budget:,}"
        f"  | corpus recorded {corpus_budget if corpus_budget is not None else 'not recorded'}"
        f"  | {'AGREES' if agrees else 'DISAGREES'}"
    )
    ok &= agrees

    lines.append("3. run shape")
    if n_train is None:
        lines.append("  n_train not recorded in manifest -- cannot compute steps")
        ok = False
        steps = warmup = 0
        checkpoints: list[int] = []
    else:
        steps = total_optimizer_steps(
            n_train,
            batch_size=config.lora.per_device_train_batch_size,
            accumulate=config.lora.gradient_accumulation_steps,
            epochs=config.lora.num_train_epochs,
        )
        warmup = resolve_warmup_steps(config.lora, n_train)
        checkpoints = scheduled_checkpoints(config.lora, n_train)
        lines.append(
            f"  n_train {n_train:,} | epochs {config.lora.num_train_epochs} | {steps} optimiser steps"
            f" | warmup {warmup} | checkpoints {len(checkpoints)} at {checkpoints}"
        )
        try:
            assert_run_is_resumable(config.lora, n_train)
            lines.append("  assert_run_is_resumable: PASS")
        except Exception as exc:
            ok = False
            lines.append(f"  assert_run_is_resumable: FAIL {type(exc).__name__}: {exc}")

    lines.append("4. trainer API compatibility")
    kwargs = build_sft_config_kwargs(
        config.lora,
        output_dir=str(config.lora.output_dir),
        n_train_examples=n_train or 0,
        eval_enabled=True,
        seed=config.project.seed,
    )
    try:
        from trl import SFTConfig

        assert_trainer_arguments_supported(kwargs, SFTConfig, label="SFTConfig")
        lines.append(
            f"  SFTConfig accepts all {len(kwargs)} keys | max_length {kwargs.get('max_length')}"
            f" | save_steps {kwargs.get('save_steps')}"
        )
    except Exception as exc:
        ok = False
        lines.append(f"  SFTConfig: FAIL {type(exc).__name__}: {exc}")

    return ok, lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ok, lines = verdict(args.train_config, args.sft_dir)
    print(f"preflight: train-config {args.train_config} | sft-dir {args.sft_dir}")
    print("\n".join(lines))
    print(f"VERDICT: {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

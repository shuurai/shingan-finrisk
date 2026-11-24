"""The preflight must fail when any of its four facts is broken.

These are mutation tests, not fixtures: each one damages a real input on disk (in a
tmp_path copy) and asserts the verdict changes. A preflight that always prints "pass"
is worse than none, because it licenses the 5-7 hour run that follows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.preflight_wide_seq4096 import verdict

OVERLAY = Path("configs/train/qlora_qwen3_14b_seq4096.yaml")
CORPUS = Path("data/processed/sft_stage2_wide_seq4096")


def _copy_corpus(dst: Path) -> Path:
    """A complete, small-enough copy: manifests plus the first 40 rows of each split."""
    dst.mkdir(parents=True)
    for name in ("ARCHIVE_MANIFEST.json", "manifest.json"):
        (dst / name).write_bytes((CORPUS / name).read_bytes())
    # The hash check needs the *exact* archived bytes, so copy the pinned files whole.
    # They are ~65 MB total; tests that mutate a byte re-pin the digest themselves.
    for name in ("train.jsonl", "valid.jsonl"):
        (dst / name).write_bytes((CORPUS / name).read_bytes())
    return dst


def test_shipped_configuration_passes() -> None:
    ok, lines = verdict(OVERLAY, CORPUS)
    assert ok, "\n".join(lines)
    assert "457 optimiser steps" in "\n".join(lines)
    assert "AGREES" in "\n".join(lines)


def test_a_corrupted_corpus_byte_is_refused(tmp_path: Path) -> None:
    corpus = _copy_corpus(tmp_path / "corpus")
    train = corpus / "train.jsonl"
    raw = bytearray(train.read_bytes())
    raw[0] ^= 0x01
    train.write_bytes(bytes(raw))
    ok, lines = verdict(OVERLAY, corpus)
    assert not ok
    assert any("MISMATCH" in line for line in lines)


def test_a_missing_manifest_is_refused(tmp_path: Path) -> None:
    corpus = _copy_corpus(tmp_path / "corpus")
    (corpus / "manifest.json").unlink()
    ok, lines = verdict(OVERLAY, corpus)
    assert not ok
    assert "not recorded" in "\n".join(lines) or "unreadable" in "\n".join(lines)


def test_a_budget_mismatch_is_refused(tmp_path: Path) -> None:
    corpus = _copy_corpus(tmp_path / "corpus")
    manifest = corpus / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["character_budget"] = 13764  # the 8192 corpus's budget
    manifest.write_text(json.dumps(data), encoding="utf-8")
    ok, lines = verdict(OVERLAY, corpus)
    assert not ok
    assert "DISAGREES" in "\n".join(lines)


def test_an_overlay_that_schedules_no_checkpoint_is_refused(tmp_path: Path) -> None:
    """The 2026-10-01 failure mode, replayed against the gate in under a second."""
    overlay = tmp_path / "bad.yaml"
    overlay.write_text(
        "lora:\n"
        "  max_seq_length: 4096\n"
        "  num_train_epochs: 1\n"
        "  save_strategy: steps\n"
        "  save_steps: 500\n",
        encoding="utf-8",
    )
    ok, lines = verdict(overlay, CORPUS)
    assert not ok
    assert any("assert_run_is_resumable: FAIL" in line for line in lines)


@pytest.mark.parametrize("budget", [None, 5981, 13764])
def test_only_the_recorded_budget_agrees(budget: int | None, tmp_path: Path) -> None:
    corpus = _copy_corpus(tmp_path / "corpus")
    manifest = corpus / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if budget is None:
        data.pop("character_budget", None)
    else:
        data["character_budget"] = budget
    manifest.write_text(json.dumps(data), encoding="utf-8")
    ok, lines = verdict(OVERLAY, corpus)
    assert not ok
    assert "DISAGREES" in "\n".join(lines) or "not recorded" in "\n".join(lines)

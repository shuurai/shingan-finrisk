"""The placebo corpus script is part of the experiment record, not a throwaway.

CI smoke-runs every script's ``--help``, but the permutation and the manifest are
persistence contracts: a silent change to either would make two placebo runs
incomparable while every metric still renders. The script lives in ``scripts/``
(not a package), so it is loaded here by file path — one loader, stated once.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "placebo_corpus.py"

_spec = importlib.util.spec_from_file_location("placebo_corpus_under_test", SCRIPT)
if _spec is None or _spec.loader is None:  # pragma: no cover - only on a broken checkout
    pytest.fail(f"cannot load {SCRIPT}")
placebo_corpus = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, placebo_corpus)
_spec.loader.exec_module(placebo_corpus)


def _record(index: int, positive: bool) -> dict:
    """One SFT-shaped record. The system turn marks which record it *was*."""
    return {
        "messages": [
            {"role": "system", "content": f"system-for-row-{index}"},
            {"role": "user", "content": f"text-for-row-{index}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"score": 8 if positive else 2, "source_ref": f"10-K:row-{index}"}
                ),
            },
        ],
        "meta": {"sample_id": f"ROW-{index:04d}", "label": "tail_risk"},
    }


def _write_source(directory: Path, split: str, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run(tmp_path: Path, dst_name: str, seed: int = 7) -> Path:
    src = tmp_path / "src"
    _write_source(src, "train", [_record(i, i % 4 == 0) for i in range(20)])
    _write_source(src, "valid", [_record(i, i % 5 == 0) for i in range(10)])
    dst = tmp_path / dst_name
    placebo_corpus.main(["--src", str(src), "--dst", str(dst), "--seed", str(seed)])
    return dst


def test_the_permutation_moves_user_turns_but_touches_nothing_else(tmp_path: Path) -> None:
    dst = _run(tmp_path, "dst")
    for split in ("train", "valid"):
        out = [json.loads(line) for line in (dst / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert out, f"{split} came back empty"
        for record in out:
            roles = [message["role"] for message in record["messages"]]
            assert roles == ["system", "user", "assistant"], "the turn order changed"
            # The system turn names the row whose *target* is on this line, so the
            # assistant target and meta must still agree with it even after the user
            # turn was reassigned.
            row = record["messages"][0]["content"].removeprefix("system-for-row-")
            assert f"10-K:row-{row}" in record["messages"][2]["content"]
            assert record["meta"]["sample_id"] == f"ROW-{int(row):04d}"
        # The label distribution is preserved because the assistant targets are:
        # train writes a positive every 4th row, valid every 5th.
        scores = [json.loads(r["messages"][2]["content"])["score"] for r in out]
        expected_positives = len(out) // 4 if split == "train" else len(out) // 5
        assert scores.count(8) == expected_positives
        assert scores.count(2) == len(out) - scores.count(8)


def test_the_manifest_records_the_seed_hashes_and_counts(tmp_path: Path) -> None:
    dst = _run(tmp_path, "dst")
    manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "placebo"
    assert manifest["seed"] == 7
    for split in ("train", "valid"):
        payload = (dst / f"{split}.jsonl").read_bytes()
        assert manifest["output_files"][split]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert manifest["output_files"][split]["bytes"] == len(payload)
        assert manifest["output_files"][split]["n_records"] > 0
        source = tmp_path / "src" / f"{split}.jsonl"
        assert manifest["source_files"][split]["sha256"] == placebo_corpus.sha256_file(source)


def test_rebuilding_is_byte_identical_so_reruns_close_the_provenance_chain(
    tmp_path: Path,
) -> None:
    """A manifest that drifts between reruns would make two placebo arms incomparable."""
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")

    for name in ("train.jsonl", "valid.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_a_source_without_exactly_one_user_turn_per_record_is_refused(tmp_path: Path) -> None:
    src = tmp_path / "src"
    broken = _record(0, positive=False)
    broken["messages"][1]["role"] = "assistant"  # now two assistant turns, no user turn
    _write_source(src, "train", [broken, _record(1, positive=False)])
    _write_source(src, "valid", [_record(2, positive=True)])

    with pytest.raises(SystemExit, match="one user turn per record"):
        placebo_corpus.main(["--src", str(src), "--dst", str(tmp_path / "dst"), "--seed", "7"])

"""Provenance must travel with the artifact that describes the data.

Before this module existed, an SFT manifest recorded split counts but not which panel
they came from, a training ``run.json`` recorded hyperparameters but not which files it
read, and an adapter trained on the synthetic generator produced the same provenance
fields as one trained on SEC filings. Every test here pins a chain link:
``run.json -> SFT manifest -> raw-table hashes``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from shingan.config import ProjectConfig
from shingan.data.provenance import (
    data_provenance,
    file_record,
    sha256_file,
    training_data_block,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"shingan provenance\n"
    target = tmp_path / "sample.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_a_missing_file_records_none_rather_than_an_empty_record(tmp_path: Path) -> None:
    assert file_record(tmp_path / "absent.parquet") is None
    assert file_record(None) is None


def test_a_present_file_records_identity(tmp_path: Path) -> None:
    target = tmp_path / "filings.parquet"
    target.write_bytes(b"parquet-bytes")
    record = file_record(target)
    assert record is not None
    assert record["path"] == str(target)
    assert record["bytes"] == len(b"parquet-bytes")
    assert record["sha256"] == hashlib.sha256(b"parquet-bytes").hexdigest()


def test_a_synthetic_source_has_no_raw_files() -> None:
    config = ProjectConfig()
    assert list(config.data.sources) == ["synthetic"]
    assert data_provenance(pd.DataFrame(columns=["ticker"]), config)["raw_files"] == {}


def test_data_provenance_follows_the_panel(tmp_path: Path) -> None:
    config = ProjectConfig()
    panel = pd.DataFrame(
        {"ticker": ["JPM", "GS", "JPM"], "is_synthetic": [True, True, True]}
    )
    block = data_provenance(panel, config, data_config_path=tmp_path / "overlay.yaml")

    assert block["sources"] == ["synthetic"]
    assert block["is_synthetic"] is True
    assert block["n_rows"] == 3
    assert block["n_tickers"] == 2
    assert block["data_config"].endswith("overlay.yaml")
    # A synthetic build reads no files; recording file paths would imply a provenance
    # the data does not have.
    assert block["raw_files"] == {}


def test_a_panel_without_the_flag_records_none() -> None:
    config = ProjectConfig()
    panel = pd.DataFrame({"ticker": ["JPM"]})
    assert data_provenance(panel, config)["is_synthetic"] is None


def test_training_data_block_hashes_the_files_and_embeds_the_manifest(tmp_path: Path) -> None:
    payload = b'{"messages": []}\n'
    train = tmp_path / "sft" / "train.jsonl"
    train.parent.mkdir(parents=True)
    # Bytes, not write_text: Windows text mode would translate \n to \r\n and the
    # recorded hash would not be the hash of what a reader sees.
    train.write_bytes(payload)
    manifest = {"per_label": {"tail_risk": {"train": 1}}, "data": {"is_synthetic": True}}
    (train.parent / "manifest.json").write_bytes(json.dumps(manifest).encode("utf-8"))
    evaluation = tmp_path / "sft" / "valid.jsonl"
    evaluation.write_bytes(payload)

    block = training_data_block(train, evaluation)

    assert block["train_file"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert block["eval_file"]["bytes"] == len(payload)
    assert block["sft_manifest"] == manifest
    assert "sft_manifest_error" not in block
    assert "sft_manifest_note" not in block


def test_a_missing_manifest_is_recorded_as_a_reason(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    block = training_data_block(train, None)

    assert block["sft_manifest"] is None
    assert "predates the provenance requirement" in block["sft_manifest_note"]


def test_an_unreadable_manifest_is_recorded_not_dropped(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    block = training_data_block(train, None)

    assert block["sft_manifest"] is None
    assert "unreadable" in block["sft_manifest_error"]


def test_everything_stays_json_serialisable(tmp_path: Path) -> None:
    """The block is written into run.json by ``json.dumps``; it must survive that."""
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    block = training_data_block(train, None)
    assert json.loads(json.dumps(block)) == block


@pytest.mark.parametrize("field", ["sources", "is_synthetic", "n_rows", "data_version"])
def test_the_minimal_block_names_its_sources(field: str) -> None:
    config = ProjectConfig()
    panel = pd.DataFrame({"ticker": ["JPM"], "is_synthetic": [True]})
    assert field in data_provenance(panel, config)

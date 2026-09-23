"""Data provenance records: what an artifact's inputs were, and where they came from.

Three artifacts claimed to describe their own data before this module existed, and
none of them could. ``sft/manifest.json`` recorded the split counts but not which
panel the splits came from; ``run.json`` recorded the trainer's arguments but not
which files it trained on; and the model card template asserted a source list in
prose regardless of what any run actually read. An adapter trained on the synthetic
generator and one trained on SEC filings produced byte-identical provenance
fields — which is the defect class this repository treats as disqualifying
everywhere else.

Everything here is deliberately dependency-light (hashlib, pathlib, json). The
training module imports :func:`training_data_block` next to the trainer, so a
machine without the ``train`` extra must still be able to execute and test it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from shingan.__about__ import DATA_SCHEMA_VERSION

#: The raw tables a real-data build reads, mirroring
#: :func:`shingan.data.builder.load_cached_tables`. The first three are required
#: there; the last two may be absent, which the record makes visible instead of
#: implying by omission.
RAW_TABLE_NAMES: tuple[str, ...] = ("prices", "fundamentals", "filings", "news", "events")


def sha256_file(path: Path | str) -> str:
    """The SHA-256 of a file, streamed so a multi-GB parquet costs little memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | str | None) -> dict[str, Any] | None:
    """A JSON-safe identity record for one file, or ``None`` when it does not exist.

    ``None`` rather than an empty dict: a missing input must stay visibly absent in
    the JSON (``null``), not collapse into a record that reads as if it were taken.
    """
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return {
        "path": str(candidate),
        "sha256": sha256_file(candidate),
        "bytes": candidate.stat().st_size,
    }


def raw_table_files(config: Any) -> dict[str, Any]:
    """Identity records for the raw tables a real build would read.

    Args:
        config: The project configuration; only ``config.data`` is read.

    Returns:
        ``{"cache_dir": str, "files": {name: record}}`` for a real source set.
        An empty dict for a synthetic one — the generator is deterministic code, not
        a file, and recording absent file paths beside it would imply a provenance
        the data does not have.
    """
    if "synthetic" in getattr(config.data, "sources", []):
        return {}
    from shingan.paths import ProjectPaths

    candidates: list[Path] = []
    cache_dir = getattr(config.data, "cache_dir", None)
    if cache_dir:
        candidate = Path(cache_dir)
        root = ProjectPaths.from_root(config.project.root).root
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    candidates.append(ProjectPaths.from_root(config.project.root).root / "data" / "raw" / "real")
    directory = next((item for item in candidates if (item / "prices.parquet").is_file()), None)
    if directory is None:
        return {"cache_dir": None, "files": {}, "note": "no raw-table directory found"}
    files = {
        name: file_record(directory / f"{name}.parquet")
        for name in RAW_TABLE_NAMES
    }
    return {"cache_dir": str(directory), "files": files}


def data_provenance(
    panel: Any,
    config: Any,
    *,
    data_config_path: Path | str | None = None,
) -> dict[str, Any]:
    """The ``data`` block an SFT manifest (or any panel-derived artifact) carries.

    Args:
        panel: The assembled panel; read for ``is_synthetic`` and shape only.
        config: The project configuration.
        data_config_path: The overlay that selected the sources, when one was passed.

    Returns:
        A JSON-serialisable dict. Field presence follows the data: a synthetic build
        has no ``raw_files`` entries, and a panel without an ``is_synthetic`` column
        records ``None`` rather than a guess.
    """
    is_synthetic: bool | None = None
    if "is_synthetic" in getattr(panel, "columns", []) and len(panel):
        is_synthetic = bool(panel["is_synthetic"].any())
    return {
        "sources": [str(source) for source in config.data.sources],
        "is_synthetic": is_synthetic,
        "n_rows": len(panel),
        "n_tickers": int(panel["ticker"].nunique()) if "ticker" in panel.columns else None,
        "universe": [str(item) for item in config.data.universe],
        "window": {"start": str(config.data.start), "end": str(config.data.end)},
        "data_version": str(config.project.data_version),
        "data_schema_version": DATA_SCHEMA_VERSION,
        "data_config": str(data_config_path) if data_config_path is not None else None,
        "raw_files": raw_table_files(config),
    }


def training_data_block(
    train_file: Path | str | None,
    eval_file: Path | str | None = None,
) -> dict[str, Any]:
    """The ``data`` block a training run's ``run.json`` carries.

    Beyond hashing the two JSONL files, this embeds the SFT manifest that sits next
    to the training file, which is what chains an adapter to the panel it was built
    from: ``run.json -> sft manifest -> manifest.data -> raw table hashes``. A
    manifest that cannot be read is recorded as a reason, not dropped — an
    unexplained ``null`` in a provenance chain reads the same way as no attempt.

    Args:
        train_file: The training JSONL actually passed to the trainer.
        eval_file: The validation JSONL, when one was supplied.

    Returns:
        A JSON-serialisable dict.
    """
    block: dict[str, Any] = {
        "train_file": file_record(train_file),
        "eval_file": file_record(eval_file),
        "sft_manifest": None,
    }
    if train_file is not None:
        manifest_path = Path(train_file).parent / "manifest.json"
        if manifest_path.is_file():
            try:
                block["sft_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                block["sft_manifest_error"] = f"manifest at {manifest_path} is unreadable: {exc}"
        else:
            block["sft_manifest_note"] = (
                f"no manifest.json beside {train_file}; the SFT file predates the "
                "provenance requirement or was written without one"
            )
    return block


__all__ = [
    "RAW_TABLE_NAMES",
    "data_provenance",
    "file_record",
    "raw_table_files",
    "sha256_file",
    "training_data_block",
]

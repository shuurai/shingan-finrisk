"""Tests for ``publish hf``: value injection, per-card gating, report selection.

The command's contract is deliberately conservative — it refuses to upload a card
that still has an unfilled ``{{...}}`` slot. These tests pin the pieces that make
the refusal *selective* (``--only``), the values injection path (``--values-file``),
the report detection that name order alone got wrong, and the invariant that the
dataset card is completable today while the model card is not.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from shingan.__about__ import GITHUB_URL, HF_DATASET_ID
from shingan.cli import app
from shingan.eval.report import render_dataset_card

runner = CliRunner()

#: The 23 dataset-card fields ``scripts/card_values.py`` produces. Real numbers
#: from the Stage 2 run — kept literal here so the test does not depend on
#: untracked artifacts under ``artifacts/``.
DATASET_VALUES: dict[str, str] = {
    "n_rows": "2221",
    "n_companies": "34",
    "date_range": "2009-07-15 → 2026-09-10",
    "size_category": "1K<n<10K",
    "contains_synthetic": "no",
    "synthetic_share": "0%",
    "pos_default": "not measured (event source not connected)",
    "rate_default": "not measured (event source not connected)",
    "pos_fraud": "not measured (event source not connected)",
    "rate_fraud": "not measured (event source not connected)",
    "pos_tail": "39",
    "rate_tail": "1.76% of all rows (2.19% of the 1778 mask-true rows)",
    "label_columns": (
        "1 label(s) with columns in the panel: tail_risk; "
        "no column for default_risk, fraud_risk (event sources not connected)"
    ),
    "audit_sample_size": "1778",
    "audit_disagreement_rate": "0.00%",
    "market_cap_scope": "34 US large-caps; small-cap behaviour untested",
    "pit_membership_status": "not applied",
    "rating_history_status": "unavailable",
    "vendor_lookahead_status": "not used (XBRL facts read from EDGAR with filed <= as_of)",
    "git_commit": "267b56a",
    "build_date": "2026-09-21",
    "source_snapshot_date": "not recorded",
    "known_issues_url": f"{GITHUB_URL}/issues",
    "changelog_url": f"{GITHUB_URL}/blob/main/CHANGELOG.md",
}


def _values_file(tmp_path: Path, values: dict[str, str]) -> Path:
    """Write a values file and return its path."""
    path = tmp_path / "card_values.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def _flat(text: str) -> str:
    """Collapse all whitespace: rich wraps console lines at terminal width, and the
    wrap point must not decide whether a message substring counts as present."""
    return " ".join(text.split())


def _install_fake_hub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stand in for ``huggingface_hub`` so no test can reach the network.

    Returns the mock API instance its ``HfApi()`` constructor hands back.
    """
    api = MagicMock()
    module = types.ModuleType("huggingface_hub")
    module.HfApi = MagicMock(return_value=api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    return api


def test_the_github_constant_points_at_the_renamed_repository():
    """The card templates embed ``repo_url`` links; a stale slug ships dead links."""
    assert GITHUB_URL == "https://github.com/shuurai/shingan-finrisk"


def test_the_dataset_card_is_completable_without_any_training_run():
    """The labelled panel is real today: every slot has a measured or declared value."""
    card, missing = render_dataset_card(DATASET_VALUES)
    assert missing == []
    assert "not measured (event source not connected)" in card


def test_rendered_cards_do_not_ship_the_editorial_comment_block():
    """A card containing 'delete this block before publishing' contradicts itself."""
    card, _ = render_dataset_card(DATASET_VALUES)
    assert "<!--" not in card
    assert "模板使用方式" not in card


def test_values_file_takes_the_dataset_card_to_zero_placeholders(tmp_path):
    values_file = _values_file(tmp_path, DATASET_VALUES)
    result = runner.invoke(
        app,
        ["publish", "hf", "--values-file", str(values_file), "--only", "dataset", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "dataset card" in result.output


def test_only_dataset_uploads_one_card_and_never_touches_the_model_repo(
    tmp_path, monkeypatch
):
    api = _install_fake_hub(monkeypatch)
    values_file = _values_file(tmp_path, DATASET_VALUES)
    result = runner.invoke(
        app,
        ["publish", "hf", "--values-file", str(values_file), "--only", "dataset"],
    )
    assert result.exit_code == 0, result.output
    assert api.upload_file.call_count == 1
    kwargs = api.upload_file.call_args.kwargs
    assert kwargs["repo_id"] == HF_DATASET_ID
    assert kwargs["repo_type"] == "dataset"


def test_the_refusal_still_blocks_an_incomplete_selected_card(tmp_path):
    values = dict(DATASET_VALUES)
    del values["pos_tail"]  # one slot stays unfilled
    values_file = _values_file(tmp_path, values)
    result = runner.invoke(
        app,
        ["publish", "hf", "--values-file", str(values_file), "--only", "dataset"],
    )
    assert result.exit_code == 1
    assert "refusing to upload" in _flat(result.output)


def test_only_model_still_refuses_while_the_adapter_does_not_exist():
    """The model card has no adapter behind it; gating it alone must still refuse."""
    result = runner.invoke(app, ["publish", "hf", "--only", "model"])
    assert result.exit_code == 1
    assert "refusing to upload" in _flat(result.output)


def test_run_dir_prefers_the_report_shape_over_name_order(tmp_path, monkeypatch):
    """Aux JSONs sit next to the report; the report is the one with metadata+comparison."""
    (tmp_path / "aaa_report.json").write_text(
        json.dumps({"metadata": {"run_id": "r1"}, "comparison": []}),
        encoding="utf-8",
    )
    (tmp_path / "zzz_aux.json").write_text(
        json.dumps({"generated_at": "2026-09-21T00:00:00Z"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)  # keep the printed path short enough not to wrap
    result = runner.invoke(app, ["publish", "hf", "--run-dir", ".", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "aaa_report.json" in result.output
    assert "zzz_aux.json" not in result.output


def test_values_file_must_be_a_json_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    result = runner.invoke(
        app, ["publish", "hf", "--values-file", str(path), "--dry-run"]
    )
    assert result.exit_code == 1
    assert "must contain a JSON object" in _flat(result.output)


def test_values_file_must_exist(tmp_path):
    result = runner.invoke(
        app, ["publish", "hf", "--values-file", str(tmp_path / "nope.json"), "--dry-run"]
    )
    assert result.exit_code == 1
    assert "cannot read values file" in _flat(result.output)


def test_only_rejects_an_unknown_target():
    result = runner.invoke(app, ["publish", "hf", "--only", "universe", "--dry-run"])
    assert result.exit_code == 1
    assert "unknown --only value" in _flat(result.output)


# -- data files -----------------------------------------------------------------


def _write_parquet(path: Path, n_rows: int) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"ticker": [f"T{i}" for i in range(n_rows)]}), path)
    return path


def test_a_data_file_matching_the_card_uploads_after_it(tmp_path, monkeypatch):
    api = _install_fake_hub(monkeypatch)
    values_file = _values_file(tmp_path, DATASET_VALUES)
    panel = _write_parquet(tmp_path / "panel.parquet", int(DATASET_VALUES["n_rows"]))
    result = runner.invoke(
        app,
        [
            "publish", "hf", "--values-file", str(values_file),
            "--only", "dataset", "--data-file", str(panel),
        ],
    )
    assert result.exit_code == 0, result.output
    assert api.upload_file.call_count == 2, "card first, then the data file"
    card_call, file_call = api.upload_file.call_args_list
    assert card_call.kwargs["path_in_repo"] == "README.md"
    assert file_call.kwargs["path_in_repo"] == "panel.parquet"
    assert file_call.kwargs["repo_type"] == "dataset"
    assert "matches the card's n_rows" in _flat(result.output)


def test_a_data_file_contradicting_the_card_refuses_before_any_upload(tmp_path, monkeypatch):
    """A card saying 2221 rows next to a 3-row file is a published lie; catch it locally."""
    api = _install_fake_hub(monkeypatch)
    values_file = _values_file(tmp_path, DATASET_VALUES)
    panel = _write_parquet(tmp_path / "panel.parquet", 3)
    result = runner.invoke(
        app,
        [
            "publish", "hf", "--values-file", str(values_file),
            "--only", "dataset", "--data-file", str(panel),
        ],
    )
    assert result.exit_code == 1
    assert "contradicts its card" in _flat(result.output)
    assert api.upload_file.call_count == 0, "nothing may reach the network on a mismatch"


def test_a_missing_data_file_refuses_before_anything_uploads(tmp_path, monkeypatch):
    api = _install_fake_hub(monkeypatch)
    values_file = _values_file(tmp_path, DATASET_VALUES)
    result = runner.invoke(
        app,
        [
            "publish", "hf", "--values-file", str(values_file),
            "--only", "dataset", "--data-file", str(tmp_path / "nope.parquet"),
        ],
    )
    assert result.exit_code == 1
    assert "--data-file not found" in _flat(result.output)
    assert api.upload_file.call_count == 0


def test_dry_run_reports_the_data_file_check_without_uploading(tmp_path):
    values_file = _values_file(tmp_path, DATASET_VALUES)
    panel = _write_parquet(tmp_path / "panel.parquet", 3)  # mismatched on purpose
    result = runner.invoke(
        app,
        [
            "publish", "hf", "--values-file", str(values_file),
            "--only", "dataset", "--data-file", str(panel), "--dry-run",
        ],
    )
    assert result.exit_code == 1, "the mismatch must surface in dry-run too"
    assert "contradicts its card" in _flat(result.output)

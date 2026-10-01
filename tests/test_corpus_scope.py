"""The check that runs before a scoring run renders anything.

`eval lora` compares its rebuilt prompts against an SFT file byte for byte. That gate
names a symptom; this one names the cause, one step upstream and before the expensive
part.

The failure being guarded is the one measured on this repository on 2026-09-30: the
corpus at `data/processed/sft_stage2_wide` was rendered at a 13,764-character budget
because `lora.max_seq_length` is 8192 in the training overlay — and `eval lora` could
not read that overlay, computed 5,982 from the base config, and would have reported a
prompt mismatch on every row that admitted the extra text. The hint attached to that
failure tells the reader to rebuild the corpus, which is already correct.

Three values, not two: a corpus with no manifest, or one whose manifest predates the
budget fields, is *unverified* — and unverified must not read as a pass, because the
byte-for-byte comparison is then the only gate standing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shingan.eval.lora import corpus_scope


def write_manifest(directory: Path, **fields: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(fields), encoding="utf-8")
    return directory


def test_a_corpus_built_at_the_runs_own_budget_agrees(tmp_path: Path) -> None:
    directory = write_manifest(tmp_path / "sft", character_budget=13_764, max_seq_length=8192)
    scope = corpus_scope(directory, run_budget=13_764, run_seq_length=8192)
    assert scope.agrees is True
    assert scope.manifest_present is True


def test_a_corpus_built_at_another_budget_disagrees(tmp_path: Path) -> None:
    """The measured case: the corpus holds 13,764 and the run computes 5,982."""
    directory = write_manifest(
        tmp_path / "sft",
        character_budget=13_764,
        max_seq_length=8192,
        train_config="configs/train/qlora_qwen3_14b.yaml",
    )
    scope = corpus_scope(directory, run_budget=5_982, run_seq_length=4096)
    assert scope.agrees is False
    description = scope.describe()
    assert "13,764" in description and "5,982" in description
    # The file that set the corpus's budget is what makes the refusal actionable: the
    # repair is to pass that overlay, not to rebuild the corpus.
    assert "configs/train/qlora_qwen3_14b.yaml" in description


def test_a_missing_manifest_is_unverified_rather_than_a_pass(tmp_path: Path) -> None:
    scope = corpus_scope(tmp_path / "absent", run_budget=5_982, run_seq_length=4096)
    assert scope.agrees is None, "no manifest states no budget; that is not agreement"
    assert scope.manifest_present is False
    assert "unverified" in scope.describe()


def test_a_manifest_without_a_budget_is_unverified(tmp_path: Path) -> None:
    """An older corpus records `max_seq_length` but not the derived budget."""
    directory = write_manifest(tmp_path / "sft", max_seq_length=4096)
    scope = corpus_scope(directory, run_budget=5_982, run_seq_length=4096)
    assert scope.agrees is None
    assert scope.manifest_present is True
    assert scope.corpus_budget is None


def test_an_unreadable_manifest_is_unverified_not_an_exception(tmp_path: Path) -> None:
    """Scoring must not be refused because metadata is absent — only because it disagrees.

    Turning a missing record into a failed model is the mistake this keeps separate.
    """
    directory = tmp_path / "sft"
    directory.mkdir()
    (directory / "manifest.json").write_text("{not json", encoding="utf-8")
    scope = corpus_scope(directory, run_budget=5_982, run_seq_length=4096)
    assert scope.agrees is None
    assert scope.manifest_present is False


def test_a_manifest_holding_a_list_is_unverified(tmp_path: Path) -> None:
    directory = tmp_path / "sft"
    directory.mkdir()
    (directory / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    scope = corpus_scope(directory, run_budget=5_982, run_seq_length=4096)
    assert scope.agrees is None


def test_the_payload_records_both_sides_of_the_comparison(tmp_path: Path) -> None:
    directory = write_manifest(
        tmp_path / "sft",
        character_budget=5_982,
        max_seq_length=4096,
        train_config=None,
    )
    scope = corpus_scope(directory, run_budget=5_982, run_seq_length=4096)
    payload = scope.as_dict()
    assert payload["corpus_budget"] == 5_982
    assert payload["run_budget"] == 5_982
    assert payload["corpus_max_seq_length"] == 4_096
    assert payload["run_max_seq_length"] == 4_096
    assert payload["agrees"] is True
    # `null` rather than a string, so a reader cannot mistake it for a path.
    assert payload["corpus_train_config"] is None


@pytest.mark.parametrize("budget", [5_982, 13_764, 15_710])
def test_a_null_train_config_does_not_become_the_string_none(tmp_path: Path, budget: int) -> None:
    """`json` null must survive as None. `str(None)` would print 'None' as a file name."""
    directory = write_manifest(tmp_path / "sft", character_budget=budget, train_config=None)
    scope = corpus_scope(directory, run_budget=budget, run_seq_length=8192)
    assert scope.corpus_train_config is None
    assert "None" not in scope.describe()


def test_the_artifact_says_which_rows_the_budget_affects() -> None:
    """The fitted baselines are re-rendered at this run's budget, so a `text_baseline` row
    from two artifacts can be two measurements under one name. Only the prompt-reading
    rows are affected, and the caveat has to say which."""
    from test_lora_eval import sample_payload

    artifact = sample_payload(
        prompt={
            "integrity": {},
            "chars_budget": 13_764,
            "corpus": {
                "corpus_budget": 13_764,
                "run_budget": 13_764,
                "corpus_train_config": "configs/train/qlora_qwen3_14b.yaml",
                "agrees": True,
            },
        }
    )
    caveat = next((c for c in artifact["caveats"] if "13,764" in c), None)
    assert caveat is not None, "the artifact must state the budget its baselines used"
    assert "text_baseline" in caveat and "fused" in caveat
    assert "structured_matched" in caveat, (
        "the row the LoRA row is subtracted from must be named as unaffected, or the "
        "caveat reads as though the headline subtraction were in doubt"
    )
    assert "configs/train/qlora_qwen3_14b.yaml" in caveat


def test_no_such_caveat_when_the_corpus_names_no_overlay() -> None:
    """A corpus built from the base config cannot disagree with an `eval run` that used no
    overlay, so the caveat would be noise on the one case where nothing can differ."""
    from test_lora_eval import sample_payload

    artifact = sample_payload(
        prompt={
            "integrity": {},
            "chars_budget": 5_982,
            "corpus": {
                "corpus_budget": 5_982,
                "run_budget": 5_982,
                "corpus_train_config": None,
                "agrees": True,
            },
        }
    )
    assert not any("character budget, set by" in caveat for caveat in artifact["caveats"])

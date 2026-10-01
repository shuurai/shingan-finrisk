"""`max_seq_length` is a content budget, and the corpus must be built at the same one.

`lora.max_seq_length` does not only cap training-time sequence length. It is translated
by `chars_budget_for_seq_length` into the character budget the prompt's filing and news
sections are trimmed to fit, so it decides **how much source text reaches the text
track at all**.

That number lives in the training overlay (`configs/train/qlora_qwen3_14b.yaml`), and
until `data sft --train-config` existed the SFT command had no way to read it: `data sft`
passed `train_config=None` and rendered every prompt at the *base* config's limit while
the trainer truncated at the overlay's. The two numbers disagree, nothing raises, and the
corpus silently carries less text than either number suggests — the file a reader would
consult to answer "how much text does the model see" did not control it.

These tests pin the wiring, not the plumbing: the overlay has to be reachable from
`data sft`, forwarded to the manifest, and the manifest's budget has to follow it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from shingan.cli import app
from shingan.config import ProjectConfig, load_config
from shingan.paths import find_project_root
from shingan.pipeline import sft_examples
from shingan.prompts import chars_budget_for_seq_length

ROOT = find_project_root()
TRAIN_OVERLAY = ROOT / "configs" / "train" / "qlora_qwen3_14b.yaml"


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def test_data_sft_can_read_the_training_overlay() -> None:
    """The flag has to exist, or the value that sets the budget is unreachable from the
    command that spends it."""
    parameters = inspect.signature(__import__("shingan.cli", fromlist=["data_sft"]).data_sft).parameters
    assert "train_config" in parameters, (
        "data sft must accept the training overlay: lora.max_seq_length is a training "
        "setting that also governs this corpus"
    )


def test_the_budget_is_a_function_of_max_seq_length_and_grows_with_it() -> None:
    """The link the wiring depends on. If the budget ignored the sequence limit, passing
    the overlay through would be bookkeeping with no effect."""
    small = chars_budget_for_seq_length(4096)
    large = chars_budget_for_seq_length(8192)
    assert large > small, "a longer sequence limit must admit more source text"
    assert 0 < small < large


def test_the_training_overlay_changes_the_budget() -> None:
    """The overlay is the *only* place the wider budget exists, which is why forwarding
    the file is load-bearing rather than tidy.

    When this test was written the two files agreed at 4096, and it asserted the
    agreement so that the day they stopped agreeing would be noticed. That day is
    2026-09-30. The assertion is inverted rather than deleted: if someone sets the
    overlay back to the base value, this fails and says the flag is currently harmless —
    a reader then updates it deliberately instead of trusting a green run to mean the
    wiring is still doing something.
    """
    base = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    if not TRAIN_OVERLAY.is_file():
        pytest.skip("training overlay not present")
    merged = load_config(ROOT / "configs" / "default.yaml", [TRAIN_OVERLAY], root=ROOT)
    assert merged.lora.max_seq_length != base.lora.max_seq_length, (
        f"the overlay and the base config agree again at "
        f"{base.lora.max_seq_length}; `--train-config` remains correct but is no longer "
        "load-bearing anywhere. Update this test deliberately."
    )
    assert chars_budget_for_seq_length(merged.lora.max_seq_length) > chars_budget_for_seq_length(
        base.lora.max_seq_length
    ), "the divergence must be in the direction that admits more source text"


def test_a_raised_limit_would_reach_the_corpus_through_the_overlay(tmp_path) -> None:
    """End to end on the value that matters: an overlay that raises the limit produces a
    larger budget once it is merged, which is exactly what `data sft --train-config` does."""
    base = load_config(ROOT / "configs" / "default.yaml", root=ROOT)
    overlay = tmp_path / "long.yaml"
    overlay.write_text("lora:\n  max_seq_length: 8192\n", encoding="utf-8")
    merged = load_config(ROOT / "configs" / "default.yaml", [overlay], root=ROOT)

    assert chars_budget_for_seq_length(merged.lora.max_seq_length) > chars_budget_for_seq_length(
        base.lora.max_seq_length
    )


def test_the_cli_forwards_the_overlay_rather_than_dropping_it(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression itself. `data sft` used to call `_load_stack(train_config=None)`,
    so the overlay the user passed was accepted and then discarded."""
    import shingan.cli as cli_module
    import shingan.data.builder as builder_module
    import shingan.pipeline as pipeline_module

    overlay = tmp_path / "long.yaml"
    overlay.write_text("lora:\n  max_seq_length: 8192\n", encoding="utf-8")

    seen: dict[str, object] = {}

    class FakeBuild:
        def __init__(self, panel: pd.DataFrame) -> None:
            self.panel = panel
            self.n_sources: list[str] = []

    def fake_build_panel(config: ProjectConfig, write: bool = False):
        seen["build_max_seq_length"] = config.lora.max_seq_length
        return FakeBuild(pd.DataFrame({"ticker": ["TEST"], "as_of": [pd.Timestamp("2020-01-01")]}))

    def fake_sft_examples(panel, build, config, **kwargs):
        seen["sft_max_seq_length"] = config.lora.max_seq_length
        seen["train_config_path"] = kwargs.get("train_config_path")
        return {"train": [], "valid": []}, {
            "per_label": {},
            "prompts_truncated": 0,
            "character_budget": chars_budget_for_seq_length(config.lora.max_seq_length),
            "max_seq_length": config.lora.max_seq_length,
            "target_rule": "score = label_binary",
            "train_config": str(kwargs.get("train_config_path")),
            "data": {},
        }

    monkeypatch.setattr(builder_module, "build_panel", fake_build_panel)
    monkeypatch.setattr(pipeline_module, "sft_examples", fake_sft_examples)

    result = runner.invoke(
        app,
        ["data", "sft", "--train-config", str(overlay), "--out", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    assert seen.get("sft_max_seq_length") == 8192, (
        "the overlay reached the SFT build; before the fix this was the base config's 4096"
    )
    assert Path(str(seen.get("train_config_path"))) == overlay, "the path itself is forwarded"
    assert cli_module  # imported for the app; keeps the reference explicit


def test_the_manifest_records_which_config_set_the_budget() -> None:
    """Absent has to stay absent: `None` says "the base config set this", which is a
    different statement from naming a file that was never read."""
    assert "train_config_path" in inspect.signature(sft_examples).parameters
    parameter = inspect.signature(sft_examples).parameters["train_config_path"]
    assert parameter.default is None

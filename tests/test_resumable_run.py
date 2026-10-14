"""A run that cannot be resumed is a run that gets repeated, so refuse to start it.

These tests exist because of one specific lost run, and this time the loss was not a
``TypeError`` but 6.5 days of GPU time. On 2026-10-01 the wide-pool retrain was launched
with ``save_strategy`` and ``save_steps`` absent from the overlay, which left the
trainer's own defaults in force: **steps, every 500**. The run was 457 steps. So it
scheduled no checkpoint until its final save, reached step 246, and died -- leaving no
checkpoint, no loss value and no ``run.json``. Nothing to resume from, and nothing to
read the failure out of afterwards.

The gap was invisible for the same reason it was fatal: every run this project had done
before it was 108 steps, and a 108-step run also schedules nothing before its end save.
What changed was only the length of the run, and no check knew that checkpointing has to
scale with it.

Two things are pinned here, and neither needs a GPU:

* **The arithmetic.** ``scheduled_checkpoints`` answers "at which steps, before the last
  one, would a crash have been recoverable?" -- including the case that reads as
  configured but is not, one epoch with ``save_strategy: epoch``.
* **The refusal.** ``assert_run_is_resumable`` has to fire on the exact combination that
  lost the run, name the numbers, and offer both ways out.
"""

from __future__ import annotations

import pytest

from shingan.config import LoraConfig, load_config
from shingan.models.lora import (
    REDOABLE_STEPS,
    UnresumableRun,
    assert_run_is_resumable,
    build_sft_config_kwargs,
    scheduled_checkpoints,
    total_optimizer_steps,
)

#: The wide-pool training fold, and the run length that lost the 6.5 days.
WIDE_TRAIN_ROWS = 7307
WIDE_STEPS = 457


def test_the_wide_run_length_is_what_the_docs_claim() -> None:
    """457 is quoted in the config comment, in docs/09 section 24 and above. Pin the sum."""
    assert (
        total_optimizer_steps(WIDE_TRAIN_ROWS, batch_size=1, accumulate=16, epochs=1) == WIDE_STEPS
    )


def test_the_default_interval_schedules_checkpoints_inside_the_wide_run() -> None:
    """The regression guard. This is the default the defect was hiding in.

    One epoch, because that is what the shipped overlay sets: the dataclass default is 3
    and would triple the run, which is a different question from the one being asked here.
    """
    marks = scheduled_checkpoints(LoraConfig(num_train_epochs=1), WIDE_TRAIN_ROWS)

    assert marks, "the shipped default must put at least one checkpoint inside a 457-step run"
    assert marks == list(range(50, WIDE_STEPS, 50))
    assert len(marks) == 9


def test_the_trainers_own_default_never_fires_and_is_refused() -> None:
    """``save_steps: 500`` in a 457-step run: the exact configuration that lost the run."""
    config = LoraConfig(num_train_epochs=1, save_strategy="steps", save_steps=500)

    assert scheduled_checkpoints(config, WIDE_TRAIN_ROWS) == []

    with pytest.raises(UnresumableRun) as excinfo:
        assert_run_is_resumable(config, WIDE_TRAIN_ROWS)

    message = str(excinfo.value)
    assert "457" in message, "the refusal must say how long the run is"
    assert "save_steps=500" in message, "and which interval was in force"
    assert "lora.save_steps" in message and "num_train_epochs" in message, (
        "and both ways out: a smaller interval, or more than one epoch"
    )


def test_one_epoch_with_an_epoch_strategy_is_also_unresumable() -> None:
    """The case that reads as configured but is not.

    ``save_strategy: epoch`` sounds safer than ``steps``, and with one epoch it is exactly
    as unresumable: the end-of-epoch save *is* the final save.
    """
    config = LoraConfig(save_strategy="epoch", num_train_epochs=1)

    assert scheduled_checkpoints(config, WIDE_TRAIN_ROWS) == []

    with pytest.raises(UnresumableRun):
        assert_run_is_resumable(config, WIDE_TRAIN_ROWS)


def test_an_epoch_strategy_with_more_epochs_does_schedule_inside_the_run() -> None:
    """The remedy the refusal offers has to actually work.

    Three epochs of a 457-step run are 1,371 steps, so the epoch boundaries fall inside it.
    """
    config = LoraConfig(save_strategy="epoch", num_train_epochs=3)

    marks = scheduled_checkpoints(config, WIDE_TRAIN_ROWS)

    assert marks == [457, 914]
    assert_run_is_resumable(config, WIDE_TRAIN_ROWS)  # does not raise


def test_a_run_short_enough_to_redo_is_not_gated() -> None:
    """Redoing 2 steps is cheaper than resuming them, so the gate must stay out of the way.

    This is what keeps the probe and the smoke tests usable: they are deliberately shorter
    than the interval that protects the long runs.
    """
    short = REDOABLE_STEPS * 16  # exactly REDOABLE_STEPS optimisation steps, at one epoch
    unusable = LoraConfig(num_train_epochs=1, save_strategy="steps", save_steps=500)

    assert scheduled_checkpoints(unusable, short) == []
    assert_run_is_resumable(unusable, short)  # no raise
    assert_run_is_resumable(LoraConfig(save_strategy="no", num_train_epochs=1), short)  # no raise


def test_turning_checkpointing_off_outright_is_refused_on_a_long_run() -> None:
    """``no`` is a legitimate choice for a smoke test and not for a 457-step run."""
    config = LoraConfig(save_strategy="no", num_train_epochs=1)

    with pytest.raises(UnresumableRun) as excinfo:
        assert_run_is_resumable(config, WIDE_TRAIN_ROWS)

    assert "nothing to resume from at any point" in str(excinfo.value)


def test_the_mapping_carries_the_checkpoint_policy_to_the_trainer() -> None:
    """A config key the mapping drops is a key that does not exist."""
    kwargs = build_sft_config_kwargs(
        LoraConfig(),
        output_dir="artifacts/x",
        n_train_examples=WIDE_TRAIN_ROWS,
        eval_enabled=True,
        seed=13,
    )

    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 50
    assert kwargs["save_total_limit"] == 3


def test_the_shipped_overlay_produces_a_resumable_run() -> None:
    """The file, not the dataclass default: this is what the next launch will actually read."""
    project = load_config(overlays=["configs/train/qlora_qwen3_14b.yaml"])

    assert scheduled_checkpoints(project.lora, WIDE_TRAIN_ROWS), (
        "configs/train/qlora_qwen3_14b.yaml must schedule checkpoints inside the wide run"
    )
    assert_run_is_resumable(project.lora, WIDE_TRAIN_ROWS)  # does not raise

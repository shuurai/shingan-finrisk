"""Config-to-trainer-argument mapping, and the check that keeps it inside the API.

These tests exist because of one specific lost run. `shingan train lora` read
``lora.warmup_ratio`` from the YAML and passed it straight to ``trl.SFTConfig``, which
inherited the key from ``transformers.TrainingArguments`` — until ``transformers`` v5
removed it. The key was valid when the code was written and rejected by the time it ran,
and the failure surfaced *after* a 25 GB base-model download: two hours of network for a
``TypeError`` on a keyword.

So there are two things to keep true, and neither needs a GPU or the training extra:

* **The mapping.** ``build_sft_config_kwargs`` is a pure function of the config, so its
  contents can be asserted here. The regression guard is the first test: ``warmup_ratio``
  must never appear in the output, and ``warmup_steps`` must always carry its meaning.
* **The boundary.** ``assert_trainer_arguments_supported`` has to catch a key the
  installed class does not take, name *all* of them, and say which versions are in play.
  The last test runs it against the real ``SFTConfig`` when the extra is installed — it is
  the one that would have failed before the run instead of after it, and it skips in CI,
  which is why the pure-function tests above it carry the weight there.
"""

from __future__ import annotations

import json
import math

import pytest

from shingan.config import LoraConfig
from shingan.models.lora import (
    TrainingStackMismatch,
    accepted_arguments,
    assert_trainer_arguments_supported,
    build_sft_config_kwargs,
    dtype_keyword,
    library_versions,
    parse_major,
    resolve_warmup_steps,
    total_optimizer_steps,
    unsupported_arguments,
)


def _kwargs(**overrides: object) -> dict[str, object]:
    """The mapped arguments for a small, deterministic run."""
    config = LoraConfig(**overrides)  # type: ignore[arg-type]
    return build_sft_config_kwargs(
        config,
        output_dir="artifacts/lora",
        n_train_examples=160,
        eval_enabled=True,
        seed=7,
    )


# --------------------------------------------------------------------------- #
# The mapping
# --------------------------------------------------------------------------- #


def test_mapping_never_emits_a_removed_key() -> None:
    """The regression guard: `warmup_ratio` is gone from transformers >= 5.

    This is the assertion that fails on the old code, which is the point — a test suite
    that cannot reproduce the failure it was written for is decoration.
    """
    arguments = _kwargs()

    assert "warmup_ratio" not in arguments
    assert "warmup_steps" in arguments


def test_mapping_preserves_the_warmup_intent() -> None:
    """0.03 over 30 steps is 0.9 steps, and a run that short gets 1, not 0."""
    arguments = _kwargs()

    # 160 examples / (batch 1 * accumulate 16) = 10 steps per epoch, 3 epochs.
    assert total_optimizer_steps(160, batch_size=1, accumulate=16, epochs=3) == 30
    assert arguments["warmup_steps"] == 1


def test_warmup_ratio_of_zero_means_no_warmup() -> None:
    assert resolve_warmup_steps(LoraConfig(warmup_ratio=0.0), 160) == 0


def test_warmup_rounds_half_up_not_to_even() -> None:
    """A 0.5-step warmup is warmup the config asked for.

    The built-in ``round`` would return 0 here (round-half-to-even), silently turning a
    requested warmup into none.
    """
    # 10 steps per epoch * 1 epoch = 10 steps; 0.05 * 10 = 0.5.
    config = LoraConfig(num_train_epochs=1, warmup_ratio=0.05)

    assert resolve_warmup_steps(config, 160) == 1


def test_warmup_scales_with_the_ratio() -> None:
    """Half the run warmed up, when that is what the config says."""
    config = LoraConfig(num_train_epochs=1, warmup_ratio=0.5)

    assert resolve_warmup_steps(config, 160) == 5


def test_total_steps_composes_ceilings_identically() -> None:
    """The two-ceiling spelling equals the one-ceiling formula it stands for.

    Checked rather than asserted in a comment, because the docstring claims equality and
    a reader is entitled to see it hold at the boundaries where ceilings bite.
    """
    for n_examples in range(1, 200):
        for batch_size in (1, 2, 3, 8):
            for accumulate in (1, 2, 3, 16):
                composed = total_optimizer_steps(
                    n_examples, batch_size=batch_size, accumulate=accumulate, epochs=3
                )
                single = math.ceil(n_examples / (batch_size * accumulate)) * 3

                assert composed == single, (n_examples, batch_size, accumulate)


def test_eval_strategy_is_off_without_an_eval_file() -> None:
    """No validation fold means no evaluation, not an evaluation on the train fold."""
    config = LoraConfig()
    kwargs = build_sft_config_kwargs(
        config,
        output_dir="artifacts/lora",
        n_train_examples=160,
        eval_enabled=False,
        seed=7,
    )

    assert kwargs["eval_strategy"] == "no"
    assert _kwargs()["eval_strategy"] == "epoch"


def test_packing_stays_false_against_a_hand_edited_config() -> None:
    """The mapped arguments refuse packing even when the config asks for it.

    `assert_packing_disabled` raises first, so this is the second lock on the same door:
    it holds for a caller that reaches the trainer by another route.
    """
    arguments = _kwargs(packing=True)

    assert arguments["packing"] is False


def test_dataloader_workers_are_zero_regardless_of_config() -> None:
    """Windows does not fork; the argument is a constant, not a config read."""
    assert _kwargs(dataloader_num_workers=8)["dataloader_num_workers"] == 0


def test_seed_travels_into_both_seed_arguments() -> None:
    """One seed for the run, applied to both the data order and the init."""
    arguments = _kwargs()

    assert arguments["seed"] == 7
    assert arguments["data_seed"] == 7


def test_every_mapped_value_is_a_json_primitive() -> None:
    """The mapping is written into run.json, so it has to serialise."""
    arguments = _kwargs()

    assert json.loads(json.dumps(arguments)) == arguments


# --------------------------------------------------------------------------- #
# The boundary
# --------------------------------------------------------------------------- #


class _StrictConfig:
    """A stand-in with an explicit keyword list, like the real trainer config."""

    def __init__(self, alpha: int = 1, beta: str = "b") -> None:
        self.alpha = alpha
        self.beta = beta


class _LooseConfig:
    """A stand-in that accepts anything, so nothing can be rejected."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_accepted_arguments_reads_the_signature() -> None:
    assert accepted_arguments(_StrictConfig) == {"alpha", "beta"}


def test_accepted_arguments_is_none_for_arbitrary_keywords() -> None:
    """`None`, not an empty set: "cannot check" is not "nothing is allowed"."""
    assert accepted_arguments(_LooseConfig) is None
    assert unsupported_arguments({"anything": 1}, _LooseConfig) == []


def test_unsupported_arguments_lists_every_offender_sorted() -> None:
    assert unsupported_arguments({"beta": "b", "gamma": 1, "alpha": 1}, _StrictConfig) == ["gamma"]
    assert unsupported_arguments({"a": 1, "b": 2}, _StrictConfig) == ["a", "b"]


def test_mismatch_names_the_keys_and_the_versions() -> None:
    """The error has to be actionable on its own: which key, which versions, what next."""
    with pytest.raises(TrainingStackMismatch) as caught:
        assert_trainer_arguments_supported(
            {"warmup_ratio": 0.03, "beta": "b"}, _StrictConfig, label="SFTConfig"
        )

    message = str(caught.value)
    assert "SFTConfig" in message
    assert "warmup_ratio" in message
    assert "transformers=" in message
    assert ".[train]" in message


def test_no_mismatch_passes_silently() -> None:
    assert_trainer_arguments_supported({"alpha": 1}, _StrictConfig)


def test_mismatch_is_not_confused_with_a_missing_extra() -> None:
    """The two failures need different repairs, so they are different classes."""
    from shingan.models.lora import MissingTrainDependencies

    assert not issubclass(TrainingStackMismatch, MissingTrainDependencies)


def test_parse_major_handles_unparseable_versions() -> None:
    assert parse_major("5.17.0") == 5
    assert parse_major("2.11.0+cu128") == 2
    assert parse_major("0.21.0") == 0
    assert parse_major("unknown") is None
    assert parse_major("") is None


def test_dtype_keyword_follows_the_rename() -> None:
    """`dtype` on transformers >= 5, `torch_dtype` before — and when unknown, the old one.

    The old spelling is the only one valid on both sides of the rename, so an
    unclassifiable version has to fall back to it rather than to the new name.
    """
    assert dtype_keyword("5.17.0") == "dtype"
    assert dtype_keyword("6.0.0.dev0") == "dtype"
    assert dtype_keyword("4.53.2") == "torch_dtype"
    assert dtype_keyword("unknown") == "torch_dtype"


def test_library_versions_reports_absence_instead_of_raising() -> None:
    versions = library_versions("pytest", "a-distribution-that-does-not-exist")

    assert versions["a-distribution-that-does-not-exist"] == "not installed"
    assert versions["pytest"] not in ("", "not installed")


@pytest.mark.parametrize("key", ["warmup_ratio"])
def test_the_removed_key_is_actually_missing_here(key: str) -> None:
    """Documents *why* the mapping translates, against the installed library.

    Skipped when the training extra is absent, since the claim is about the installed
    transformer version rather than about this package. If a future release puts the
    ratio back, this test says so instead of leaving the translation unexplained.
    """
    transformers = pytest.importorskip("transformers")
    from trl import SFTConfig

    accepted = accepted_arguments(SFTConfig)
    assert accepted is not None

    major = parse_major(transformers.__version__)
    if major is not None and major >= 5:
        assert key not in accepted
    else:
        pytest.skip(f"transformers {transformers.__version__} predates the removal")


def test_mapping_is_accepted_by_the_installed_trainer_config() -> None:
    """The test that would have failed before the run rather than after it.

    Needs the `train` extra, so it skips in CI. It is still the most valuable test in the
    file on the machine that actually trains: it runs the real mapping against the real
    class, which is the only way to notice a key disappearing in a new major without
    paying for a download first.
    """
    pytest.importorskip("trl")
    from trl import SFTConfig

    assert_trainer_arguments_supported(_kwargs(), SFTConfig, label="SFTConfig")

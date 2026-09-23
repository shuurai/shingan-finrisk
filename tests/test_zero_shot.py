"""The base model as a scored row, and the difference between the two text arms.

``text_only_zero_shot`` is in docs/05's ablation table as the row that shows "the
increment fine-tuning bought". It did not exist: ``eval run`` had no text-model arm at
all, and the adapter row had nothing to be subtracted from. Adding it introduces three
decisions that are not obvious from the code and are therefore pinned here:

* **Which base weights.** A run that loads an adapter for its base and a base model for
  its arm is comparing two models, not measuring an adapter. Naming both is therefore
  allowed only when they agree, and refused with both values in the message when they do
  not.
* **Which tokenizer.** The tokenizer saved with the adapter disagrees with the base
  model's about BOS (the training log recorded ``Updated tokens: {'bos_token_id': None}``).
  A zero-shot arm rendered with the base model's own tokenizer would differ from the
  adapter arm in tokenisation *and* weights, so the identity keeps the two facts separate:
  no adapter is named, but the prompt is still rendered by the adapter directory's
  tokenizer.
* **Which arm is a baseline.** Two arms are not two results — the quantity of interest is
  their difference, so the plan pairs them, and the pairing only exists if both arms were
  generated in one process from one prompt set.

Nothing here imports torch: the identity is pure, and the two functions that do need torch
refuse their bad arguments before probing for it, which is exactly the behaviour a CI
machine without the training stack can check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from shingan.eval.lora import (
    ARMS_BY_MODE,
    PATH_LORA,
    PATH_ZERO_SHOT,
    PROMPT_DISCLOSURE,
    ZERO_SHOT_DISCLOSURE,
    arms_for_mode,
    build_payload,
    difference_plan,
    paired_differences,
    render_markdown,
    score_from_attempts,
)
from shingan.models.lora_inference import (
    MODE_ADAPTER,
    MODE_BOTH,
    MODE_ZERO_SHOT,
    attach_adapter,
    describe_model,
    load_for_inference,
)

BASE = "Qwen/Qwen3-14B"


def write_adapter(directory: Path, *, base_model: str = BASE, tokenizer: bool = True) -> Path:
    """A directory shaped like the one ``shingan train lora`` writes."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model,
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "k_proj"],
            }
        ),
        encoding="utf-8",
    )
    (directory / "adapter_model.safetensors").write_bytes(b"not really weights")
    if tokenizer:
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return directory


# -- which rows each mode produces, and in what order ---------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (MODE_ADAPTER, (PATH_LORA,)),
        (MODE_ZERO_SHOT, (PATH_ZERO_SHOT,)),
        (MODE_BOTH, (PATH_ZERO_SHOT, PATH_LORA)),
    ],
)
def test_each_mode_names_its_arms(mode: str, expected: tuple[str, ...]) -> None:
    assert arms_for_mode(mode) == expected


def test_the_adapter_arm_is_last_because_attaching_mutates_the_model() -> None:
    """The base arm has to be generated before the adapter is wrapped around it.

    This is an ordering assertion about a data structure because that is the only place
    the ordering is written down. `eval_lora` iterates this tuple and attaches the adapter
    at the arm whose path is `text_only_lora`; reversing it would score the adapter twice
    and publish the second pass as the base model's.
    """
    arms = arms_for_mode(MODE_BOTH)

    assert arms.index(PATH_LORA) == len(arms) - 1
    assert ARMS_BY_MODE["adapter"] == (PATH_LORA,)


def test_an_unknown_mode_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match=r"zero_shot.*both|both.*zero_shot"):
        arms_for_mode("zero-shot")


# -- which arm is a baseline ----------------------------------------------------


def test_a_single_arm_is_compared_against_the_fitted_baselines() -> None:
    plan = difference_plan((PATH_ZERO_SHOT,), ("structured_matched", "text_baseline"))

    assert plan == [
        {
            "candidate": PATH_ZERO_SHOT,
            "baselines": ("structured_matched", "text_baseline"),
        }
    ]


def test_a_two_arm_run_pairs_the_arms_in_their_own_group() -> None:
    """The arm-versus-arm comparison and the arm-versus-baseline one are separate groups.

    Not a stylistic choice: `paired_differences` restricts a comparison to rows where every
    arm it was handed has a score, so one group containing both would let a text arm that
    parsed almost nothing shorten the adapter arm's comparison against the *fitted*
    baselines — a comparison that does not involve it.
    """
    plan = difference_plan((PATH_ZERO_SHOT, PATH_LORA), ("structured_matched",))

    assert plan == [
        {"candidate": PATH_ZERO_SHOT, "baselines": ("structured_matched",)},
        {"candidate": PATH_LORA, "baselines": (PATH_ZERO_SHOT,)},
        {"candidate": PATH_LORA, "baselines": ("structured_matched",)},
    ]


def test_one_arms_failures_do_not_shorten_another_arms_baseline_comparison() -> None:
    """Measured: the zero-shot arm parsed 4 of 64 rows, all negatives.

    Put in one group, that collapsed `lora - structured_matched` from 64 rows to 4 and
    reported "not measured" for it. In separate groups the adapter arm keeps its 64 rows
    and only the arm-versus-arm comparison shrinks to the intersection.
    """
    n = 24
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime("2020-01-01") + pd.to_timedelta(pd.RangeIndex(n) * 7, unit="D"),
            "y_true": [1 if i % 4 == 0 else 0 for i in range(n)],
            PATH_LORA: [0.2] * n,
            "structured_matched": [0.6 if i % 4 == 0 else 0.25 for i in range(n)],
            PATH_ZERO_SHOT: [float("nan")] * n,
        }
    )
    frame.loc[:5, PATH_ZERO_SHOT] = 0.4  # six rows the base model managed to answer

    by_group = {}
    for group in difference_plan((PATH_ZERO_SHOT, PATH_LORA), ("structured_matched",)):
        records = paired_differences(
            frame,
            candidate=group["candidate"],
            baselines=group["baselines"],
            n_boot=20,
            block_days=30,
            alpha=0.05,
            seed=3,
        )
        by_group[(group["candidate"], group["baselines"])] = {item["n_rows"] for item in records}

    assert by_group[(PATH_LORA, ("structured_matched",))] == {n}
    assert by_group[(PATH_LORA, (PATH_ZERO_SHOT,))] == {6}
    assert by_group[(PATH_ZERO_SHOT, ("structured_matched",))] == {6}


def test_no_arm_is_its_own_baseline() -> None:
    for group in difference_plan((PATH_ZERO_SHOT, PATH_LORA), ("text_baseline",)):
        assert group["candidate"] not in group["baselines"]


def test_an_empty_plan_is_refused() -> None:
    with pytest.raises(ValueError, match="no arms to compare"):
        difference_plan((), ("text_baseline",))


def test_a_repeated_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="listed twice"):
        difference_plan((PATH_LORA, PATH_LORA), ())


def test_an_arm_cannot_also_be_a_fitted_baseline() -> None:
    with pytest.raises(ValueError, match="both an arm and a baseline"):
        difference_plan((PATH_LORA,), (PATH_LORA,))


# -- what will be loaded --------------------------------------------------------


def test_a_zero_shot_identity_names_no_adapter_but_keeps_its_tokenizer(tmp_path: Path) -> None:
    """No adapter is scored, so none is named — but the prompt is still its tokenizer's.

    Both halves matter. Naming an adapter would put its hash next to numbers it never
    touched; taking the base model's own tokenizer would make the two arms differ in
    tokenisation as well as in weights, and the comparison would measure both.
    """
    directory = write_adapter(tmp_path / "adapter")

    identity = describe_model(mode=MODE_ZERO_SHOT, adapter_dir=directory)

    assert identity.adapter is None
    assert identity.base_model == BASE
    assert identity.base_model_source.startswith("adapter directory")
    assert identity.tokenizer_source == "adapter directory"
    assert identity.tokenizer_path == str(directory)


def test_an_adapter_identity_names_the_adapter_and_hashes_it(tmp_path: Path) -> None:
    directory = write_adapter(tmp_path / "adapter")

    identity = describe_model(mode=MODE_ADAPTER, adapter_dir=directory)

    assert identity.adapter is not None
    assert len(identity.adapter.weights_sha256) == 64
    assert identity.adapter.rank == 32


def test_both_mode_keeps_the_adapter_because_it_scores_that_arm(tmp_path: Path) -> None:
    identity = describe_model(mode=MODE_BOTH, adapter_dir=write_adapter(tmp_path / "adapter"))

    assert identity.adapter is not None


def test_a_tokenizer_less_directory_falls_back_and_says_so(tmp_path: Path) -> None:
    directory = write_adapter(tmp_path / "adapter", tokenizer=False)

    identity = describe_model(mode=MODE_ZERO_SHOT, adapter_dir=directory)

    assert identity.tokenizer_source == "base model"
    assert identity.tokenizer_path == BASE


def test_a_base_model_alone_is_enough(tmp_path: Path) -> None:
    identity = describe_model(mode=MODE_ZERO_SHOT, base_model=BASE)

    assert identity.base_model_source == "named explicitly"
    assert identity.tokenizer_source == "base model"
    assert identity.adapter is None


def test_a_disagreeing_base_model_is_refused_with_both_names(tmp_path: Path) -> None:
    """The failure this prevents is silent: two arms on different weights look normal."""
    directory = write_adapter(tmp_path / "adapter", base_model=BASE)

    with pytest.raises(ValueError, match=r"Qwen/Qwen3-8B.*Qwen/Qwen3-14B"):
        describe_model(mode=MODE_ZERO_SHOT, adapter_dir=directory, base_model="Qwen/Qwen3-8B")


def test_nothing_to_describe_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to describe"):
        describe_model(mode=MODE_ZERO_SHOT)


def test_an_unknown_mode_is_refused_by_the_identity_too() -> None:
    with pytest.raises(ValueError, match="unknown scoring mode"):
        describe_model(mode="fine-tuned", base_model=BASE)


def test_an_unquantised_run_says_so() -> None:
    identity = describe_model(mode=MODE_ZERO_SHOT, base_model=BASE, load_in_4bit=False)

    assert identity.quantization == "bfloat16, unquantised"


def test_the_recorded_quantisation_names_the_one_the_loader_uses(tmp_path: Path) -> None:
    directory = write_adapter(tmp_path / "adapter")

    identity = describe_model(mode=MODE_ADAPTER, adapter_dir=directory)

    assert identity.quantization == "4-bit nf4 double-quantised, bfloat16 compute"


def test_attaching_without_an_adapter_is_refused_before_importing_peft() -> None:
    """A no-op here would publish a base-model score under the adapter's name."""
    identity = describe_model(mode=MODE_ZERO_SHOT, base_model=BASE)

    with pytest.raises(ValueError, match="names no adapter"):
        attach_adapter(object(), identity)


def test_loading_with_attach_on_a_zero_shot_identity_is_refused_before_torch() -> None:
    """Runs on a machine with no training stack, which is where this is easy to get wrong."""
    identity = describe_model(mode=MODE_ZERO_SHOT, base_model=BASE)

    with pytest.raises(ValueError, match="mode 'zero_shot' does not score an adapter arm"):
        load_for_inference(identity, attach=True)


# -- the artifact ---------------------------------------------------------------


def zero_shot_payload(arms: list[str], runs: dict) -> dict:
    return build_payload(
        label="tail_risk",
        split="test",
        rows=[run.as_dict() for run in runs.values()],
        scored=runs,
        arms=arms,
        model={
            "mode": "both",
            "base_model": BASE,
            "base_model_source": "adapter directory (artifacts/lora/adapter)",
            "tokenizer_source": "adapter directory",
            "tokenizer_path": "artifacts/lora/adapter",
            "quantization": "4-bit nf4 double-quantised, bfloat16 compute",
            "adapter": None,
        },
        generation={"policy": "greedy"},
        prompt={"integrity": {"checked": False}, "integrity_enforced": False},
        data={"is_synthetic": False},
        predictions={PATH_ZERO_SHOT: [{"ticker": "AAA", "score": None}]},
    )


def an_arm(path: str, scores: list, reasons: list[str]):
    """A two-row run for one arm. ``path`` is not cosmetic: the comparison table marks a
    row as an arm by matching this against the run's arm list, so a run scored under the
    default path would be rendered as a baseline."""
    return score_from_attempts([0, 1], scores, reasons, label="x", path=path)


def test_the_zero_shot_row_carries_its_own_disclosure() -> None:
    run = an_arm(PATH_ZERO_SHOT, [None, None], ["rejected", "rejected"])

    payload = zero_shot_payload([PATH_ZERO_SHOT], {PATH_ZERO_SHOT: run})

    assert PROMPT_DISCLOSURE in payload["caveats"]
    assert ZERO_SHOT_DISCLOSURE in payload["caveats"]
    assert "no adapter" in render_markdown(payload)


def test_the_parse_accounting_is_per_arm() -> None:
    """Two arms have two failure rates; "the" failure rate stopped being one number."""
    base = an_arm(PATH_ZERO_SHOT, [None, 0.8], ["rejected", ""])
    adapter = an_arm(PATH_LORA, [0.2, 0.8], ["", ""])

    payload = zero_shot_payload(
        [PATH_ZERO_SHOT, PATH_LORA], {PATH_ZERO_SHOT: base, PATH_LORA: adapter}
    )

    assert payload["parse"][PATH_ZERO_SHOT]["n_dropped"] == 1
    assert payload["parse"][PATH_LORA]["n_dropped"] == 0
    assert payload["predictions"][PATH_LORA] == []


def test_only_the_arm_that_dropped_something_gets_the_caveat() -> None:
    base = an_arm(PATH_ZERO_SHOT, [None, 0.8], ["rejected", ""])
    adapter = an_arm(PATH_LORA, [0.2, 0.8], ["", ""])

    payload = zero_shot_payload(
        [PATH_ZERO_SHOT, PATH_LORA], {PATH_ZERO_SHOT: base, PATH_LORA: adapter}
    )
    mentioned = [c for c in payload["caveats"] if "did not parse" in c]

    assert len(mentioned) == 1
    assert PATH_ZERO_SHOT in mentioned[0]


def test_the_table_marks_which_rows_came_from_a_model() -> None:
    run = an_arm(PATH_ZERO_SHOT, [0.2, 0.8], ["", ""])
    payload = zero_shot_payload([PATH_ZERO_SHOT], {PATH_ZERO_SHOT: run})
    payload["rows"].append({"path": "structured", "auc": 0.6})

    markdown = render_markdown(payload)

    assert f"| {PATH_ZERO_SHOT} | arm |" in markdown
    assert "| structured | baseline |" in markdown


def test_arms_that_disagree_with_the_runs_are_refused() -> None:
    """An arm in the table but not in the accounting is a row nothing explains."""
    run = an_arm(PATH_ZERO_SHOT, [0.2, 0.8], ["", ""])

    with pytest.raises(ValueError, match="not the same set"):
        zero_shot_payload([PATH_ZERO_SHOT, PATH_LORA], {PATH_ZERO_SHOT: run})


def test_a_run_with_no_arms_is_refused() -> None:
    with pytest.raises(ValueError, match="no arm was scored"):
        zero_shot_payload([], {})

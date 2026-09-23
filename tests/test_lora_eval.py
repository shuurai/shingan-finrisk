"""Scoring an adapter: the parts that decide whether a number may be quoted.

The regression these tests exist for is not a crash. Scoring the fine-tuned model was
missing entirely — ``eval run`` fitted structured, TF-IDF and fusion, and the adapter was
never run — so the project could not answer the question it exists to ask. Adding a path
therefore has to come with the accounting that makes its output readable, and that
accounting is what is tested here:

* the parse taxonomy, because ``None`` from the parser cannot distinguish "the model
  wrote prose" from "the model answered a different label", and the two have different
  fixes;
* the drop accounting, because a dropped row is not a random row — a model that fails on
  its longest prompts looks better after the failures are removed;
* the prompt-comparison key, which is ``(sample_id, label)``. Keyed on the id alone, a
  correct three-label SFT file reports two mismatches in every three rows, which is how
  this was found;
* strict JSON on the way out. An artifact whose AUC is unmeasurable serialises as
  ``NaN`` by default, and no strict parser will read the file back.

Nothing here imports torch. Data loading and generation are exercised by running the
command; these are the decision rules underneath it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from shingan.eval.lora import (
    PATH_LORA,
    PROMPT_DISCLOSURE,
    PromptIntegrity,
    _json_safe,
    build_payload,
    compare_prompts,
    paired_differences,
    render_markdown,
    score_from_attempts,
    write_artifact,
)
from shingan.models.lora_inference import (
    MissingInferenceDependencies,
    adapter_facts,
    missing_inference_modules,
    score_generation,
)
from shingan.prompts import SYSTEM_PROMPT

HORIZON = 30


def payload_for(
    *,
    label: str = "tail_risk",
    score: float = 0.2,
    horizon_days: int = HORIZON,
    reasons: list[str] | None = None,
    extra: dict | None = None,
) -> str:
    body = {
        "label": label,
        "severity": "medium",
        "score": score,
        "horizon_days": horizon_days,
        "reasons": ["a reason"] if reasons is None else reasons,
        "evidence": [],
    }
    body.update(extra or {})
    return json.dumps(body)


def sft_record(sample_id: str, label: str, content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "meta": {"sample_id": sample_id, "label": label},
    }


# -- the parse taxonomy ---------------------------------------------------------


def test_a_well_formed_generation_scores() -> None:
    attempt = score_generation(
        payload_for(score=0.37), expected_label="tail_risk", expected_horizon_days=HORIZON
    )

    assert attempt.parsed
    assert attempt.score == pytest.approx(0.37)
    assert attempt.assessment is not None
    assert attempt.assessment.severity.value == "medium"


def test_a_fenced_generation_still_scores() -> None:
    """Models fence their JSON despite being told not to, and a fence is not a failure."""
    fenced = f"Here is the assessment:\n```json\n{payload_for(score=0.4)}\n```\n"

    attempt = score_generation(fenced, expected_label="tail_risk", expected_horizon_days=HORIZON)

    assert attempt.parsed
    assert attempt.score == pytest.approx(0.4)


def test_prose_is_rejected_with_a_reason_that_names_the_problem() -> None:
    attempt = score_generation("I cannot assess this company.", expected_label="tail_risk")

    assert not attempt.parsed
    assert attempt.score is None
    assert "no JSON object" in attempt.reason


def test_the_wrong_label_is_rejected_and_says_which_was_asked_for() -> None:
    """A different label is a different question, so its score is not comparable.

    The distinction is why the parser hands back a reason: filling a score in anyway
    would mix answers to three questions into one metric.
    """
    attempt = score_generation(
        payload_for(label="default_risk"), expected_label="tail_risk", expected_horizon_days=HORIZON
    )

    assert not attempt.parsed
    assert "default_risk" in attempt.reason
    assert "tail_risk" in attempt.reason


def test_the_wrong_horizon_is_rejected() -> None:
    attempt = score_generation(
        payload_for(horizon_days=365), expected_label="tail_risk", expected_horizon_days=HORIZON
    )

    assert not attempt.parsed
    assert "horizon" in attempt.reason


def test_a_schema_violation_is_rejected() -> None:
    attempt = score_generation(
        payload_for(score=1.5), expected_label="tail_risk", expected_horizon_days=HORIZON
    )

    assert not attempt.parsed
    assert "schema validation" in attempt.reason


def test_an_assessment_without_reasons_is_rejected() -> None:
    """The schema refuses it, and the refusal has to survive the parse path."""
    attempt = score_generation(
        payload_for(reasons=[]), expected_label="tail_risk", expected_horizon_days=HORIZON
    )

    assert not attempt.parsed


# -- the drop accounting --------------------------------------------------------


def test_unparsed_rows_are_dropped_and_counted() -> None:
    """The drop count and the dropped *positives* are part of the metric's meaning.

    Dropping a positive because the model produced prose biases the metric downwards in
    the direction a reader will not guess, so the count sits next to the number.
    """
    run = score_from_attempts(
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0.1, 0.9, None, 0.8, 0.2, None, 0.3, 0.7],
        [
            "",
            "",
            "no JSON object found in model output",
            "",
            "",
            "no JSON object found in model output",
            "",
            "",
        ],
        label="tail_risk",
        path=PATH_LORA,
    )

    assert run.n_attempted == 8
    assert run.n_parsed == 6
    assert run.n_dropped == 2
    assert run.n_dropped_positives == 1
    assert run.failure_rate == pytest.approx(0.25)
    assert run.failure_reasons == {"no JSON object found in model output": 2}
    assert run.report.n_rows == 6


def test_a_run_with_nothing_parsing_produces_no_numbers_rather_than_an_error() -> None:
    """Zero usable rows is a result about the model, not a crash in the harness."""
    run = score_from_attempts(
        [0, 1, 1, 0],
        [None, None, None, None],
        ["rejected"] * 4,
        label="tail_risk",
        path=PATH_LORA,
    )

    assert run.n_parsed == 0
    assert run.failure_rate == pytest.approx(1.0)
    assert run.report.n_rows == 0
    assert not math.isfinite(run.report.auc)


def test_the_failure_rate_of_an_empty_run_is_zero_not_a_division_error() -> None:
    run = score_from_attempts([], [], [], label="tail_risk", path=PATH_LORA)

    assert run.n_attempted == 0
    assert run.failure_rate == 0.0


# -- the prompt comparison ------------------------------------------------------


def test_one_label_out_of_three_compares_only_its_own_rows() -> None:
    """The key is (sample_id, label), because the id repeats across labels.

    This is the regression: the SFT file holds one example per row *and* label, so a
    comparison keyed on the sample id alone finds two rows under every id whose task tag
    differs and reports a correct file as broken.
    """
    records = [
        sft_record("AAA-20200101", "default_risk", "prompt for default"),
        sft_record("AAA-20200101", "tail_risk", "prompt for tail"),
        sft_record("AAA-20200101", "fraud_risk", "prompt for fraud"),
    ]
    rebuilt = {("AAA-20200101", "tail_risk"): "prompt for tail"}

    integrity = compare_prompts(records, rebuilt)

    assert integrity.n_scanned == 3
    assert integrity.n_compared == 1
    assert integrity.n_matching == 1
    assert integrity.n_other_label == 2
    assert integrity.ok


def test_a_mismatch_is_reported_with_the_offset() -> None:
    records = [
        sft_record("AAA-20200101", "tail_risk", "<TASK>label=tail_risk horizon_days=30</TASK>")
    ]
    rebuilt = {("AAA-20200101", "tail_risk"): "<TASK>label=default_risk horizon_days=365</TASK>"}

    integrity = compare_prompts(records, rebuilt)

    assert not integrity.ok
    assert integrity.n_compared == 1
    assert integrity.n_matching == 0
    assert integrity.examples[0]["first_difference"]["offset"] > 0


def test_a_row_the_adapter_trained_on_but_scoring_cannot_rebuild_is_a_failure() -> None:
    """Missing is not the same as other-label, and only one of them is acceptable."""
    records = [
        sft_record("AAA-20200101", "tail_risk", "p1"),
        sft_record("BBB-20200101", "tail_risk", "p2"),
    ]
    rebuilt = {("AAA-20200101", "tail_risk"): "p1"}

    integrity = compare_prompts(records, rebuilt)

    assert integrity.n_rebuilt_missing == 1
    assert integrity.n_other_label == 0
    assert not integrity.ok


def test_records_without_metadata_are_counted_not_silently_skipped() -> None:
    records = [{"messages": [{"role": "user", "content": "x"}]}, {"meta": {}, "messages": []}]

    integrity = compare_prompts(records, {})

    assert integrity.n_unkeyed == 2
    assert integrity.n_compared == 0
    assert not integrity.ok


def test_an_empty_comparison_is_not_a_pass() -> None:
    """No rows compared is a check that did not run, which must not read as success."""
    assert not PromptIntegrity().ok


def test_the_verdict_is_exposed_as_a_plain_bool() -> None:
    """``ok`` decides whether a run proceeds, so it must not be truthy-by-accident."""
    integrity = compare_prompts([sft_record("A", "tail_risk", "p")], {("A", "tail_risk"): "p"})

    assert integrity.as_dict()["ok"] is True
    assert isinstance(integrity.as_dict()["ok"], bool)


def test_an_edited_instruction_contract_is_caught() -> None:
    """The system turn was not compared at all until this check existed.

    Every example in an SFT file carries the same system turn, so editing `SYSTEM_PROMPT`
    between training and scoring changes what the adapter is shown without changing any
    user turn — the row-by-row comparison stays byte-identical and reports success. This
    is not hypothetical: the reason to edit it is a defect the base model exposed
    (`source_type` is not enumerated, see docs/09 section 12.6), and the person fixing it
    would otherwise rescore the adapter on a contract it never trained under.
    """
    record = sft_record("A", "tail_risk", "p")
    record["messages"][0]["content"] = "an older instruction contract"

    integrity = compare_prompts([record], {("A", "tail_risk"): "p"})

    assert integrity.n_system_checked == 1
    assert integrity.n_system_mismatch == 1
    assert integrity.system_example is not None
    assert integrity.system_example["first_difference"]["offset"] == 0
    assert not integrity.ok
    # The user turn still matched: this failure is invisible without the system check.
    assert integrity.n_matching == integrity.n_compared


# -- the artifact ---------------------------------------------------------------


def sample_payload(**overrides: object) -> dict:
    """``build_payload`` with the uneventful arguments filled in.

    Three of its arguments became plural when the zero-shot arm arrived, and the artifact
    tests all need a payload. Spelling the same eight lines out in each of them is how one
    of them ends up not exercising what its name claims.
    """
    run = score_from_attempts([0, 1], [0.2, 0.8], ["", ""], label="tail_risk")
    defaults: dict[str, object] = {
        "label": "tail_risk",
        "split": "test",
        "rows": [run.as_dict()],
        "scored": {PATH_LORA: run},
        "arms": [PATH_LORA],
        "model": {
            "mode": "adapter",
            "base_model": "Qwen/Qwen3-14B",
            "base_model_source": "adapter directory (artifacts/lora/adapter)",
            "tokenizer_source": "adapter directory",
            "tokenizer_path": "artifacts/lora/adapter",
            "quantization": "4-bit nf4 double-quantised, bfloat16 compute",
            "adapter": {
                "weights_file": "adapter_model.safetensors",
                "weights_sha256": "ab" * 32,
                "weights_bytes": 17,
                "rank": 32,
            },
        },
        "generation": {"policy": "greedy"},
        "prompt": {"integrity": {}},
        "data": {"is_synthetic": True},
    }
    defaults.update(overrides)
    return build_payload(**defaults)  # type: ignore[arg-type]


def test_non_finite_metrics_become_null_rather_than_nan() -> None:
    """``json.dumps`` writes bare ``NaN``, which is not JSON and cannot be read back."""
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe({"a": [float("nan"), 1.5], "b": {"c": float("-inf")}}) == {
        "a": [None, 1.5],
        "b": {"c": None},
    }
    assert _json_safe(2.5) == 2.5


def test_written_artifact_parses_strictly(tmp_path: Path) -> None:
    run = score_from_attempts([0, 1], [None, None], ["rejected", "rejected"], label="x")

    written = write_artifact(
        sample_payload(rows=[run.as_dict()], scored={PATH_LORA: run}), tmp_path
    )
    text = written["lora_eval_json"].read_text(encoding="utf-8")

    assert "NaN" not in text
    assert json.loads(text)["rows"][0]["auc"] is None
    assert written["lora_eval_markdown"].is_file()


def test_the_payload_carries_the_disclosure_that_makes_the_row_readable() -> None:
    """The row is named ``text_only_lora`` and is not text-only. The payload must say so."""
    payload = sample_payload(prompt={"includes_structured_signals": True})

    assert PROMPT_DISCLOSURE in payload["caveats"]
    assert any("synthetic" in caveat for caveat in payload["caveats"])
    assert payload["prompt"]["includes_structured_signals"] is True


def test_a_dropped_positive_is_disclosed_in_the_caveats() -> None:
    run = score_from_attempts([1, 1], [None, 0.4], ["rejected", ""], label="tail_risk")

    payload = sample_payload(
        rows=[run.as_dict()], scored={PATH_LORA: run}, data={"is_synthetic": False}
    )

    assert run.n_dropped_positives == 1
    assert any("did not parse" in caveat for caveat in payload["caveats"])
    assert any(PATH_LORA in caveat for caveat in payload["caveats"]), (
        "with more than one arm possible, a drop count that does not name its arm is a "
        "number the reader has to guess at"
    )


def test_predictions_are_included_so_the_metrics_can_be_recomputed() -> None:
    predictions = [
        {"ticker": "AAA", "as_of": "2020-01-01", "y_true": 0, "score": 0.2},
        {"ticker": "AAA", "as_of": "2020-04-01", "y_true": 1, "score": 0.8},
    ]

    payload = sample_payload(predictions={PATH_LORA: predictions})

    assert payload["predictions"][PATH_LORA] == predictions


def test_the_markdown_says_not_measured_rather_than_printing_nan() -> None:
    run = score_from_attempts([0, 1], [None, None], ["rejected", "rejected"], label="x")
    markdown = render_markdown(sample_payload(rows=[run.as_dict()], scored={PATH_LORA: run}))

    assert "not measured" in markdown
    assert "nan" not in markdown.lower()


# -- adapter identity -----------------------------------------------------------


def write_adapter(directory: Path, *, base_model: str = "Qwen/Qwen3-14B") -> Path:
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
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return directory


def test_adapter_facts_hash_the_weights_and_name_the_tokenizer(tmp_path: Path) -> None:
    facts = adapter_facts(write_adapter(tmp_path / "adapter"))

    assert facts.base_model == "Qwen/Qwen3-14B"
    assert facts.rank == 32
    assert facts.weights_bytes == len(b"not really weights")
    assert len(facts.weights_sha256) == 64
    assert facts.tokenizer_source == "adapter directory"


def test_a_tokenizer_less_adapter_says_it_falls_back_to_the_base_model(tmp_path: Path) -> None:
    """Recorded rather than assumed: the two tokenizers can disagree about BOS."""
    directory = write_adapter(tmp_path / "adapter")
    (directory / "tokenizer_config.json").unlink()

    assert adapter_facts(directory).tokenizer_source == "base model"


def test_a_directory_without_weights_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"no base_model_name_or_path|no weights"):
        adapter_facts(directory)


def test_a_missing_directory_is_refused_with_its_path() -> None:
    with pytest.raises(FileNotFoundError, match="adapter directory not found"):
        adapter_facts(Path("does/not/exist"))


def test_missing_inference_modules_is_empty_or_a_subset() -> None:
    """The probe never raises, whether or not the train extra is installed."""
    missing = missing_inference_modules()

    assert isinstance(missing, list)
    assert set(missing) <= {"torch", "transformers", "peft", "bitsandbytes"}


def test_the_inference_error_is_not_the_training_error() -> None:
    """Scoring must not send a user to install a trainer it will never call."""
    assert issubclass(MissingInferenceDependencies, RuntimeError)
    assert not issubclass(MissingInferenceDependencies, ImportError)


# -- the matched baseline and the paired differences ----------------------------


def comparison_frame(n_rows: int = 60) -> pd.DataFrame:
    """A tidy frame: one row per evaluation row, one column per path.

    Dates step a week apart so that a 30-day block length produces many blocks; a single
    block makes the bootstrap interval degenerate, which is a different test.

    The candidate separates the classes cleanly and the baselines overlap them, so the
    differences have a known sign. Perfect separation on *both* arms would give a
    difference of exactly zero and make a sign assertion vacuous.
    """
    index = pd.RangeIndex(n_rows)
    truth = [1 if i % 6 == 0 else 0 for i in range(n_rows)]
    as_of = pd.to_datetime("2020-01-01") + pd.to_timedelta(index * 7, unit="D")
    return pd.DataFrame(
        {
            "as_of": as_of,
            "y_true": truth,
            "text_only_lora": [0.9 if t else 0.1 + 0.001 * i for i, t in enumerate(truth)],
            "structured_matched": [0.6 if t else 0.25 + 0.005 * i for i, t in enumerate(truth)],
            "text_baseline": [0.45 if t else 0.30 + 0.006 * i for i, t in enumerate(truth)],
        },
        index=index,
    )


def test_paired_differences_are_reported_per_baseline_and_metric() -> None:
    records = paired_differences(
        comparison_frame(),
        candidate="text_only_lora",
        baselines=("structured_matched", "text_baseline"),
        n_boot=200,
        block_days=30,
        alpha=0.05,
        seed=7,
    )

    assert {(item["b"], item["metric"]) for item in records} == {
        ("structured_matched", "pr_auc"),
        ("structured_matched", "auc"),
        ("text_baseline", "pr_auc"),
        ("text_baseline", "auc"),
    }
    assert all(item["a"] == "text_only_lora" for item in records)
    assert all(item["n_rows"] == 60 for item in records)


def test_a_better_candidate_produces_a_positive_difference() -> None:
    """The sign convention has to be the one a reader assumes: candidate minus baseline."""
    records = paired_differences(
        comparison_frame(),
        candidate="text_only_lora",
        baselines=("text_baseline",),
        n_boot=200,
        block_days=30,
        alpha=0.05,
        seed=7,
    )

    for item in records:
        assert item["estimate"] > 0


def test_rows_missing_a_score_are_dropped_for_every_arm() -> None:
    """A difference between different row sets is not a difference, so it is refused.

    The LoRA column is NaN exactly where a generation failed to parse, and dropping that
    row only from the LoRA arm would compare two models on two different samples.
    """
    frame = comparison_frame()
    frame.loc[0:5, "text_only_lora"] = float("nan")

    records = paired_differences(
        frame,
        candidate="text_only_lora",
        baselines=("text_baseline",),
        n_boot=100,
        block_days=30,
        alpha=0.05,
        seed=7,
    )

    assert all(item["n_rows"] == 54 for item in records)


def test_the_row_set_of_a_comparison_is_stated_not_just_counted() -> None:
    """``n_rows`` alone does not say *why* the rows are missing.

    Rows leave a comparison because some arm failed to parse there, and that is not a
    random subset — a run whose surviving rows are all negatives is exactly the case the
    parse accounting exists for. The count is in ``n_rows``; the reason has to be in words.
    """
    frame = comparison_frame()
    frame.loc[0:5, "text_only_lora"] = float("nan")

    records = paired_differences(
        frame,
        candidate="text_only_lora",
        baselines=("text_baseline",),
        n_boot=100,
        block_days=30,
        alpha=0.05,
        seed=7,
    )

    assert all("6 of 60 rows are outside this comparison" in item["note"] for item in records)


def test_a_single_class_sample_yields_no_estimate_rather_than_zero() -> None:
    """No interval means no interval. Reporting 0.0 would read as "measured, no effect"."""
    frame = comparison_frame()
    frame["y_true"] = 0

    records = paired_differences(
        frame,
        candidate="text_only_lora",
        baselines=("text_baseline",),
        n_boot=50,
        block_days=30,
        alpha=0.05,
        seed=7,
    )

    assert len(records) == 2
    for item in records:
        assert item["estimate"] is None
        assert item["crosses_zero"] is None
        assert "one class" in item["note"]


def test_a_path_cannot_be_its_own_baseline() -> None:
    with pytest.raises(ValueError, match="its own baseline"):
        paired_differences(
            comparison_frame(),
            candidate="text_only_lora",
            baselines=("text_only_lora",),
            n_boot=10,
            block_days=30,
            alpha=0.05,
            seed=7,
        )


def test_a_missing_column_is_named() -> None:
    with pytest.raises(ValueError, match="missing column"):
        paired_differences(
            comparison_frame(),
            candidate="text_only_lora",
            baselines=("not_a_path",),
            n_boot=10,
            block_days=30,
            alpha=0.05,
            seed=7,
        )


def test_the_matched_baseline_refuses_a_panel_without_the_prompt_signals() -> None:
    """Dropping a signal silently would make the "same information" claim false.

    The whole point of this baseline is that it sees the columns the prompt shows. A
    reduced column list would produce a row that still says ``structured_matched`` while
    no longer being the thing that name means.
    """
    from shingan.config import load_config
    from shingan.paths import find_project_root
    from shingan.pipeline import matched_signal_report

    config = load_config(find_project_root() / "configs" / "default.yaml")
    panel = pd.DataFrame({"ticker": ["A"], "as_of": [pd.Timestamp("2020-01-01")]})

    with pytest.raises(ValueError, match="debt_to_equity"):
        matched_signal_report(panel, config, "tail_risk")


def test_the_matched_baseline_uses_the_prompt_signal_list() -> None:
    """The default column set must be the prompt's own, not a copy that can drift."""
    import inspect

    from shingan.data.builder import PROMPT_SIGNAL_COLUMNS
    from shingan.pipeline import matched_signal_report

    default = inspect.signature(matched_signal_report).parameters["signals"].default

    assert tuple(default) == tuple(PROMPT_SIGNAL_COLUMNS)

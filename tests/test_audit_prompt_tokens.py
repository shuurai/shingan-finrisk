"""Tests for the prompt token audit: the check that decides whether a bigger budget fits.

`scripts/audit_prompt_tokens.py` answers a question that costs GPU hours to get wrong, and
it had two ways to be confidently useless:

* **A circular projection.** The budget is `limit * c - overhead`, so projecting the extra
  text at the same constant `c` consumes exactly the extra tokens by construction — the
  answer is the limit itself, whatever the corpus, whatever the limit. The first version
  of this script did that and reported `slack = -47` for every candidate. The tests below
  pin that the projection can disagree with the constant.
* **A table that re-checks a fixed corpus against limits it could never exceed.** The
  first version printed an identical row for every candidate limit, which reads as three
  independent confirmations and is one measurement repeated.

The tokenizer is faked throughout: the arithmetic is what is under test, and a test that
needs a 14B model's tokenizer on disk is a test that does not run.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

from shingan.paths import find_project_root
from shingan.prompts import CHARS_PER_TOKEN, chars_budget_for_seq_length

ROOT = find_project_root()


@pytest.fixture(scope="module")
def audit():
    name = "audit_prompt_tokens_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "audit_prompt_tokens.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    """One token per ``ratio`` characters, rounded up. Deterministic and offline."""

    def __init__(self, characters_per_token: float = 2.0) -> None:
        self.ratio = characters_per_token

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        count = max(1, int(np.ceil(len(text) / self.ratio)))
        return {"input_ids": list(range(count))}


def write_sft(directory, rows: list[str], split: str = "train") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for text in rows:
            handle.write(json.dumps({"messages": [{"role": "user", "content": text}]}) + "\n")


# -- reading ------------------------------------------------------------------


def test_messages_are_concatenated_in_order(audit, tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "abc"},
                    {"role": "assistant", "content": "de"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert list(audit.iter_messages(path)) == ["abcde"]


def test_a_file_without_messages_says_so_rather_than_measuring_zero(audit, tmp_path) -> None:
    """Zero rows measured would read as "nothing overflows", which is the opposite of the
    truth about a file the script could not read."""
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"prompt": "x", "completion": "y"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="messages"):
        list(audit.iter_messages(path))


# -- measuring ----------------------------------------------------------------


def test_token_lengths_and_the_effective_ratio_use_sums_not_a_mean_of_ratios(audit) -> None:
    """A mean of per-row ratios is dominated by the shortest prompts, which is the wrong
    direction: it would understate how much text a token buys."""
    texts = ["a" * 100, "b" * 10]
    lengths, characters = audit.token_lengths(texts, FakeTokenizer(2.0))
    assert list(lengths) == [50, 5]
    assert list(characters) == [100, 10]
    assert audit.effective_chars_per_token(lengths, characters) == pytest.approx(110 / 55)


def test_a_prompt_over_the_limit_is_reported_with_negative_headroom(audit) -> None:
    """Both numbers together: a corpus with one over-limit row and a large median passes
    either test alone."""
    lengths = np.array([100, 200, 9_000])
    block = audit.summarise(lengths, 4_096)
    assert block["over"] == 1
    assert block["headroom"] == 4_096 - 9_000
    assert block["percentiles"]["p100"] == 9_000


def test_an_empty_corpus_reports_not_measured(audit) -> None:
    block = audit.summarise(np.array([], dtype=int), 4_096)
    assert block["over"] is None
    assert block["headroom"] is None


# -- projecting ---------------------------------------------------------------


def test_the_projection_is_not_the_budget_formula_restated(audit) -> None:
    """The defect the first version shipped: at ratio == CHARS_PER_TOKEN the projected
    length is `limit`, for every limit, because the growth and the budget are the same
    arithmetic. A projection that cannot disagree with the formula is not a measurement."""
    lengths = np.array([4_096])
    ratios = {"planning constant": CHARS_PER_TOKEN, "dense measured end": audit.DENSE_RATIO}
    for limit in (6_144, 8_192, 16_384):
        block = audit.project_after_budget_raise(lengths, 4_096, limit, ratios)
        # Exact up to the integer truncation inside `chars_budget_for_seq_length`, which
        # is the only place the two sides of the identity differ.
        assert block["per_ratio"]["planning constant"]["projected_max_tokens"] == pytest.approx(
            limit, abs=1.0
        )
        # And the dense end must disagree with it, or the range carries no information.
        assert block["per_ratio"]["dense measured end"]["projected_max_tokens"] > limit


def test_the_verdict_is_the_conjunction_over_ratios(audit) -> None:
    """`fits` must be False when any plausible ratio overflows. Reporting the friendly
    ratio alone is how the first version said "fits, slack 2,018" about a limit it had
    just computed as overflowing."""
    lengths = np.array([4_096])
    ratios = {"optimistic": 4.0, "pessimistic": 1.5}
    block = audit.project_after_budget_raise(lengths, 4_096, 8_192, ratios)
    assert block["per_ratio"]["optimistic"]["fits"] is True
    assert block["per_ratio"]["pessimistic"]["fits"] is False
    assert block["fits"] is False
    assert block["worst_slack"] < 0


def test_a_ratio_of_zero_is_not_measured_rather_than_infinite(audit) -> None:
    """Dividing by zero would report `inf` tokens and read as a failure; an unmeasurable
    ratio is a third state."""
    block = audit.project_after_budget_raise(np.array([10]), 4_096, 8_192, {"bad": 0.0})
    assert block["per_ratio"]["bad"]["fits"] is None
    assert block["per_ratio"]["bad"]["slack_tokens"] is None


def test_the_growth_is_the_budget_difference_not_the_limit_difference(audit) -> None:
    block = audit.project_after_budget_raise(np.array([10]), 4_096, 8_192, {"c": 2.0})
    expected = chars_budget_for_seq_length(8_192) - chars_budget_for_seq_length(4_096)
    assert block["growth_chars"] == expected
    assert block["growth_chars"] != 8_192 - 4_096


# -- rendering ----------------------------------------------------------------


def fake_projection(audit) -> dict:
    return audit.project_after_budget_raise(
        np.array([4_143]), 4_096, 8_192, {"planning constant": 1.9, "dense measured end": 1.84}
    )


def test_the_report_states_the_verdict_in_words(audit) -> None:
    """The table has a per-ratio verdict; the paragraph is what a reader quotes, and it
    must be the conjunction rather than the best row."""
    projections = {8_192: fake_projection(audit)}
    text = audit.render({"train": audit.summarise(np.array([4_143]), 4_096)}, 3.8, 4_096, projections, None)
    assert "overflows under at least one" in text
    assert "fits under every ratio" not in text


def test_the_report_prints_the_manifest_counter_that_is_easy_to_misread(audit) -> None:
    """`prompts_truncated` names how many prompts had a section trimmed by the budget. It
    is not the number of prompts that do not fit, and the report has to show both."""
    manifest = {
        "max_seq_length": 4_096,
        "character_budget": 5_982,
        "prompts_truncated": 8_971,
        "train_config": None,
    }
    text = audit.render(
        {"train": audit.summarise(np.array([4_143, 1_000]), 4_096)}, 3.8, 4_096, {}, manifest
    )
    assert "prompts_truncated` 8971" in text
    assert "1 prompt(s) over" in text
    assert "does not fit" in text


def test_a_fitting_corpus_is_reported_as_fitting(audit) -> None:
    text = audit.render(
        {"valid": audit.summarise(np.array([3_431, 1_687]), 4_096)}, 3.8, 4_096, {}, None
    )
    assert "0 prompt(s) over" in text
    assert "→ fits." in text

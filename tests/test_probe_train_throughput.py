"""The throughput verdict must scale to the input length, not to a truncation ceiling.

2026-11-24: the window scan (§25.5) compared three ``max_seq_length`` values on one
already-built corpus. trl truncates with ``slice(None, max_length)``, so a ceiling above
the corpus's longest row clips nothing. Three "different" windows were one experiment.
The verdict had been scaling its baseline by that ceiling, which is the same category
error in code.
"""

from __future__ import annotations

import pytest
from scripts.probe_train_throughput import (
    BASELINE_EFFECTIVE_TOKENS,
    TOLERANCE_FACTOR,
    configured_ceiling,
    corpus_build_knobs,
    verdict,
)


def test_a_verdict_without_a_measured_length_is_not_a_pass() -> None:
    """Three-valued: no effective length means no verdict, never a silent pass."""
    ok, message = verdict(40.0, None, TOLERANCE_FACTOR)

    assert ok is None
    assert "not measured" in message
    assert "audit_window_sensitivity" in message


def test_zero_or_negative_length_is_also_not_measured() -> None:
    for bad in (0, -1):
        assert verdict(40.0, bad, TOLERANCE_FACTOR)[0] is None


def test_the_baseline_scales_with_effective_tokens_not_the_ceiling() -> None:
    """The old code would have used 4096 here. At 1417 the same rate is a wider miss."""
    at_baseline, _ = verdict(30.0, BASELINE_EFFECTIVE_TOKENS, TOLERANCE_FACTOR)
    at_double, _ = verdict(30.0, 2 * BASELINE_EFFECTIVE_TOKENS, TOLERANCE_FACTOR)

    assert at_baseline is True
    assert at_double is True
    # The expected value must actually move with the length; a fixed ceiling would not.
    _, msg_short = verdict(100.0, BASELINE_EFFECTIVE_TOKENS, TOLERANCE_FACTOR)
    _, msg_long = verdict(100.0, 4 * BASELINE_EFFECTIVE_TOKENS, TOLERANCE_FACTOR)
    assert msg_short != msg_long


def test_mutation_the_ceiling_must_not_be_accepted_as_a_length() -> None:
    """Mutation test: if the ceiling were passed as the length, the verdict would flip.

    A pathological 200 s/step on a short corpus must FAIL. Scaling by a generous 8192
    ceiling would have made it pass -- that is precisely the error being guarded.
    """
    ok, _ = verdict(200.0, BASELINE_EFFECTIVE_TOKENS, TOLERANCE_FACTOR)
    assert ok is False

    ok_if_scaled_by_ceiling, _ = verdict(200.0, 8192, TOLERANCE_FACTOR)
    assert ok_if_scaled_by_ceiling is True  # the wrong computation produces a false pass


def test_pathological_rate_fails_even_at_a_long_corpus() -> None:
    ok, message = verdict(1361.0, 3366, TOLERANCE_FACTOR)
    assert ok is False
    assert "do not launch" in message


def test_configured_ceiling_reads_the_overlay(tmp_path) -> None:
    p = tmp_path / "overlay.yaml"
    p.write_text("lora:\n  max_seq_length: 8192\n", encoding="utf-8")
    assert configured_ceiling(p) == 8192.0


def test_configured_ceiling_refuses_a_missing_value(tmp_path) -> None:
    p = tmp_path / "overlay.yaml"
    p.write_text("lora:\n  learning_rate: 0.0001\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no max_seq_length"):
        configured_ceiling(p)


def test_absent_manifest_is_not_measured(tmp_path) -> None:
    assert corpus_build_knobs(tmp_path) == {"manifest": "absent (not measured)"}


def test_manifest_exposes_both_knobs(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"max_seq_length": 8192, "character_budget": 13764}',
        encoding="utf-8",
    )
    knobs = corpus_build_knobs(tmp_path)
    assert knobs["max_seq_length"] == 8192
    assert knobs["character_budget"] == 13764
    assert knobs["train_config"] == "not recorded"

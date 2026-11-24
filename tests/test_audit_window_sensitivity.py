"""The window scan has to be able to vary the input before it can say anything.

Regression guard for the 2026-10-14 mistake: three windows were compared on one corpus
that was built at a *fixed* character budget, so the trainer truncated almost nothing and
the three probes were fed near-identical inputs. "All three are slow" was then read as
"the window is not the variable" and, one step later, "the machine/driver changed".
"""

from __future__ import annotations

import pytest
from scripts.audit_window_sensitivity import (
    MIN_CONTRAST,
    effective_lengths,
    manifest_budget,
    percentile,
    scan_contrast,
)

# The real 32-row probe slice of data/processed/sft_stage2_wide, token lengths measured
# with the Qwen3-14B chat template on 2026-11-24 (sorted for readability). Only one row
# exceeds 4096; none exceeds 6144.
WIDE_PROBE_LENGTHS = [
    1525, 1616, 1659, 1999, 2400, 2406, 2553, 2565, 3274, 3444, 3444,
    3456, 3456, 3456, 3456, 3474, 3793, 3801, 3815, 3823, 3830, 3860,
    3860, 3896, 3903, 3922, 3927, 3955, 4025, 4050, 4081, 5003,
]


def test_truncation_is_a_ceiling_not_a_pad() -> None:
    """trl slices to max_length; short rows stay short. A larger window changes nothing."""
    lengths = [100, 4096, 8192]

    assert effective_lengths(lengths, 4096) == [100, 4096, 4096]
    assert effective_lengths(lengths, 8192) == [100, 4096, 8192]


def test_a_non_positive_window_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        effective_lengths([10], 0)


def test_the_wide_probe_slice_cannot_support_a_window_scan() -> None:
    """The regression guard, on the measured lengths of the exact slice that was scanned."""
    per_window = {
        window: effective_lengths(WIDE_PROBE_LENGTHS, window)
        for window in (4096, 6144, 8192)
    }
    verdict, detail = scan_contrast(per_window)

    assert verdict is False
    assert detail["spread"] < MIN_CONTRAST
    # 6144 and 8192 clip nothing at all: identical inputs, not "two more experiments".
    assert per_window[6144] == per_window[8192]
    assert sum(1 for length in WIDE_PROBE_LENGTHS if length > 6144) == 0


def test_a_scan_on_a_corpus_wider_than_its_ceiling_is_informative() -> None:
    """Opposite case: genuinely longer rows make the windows different experiments."""
    lengths = [1000, 5000, 9000, 13000]
    per_window = {window: effective_lengths(lengths, window) for window in (4096, 8192)}

    verdict, detail = scan_contrast(per_window)

    assert verdict is True
    assert detail["spread"] > MIN_CONTRAST


def test_mutation_no_truncation_makes_the_informative_scan_vanish() -> None:
    """Mutation test: if truncation stops, the two 'windows' collapse to one corpus.

    This is the failure the guard exists to catch -- a scan that silently never varies
    the input. Breaking `min` must turn an informative verdict into `False`.
    """
    lengths = [1000, 5000, 9000, 13000]
    broken = {window: list(lengths) for window in (4096, 8192)}  # ceiling never applied

    verdict, detail = scan_contrast(broken)

    assert verdict is False
    assert detail["spread"] == 0.0


def test_absent_data_is_not_a_pass() -> None:
    """Three-valued: no rows / no windows is None, never silently 'no difference'."""
    assert scan_contrast({})[0] is None
    assert scan_contrast({4096: [], 8192: [100]})[0] is None


def test_a_missing_manifest_is_not_measured(tmp_path) -> None:
    assert manifest_budget(tmp_path) == {}


def test_a_broken_manifest_is_not_measured(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert manifest_budget(tmp_path) == {}


def test_manifest_reports_the_build_time_budget(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"max_seq_length": 8192, "character_budget": 13764, '
        '"train_config": "configs/train/qlora_qwen3_14b.yaml"}',
        encoding="utf-8",
    )
    assert manifest_budget(tmp_path) == {
        "max_seq_length": 8192,
        "character_budget": 13764,
        "train_config": "configs/train/qlora_qwen3_14b.yaml",
    }


def test_percentile_never_indexes_past_the_end() -> None:
    assert percentile([], 0.5) == 0
    assert percentile([7], 1.0) == 7
    assert percentile([1, 2, 3, 4], 0.9) == 4

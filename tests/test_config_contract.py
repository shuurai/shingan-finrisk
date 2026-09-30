"""Configuration invariants that no other test can see.

The config files are data: a wrong number in them changes every downstream measurement
without raising anything. These tests pin the relationships *between* files -- the parts
a reviewer cannot check by reading one file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shingan.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_the_wide_overlay_universe_matches_its_names_file() -> None:
    """The 180-name list exists twice: as a config block and as a plain text file.

    Hand-copying it once already dropped nine names (AMZN, BKR, BXP, ESS, LYB, NSC,
    NTRS, NUE, ORCL) without any error -- a silently smaller universe is invisible in
    every downstream number. This pins the two copies together.
    """
    names_file = ROOT / "configs" / "universes" / "wide_stage2.txt"
    overlay_file = ROOT / "configs" / "data" / "stage2_wide.yaml"
    if not names_file.is_file() or not overlay_file.is_file():
        pytest.skip("wide overlay not present")

    from_file = {
        word
        for line in names_file.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for word in line.split()
    }
    config = load_config(ROOT / "configs" / "default.yaml", [str(overlay_file)])
    assert {str(item) for item in config.data.universe} == from_file

"""Tests for the FNSPID ingest script.

The script exists because the adapter in ``shingan.data.news`` was written
against the dataset's *published description*; these tests pin the mapping
against what the real files were verified to contain (see the script
docstring). A regression here would silently empty the text track.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "ingest_fnspid", Path(__file__).resolve().parents[1] / "scripts" / "ingest_fnspid.py"
)
ingest_fnspid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ingest_fnspid)

UNIVERSE = frozenset({"A", "BA"})


def test_date_parser_strips_utc_and_coerces_garbage():
    series = pd.Series(["2023-12-16 23:00:00 UTC", "2020-06-05 06:30:54 UTC", "not a date", None])
    parsed = ingest_fnspid.parse_fnspid_date(series)
    assert str(parsed.iloc[0]) == "2023-12-16 23:00:00"
    assert str(parsed.iloc[1]) == "2020-06-05 06:30:54"
    assert parsed.iloc[2] is pd.NaT
    assert parsed.iloc[3] is pd.NaT


def test_chunk_mapping_filters_universe_and_maps_columns():
    chunk = pd.DataFrame(
        {
            "Unnamed: 0": ["0.0", "1.0", "2.0"],
            "Date": ["2023-12-16 23:00:00 UTC", "2023-12-17 12:00:00 UTC", "2023-12-18 00:00:00 UTC"],
            "Article_title": ["Title A", "Title ZZZ", "Title BA"],
            "Stock_symbol": ["a", "ZZZZ", "BA"],
            "Publisher": ["Nasdaq", "Benzinga", "Benzinga"],
            "Article": ["Body A", "Body ZZZ", None],
        }
    )
    mapped = ingest_fnspid.map_fnspid_chunk(chunk, UNIVERSE)
    assert list(mapped.columns) == ingest_fnspid.OUTPUT_COLUMNS
    # lower-case ticker is normalised onto the universe; ZZZZ is dropped
    assert sorted(mapped["ticker"]) == ["A", "BA"]
    row_a = mapped.loc[mapped["ticker"] == "A"].iloc[0]
    assert row_a["title"] == "Title A"
    assert row_a["source"] == "Nasdaq"
    assert row_a["body"] == "Body A"
    # sentiment is absent from FNSPID headers and must stay NA, never zero
    assert mapped["sentiment"].isna().all()


def test_chunk_mapping_refuses_unexpected_columns():
    chunk = pd.DataFrame({"Date": ["2023-12-16 23:00:00 UTC"], "Article_title": ["t"]})
    with pytest.raises(KeyError, match="missing columns"):
        ingest_fnspid.map_fnspid_chunk(chunk, UNIVERSE)


def test_ingest_end_to_end_filters_window_and_deduplicates(tmp_path):
    src = tmp_path / "Stock_news"
    src.mkdir()
    rows = [
        # inside window, kept
        ("2022-03-01 10:00:00 UTC", "Wire story on A", "A", "Benzinga", ""),
        # syndicated copy of the same title, dropped by exact-title match
        ("2022-03-01 11:00:00 UTC", "Wire story on A", "A", "Nasdaq", ""),
        # outside the window (before the lead-in), dropped
        ("2005-01-01 10:00:00 UTC", "Ancient story on BA", "BA", "Benzinga", ""),
    ]
    pd.DataFrame(
        rows,
        columns=["Date", "Article_title", "Stock_symbol", "Publisher", "Article"],
    ).to_csv(src / "All_external.csv", index=False)
    out = tmp_path / "news.parquet"

    ingest_fnspid.ingest(src, out, UNIVERSE, __import__("datetime").date(2021, 1, 1), __import__("datetime").date(2024, 12, 31))

    result = pd.read_parquet(out)
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "A"
    assert result.iloc[0]["title"] == "Wire story on A"
    assert list(result.columns) == ingest_fnspid.OUTPUT_COLUMNS

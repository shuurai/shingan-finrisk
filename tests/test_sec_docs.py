"""Tests for the SEC document fetcher: addressing, document choice, and cache safety.

This module covers ``scripts/fetch_sec_docs.py``, which is not a package and so is loaded
from its path. Everything here is offline — the two functions that touch the network
(``fetch_bytes`` and, through it, ``verify_user_agent``) are exercised through a stub.

What is checked, and why each one earned a test:

* **Addressing.** ``{ARCHIVE}/{cik-without-leading-zeros}/{accession-without-dashes}/`` is
  built in two places — here and in :mod:`shingan.data.edgar` — and the two must agree.
  A drift between them produces 404s that look like missing filings.
* **Primary-document choice.** An ``index.json`` lists the filing *and* the EDGAR index
  page and the XBRL viewer renderings. Picking the largest ``.htm`` picks the index page,
  which is a table of contents; the fetch "succeeds" and the sectioner then finds nothing,
  so the document silently enters the corpus as unsegmented. This is the failure that
  costs a full re-download to notice.
* **The partial-write guard.** ``rebuild_filings`` rewrites the table that defines the
  panel's decision grid. A partial rewrite shrinks the training set without failing
  anything. This guard exists because a smoke test did exactly that to the real table.
* **Cache atomicity.** A run that dies mid-write must not leave a torn ``.json.gz`` that
  the next run treats as a cached document.
"""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest

from shingan.data.edgar import ARCHIVE_BASE, _archive_url
from shingan.paths import find_project_root

ROOT = find_project_root()


def _load_downloader():
    """Import ``scripts/fetch_sec_docs.py`` by path; ``scripts`` is not a package.

    The module is registered in ``sys.modules`` before execution because it enables
    ``from __future__ import annotations``, and under that flag ``@dataclass(slots=True)``
    resolves its annotations by looking the module up in ``sys.modules``. Loading without
    registering fails inside the dataclasses machinery rather than anywhere in this file.
    """
    import sys

    name = "fetch_sec_docs_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "fetch_sec_docs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def downloader():
    return _load_downloader()


def filing_task(downloader, *, cik=320193, accession="0000320193-24-000123", doc_type="10-K"):
    return downloader.FilingTask(
        ticker="AAPL", cik=cik, accession=accession, doc_type=doc_type, filed="2024-11-01"
    )


# -- addressing -----------------------------------------------------------------


def test_directory_url_matches_the_other_address_builder(downloader) -> None:
    """Two modules build this address; a drift between them reads as missing filings."""
    cik, accession = "0000320193", "0000320193-24-000123"
    directory = downloader.directory_url(int(cik), accession)
    document = _archive_url(cik, accession, "aapl-20240928.htm")
    assert document == f"{directory}aapl-20240928.htm"


def test_directory_url_drops_leading_zeros_and_dashes(downloader) -> None:
    url = downloader.directory_url(320193, "0000320193-24-000123")
    assert url == f"{ARCHIVE_BASE}/320193/000032019324000123/"


def test_directory_url_keeps_a_cik_that_is_all_zeros(downloader) -> None:
    """``int()`` of 0 is 0, not an empty path segment."""
    assert downloader.directory_url(0, "0000000000-00-000000").endswith("/0/000000000000000000/")


# -- primary document choice -----------------------------------------------------


def test_index_and_xbrl_artifacts_are_never_the_primary_document(downloader) -> None:
    """The trap: the index page is often the largest ``.htm`` in the directory."""
    items = [
        {"name": "aapl-20240928-index.html", "size": "900000", "type": ""},
        {"name": "R2.htm", "size": "800000", "type": ""},
        {"name": "aapl-20240928.htm", "size": "120000", "type": "10-K"},
        {"name": "exhibit101.htm", "size": "30000", "type": "EX-101.INS"},
    ]
    assert downloader.pick_primary_document(items, "10-K") == "aapl-20240928.htm"


def test_the_form_type_beats_raw_size(downloader) -> None:
    """A large untagged exhibit must not outrank the document tagged with the form."""
    items = [
        {"name": "exhibit991.htm", "size": "500000", "type": "EX-99.1"},
        {"name": "form.htm", "size": "40000", "type": "10-Q"},
    ]
    assert downloader.pick_primary_document(items, "10-Q") == "form.htm"


def test_size_decides_when_nothing_is_tagged(downloader) -> None:
    items = [
        {"name": "small.htm", "size": "1000", "type": ""},
        {"name": "large.htm", "size": "5000", "type": ""},
    ]
    assert downloader.pick_primary_document(items, "10-K") == "large.htm"


def test_a_tie_is_broken_deterministically(downloader) -> None:
    """Same size, different names: the choice must not depend on dict ordering."""
    items = [
        {"name": "b.htm", "size": "1000", "type": ""},
        {"name": "a.htm", "size": "1000", "type": ""},
    ]
    first = downloader.pick_primary_document(items, "10-K")
    second = downloader.pick_primary_document(list(reversed(items)), "10-K")
    assert first == second


def test_no_document_when_the_directory_holds_only_artifacts(downloader) -> None:
    items = [
        {"name": "x-index.html", "size": "10", "type": ""},
        {"name": "R1.htm", "size": "20", "type": ""},
    ]
    assert downloader.pick_primary_document(items, "10-K") is None


def test_non_html_files_are_not_documents(downloader) -> None:
    items = [
        {"name": "aapl-20240928.xsd", "size": "99999", "type": ""},
        {"name": "tables.xml", "size": "88888", "type": ""},
    ]
    assert downloader.pick_primary_document(items, "10-K") is None


# -- tasks from the filings table -------------------------------------------------


def write_filings(path, rows) -> None:
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)


def test_load_tasks_deduplicates_the_same_filing(downloader, tmp_path) -> None:
    """The table holds one row per filing *section*; the fetcher wants one per document."""
    path = tmp_path / "filings.parquet"
    write_filings(
        path,
        [
            {"ticker": "AAPL", "cik": 320193, "accession": "a-1", "doc_type": "10-K", "filed": "2024-11-01"},
            {"ticker": "AAPL", "cik": 320193, "accession": "a-1", "doc_type": "10-K", "filed": "2024-11-01"},
            {"ticker": "MSFT", "cik": 789019, "accession": "m-1", "doc_type": "10-Q", "filed": "2024-10-01"},
        ],
    )
    tasks = downloader.load_tasks(path)
    assert len(tasks) == 2
    assert {task.accession for task in tasks} == {"a-1", "m-1"}
    assert all(isinstance(task.cik, int) for task in tasks)


def test_load_tasks_skips_rows_without_an_address(downloader, tmp_path) -> None:
    """A filing with no accession has no address, so it cannot be a task."""
    path = tmp_path / "filings.parquet"
    write_filings(
        path,
        [
            {"ticker": "AAPL", "cik": 320193, "accession": None, "doc_type": "10-K", "filed": "2024-11-01"},
            {"ticker": "MSFT", "cik": 789019, "accession": "m-1", "doc_type": "10-Q", "filed": "2024-10-01"},
        ],
    )
    tasks = downloader.load_tasks(path)
    assert [task.accession for task in tasks] == ["m-1"]


def test_load_tasks_names_the_missing_columns(downloader, tmp_path) -> None:
    path = tmp_path / "filings.parquet"
    pd.DataFrame({"ticker": ["AAPL"]}).to_parquet(path, index=False)
    with pytest.raises(SystemExit) as error:
        downloader.load_tasks(path)
    message = str(error.value)
    for column in ("cik", "accession", "doc_type", "filed"):
        assert column in message


def test_load_tasks_points_at_the_fetch_that_creates_the_table(downloader, tmp_path) -> None:
    with pytest.raises(SystemExit, match="fetch_real.py"):
        downloader.load_tasks(tmp_path / "absent.parquet")


# -- the cache -------------------------------------------------------------------


def test_cache_round_trips_and_leaves_no_part_file(downloader, tmp_path) -> None:
    path = tmp_path / "doc.json.gz"
    record = {"accession": "a-1", "text": "Risk factors. \u00a7 1A", "sections": {"Item 1A": "Risk factors."}}
    downloader.write_cache(path, record)

    assert downloader.read_cache(path) == record
    assert not list(tmp_path.glob("*.part")), "an atomic write must not leave its temporary behind"


def test_a_torn_cache_file_is_refetched_rather_than_crashing_the_rebuild(downloader, tmp_path) -> None:
    """A half-written cache entry must be skipped, not raised out of the rebuild."""
    cache = tmp_path / "cache"
    cache.mkdir()
    task = filing_task(downloader)
    (cache / task.cache_path).write_bytes(b"\x1f\x8b truncated garbage")

    good = downloader.FilingTask(
        ticker="MSFT", cik=789019, accession="0000789019-24-000001", doc_type="10-Q", filed="2024-10-01"
    )
    downloader.write_cache(
        cache / good.cache_path,
        {
            "ticker": "MSFT",
            "cik": 789019,
            "accession": good.accession,
            "doc_type": "10-Q",
            "filed": "2024-10-01",
            "sections": {"Item 1A": "Risk factors."},
            "text": "Risk factors.",
        },
    )

    cached, rows = downloader.rebuild_filings(
        [task, good], cache_dir=cache, out_path=tmp_path / "out.parquet", allow_partial=True
    )
    assert cached == 1, "the corrupt entry must not count as coverage"
    assert rows == 1


# -- rebuild: section selection and the partial-write guard -----------------------


def stage_record(downloader, cache, *, accession, sections, text="whole document"):
    record = {
        "ticker": "AAPL",
        "cik": 320193,
        "accession": accession,
        "doc_type": "10-K",
        "filed": "2024-11-01",
        "sections": sections,
        "text": text,
    }
    downloader.write_cache(cache / f"{accession}.json.gz", record)
    return record


def test_rebuild_keeps_only_the_sections_the_text_features_read(downloader, tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    stage_record(
        downloader,
        cache,
        accession="a-1",
        sections={"Item 1A": "risk body", "Item 7": "mdna body", "Item 3": "legal body"},
    )
    tasks = [filing_task(downloader, accession="a-1")]
    _, rows = downloader.rebuild_filings(
        tasks, cache_dir=cache, out_path=tmp_path / "out.parquet", allow_partial=True
    )
    frame = pd.read_parquet(tmp_path / "out.parquet")
    assert set(frame["section"]) == {"Item 1A", "Item 7"}, "Item 3 is not read by any text feature"
    assert rows == 2


def test_rebuild_keeps_a_document_with_no_headings_under_the_full_label(downloader, tmp_path) -> None:
    """The document still reaches the text track; disabling the sectioner is not a fix."""
    cache = tmp_path / "cache"
    cache.mkdir()
    stage_record(downloader, cache, accession="a-1", sections={}, text="plain text with no headings")
    tasks = [filing_task(downloader, accession="a-1")]
    downloader.rebuild_filings(
        tasks, cache_dir=cache, out_path=tmp_path / "out.parquet", allow_partial=True
    )
    frame = pd.read_parquet(tmp_path / "out.parquet")
    assert list(frame["section"]) == [downloader.UNSPLIT_LABEL]
    assert frame.iloc[0]["text"] == "plain text with no headings"


def test_partial_coverage_does_not_overwrite_the_live_filings_table(downloader, tmp_path) -> None:
    """Regression: a smoke test rewrote the real 2222-row table with 11 rows.

    The table defines the panel's decision grid, so a partial write shrinks the training
    set without failing anything. It must go to a differently-named file instead.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    stage_record(downloader, cache, accession="a-1", sections={"Item 1A": "body"})

    live = tmp_path / "filings.parquet"
    write_filings(
        live,
        [{"ticker": "AAPL", "cik": 320193, "accession": "a-1", "doc_type": "10-K", "filed": "2024-11-01"}],
    )
    before = live.read_bytes()

    tasks = [filing_task(downloader, accession="a-1"), filing_task(downloader, accession="a-2")]
    cached, _ = downloader.rebuild_filings(tasks, cache_dir=cache, out_path=live)

    assert cached == 1
    assert live.read_bytes() == before, "the live table must be untouched"
    assert (tmp_path / "filings.partial.parquet").is_file()


def test_complete_coverage_does_overwrite_the_live_table(downloader, tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    stage_record(downloader, cache, accession="a-1", sections={"Item 1A": "body"})
    live = tmp_path / "filings.parquet"

    downloader.rebuild_filings([filing_task(downloader, accession="a-1")], cache_dir=cache, out_path=live)

    assert live.is_file()
    assert not (tmp_path / "filings.partial.parquet").exists()
    assert pd.read_parquet(live)["accession"].tolist() == ["a-1"]


def test_rebuild_refuses_to_write_nothing(downloader, tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(SystemExit, match="no cached documents"):
        downloader.rebuild_filings(
            [filing_task(downloader)], cache_dir=cache, out_path=tmp_path / "out.parquet"
        )


# -- the user-agent probe ---------------------------------------------------------


def test_user_agent_probe_explains_a_rejected_contact(downloader, monkeypatch) -> None:
    """A 403 from every request must be diagnosed as a UA problem, not reported as one."""
    def refuse(*args, **kwargs):
        raise downloader.DownloadFailed("HTTP Error 403: Forbidden")

    monkeypatch.setattr(downloader, "fetch_bytes", refuse)
    with pytest.raises(SystemExit) as error:
        downloader.verify_user_agent("shingan/0.1 (contact: someone@github.com)")

    message = str(error.value)
    assert "someone@github.com" in message, "the rejected UA must be echoed back"
    assert "domain" in message, "the cause is the contact domain, and the message must say so"
    assert "SHINGAN_SEC_USER_AGENT" in message, "there must be a concrete remedy"


def test_user_agent_probe_passes_a_contact_that_is_accepted(downloader, monkeypatch) -> None:
    monkeypatch.setattr(downloader, "fetch_bytes", lambda *a, **k: json.dumps({"directory": {}}).encode())
    downloader.verify_user_agent("shingan/0.1 (contact: research@shingan.dev)")

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


# -- the third artifact: -index-headers.html --------------------------------------


def test_the_edgar_header_page_is_never_the_primary_document(downloader) -> None:
    """Measured 2026-09-30: 170 of the 171 unfetched documents failed on exactly this file.

    In a pre-2009 directory it is the only ``.htm`` left once ``-index.html`` is excluded,
    so it was selected, the archive returned 404 (it is listed but not served), and the
    fetch failed. For two later accessions it *was* served, so a 22,082-character header
    page entered the cache as a 10-K with nothing failing at all.
    """
    items = [
        {"name": "0000092380-07-000014-index-headers.html", "size": "", "type": "text.gif"},
        {"name": "0000092380-07-000014-index.html", "size": "", "type": "text.gif"},
    ]
    assert downloader.pick_primary_document(items, "10-Q") is None


# -- the SGML full-submission route ----------------------------------------------

#: The shape of a pre-iXBRL full submission, copied from the AAL 2003 10-Q directory:
#: a PGP-style preamble, then one ``<DOCUMENT>`` block per document with the form in
#: ``<TYPE>``. The primary document is a ``.txt`` file, which is why a fetcher that only
#: accepts ``.htm`` cannot reach it.
FULL_SUBMISSION = """-----BEGIN PRIVACY-ENHANCED MESSAGE-----
Proc-Type: 2001,MIC-CLEAR

<DOCUMENT>
<TYPE>10-Q
<SEQUENCE>1
<FILENAME>ar2q03f.txt
<TEXT>
<HTML><BODY>Item 1. Financial Statements. A body long enough to beat an exhibit stub.</BODY></HTML>
</DOCUMENT>

<DOCUMENT>
<TYPE>EX-31.1
<SEQUENCE>2
<FILENAME>ar2q03ex311.txt
<TEXT>
<HTML><BODY>Certification of the chief executive officer</BODY></HTML>
</DOCUMENT>
"""


def test_the_sgml_route_selects_by_declared_form_not_by_filename(downloader) -> None:
    picked = downloader.select_sgml_document(FULL_SUBMISSION, "10-Q")
    assert picked is not None
    name, text, usable = picked
    assert usable
    assert name == "ar2q03f.txt"
    assert "Item 1. Financial Statements" in text
    assert "Certification" not in text, "an exhibit must not be selected for a 10-Q"


def test_the_sgml_route_returns_nothing_when_the_form_is_absent(downloader) -> None:
    assert downloader.select_sgml_document(FULL_SUBMISSION, "8-K") is None


def test_the_sgml_route_keeps_the_longest_block_for_a_repeated_form(downloader) -> None:
    """A submission can carry the same form twice; the body is the larger one."""
    submission = FULL_SUBMISSION + """
<DOCUMENT>
<TYPE>10-Q
<SEQUENCE>3
<FILENAME>supplement.txt
<TEXT>
<HTML><BODY>Risk factors. """ + ("Additional disclosure. " * 40) + """</BODY></HTML>
</DOCUMENT>
"""
    picked = downloader.select_sgml_document(submission, "10-Q")
    assert picked is not None
    assert picked[0] == "supplement.txt"
    assert picked[2]


#: The head of CCL's 2004 10-K as the archive serves it: EDGAR stores a PDF-only filing by
#: uuencoding the bytes inside ``<TEXT>``, so the "text layer" is 4.8 MB of ``M)5!$1B...``.
#: Measured 2026-09-30: 27 of 14,974 documents are this shape and this marker matched
#: nothing else. Feeding it to the panel is how the XBRL soup entered (docs/09 section 19).
PDF_ONLY_SUBMISSION = """<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<FILENAME>d58480_unofficial.pdf
<TEXT>
begin 644 d58480_unofficial.pdf
M)5!$1BTQ+C(*)>+CS],-"CDY,````````````````````````````````````
M`````````````````````````````````````````````````````````````
`
end
</DOCUMENT>
"""


def test_a_pdf_only_filing_is_reported_as_unusable(downloader) -> None:
    picked = downloader.select_sgml_document(PDF_ONLY_SUBMISSION, "10-K")
    assert picked is not None, "the block exists and must be reported, not silently dropped"
    assert picked[2] is False, "its body is an encoded binary file, not a text layer"
    assert downloader.is_binary_payload("Item 1A. Risk factors.") is False


def test_a_usable_block_beats_a_binary_one_for_the_same_form(downloader) -> None:
    """Some submissions carry a PDF-only 10-K *and* a text 10-K; take the text one."""
    combined = PDF_ONLY_SUBMISSION + FULL_SUBMISSION.replace("<TYPE>10-Q", "<TYPE>10-K")
    picked = downloader.select_sgml_document(combined, "10-K")
    assert picked is not None
    assert picked[2] is True
    assert picked[0] == "ar2q03f.txt"


# -- the integrity gate -----------------------------------------------------------


def test_the_integrity_gate_flags_an_index_page_and_a_stub(downloader) -> None:
    index_page = {
        "primary_document": "0000316709-18-000009-index-headers.html",
        "n_chars": 22082,
    }
    stub = {"primary_document": "aepex12.htm", "n_chars": 947}
    assert "index artifact" in str(downloader.cache_integrity_problem(index_page))
    assert "below the" in str(downloader.cache_integrity_problem(stub))


def test_the_integrity_gate_accepts_exhibit_13_as_a_filing(downloader) -> None:
    """198 cached documents are this shape, and for CAT, GE, LOW and EMR it is correct.

    Exhibit 13 is the annual report to shareholders, filed *as* the 10-K. A rule keyed on
    exhibit naming would quarantine these to catch seven genuinely bad 900-character
    picks, so the gate keys on the index artifacts and on size instead.
    """
    assert downloader.cache_integrity_problem(
        {"primary_document": "ex13.htm", "n_chars": 302590}
    ) is None
    assert downloader.cache_integrity_problem(
        {"primary_document": "schw-12312017x10k.htm", "n_chars": 1200000}
    ) is None


def test_the_route_recheck_targets_only_pre_sgml_short_documents(downloader) -> None:
    """`--recheck-short` exists so the corpus does not carry two picking logics at once.

    A document below the fallback threshold that predates the SGML route may hold an
    exhibit where the filing should be, and it passes the integrity gate because most of
    that population is genuinely short. Anything fetched with a recorded route, or long
    enough that route B was never worth consulting, is left alone.
    """
    assert downloader.needs_route_recheck({"n_chars": 24742, "primary_document": "ex10-1.htm"})
    assert not downloader.needs_route_recheck({"n_chars": 24742, "source_route": "html-primary"})
    assert not downloader.needs_route_recheck({"n_chars": 900000, "primary_document": "aapl.htm"})
    assert not downloader.needs_route_recheck(
        {"n_chars": 24742, "source_route": "sgml-submission", "unusable_reason": "x"}
    )


def test_a_truthy_predicate_return_cannot_quarantine_a_sound_document(downloader, tmp_path) -> None:
    """Regression: the first `--recheck-short` predicate returned bool, the caller skipped
    only on `None`, and a repair run moved 9,786 of 14,974 documents into quarantine.

    The contract is truthy-means-reject. Anything falsy must leave the document alone.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    task = filing_task(downloader, accession="a-1")
    downloader.write_cache(
        cache / task.cache_path,
        {"ticker": "AAPL", "doc_type": "10-K", "filed": "2024-11-01", "primary_document": "aapl.htm", "n_chars": 900000},
    )

    quarantined = downloader.refetch_invalid([task], cache_dir=cache, predicate=lambda record: False)

    assert quarantined == 0
    assert (cache / task.cache_path).is_file(), "a false answer must not move the file"


def test_a_predicate_that_answers_with_a_bool_still_quarantines_what_it_rejects(downloader, tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    task = filing_task(downloader, accession="a-1")
    downloader.write_cache(
        cache / task.cache_path,
        {"ticker": "AAPL", "doc_type": "10-K", "filed": "2024-11-01", "primary_document": "aapl.htm", "n_chars": 900000},
    )

    quarantined = downloader.refetch_invalid([task], cache_dir=cache, predicate=lambda record: True)

    assert quarantined == 1
    assert not (cache / task.cache_path).exists()


def test_refetch_invalid_quarantines_only_the_rejected_documents(downloader, tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    good = filing_task(downloader, accession="a-good")
    bad = filing_task(downloader, accession="a-bad")
    downloader.write_cache(
        cache / good.cache_path,
        {"ticker": "AAPL", "doc_type": "10-K", "filed": "2024-11-01", "primary_document": "aapl.htm", "n_chars": 900000},
    )
    downloader.write_cache(
        cache / bad.cache_path,
        {"ticker": "AAPL", "doc_type": "10-K", "filed": "2024-11-01", "primary_document": "x-index.html", "n_chars": 900},
    )

    quarantined = downloader.refetch_invalid([good, bad], cache_dir=cache)

    assert quarantined == 1
    assert (cache / good.cache_path).is_file(), "a sound document must survive"
    assert not (cache / bad.cache_path).exists(), "quarantining is what makes it pending again"
    # Timestamped, not a flat _invalid/ directory: a second repair run must not have to
    # overwrite the first run's evidence to store its own.
    kept = list((cache / "_invalid").glob(f"*/{bad.cache_path}"))
    assert len(kept) == 1, "the wrong document stays inspectable under its run stamp"


# -- route selection inside fetch_one --------------------------------------------


def fake_fetcher(downloader, monkeypatch, *, items, files):
    """Stub ``fetch_bytes`` with a directory listing and a map of file bodies."""
    calls: list[str] = []

    def fake(url, *args, **kwargs):
        calls.append(url)
        if url.endswith("index.json"):
            return json.dumps({"directory": {"item": items}}).encode()
        for suffix, body in files.items():
            if url.endswith(suffix):
                return body
        raise downloader.DownloadFailed(f"HTTP Error 404: Not Found ({url})")

    monkeypatch.setattr(downloader, "fetch_bytes", fake)
    return calls


def big_html(body: str) -> bytes:
    return f"<html><body>{body}</body></html>".encode()


def test_a_healthy_html_primary_never_pays_for_the_second_request(downloader, monkeypatch, tmp_path) -> None:
    """The SGML route is a fallback, not the default: 14,675 documents at +1 request each
    would cost hours for nothing, since the HTML primary is the right document whenever
    it is large enough to be one."""
    task = filing_task(downloader)
    calls = fake_fetcher(
        downloader,
        monkeypatch,
        items=[{"name": "big.htm", "size": "200000", "type": "10-K"}],
        files={"big.htm": big_html("Item 1A. Risk factors. " * 3000)},
    )
    record = downloader.fetch_one(task, downloader.RateLimiter(1000.0), tmp_path, "ua")

    assert record["source_route"] == "html-primary"
    assert not any(url.endswith(".txt") for url in calls), calls
    assert record["route_sgml"] == {"document": "", "chars": 0}


def test_a_stub_html_primary_loses_to_the_sgml_document(downloader, monkeypatch, tmp_path) -> None:
    """The AEP failure: the directory's only ``.htm`` is a 900-character exhibit."""
    task = filing_task(downloader, doc_type="10-Q")
    fake_fetcher(
        downloader,
        monkeypatch,
        items=[{"name": "aepex12.htm", "size": "947", "type": "text.gif"}],
        files={
            "aepex12.htm": big_html("Exhibit 12 - Computation of ratios"),
            f"{task.accession}.txt": FULL_SUBMISSION.encode(),
        },
    )
    record = downloader.fetch_one(task, downloader.RateLimiter(1000.0), tmp_path, "ua")

    assert record["source_route"] == "sgml-submission"
    assert record["primary_document"] == "ar2q03f.txt"
    assert record["route_html"]["chars"] < record["route_sgml"]["chars"] < downloader._SGML_FALLBACK_BELOW_CHARS


def test_a_directory_of_artifacts_alone_still_yields_the_filing(downloader, monkeypatch, tmp_path) -> None:
    """The 170-document failure, end to end: route A has no candidate, route B does."""
    task = filing_task(downloader, doc_type="10-Q")
    fake_fetcher(
        downloader,
        monkeypatch,
        items=[
            {"name": f"{task.accession}-index-headers.html", "size": "", "type": "text.gif"},
            {"name": f"{task.accession}-index.html", "size": "", "type": "text.gif"},
        ],
        files={f"{task.accession}.txt": FULL_SUBMISSION.encode()},
    )
    record = downloader.fetch_one(task, downloader.RateLimiter(1000.0), tmp_path, "ua")

    assert record["source_route"] == "sgml-submission"
    assert record["n_chars"] > 0
    assert record["sections"], "the fetched text must have been segmented"


def test_a_404_on_the_listed_primary_falls_through_to_the_sgml_route(downloader, monkeypatch, tmp_path) -> None:
    """The listing can name a file the archive no longer serves. That must not be fatal."""
    task = filing_task(downloader, doc_type="10-Q")
    fake_fetcher(
        downloader,
        monkeypatch,
        items=[{"name": "gone.htm", "size": "300000", "type": "10-K"}],
        files={f"{task.accession}.txt": FULL_SUBMISSION.encode()},
    )
    record = downloader.fetch_one(task, downloader.RateLimiter(1000.0), tmp_path, "ua")

    assert record["source_route"] == "sgml-submission"
    assert record["route_html"]["document"] is None


def test_neither_route_producing_text_is_an_error(downloader, monkeypatch, tmp_path) -> None:
    task = filing_task(downloader, doc_type="8-K")
    fake_fetcher(
        downloader,
        monkeypatch,
        items=[{"name": "other.htm", "size": "10", "type": ""}],
        files={"other.htm": big_html(""), f"{task.accession}.txt": FULL_SUBMISSION.encode()},
    )
    with pytest.raises(downloader.DownloadFailed, match="neither route"):
        downloader.fetch_one(task, downloader.RateLimiter(1000.0), tmp_path, "ua")


def test_a_pdf_only_filing_is_cached_as_text_less_rather_than_failing(downloader, monkeypatch, tmp_path) -> None:
    """27 real filings are like this. Failing the fetch would drop the filing date from the
    decision grid; caching the uuencoded body would put megabytes of binary in the corpus.
    Neither: the record says the text is unusable and ``rebuild_filings`` leaves it out."""
    task = filing_task(downloader, doc_type="10-Q")
    fake = fake_fetcher(
        downloader,
        monkeypatch,
        items=[{"name": f"{task.accession}-index.html", "size": "", "type": "text.gif"}],
        files={f"{task.accession}.txt": PDF_ONLY_SUBMISSION.replace("<TYPE>10-K", "<TYPE>10-Q").encode()},
    )
    record = downloader.fetch_one(task, downloader.RateLimiter(1000.0), tmp_path, "ua")

    assert record["n_chars"] == 0
    assert record["text"] == ""
    assert "encoded binary" in record["unusable_reason"]
    assert record["sections"] == {}
    assert any(url.endswith(".txt") for url in fake)


def test_the_rebuild_drops_textless_documents_and_says_how_many(downloader, tmp_path, capsys) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    stage_record(downloader, cache, accession="a-good", sections={"Item 1A": "risk body"})
    downloader.write_cache(
        cache / "a-pdf.json.gz",
        {
            "ticker": "CCL",
            "cik": 815097,
            "accession": "a-pdf",
            "doc_type": "10-K",
            "filed": "2004-02-25",
            "sections": {},
            "text": "",
            "unusable_reason": "the archive holds x.pdf as an encoded binary file",
        },
    )

    tasks = [filing_task(downloader, accession="a-good"), filing_task(downloader, accession="a-pdf")]
    cached, rows = downloader.rebuild_filings(
        tasks, cache_dir=cache, out_path=tmp_path / "out.parquet", allow_partial=True
    )
    printed = capsys.readouterr().out

    assert cached == 2, "a text-less document is still fetched coverage"
    assert rows == 1, "but it must not reach the grid as a row of empty text"
    assert "no usable text layer" in printed
    assert "CCL 10-K 2004-02-25" in printed, "the dropped filings must be named, not counted"

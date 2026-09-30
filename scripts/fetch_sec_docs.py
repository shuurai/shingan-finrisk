"""Download SEC filing documents for the Stage 2 universe and cut them into Item sections.

Why this script exists
----------------------
``data/raw/real/filings.parquet`` holds 2222 filing *index* records — ticker, CIK,
accession, form and filing date are all 100% populated, while ``section`` is the
literal string ``"unavailable"`` and ``text`` is empty. That is the honest record of a
structured-only fetch: the decision grid exists, the documents do not. Every
text-derived feature in the Stage 2 panel is therefore NaN, which is why 8 of the 16
zero-coverage features in that run could only ever have been NaN.

Addressing a document needs no new source of truth. The accession number *is* the
address::

    {ARCHIVE_BASE}/{cik without leading zeros}/{accession without dashes}/

That directory listing (``index.json``) names every file in the filing; the primary
document is picked from it. The fetch is the expensive part, not the addressing.

Design notes that matter
------------------------
* **Resumable and cached.** Each document is written to its own gzip JSON the moment it
  is parsed. A run that dies at document 1800 restarts at 1800. The SEC archive is
  occasionally throttled for a whole network (HTTP 403 "Undeclared Automated Tool"), and
  that block clears on its own — a fetcher without a cache has to start over each time.
* **Three workers, not ten.** Measured on this network: 0.11 MB/s at one worker,
  0.38 MB/s at three, 0.24 MB/s at six. The constraint is the link, not SEC's published
  10 req/s ceiling, and going wider makes it slower.
* **Sections are stored raw, not selected.** Every ``Item`` heading the segmenter
  recognises is kept. Selection is a downstream concern and has already changed once.
* **No truncation.** Full document text is kept so the corpus can be re-segmented
  without refetching. Truncation is applied when the panel is built, where the cost is
  visible.

Usage (repository root, project venv)::

    python scripts/fetch_sec_docs.py --limit 5      # smoke test
    python scripts/fetch_sec_docs.py                # full corpus, resumable
    python scripts/fetch_sec_docs.py --rebuild-only # rewrite filings.parquet from cache
    python scripts/fetch_sec_docs.py --resegment    # re-cut cached docs with the current
                                                    # segmenter, then rebuild the parquet
    python scripts/fetch_sec_docs.py --report       # coverage summary only

What it writes under ``data/raw/real/``:

    sec_docs/<accession>.json.gz   one document: metadata, sections, full text
    sec_docs/_failures.json        documents that stayed unreachable, with the reason
    filings.parquet                rebuilt in place, with real ``section`` and ``text``
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shingan.config import ProjectConfig, load_config  # noqa: E402
from shingan.data.edgar import ARCHIVE_BASE, strip_html  # noqa: E402
from shingan.features.text import segment_items  # noqa: E402

DEFAULT_OVERLAY = ROOT / "configs" / "data" / "stage2_real.yaml"

#: Sections written into ``filings.parquet``, in the order the repo's own text features
#: read them (``RISK_FACTOR_SECTIONS`` then ``MDNA_SECTIONS`` then financial statements).
#: A section absent from a given filing simply produces no row.
SECTIONS_KEPT: tuple[str, ...] = (
    "Item 1A",
    "Item 1B",
    "Item 1C",
    "Item 2",
    "Item 7",
    "Item 7A",
    "Item 8",
    "Item 9A",
)

#: Fallback label used when the segmenter finds no ``Item`` headings at all. The row
#: holds the whole document, and the label says so rather than inventing a section.
UNSPLIT_LABEL = "full"

#: ``-index.html``, ``-index-headers.html`` and ``R<n>.htm`` are index/XBRL-viewer
#: artifacts, never the filing. ``-index-headers.html`` earns its place here from
#: measurement, not from symmetry: in the 2026-09-30 wide run, 170 of the 171 unfetched
#: documents failed with a 404 on exactly that filename (the archive lists it in
#: ``index.json`` but does not serve it for pre-2009 accessions), and two cached
#: documents had it standing in as their "primary" — a 22,082-character header page
#: recorded as a 10-K.
_INDEX_ARTIFACT_RE = re.compile(r"(-index(-headers)?\.html?$)|(^r\d+\.htm$)", re.IGNORECASE)

#: A fetched document shorter than this is not a filing. Measured on the cached corpus:
#: the ten documents below 20,000 characters are all one of two known defects — a
#: 900-character exhibit picked because it was the only ``.htm`` in the directory, or an
#: 11,520-character index page — while the smallest legitimate filing in the corpus is a
#: 10-K at 22,082 characters. The margin is thin, so :func:`report_coverage` prints the
#: smallest documents on every run: an outlier that *passes* this gate has to stay
#: visible rather than be silently trusted.
_MIN_PRIMARY_CHARS = 20_000

#: Below this, the SGML full-submission copy is fetched too and the longer text wins.
#: Pre-iXBRL directories often hold no usable ``.htm`` primary at all — the filing is a
#: ``.txt`` SGML file and the only HTML present is exhibits, so picking by size silently
#: returns a supply contract where the 10-K should be. The full-submission file carries
#: ``<TYPE>`` metadata, so the right document can be selected by form instead of by size.
#: Cost is one extra request, bounded by measurement: the 1st percentile of modern 10-Q
#: length is 55,874 characters, so this fires for roughly 1% of post-2010 filings.
_SGML_FALLBACK_BELOW_CHARS = 60_000

#: Blocks of an EDGAR full-submission text file. ``<TYPE>`` is the form, so the primary
#: document is selected by metadata rather than guessed from a filename.
_SGML_DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)(?=<DOCUMENT>|\Z)", re.DOTALL)
_SGML_TYPE_RE = re.compile(r"<TYPE>\s*([^\n<]+)")
_SGML_FILENAME_RE = re.compile(r"<FILENAME>\s*([^\n<]+)")
_SGML_TEXT_RE = re.compile(r"<TEXT>(.*)", re.DOTALL)

#: A filing the archive holds as a PDF has no text layer at all: EDGAR stores the PDF
#: uuencoded inside ``<TEXT>``, so ``strip_html`` happily returns megabytes of
#: ``M)5!$1BTQ+C(*)...`` and calls it prose. Measured 2026-09-30 on the repaired wide
#: corpus: 27 of 14,974 documents are this shape (every one of them PDF-only, e.g. CCL's
#: 2003-2007 filings and AEP's 2004 10-K — 8.9 MB of uuencoded bytes in the worst case),
#: and the marker matched nothing else. This is the same failure the XBRL soup caused in
#: docs/09 section 19, one era earlier: a machine artefact entering the corpus as text.
_UUENCODE_RE = re.compile(r"^begin [0-7]{3} ", re.MULTILINE)
_PDF_MAGIC = "%PDF-"



RETRY_BACKOFF_SECONDS = (20.0, 60.0, 150.0)


class DownloadFailed(RuntimeError):
    """A document could not be retrieved after every retry."""


class RateLimiter:
    """A token bucket shared by every worker thread.

    Threads sleep inside the lock on purpose: the point is a ceiling on outbound
    requests for the process as a whole, and a limiter that lets three threads each
    wait independently would send three requests in the same instant.
    """

    def __init__(self, per_second: float) -> None:
        self._minimum_interval = 1.0 / max(per_second, 0.01)
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        with self._lock:
            gap = self._minimum_interval - (time.monotonic() - self._last_request)
            if gap > 0:
                time.sleep(gap)
            self._last_request = time.monotonic()


@dataclass(slots=True)
class Progress:
    """Counts shared by the reporting loop."""

    done: int = 0
    ok: int = 0
    failed: int = 0
    bytes: int = 0
    started: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, ok: bool, n_bytes: int = 0) -> None:
        with self._lock:
            self.done += 1
            self.bytes += n_bytes
            if ok:
                self.ok += 1
            else:
                self.failed += 1

    def line(self, total: int) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-6)
        rate = self.done / elapsed
        remaining = (total - self.done) / rate if rate else float("inf")
        return (
            f"  {self.done}/{total}  ok={self.ok} failed={self.failed}  "
            f"{self.bytes / 1e6:.0f} MB  {rate * 60:.1f} docs/min  "
            f"elapsed {elapsed / 60:.0f}m  eta {remaining / 60:.0f}m"
        )


def fetch_bytes(url: str, limiter: RateLimiter, *, user_agent: str, timeout: float = 120.0) -> bytes:
    """GET ``url`` with retries. 404 is permanent; everything else is retried."""
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Accept": "*/*"}
    last_error: Exception | None = None
    for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
        limiter.wait()
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    import gzip as _gzip

                    payload = _gzip.decompress(payload)
                return payload
        except HTTPError as exc:
            if exc.code == 404:
                raise DownloadFailed(f"404 {url}") from exc
            last_error = exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < len(RETRY_BACKOFF_SECONDS):
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    raise DownloadFailed(f"{type(last_error).__name__}: {last_error}")


def directory_url(cik: int, accession: str) -> str:
    """The archive directory that holds every file of one filing."""
    compact = str(accession).replace("-", "").strip()
    return f"{ARCHIVE_BASE}/{int(cik)}/{compact}/"


def pick_primary_document(items: list[dict[str, Any]], doc_type: str) -> str | None:
    """Choose the primary document from an ``index.json`` file list.

    Two signals, in order: the ``type`` field when it equals the form (many filings
    tag the primary document with it), then raw size. Index pages and XBRL viewer
    renderings are excluded first — they are HTML, they are sometimes large, and
    picking one would silently produce a "filing" that is really a table of contents.
    """
    candidates: list[tuple[int, str, str]] = []
    for item in items:
        name = str(item.get("name", ""))
        if not name.lower().endswith((".htm", ".html")) or _INDEX_ARTIFACT_RE.search(name):
            continue
        raw_size = str(item.get("size", "")).strip()
        size = int(raw_size) if raw_size.isdigit() else 0
        candidates.append((size, name, str(item.get("type", ""))))
    if not candidates:
        return None
    typed = [entry for entry in candidates if doc_type and entry[2].upper() == doc_type.upper()]
    return max(typed or candidates)[1]


@dataclass(slots=True)
class FilingTask:
    """One document to fetch."""

    ticker: str
    cik: int
    accession: str
    doc_type: str
    filed: str

    @property
    def cache_path(self) -> str:
        return f"{self.accession}.json.gz"


def is_binary_payload(text: str) -> bool:
    """Whether a document body is an encoded binary file rather than a text layer."""
    if not text:
        return False
    return bool(_UUENCODE_RE.search(text)) or _PDF_MAGIC in text[:512]


def select_sgml_document(raw: str, doc_type: str) -> tuple[str, str, bool] | None:
    """Pick the primary document out of an EDGAR full-submission text file.

    Returns ``(filename, text, usable)`` for the block whose ``<TYPE>`` equals
    ``doc_type``, or ``None`` when no block declares that form. A usable block always
    beats an unusable one, and the longest of each wins within its own class, so a
    submission holding both a PDF-only 10-K and a text-layer 10-K yields the text one.
    ``usable=False`` means the block exists but its body is an encoded binary file; the
    caller must not treat the returned text as prose.
    """
    wanted = doc_type.strip().upper()
    best: tuple[str, str, bool] | None = None
    for block in _SGML_DOCUMENT_RE.findall(raw):
        declared = _SGML_TYPE_RE.search(block)
        if declared is None or declared.group(1).strip().upper() != wanted:
            continue
        name = _SGML_FILENAME_RE.search(block)
        body = _SGML_TEXT_RE.search(block)
        text = strip_html(body.group(1) if body else block)
        usable = not is_binary_payload(text)
        candidate = (name.group(1).strip() if name else "", text, usable)
        if best is None:
            best = candidate
            continue
        if usable and not best[2]:
            best = candidate
        elif usable == best[2] and len(text) > len(best[1]):
            best = candidate
    return best


def cache_integrity_problem(record: dict[str, Any]) -> str | None:
    """Why a cached record should be re-fetched, or ``None`` when it looks sound.

    Only two signals are used, because both are unambiguous. Exhibit naming is
    deliberately *not* one of them: 198 cached documents have ``ex13.htm`` /
    ``exhibit13.htm`` as their primary, and for CAT, GE, LOW and EMR that is the correct
    document — the annual report filed as Exhibit 13 — so a name rule would throw away
    good filings to catch seven bad ones.
    """
    name = str(record.get("primary_document") or "")
    if is_binary_payload(str(record.get("text") or "")):
        # Fetched before the PDF-only case was handled: the body is an encoded binary.
        return f"text layer is an encoded binary file: {name}"
    if _INDEX_ARTIFACT_RE.search(name):
        return f"index artifact cached as the filing: {name}"
    if record.get("source_route"):
        # Fetched by a version that consults the SGML full submission whenever the HTML
        # primary is small, so a small body now means the archive really holds nothing
        # longer. Flagging it would re-fetch it on every run and get the same answer.
        return None
    chars = int(record.get("n_chars") or 0)
    if chars < _MIN_PRIMARY_CHARS:
        return f"{chars:,} characters, below the {_MIN_PRIMARY_CHARS:,} floor: {name}"
    return None


def fetch_one(task: FilingTask, limiter: RateLimiter, cache_dir: Path, user_agent: str) -> dict[str, Any]:
    """Fetch, strip, segment and cache one filing. Returns the cached record.

    Two routes, because the archive offers two and only one of them works for a given
    era. Route A is the ``.htm`` primary named by ``index.json``; route B is the SGML
    full-submission ``<accession>.txt``, which carries ``<TYPE>`` metadata. Route B is
    consulted when route A is missing, fails, or comes back implausibly small, and then
    the longer of the two bodies wins. Both contenders are recorded so the choice is
    auditable instead of inferred.
    """
    base = directory_url(task.cik, task.accession)
    index_raw = fetch_bytes(base + "index.json", limiter, user_agent=user_agent)
    index = json.loads(index_raw)
    items = index.get("directory", {}).get("item", [])
    n_bytes = len(index_raw)

    html_name = pick_primary_document(items, task.doc_type)
    html_text = ""
    if html_name is not None:
        try:
            raw_html = fetch_bytes(base + html_name, limiter, user_agent=user_agent)
            n_bytes += len(raw_html)
            html_text = strip_html(raw_html.decode("utf-8", "replace"))
        except DownloadFailed:
            # The listing can name a file the archive no longer serves. Route B is the
            # reason this is recoverable rather than fatal, so do not give up here.
            html_name = None

    sgml_name = ""
    sgml_text = ""
    sgml_declared = ""
    if not html_text or len(html_text) < _SGML_FALLBACK_BELOW_CHARS:
        raw_sgml = fetch_bytes(base + f"{task.accession}.txt", limiter, user_agent=user_agent)
        n_bytes += len(raw_sgml)
        picked = select_sgml_document(raw_sgml.decode("utf-8", "replace"), task.doc_type)
        if picked is not None:
            sgml_name, sgml_text, usable = picked
            if not usable:
                # A declared-form block exists but its body is an encoded binary file.
                # Keep the name for the record, drop the bytes: 8.9 MB of uuencoded PDF
                # must not reach the corpus as prose.
                sgml_declared = sgml_name
                sgml_name, sgml_text = "", ""

    if not html_text and not sgml_text:
        if sgml_declared:
            # The filing is text-less, not missing. Cache the fact rather than failing,
            # so the decision grid keeps the filing date and the panel records the text
            # as not measured ("未测量 ≠ 失败"). rebuild_filings drops these rows and says
            # how many it dropped.
            record = {
                "ticker": task.ticker,
                "cik": task.cik,
                "accession": task.accession,
                "doc_type": task.doc_type,
                "filed": task.filed,
                "directory_url": base,
                "primary_document": sgml_declared,
                "document_url": base + sgml_declared,
                "source_route": "sgml-submission",
                "unusable_reason": f"the archive holds {sgml_declared} as an encoded binary file",
                "route_html": {"document": html_name, "chars": 0},
                "route_sgml": {"document": sgml_declared, "chars": 0},
                "n_files_in_directory": len(items),
                "n_html_bytes": n_bytes,
                "n_chars": 0,
                "n_words": 0,
                "section_names": [],
                "n_sections": 0,
                "sections": {},
                "text": "",
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            write_cache(cache_dir / task.cache_path, record)
            return record
        raise DownloadFailed(
            f"neither route produced text for {task.doc_type} in {base} "
            f"({len(items)} directory entries, html={html_name!r}, sgml={sgml_name!r})"
        )

    use_sgml = len(sgml_text) > len(html_text)
    document_name = sgml_name if use_sgml else html_name
    text = sgml_text if use_sgml else html_text
    url = base + (document_name or "")
    sections = segment_items(text)

    record = {
        "ticker": task.ticker,
        "cik": task.cik,
        "accession": task.accession,
        "doc_type": task.doc_type,
        "filed": task.filed,
        "directory_url": base,
        "primary_document": document_name,
        "document_url": url,
        "source_route": "sgml-submission" if use_sgml else "html-primary",
        "route_html": {"document": html_name, "chars": len(html_text)},
        "route_sgml": {"document": sgml_name, "chars": len(sgml_text)},
        "n_files_in_directory": len(items),
        "n_html_bytes": n_bytes,
        "n_chars": len(text),
        "n_words": len(text.split()),
        "section_names": sorted(sections),
        "n_sections": len(sections),
        "sections": {label: body for label, body in sorted(sections.items())},
        "text": text,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_cache(cache_dir / task.cache_path, record)
    return record


def write_cache(path: Path, record: dict[str, Any]) -> None:
    """Write a record atomically, so an interrupted run cannot leave a torn file."""
    temporary = path.with_suffix(path.suffix + ".part")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(record, handle, ensure_ascii=False)
    temporary.replace(path)


def read_cache(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        cached: dict[str, Any] = json.load(handle)
    return cached


def load_tasks(filings_path: Path) -> list[FilingTask]:
    """One task per distinct ``(cik, accession)`` in the existing filings table."""
    import pandas as pd

    if not filings_path.is_file():
        raise SystemExit(f"{filings_path} does not exist; run scripts/fetch_real.py first")
    frame = pd.read_parquet(filings_path)
    required = {"ticker", "cik", "accession", "doc_type", "filed"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{filings_path} lacks {sorted(missing)}")
    frame = frame.dropna(subset=["cik", "accession"]).drop_duplicates(["cik", "accession"])
    tasks: list[FilingTask] = []
    for row in frame.itertuples(index=False):
        tasks.append(
            FilingTask(
                ticker=str(row.ticker),
                cik=int(row.cik),
                accession=str(row.accession),
                doc_type=str(row.doc_type),
                filed=str(row.filed)[:10],
            )
        )
    tasks.sort(key=lambda task: (task.ticker, task.filed))
    return tasks


def download(
    tasks: list[FilingTask],
    *,
    cache_dir: Path,
    user_agent: str,
    workers: int,
    requests_per_second: float,
    limit: int | None,
) -> dict[str, str]:
    """Fetch every uncached task. Returns accession -> failure reason."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(requests_per_second)
    already = [task for task in tasks if (cache_dir / task.cache_path).is_file()]
    pending = [task for task in tasks if not (cache_dir / task.cache_path).is_file()]
    if limit is not None:
        pending = pending[:limit]

    print(
        f"documents: {len(tasks)} total, {len(already)} already cached, {len(pending)} to fetch "
        f"({workers} workers, {requests_per_second:g} req/s)"
    )
    if not pending:
        return {}

    progress = Progress()
    failures: dict[str, str] = {}
    last_print = time.monotonic()
    total = len(pending)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_one, task, limiter, cache_dir, user_agent): task for task in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
                progress.record(ok=True, n_bytes=int(record["n_html_bytes"]))
            except Exception as exc:
                progress.record(ok=False)
                failures[task.accession] = f"{task.ticker} {task.doc_type} {task.filed}: {exc}"
            if time.monotonic() - last_print > 20 or progress.done == total:
                last_print = time.monotonic()
                print(progress.line(total), flush=True)

    if failures:
        failure_path = cache_dir / "_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"failures written to {failure_path} ({len(failures)})")
    return failures


def rebuild_filings(
    tasks: list[FilingTask], *, cache_dir: Path, out_path: Path, allow_partial: bool = False
) -> tuple[int, int]:
    """Rewrite ``filings.parquet`` from the document cache.

    One row per ``(filing, section)``, matching the schema the builder consumes. The
    loader records a section the segmenter could not find as a single ``full`` row so
    that a document without detectable headings still reaches the text track.

    Refuses to overwrite the live table while coverage is incomplete. The filings table
    defines the panel's decision grid, so writing a partial one would silently shrink
    the training set rather than fail — the worst kind of failure here. A partial result
    goes to ``filings.partial.parquet`` instead, where it can be inspected and nothing
    downstream will pick it up by accident.
    """
    import pandas as pd

    records: list[dict[str, Any]] = []
    cached = 0
    unreadable = 0
    textless: list[str] = []
    for task in tasks:
        path = cache_dir / task.cache_path
        if not path.is_file():
            continue
        try:
            record = read_cache(path)
        except Exception:
            unreadable += 1
            continue
        cached += 1
        if record.get("unusable_reason"):
            # A filing the archive holds only as an encoded binary. It has no text to
            # contribute, and emitting the megabytes of uuencoded PDF as a ``full`` row
            # is how a corpus gets quietly poisoned. It is named in the output instead.
            textless.append(f"{record.get('ticker')} {record.get('doc_type')} {record.get('filed')}")
            continue
        sections: dict[str, str] = record.get("sections") or {}
        selected = {name: sections[name] for name in SECTIONS_KEPT if sections.get(name)}
        if not selected:
            selected = {UNSPLIT_LABEL: record.get("text", "")}
        for section, body in selected.items():
            records.append(
                {
                    "ticker": record["ticker"],
                    "cik": str(record["cik"]),
                    "filed": pd.Timestamp(record["filed"]),
                    "doc_type": record["doc_type"],
                    "section": section,
                    "text": body,
                    "accession": record["accession"],
                }
            )
    if not records:
        raise SystemExit("no cached documents to rebuild from")
    frame = pd.DataFrame.from_records(records).sort_values(
        ["ticker", "filed", "section"]
    ).reset_index(drop=True)

    complete = cached >= len(tasks)
    destination = out_path
    if not complete and not allow_partial:
        destination = out_path.with_name("filings.partial.parquet")
        print(
            f"partial coverage ({cached}/{len(tasks)} documents): writing {destination.name} "
            "and leaving the live filings table untouched. Pass --allow-partial to override."
        )
    frame.to_parquet(destination, index=False)
    print(f"wrote {destination} ({len(frame)} rows from {cached} documents)")
    if textless:
        print(
            f"{len(textless)} cached document(s) carry no usable text layer and are absent "
            f"from the grid (the archive holds them as encoded binaries): {', '.join(textless)}"
        )
    if unreadable:
        print(f"warning: {unreadable} unreadable cache entries were skipped (re-run to refetch)")
    return cached, len(frame)


def needs_route_recheck(record: dict[str, Any]) -> str | None:
    """Why a cached document should be re-fetched with both routes available, or ``None``.

    Route B is consulted whenever the HTML primary is below ``_SGML_FALLBACK_BELOW_CHARS``,
    so a document below that threshold and fetched *before* route B existed may still hold
    an exhibit where the filing should be — and it passes the integrity gate, because most
    of that population is genuinely short. Measured 2026-09-30: 294 of 14,974 cached
    documents, of which LUV's 2006-10-20 10-Q is the clearest case (a 24,742-character
    Exhibit 10.1 contract standing in for a quarterly report).

    The point is not those 294 documents individually; it is that a corpus carrying two
    generations of document-picking logic cannot be reproduced from its own cache.

    Returns a message rather than a bool on purpose. The first version returned bool, the
    caller only skipped on ``None``, and a repair run moved 9,786 documents of 14,974 into
    quarantine before a locked file stopped it.
    """
    if record.get("source_route"):
        return None
    chars = int(record.get("n_chars") or 0)
    if chars >= _SGML_FALLBACK_BELOW_CHARS:
        return None
    return f"{chars:,} characters, below the {_SGML_FALLBACK_BELOW_CHARS:,} route threshold: {record.get('primary_document')}"


def refetch_invalid(
    tasks: list[FilingTask], *, cache_dir: Path, predicate: Any = cache_integrity_problem
) -> int:
    """Quarantine cache entries a predicate rejects, so the run re-fetches them.

    Entries move to ``sec_docs/_invalid/<run stamp>/`` rather than being deleted, so a
    rejected document can be inspected after the fact, and moving it is what makes the
    normal download treat the task as pending again. The stamp matters: a second repair
    run must not collide with the first one's files, and a destructive overwrite there
    would destroy exactly the evidence a repair run exists to preserve.

    Returns the number quarantined.
    """
    quarantine = cache_dir / "_invalid" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    quarantined = 0
    for task in tasks:
        path = cache_dir / task.cache_path
        if not path.is_file():
            continue
        record: dict[str, Any] | None = None
        try:
            record = read_cache(path)
        except Exception as exc:
            problem = f"unreadable cache entry ({exc})"
        else:
            problem = predicate(record)
            # Contract: truthy means "reject this document". Normalising a non-string
            # return here is what makes a predicate that answers a question (True/False)
            # safe to pass in, instead of quarantining every record it is shown.
            if not problem:
                continue
            if not isinstance(problem, str):
                problem = "rejected by the re-fetch predicate"
        quarantine.mkdir(parents=True, exist_ok=True)
        path.replace(quarantine / path.name)
        ticker = record.get("ticker") if record else task.ticker
        form = record.get("doc_type") if record else task.doc_type
        filed = record.get("filed") if record else task.filed
        print(f"  quarantined {ticker} {form} {filed}: {problem}")
        quarantined += 1
    print(f"quarantined {quarantined} cache entr{'y' if quarantined == 1 else 'ies'} into {quarantine}")
    return quarantined


def resegment_cache(tasks: list[FilingTask], *, cache_dir: Path) -> None:
    """Re-run the *current* segmenter over every cached document's full text.

    The cache stores both the raw text and the segmentation computed at fetch time,
    and :func:`rebuild_filings` faithfully re-emits the stored segmentation -- so a
    segmenter improvement does not reach the parquet until this runs. That is not a
    hypothetical: the 2026-09-21 fetch predated the inline-heading detection, 1,722 of
    2,879 filing rows fell back to ``full`` (the whole document, whose head in newer
    inline-XBRL filings is hidden-facts tag soup), and the first real adapter eval
    failed to parse 82% of its generations because the model had been shown and had
    quoted exactly that soup. The raw text was always good; only the stored cut was
    stale.

    A document whose re-segmentation is unchanged keeps its cache file untouched, so
    re-running this is idempotent and cheap.
    """
    changed = 0
    unchanged = 0
    unreadable = 0
    for task in tasks:
        path = cache_dir / task.cache_path
        if not path.is_file():
            continue
        try:
            record = read_cache(path)
        except Exception:
            unreadable += 1
            continue
        text = str(record.get("text", ""))
        sections = segment_items(text)
        names = sorted(sections)
        if names == list(record.get("section_names") or []):
            unchanged += 1
            continue
        record["section_names"] = names
        record["n_sections"] = len(sections)
        record["sections"] = {label: body for label, body in sorted(sections.items())}
        record["resegmented_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_cache(path, record)
        changed += 1
    print(
        f"resegmented: {changed} document(s) changed, {unchanged} unchanged, "
        f"{unreadable} unreadable"
    )


def report_coverage(tasks: list[FilingTask], *, cache_dir: Path) -> None:
    """Per-form section resolution rates. This is what says whether the text track is real."""
    import statistics

    by_form: dict[str, Counter[str]] = defaultdict(Counter)
    totals: dict[str, int] = defaultdict(int)
    chars: dict[str, list[int]] = defaultdict(list)
    routes: Counter[str] = Counter()
    suspect: list[tuple[int, str, str, str, str]] = []
    all_lengths: list[tuple[int, str, str]] = []
    textless = 0
    #: Era × form split of the segmenter's failure to split at all. It is reported here
    #: rather than in a separate audit because it is the one number that says whether the
    #: window expansion bought readable text or only decision dates: pre-iXBRL primary
    #: documents wrap a heading inside a source line, so "Item 1A." and "Risk Factors"
    #: arrive on different lines and no line-anchored pattern can fire. The corpus can
    #: therefore hold a 2007 filing whose row exists and whose text features are absent.
    unsplit: dict[str, Counter[str]] = defaultdict(Counter)
    era_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        path = cache_dir / task.cache_path
        if not path.is_file():
            continue
        record = read_cache(path)
        totals[task.doc_type] += 1
        era = "2003-2008" if str(record.get("filed") or "")[:4] < "2009" else "2009+"
        era_totals[era][task.doc_type] += 1
        if int(record.get("n_sections") or 0) < 2:
            unsplit[era][task.doc_type] += 1
        if record.get("unusable_reason"):
            textless += 1
        chars[task.doc_type].append(int(record.get("n_chars", 0)))
        all_lengths.append((int(record.get("n_chars", 0)), str(record.get("ticker")), task.doc_type))
        routes[str(record.get("source_route") or "unrecorded")] += 1
        problem = cache_integrity_problem(record)
        if problem:
            suspect.append(
                (
                    int(record.get("n_chars", 0)),
                    str(record.get("ticker")),
                    task.doc_type,
                    str(record.get("filed"))[:10],
                    f"{problem}; route={record.get('source_route')}",
                )
            )
        for name in record.get("section_names", []):
            by_form[task.doc_type][name] += 1

    print("covered documents:", sum(totals.values()), "of", len(tasks))
    print("document route:", dict(routes.most_common()))
    if textless:
        print(
            f"{textless} document(s) have no usable text layer (the archive stores them as "
            "encoded binaries); they keep their filing date but contribute no text"
        )
    for doc_type in sorted(totals):
        count = totals[doc_type]
        lengths = chars[doc_type]
        print(
            f"\n{doc_type}: {count} documents, median {statistics.median(lengths):,.0f} chars, "
            f"total {sum(lengths) / 1e6:.2f} M chars"
        )
        for name, hits in by_form[doc_type].most_common():
            print(f"    {name:10s} {hits:5d}  ({hits / count:6.1%})")

    # Two lists that exist so a bad document cannot hide behind a good aggregate. The
    # smallest ten are printed whether or not they fail the integrity gate: a document
    # that passes at 24,742 characters while its siblings run to 200,000 is a defect this
    # gate cannot name, and the only honest treatment is to keep showing it.
    print(f"\nsmallest documents ({len(suspect)} fail the integrity gate outright):")
    for length, ticker, form, filed, note in sorted(suspect)[:12]:
        print(f"    {length:>9,}  {ticker:5s} {form:4s} {filed}  {note}")
    if not suspect:
        print("    none")
    everything = sorted(all_lengths)
    print("    corpus minimum, for reference:")
    for length, ticker, form in everything[:5]:
        print(f"    {length:>9,}  {ticker:5s} {form}")

    print("\ndocuments the segmenter could not split (no Item heading found):")
    for era in sorted(era_totals):
        parts = []
        for form in sorted(era_totals[era]):
            count = era_totals[era][form]
            parts.append(f"{form} {unsplit[era][form]}/{count} ({unsplit[era][form] / count:.1%})")
        print(f"    {era:10s} " + "   ".join(parts))


def verify_user_agent(user_agent: str) -> None:
    """Fail loudly and diagnostically if the archive rejects the contact address.

    The failure mode being guarded against is specific and expensive to diagnose from
    the far end: every request 403s, the retry loop turns that into a multi-minute delay
    per document, and the run ends with a failures file that says "403" and nothing
    about why. The cause is the User-Agent, and it is not the shape of the string — a
    descriptive ``name/version (contact: addr)`` is rejected or accepted depending
    purely on ``addr``'s domain. Measured on this network:

        research@shingan.dev          200
        a@example.org                 200
        a@noreply.com                 200
        a@github.com                  403
        shuurai@users.noreply...      403
        a browser User-Agent          403

    So the check is cheap, the message is explicit, and it runs before any work.
    """
    probe = f"{ARCHIVE_BASE}/320193/000032019323000106/index.json"
    try:
        fetch_bytes(probe, RateLimiter(5.0), user_agent=user_agent)
        return
    except Exception as exc:
        raise SystemExit(
            f"the SEC archive rejects this User-Agent:\n  {user_agent!r}\n  {exc}\n"
            "A descriptive UA is required, but the contact *domain* is what is filtered: "
            "addresses at github.com are refused while ordinary domains are served. "
            "Point the contact at a domain you can receive mail at, either in "
            "configs/data/stage2_real.yaml (sec_user_agent) or via the "
            "SHINGAN_SEC_USER_AGENT environment variable."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--workers", type=int, default=3, help="Concurrent downloaders (default 3).")
    parser.add_argument("--limit", type=int, default=None, help="Fetch at most N uncached documents.")
    parser.add_argument("--rebuild-only", action="store_true", help="Skip the network entirely.")
    parser.add_argument(
        "--resegment",
        action="store_true",
        help="Re-run the current segmenter over every cached document before rebuilding "
        "(implies --rebuild-only: no network). Use after a segmenter improvement.",
    )
    parser.add_argument("--report", action="store_true", help="Print coverage and exit.")
    parser.add_argument(
        "--refetch-invalid",
        action="store_true",
        help="Quarantine cached documents that fail the integrity gate (an index page, a "
        "sub-20,000-character stub, or an encoded binary) so the run re-fetches them.",
    )
    parser.add_argument(
        "--recheck-short",
        action="store_true",
        help="Also quarantine documents fetched before the SGML route existed whose body is "
        "below the fallback threshold, so one picking logic covers the whole corpus.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Overwrite the live filings table even when some documents are still missing.",
    )
    args = parser.parse_args()

    config: ProjectConfig = load_config(ROOT / "configs" / "default.yaml", [args.data_config], root=ROOT)
    out_dir = Path(config.data.cache_dir) if config.data.cache_dir else ROOT / "data" / "raw" / "real"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    cache_dir = out_dir / "sec_docs"
    filings_path = out_dir / "filings.parquet"
    user_agent = os.environ.get("SHINGAN_SEC_USER_AGENT") or str(config.data.sec_user_agent)

    tasks = load_tasks(filings_path)
    fields = Counter(task.doc_type for task in tasks)
    print(f"filings table: {len(tasks)} distinct documents {dict(fields)}")
    print(f"user agent: {user_agent}")

    if args.report:
        report_coverage(tasks, cache_dir=cache_dir)
        return

    if args.resegment:
        resegment_cache(tasks, cache_dir=cache_dir)

    if args.refetch_invalid:
        refetch_invalid(tasks, cache_dir=cache_dir)
    if args.recheck_short:
        refetch_invalid(tasks, cache_dir=cache_dir, predicate=needs_route_recheck)

    if not args.rebuild_only and not args.resegment:
        verify_user_agent(user_agent)
        download(
            tasks,
            cache_dir=cache_dir,
            user_agent=user_agent,
            workers=args.workers,
            requests_per_second=float(config.data.sec_requests_per_second),
            limit=args.limit,
        )

    rebuild_filings(
        tasks, cache_dir=cache_dir, out_path=filings_path, allow_partial=args.allow_partial
    )
    report_coverage(tasks, cache_dir=cache_dir)


if __name__ == "__main__":
    main()

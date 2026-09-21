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

#: ``-index.html`` and ``R<n>.htm`` are index/XBRL-viewer artifacts, never the filing.
_INDEX_ARTIFACT_RE = re.compile(r"(-index\.html?$)|(^r\d+\.htm$)", re.IGNORECASE)

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


def fetch_one(task: FilingTask, limiter: RateLimiter, cache_dir: Path, user_agent: str) -> dict[str, Any]:
    """Fetch, strip, segment and cache one filing. Returns the cached record."""
    base = directory_url(task.cik, task.accession)
    index_raw = fetch_bytes(base + "index.json", limiter, user_agent=user_agent)
    index = json.loads(index_raw)
    items = index.get("directory", {}).get("item", [])
    primary = pick_primary_document(items, task.doc_type)
    if primary is None:
        raise DownloadFailed(f"no .htm document in {base} ({len(items)} entries)")

    url = base + primary
    raw = fetch_bytes(url, limiter, user_agent=user_agent)
    html = raw.decode("utf-8", "replace")
    text = strip_html(html)
    sections = segment_items(text)

    record = {
        "ticker": task.ticker,
        "cik": task.cik,
        "accession": task.accession,
        "doc_type": task.doc_type,
        "filed": task.filed,
        "directory_url": base,
        "primary_document": primary,
        "document_url": url,
        "n_files_in_directory": len(items),
        "n_html_bytes": len(raw),
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
            except Exception as exc:  # noqa: BLE001 - one bad filing must not end the run
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
    for task in tasks:
        path = cache_dir / task.cache_path
        if not path.is_file():
            continue
        try:
            record = read_cache(path)
        except Exception:  # noqa: BLE001 - a corrupt cache entry is a refetch, not a crash
            unreadable += 1
            continue
        cached += 1
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
    if unreadable:
        print(f"warning: {unreadable} unreadable cache entries were skipped (re-run to refetch)")
    return cached, len(frame)


def report_coverage(tasks: list[FilingTask], *, cache_dir: Path) -> None:
    """Per-form section resolution rates. This is what says whether the text track is real."""
    import statistics

    by_form: dict[str, Counter[str]] = defaultdict(Counter)
    totals: dict[str, int] = defaultdict(int)
    chars: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        path = cache_dir / task.cache_path
        if not path.is_file():
            continue
        record = read_cache(path)
        totals[task.doc_type] += 1
        chars[task.doc_type].append(int(record.get("n_chars", 0)))
        for name in record.get("section_names", []):
            by_form[task.doc_type][name] += 1

    print("covered documents:", sum(totals.values()), "of", len(tasks))
    for doc_type in sorted(totals):
        count = totals[doc_type]
        lengths = chars[doc_type]
        print(
            f"\n{doc_type}: {count} documents, median {statistics.median(lengths):,.0f} chars, "
            f"total {sum(lengths) / 1e6:.2f} M chars"
        )
        for name, hits in by_form[doc_type].most_common():
            print(f"    {name:10s} {hits:5d}  ({hits / count:6.1%})")


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
    except Exception as exc:  # noqa: BLE001 - the message below is the point
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
    parser.add_argument("--report", action="store_true", help="Print coverage and exit.")
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

    if not args.rebuild_only:
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

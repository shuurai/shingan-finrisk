"""SEC EDGAR client.

**Status: thin adapter, not validated against the live API.** Request construction,
rate limiting, retry, caching and field mapping are implemented. No full successful
fetch has been performed from this repository. Treat it as an interface awaiting
verification — ``docs/02-data.md`` section 2.1.

Two EDGAR-specific facts that shape the implementation:

**A ``User-Agent`` is mandatory.** SEC's access policy requires automated clients to
identify themselves with a descriptive string including a contact. A missing or
generic header returns an opaque 403 rather than a helpful error, which sends people
hunting for bugs in their parsing code. The header is therefore validated at
construction time.

**Facts have two clocks.** An XBRL fact carries both ``end`` (the period it
describes) and ``filed`` (when it became public). Only ``filed`` determines
availability. Using ``end`` is the classic look-ahead bug in financial ML: it means a
model at fiscal year end already knows the 10-K that will not be filed for another
two months. Every query here returns ``filed`` and every join keys on it.
"""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from shingan.logging_utils import get_logger

logger = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
COMPANY_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

#: A handful of CIKs for the Stage 2 universe. A full mapping is a maintenance
#: burden; EDGAR's own ``company_tickers.json`` is the authoritative source and is
#: fetched by :func:`fetch_ticker_cik_map` when a broader universe is needed.
CIK_BY_TICKER: dict[str, str] = {
    "JPM": "0000019617",
    "GS": "0000886982",
    "GE": "0000040545",
    "CAT": "0000018230",
    "BA": "0000012927",
    "XOM": "0000034088",
    "T": "0000732717",
    "F": "0000037996",
    "M": "0000794367",
    "WBA": "0001618921",
}

#: Forms whose text the text track consumes.
DEFAULT_FORMS: tuple[str, ...] = ("10-K", "10-Q", "8-K")

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&#?\w+;")


class EdgarUnavailable(RuntimeError):
    """Raised when EDGAR cannot be reached, or the client is misconfigured."""


def strip_html(raw: str) -> str:
    """Convert an EDGAR HTML document to plain text.

    Not a full parser and not trying to be: the goal is a token stream for counting
    and for prompt construction, not a faithful rendering. Script and style blocks are
    removed first, because their contents would otherwise be tokenised as prose and
    would inflate every lexicon count.

    Args:
        raw: Raw HTML.

    Returns:
        Text with tags removed, entities unescaped and horizontal whitespace
        collapsed. Blank lines are preserved so that ``Item`` headings stay on their
        own line, which is what :func:`shingan.features.text.segment_items` relies on.
    """
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", raw)
    without_tags = _TAG_RE.sub(" ", without_scripts)
    unescaped = html.unescape(without_tags)
    if _ENTITY_RE.search(unescaped):
        unescaped = _ENTITY_RE.sub(" ", unescaped)
    lines = [" ".join(line.split()) for line in unescaped.splitlines()]
    return "\n".join(line for line in lines if line)


@dataclass(slots=True)
class FilingRef:
    """One filing's index metadata."""

    ticker: str
    cik: str
    accession: str
    form: str
    filed: date
    period: date | None
    primary_document: str
    url: str

    @property
    def source_ref(self) -> str:
        """Identifier used in evidence citations."""
        return f"{self.form}:{self.ticker}:{self.filed.isoformat()}"


@dataclass(slots=True)
class SecEdgarClient:
    """Minimal, polite EDGAR client.

    Rate limiting and retry are not optional extras here: EDGAR blocks clients that
    ignore its published limits, and an unattended cron job that gets blocked is
    indistinguishable from a code bug for hours.
    """

    user_agent: str
    requests_per_second: float = 5.0
    max_retries: int = 4
    timeout_seconds: float = 30.0
    cache_dir: Path | None = None
    _last_request: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.user_agent or "contact:" not in self.user_agent.lower() and "@" not in self.user_agent:
            raise EdgarUnavailable(
                "user_agent must identify this application and include a contact, e.g. "
                "'shingan-research/0.1 (contact: you@example.com)'. SEC returns 403 "
                "without one, with an error body that does not explain why. Set it via "
                "the SHINGAN_SEC_USER_AGENT environment variable."
            )
        if not 0.0 < self.requests_per_second <= 10.0:
            raise EdgarUnavailable(
                f"requests_per_second must be in (0, 10], got {self.requests_per_second}. "
                "SEC's published ceiling is 10 requests per second."
            )
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    # -- transport ---------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep if needed to stay under the configured request rate."""
        minimum_interval = 1.0 / self.requests_per_second
        elapsed = time.monotonic() - self._last_request
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
        self._last_request = time.monotonic()

    def _get_json(self, url: str, *, cache_name: str | None = None) -> dict[str, Any]:
        """Fetch and parse a JSON document, with retry and optional disk cache."""
        import httpx

        cache_path = Path(self.cache_dir) / cache_name if (self.cache_dir and cache_name) else None
        if cache_path is not None and cache_path.is_file():
            logger.debug("cache hit: %s", cache_path.name)
            with cache_path.open("r", encoding="utf-8") as handle:
                # Annotated rather than returned directly: `json.load` is typed as
                # `Any`, and annotating at the read states the expectation once
                # instead of at every return.
                cached: dict[str, Any] = json.load(handle)
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = httpx.get(
                    url,
                    headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                )
                if response.status_code == 403:
                    raise EdgarUnavailable(
                        f"EDGAR returned 403 for {url}. The User-Agent is almost certainly "
                        "unacceptable: it must be descriptive and include a contact address."
                    )
                if response.status_code == 404:
                    raise EdgarUnavailable(f"EDGAR has no document at {url}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise EdgarUnavailable(f"expected a JSON object from {url}")
                if cache_path is not None:
                    with cache_path.open("w", encoding="utf-8") as handle:
                        json.dump(payload, handle)
                return payload
            except EdgarUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - retried below, then re-raised
                last_error = exc
                backoff = min(2.0**attempt, 30.0)
                logger.warning(
                    "EDGAR request failed (attempt %d/%d): %s; retrying in %.0fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        raise EdgarUnavailable(f"EDGAR request to {url} failed after retries: {last_error}")

    # -- endpoints ---------------------------------------------------------

    def submissions(self, cik: str) -> dict[str, Any]:
        """Fetch the submissions index for a CIK."""
        padded = _pad_cik(cik)
        return self._get_json(
            SUBMISSIONS_URL.format(cik=padded), cache_name=f"submissions_{padded}.json"
        )

    def company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch all XBRL facts for a CIK via ``companyfacts``."""
        padded = _pad_cik(cik)
        return self._get_json(
            COMPANY_FACTS_URL.format(cik=padded), cache_name=f"companyfacts_{padded}.json"
        )

    def list_filings(
        self,
        ticker: str,
        *,
        forms: tuple[str, ...] = DEFAULT_FORMS,
        limit: int = 40,
        since: date | None = None,
    ) -> list[FilingRef]:
        """List recent filings of the given forms.

        Args:
            ticker: Ticker present in :data:`CIK_BY_TICKER`.
            forms: Form types to keep.
            limit: Maximum number of filings to return.
            since: Optional lower bound on the filing date.

        Returns:
            Filing references, most recent first.

        Raises:
            EdgarUnavailable: If the ticker is unknown or EDGAR is unreachable.
        """
        cik = CIK_BY_TICKER.get(ticker.upper())
        if cik is None:
            raise EdgarUnavailable(
                f"no CIK known for {ticker!r}. Add it to CIK_BY_TICKER or fetch EDGAR's "
                "company_tickers.json and use fetch_ticker_cik_map()."
            )

        payload = self.submissions(cik)
        recent = payload.get("filings", {}).get("recent", {})
        required = ("accessionNumber", "form", "filingDate", "primaryDocument")
        if not all(key in recent for key in required):
            raise EdgarUnavailable(
                f"submissions payload for CIK {cik} lacks the expected 'filings.recent' keys"
            )

        references: list[FilingRef] = []
        for position in range(len(recent["form"])):
            form = str(recent["form"][position])
            if form not in forms:
                continue
            filed = pd.Timestamp(recent["filingDate"][position]).date()
            if since is not None and filed < since:
                continue
            accession = str(recent["accessionNumber"][position])
            document = str(recent["primaryDocument"][position])
            period_raw = recent.get("reportDate", [None] * len(recent["form"]))[position]
            references.append(
                FilingRef(
                    ticker=ticker.upper(),
                    cik=cik,
                    accession=accession,
                    form=form,
                    filed=filed,
                    period=pd.Timestamp(period_raw).date() if period_raw else None,
                    primary_document=document,
                    url=_archive_url(cik, accession, document),
                )
            )
            if len(references) >= limit:
                break
        logger.debug("found %d filings for %s", len(references), ticker)
        return references

    def fetch_document(self, url: str) -> str:
        """Fetch a filing document and return it as plain text."""
        import httpx

        self._throttle()
        response = httpx.get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        return strip_html(response.text)

    def facts_to_frame(self, cik: str, *, concepts: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Flatten ``companyfacts`` into a long frame keyed on ``filed``.

        Args:
            cik: Company CIK.
            concepts: Optional allow-list of ``us-gaap`` concept names. Filtering here
                rather than after the fact keeps the frame small enough to be useful;
                a full fact dump is tens of thousands of rows per company.

        Returns:
            A frame with ``cik``, ``taxonomy``, ``concept``, ``unit``, ``start``,
            ``end``, ``value``, ``filed``, ``form`` and ``frame``. Every row keeps its
            ``filed`` date, because that is the only field that determines when the
            value became public.

        Raises:
            EdgarUnavailable: If the payload shape is unexpected.
        """
        payload = self.company_facts(cik)
        facts = payload.get("facts")
        if not isinstance(facts, dict):
            raise EdgarUnavailable(f"companyfacts for CIK {cik} has no 'facts' object")

        rows: list[dict[str, Any]] = []
        for taxonomy, concept_map in facts.items():
            if not isinstance(concept_map, dict):
                continue
            for concept, definition in concept_map.items():
                if concepts is not None and concept not in concepts:
                    continue
                units = definition.get("units") if isinstance(definition, dict) else None
                if not isinstance(units, dict):
                    continue
                for unit, observations in units.items():
                    for observation in observations:
                        rows.append(
                            {
                                "cik": payload.get("cik", cik),
                                "taxonomy": taxonomy,
                                "concept": concept,
                                "unit": unit,
                                "start": observation.get("start"),
                                "end": observation.get("end"),
                                "value": observation.get("val"),
                                "filed": observation.get("filed"),
                                "form": observation.get("form"),
                                "frame": observation.get("frame"),
                            }
                        )

        frame = pd.DataFrame.from_records(rows)
        if frame.empty:
            logger.warning("no facts extracted for CIK %s", cik)
            return frame

        for column in ("start", "end", "filed"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        logger.warning(
            "facts for CIK %s retrieved: %d rows, %d distinct (concept, end) keys. The "
            "live endpoint has not been validated; check for multi-revision restatements "
            "before joining on anything other than 'filed'.",
            cik,
            len(frame),
            frame.groupby(["concept", "end"]).ngroups,
        )
        return frame


def _pad_cik(cik: str) -> str:
    """Zero-pad a CIK to the ten digits EDGAR requires in URLs."""
    digits = "".join(character for character in str(cik) if character.isdigit())
    if not digits:
        raise EdgarUnavailable(f"{cik!r} contains no digits and cannot be a CIK")
    return digits.zfill(10)


def _archive_url(cik: str, accession: str, document: str) -> str:
    """Build the direct archive URL for a filing document."""
    digits = _pad_cik(cik).lstrip("0") or "0"
    compact = accession.replace("-", "")
    return f"{ARCHIVE_BASE}/{digits}/{compact}/{document}"


def fetch_ticker_cik_map(client: SecEdgarClient | None = None) -> dict[str, str]:
    """Fetch EDGAR's full ticker-to-CIK mapping.

    Args:
        client: An existing client, so the rate limiter is shared. A temporary one is
            created when omitted.

    Returns:
        Mapping of upper-case ticker to zero-padded CIK.

    Raises:
        EdgarUnavailable: If the document cannot be fetched.
    """
    import httpx

    active = client or SecEdgarClient(user_agent="shingan-research/0.1 (contact: unset)")
    active._throttle()  # noqa: SLF001 - same class, intentional reuse of the rate limiter
    response = httpx.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": active.user_agent},
        timeout=active.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise EdgarUnavailable("company_tickers.json is not a JSON object")
    return {
        str(entry["ticker"]).upper(): _pad_cik(str(entry["cik_str"]))
        for entry in payload.values()
        if isinstance(entry, dict) and "ticker" in entry and "cik_str" in entry
    }


__all__ = [
    "ARCHIVE_BASE",
    "CIK_BY_TICKER",
    "DEFAULT_FORMS",
    "EdgarUnavailable",
    "FilingRef",
    "SecEdgarClient",
    "fetch_ticker_cik_map",
    "strip_html",
]

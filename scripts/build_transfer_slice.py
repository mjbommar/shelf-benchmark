#!/usr/bin/env python3
"""
Build the natural-data transfer slice (Phase 6 of docs/data_plan_v0.4.md).

SHELF is 100% LLM-generated, and Phase 4's holdout is still synthetic -- newer
generators running the same generation process. Nothing in the benchmark tests
whether SHELF performance transfers to text no LLM wrote. This script builds
that test: human-authored, human-catalogued passages carrying real Library of
Congress class letters, drawn from Project Gutenberg.

Why Gutenberg
-------------
Public domain (no licensing friction), full text, and -- decisively -- a
cataloguer-assigned LC class in the RDF metadata, so the labels are not ours.
The catalogue is one 126 MB archive, which means one bulk download instead of
75,000 metadata requests.

Confirmed source layout (fetched 2026-08-26)
--------------------------------------------
Catalogue archive: ``https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2``
containing one file per ebook at ``cache/epub/<id>/pg<id>.rdf``.

Inside each RDF the class letters live in ``dcterms:subject`` blocks keyed by
``dcam:memberOf``, verbatim from ``pg1342.rdf`` (Pride and Prejudice)::

    <dcterms:subject>
      <rdf:Description rdf:nodeID="N6c38e249c10f4fbd8b45fd7afc982d7f">
        <dcam:memberOf rdf:resource="http://purl.org/dc/terms/LCC"/>
        <rdf:value>PR</rdf:value>
      </rdf:Description>
    </dcterms:subject>
    <dcterms:subject>
      <rdf:Description rdf:nodeID="Ne3fd02c3a7d24b1f83e7317cdc3700b7">
        <dcam:memberOf rdf:resource="http://purl.org/dc/terms/LCSH"/>
        <rdf:value>England -- Fiction</rdf:value>
      </rdf:Description>
    </dcterms:subject>

Note the LCC value is the *subclass* ("PR"), not the bare main class; the main
class is its first letter. Plain text is declared as a ``dcterms:hasFormat``
``pgterms:file`` with IMT ``text/plain; charset=utf-8``, and resolves to
``https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt``.

Pipeline
--------
Five resumable stages, each cached so a rerun costs nothing it already paid for::

    download  ->  <cache>/rdf-files.tar.bz2
    catalog   ->  <cache>/catalog.jsonl        one row per ebook carrying an LCC class
    select    ->  <out>/selection.json         stratified work list + per-work quota
    fetch     ->  <cache>/texts/<id>.txt       polite, rate-limited, resumable
    build     ->  <out>/records.jsonl, manifest.json, distribution_report.{txt,json}

Two constraints are wired into the code rather than left to the operator.

**Contamination.** Project Gutenberg is in the pretraining data of essentially
every model SHELF evaluates. Every record is stamped
``contamination_status="known_contaminated"``, and ``shelf.hub.transfer``
refuses to validate a natural record that claims otherwise. The slice is
*not* contamination-resistant and nothing here may describe it as such: SHELF
is the clean-synthetic condition, this is the contaminated-natural condition,
and the gap between them is the measurement.

**Skew.** Gutenberg is heavily P (Language and Literature) and pre-1929.
Selection subsamples toward the flattest achievable LCC distribution by
enforcing an identical per-class passage cap, and deliberately does *not*
redistribute the shortfall from thin classes back to fat ones (that would
simply refill P). What comes out is then measured and published verbatim in
``distribution_report.txt``. A transfer slice may be unbalanced; it may not be
unbalanced quietly.

Usage
-----
    # See what would happen; no network, no writes
    uv run python scripts/build_transfer_slice.py --dry-run

    # Small end-to-end smoke run
    uv run python scripts/build_transfer_slice.py --limit 50

    # Full build (a few thousand passages; hours of polite fetching)
    uv run python scripts/build_transfer_slice.py --target-passages 3000

    # Re-report an existing slice without refetching anything
    uv run python scripts/build_transfer_slice.py --stage build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ElementTree
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Add src to path for local development (matches scripts/prepare_hf_dataset.py).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shelf.hub.transfer import (  # noqa: E402
    CONTAMINATION_NOTICE,
    LCC_MAIN_CLASSES,
    SKEW_NOTICE,
    TRANSFER_SCHEMA_VERSION,
    ContaminationStatus,
    NaturalSource,
    SourceType,
    build_report,
)

BUILDER_VERSION = "1.0.0"

# =============================================================================
# Source constants (all verified against the live site on 2026-08-26)
# =============================================================================

GUTENBERG_BASE = "https://www.gutenberg.org"
RDF_ARCHIVE_URL = f"{GUTENBERG_BASE}/cache/epub/feeds/rdf-files.tar.bz2"
ROBOTS_URL = f"{GUTENBERG_BASE}/robots.txt"

#: Direct cache path for a book's UTF-8 plain text. Preferred over the
#: catalogue's ``/ebooks/<id>.txt.utf-8``, which 302s here -- one request
#: instead of two, which matters when being a polite bulk client.
TEXT_URL_TEMPLATE = f"{GUTENBERG_BASE}/cache/epub/{{gid}}/pg{{gid}}.txt"

DEFAULT_USER_AGENT = (
    "SHELF-benchmark/0.4 (natural transfer slice; "
    "+https://huggingface.co/datasets/mjbommar/SHELF)"
)

# RDF/XML namespaces used by the Gutenberg catalogue.
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_DCTERMS = "http://purl.org/dc/terms/"
NS_DCAM = "http://purl.org/dc/dcam/"
NS_PGTERMS = "http://www.gutenberg.org/2009/pgterms/"

MEMBER_OF_LCC = f"{NS_DCTERMS}LCC"
MEMBER_OF_LCSH = f"{NS_DCTERMS}LCSH"

LCC_URI_BASE = "http://id.loc.gov/authorities/classification/"


class Stage(str, Enum):
    """Pipeline stages, in dependency order."""

    DOWNLOAD = "download"
    CATALOG = "catalog"
    SELECT = "select"
    FETCH = "fetch"
    BUILD = "build"
    ALL = "all"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.DOWNLOAD,
    Stage.CATALOG,
    Stage.SELECT,
    Stage.FETCH,
    Stage.BUILD,
)


# =============================================================================
# LCSH -> LCGFT derivation
# =============================================================================
#
# Gutenberg RDF carries no LCGFT (655) terms at all -- only LCC and LCSH. But
# LCSH headings routinely end in a *form* subdivision ("England -- Fiction",
# "American poetry", "-- Juvenile literature"), and those subdivisions map onto
# SHELF's curated 133-form / 14-category LCGFT pool for a minority of works.
#
# The mapping below is deliberately conservative and one-directional: a form is
# assigned only when a heading contains an unambiguous marker. Anything else
# leaves ``lcgft_form`` empty rather than guessing, because a wrong form label
# is worse than a missing one on a slice whose whole purpose is label fidelity.
# Every value on the right-hand side is a real member of
# ``shelf.sampler.dimensions.LCGFT_DATA``.

#: Lowercased LCSH fragment -> (lcgft_form, lcgft_category). Order matters:
#: the first match wins, so more specific genres precede "fiction".
LCSH_FORM_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("science fiction", "Science fiction", "Literature"),
    ("fantasy fiction", "Fantasy fiction", "Literature"),
    ("detective and mystery stories", "Mystery fiction", "Literature"),
    ("short stories", "Short stories", "Literature"),
    ("drama", "Drama", "Literature"),
    ("poetry", "Poetry", "Literature"),
    ("poems", "Poetry", "Literature"),
    ("folklore", "Folk literature", "Literature"),
    ("fairy tales", "Folk literature", "Literature"),
    ("legends", "Folk literature", "Literature"),
    ("sagas", "Sagas", "Literature"),
    ("satire", "Satire", "Literature"),
    ("wit and humor", "Humor", "Recreational works"),
    ("fiction", "Fiction", "Literature"),
    ("biography", "Biographies", "Creative nonfiction"),
    ("diaries", "Diaries", "Creative nonfiction"),
    ("personal narratives", "Personal narratives", "Creative nonfiction"),
    ("autobiograph", "Memoirs", "Creative nonfiction"),
    ("description and travel", "Travel writing", "Creative nonfiction"),
    ("voyages and travels", "Travel writing", "Creative nonfiction"),
    ("essays", "Essays", "Creative nonfiction"),
    ("speeches, addresses", "Speeches", "Discursive works"),
    ("sermons", "Sermons", "Religious materials"),
    ("prayers", "Prayers", "Religious materials"),
    ("meditations", "Devotional literature", "Religious materials"),
    ("devotional", "Devotional literature", "Religious materials"),
    ("theology", "Theological works", "Religious materials"),
    ("liturg", "Liturgical texts", "Religious materials"),
    ("cookery", "Cookbooks", "Instructional and educational works"),
    (
        "handbooks, manuals",
        "Handbooks and manuals",
        "Instructional and educational works",
    ),
    ("textbooks", "Textbooks", "Instructional and educational works"),
    ("dictionaries", "Reference works", "Informational works"),
    ("encyclopedias", "Reference works", "Informational works"),
    ("maps", "Maps", "Cartographic materials"),
    ("atlases", "Atlases", "Cartographic materials"),
    ("criticism and interpretation", "Criticism", "Discursive works"),
    ("correspondence", "Personal narratives", "Creative nonfiction"),
    ("songs and music", "Songs", "Music"),
)

#: LCSH juvenile markers -> SHELF ``audience`` values. The only audience signal
#: Gutenberg cataloguing reliably provides.
LCSH_AUDIENCE_MARKERS: tuple[tuple[str, str], ...] = (
    ("juvenile fiction", "Children"),
    ("juvenile literature", "Children"),
    ("juvenile drama", "Children"),
    ("young adult", "Young adults"),
)


def derive_lcgft(headings: Sequence[str]) -> tuple[str, str]:
    """Derive ``(lcgft_form, lcgft_category)`` from LCSH headings, or ``("", "")``.

    First unambiguous marker wins; no marker means no label.
    """
    blob = " ; ".join(h.lower() for h in headings)
    for marker, form, category in LCSH_FORM_MARKERS:
        if marker in blob:
            return form, category
    return "", ""


def derive_audience(headings: Sequence[str]) -> str:
    """Derive a SHELF ``audience`` value from LCSH headings, or ``""``."""
    blob = " ; ".join(h.lower() for h in headings)
    for marker, audience in LCSH_AUDIENCE_MARKERS:
        if marker in blob:
            return audience
    return ""


def lcsh_main_headings(headings: Sequence[str]) -> list[str]:
    """Reduce ``"England -- Fiction"`` to ``"England"``, order-preserving unique.

    These populate ``topics``. They are LCSH strings, **not** members of SHELF's
    112-value ``topics`` pool -- the record's ``label_space`` field says so, and
    nothing should train a shared classifier across the two label spaces.
    """
    seen: set[str] = set()
    out: list[str] = []
    for heading in headings:
        main = heading.split(" -- ")[0].strip().rstrip(".")
        if main and main.lower() not in seen:
            seen.add(main.lower())
            out.append(main)
    return out


# =============================================================================
# Catalogue parsing
# =============================================================================


@dataclass
class CatalogEntry:
    """One Project Gutenberg ebook, reduced to the fields Phase 6 needs."""

    gutenberg_id: int
    title: str = ""
    authors: list[str] = field(default_factory=list)
    author_birth_year: int | None = None
    author_death_year: int | None = None
    languages: list[str] = field(default_factory=list)
    lcc_subclasses: list[str] = field(default_factory=list)
    lcsh_headings: list[str] = field(default_factory=list)
    bookshelves: list[str] = field(default_factory=list)
    dcmi_types: list[str] = field(default_factory=list)
    issued: str = ""
    rights: str = ""
    downloads: int = 0
    text_bytes: int = 0

    @property
    def lcc_letters(self) -> list[str]:
        """Distinct LCC main-class letters implied by the subclass codes."""
        letters = {code[0].upper() for code in self.lcc_subclasses if code}
        return sorted(letters & set(LCC_MAIN_CLASSES))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(element: Any) -> str:
    """Whitespace-normalized text of an element, or ``""``."""
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_ebook_rdf(data: bytes) -> CatalogEntry | None:
    """Parse one ``pg<id>.rdf`` payload into a :class:`CatalogEntry`.

    Returns ``None`` when the file has no ``pgterms:ebook`` element or its
    identifier is not a plain integer (Gutenberg also ships a handful of
    non-book records).
    """
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return None

    ebook = root.find(f"{{{NS_PGTERMS}}}ebook")
    if ebook is None:
        return None

    about = ebook.get(f"{{{NS_RDF}}}about", "")
    match = re.search(r"ebooks/(\d+)$", about)
    if not match:
        return None

    entry = CatalogEntry(gutenberg_id=int(match.group(1)))
    entry.title = _text(ebook.find(f"{{{NS_DCTERMS}}}title"))
    entry.issued = _text(ebook.find(f"{{{NS_DCTERMS}}}issued"))
    entry.rights = _text(ebook.find(f"{{{NS_DCTERMS}}}rights"))
    entry.downloads = _int_or_none(_text(ebook.find(f"{{{NS_PGTERMS}}}downloads"))) or 0

    for creator in ebook.findall(f"{{{NS_DCTERMS}}}creator"):
        for agent in creator.findall(f"{{{NS_PGTERMS}}}agent"):
            name = _text(agent.find(f"{{{NS_PGTERMS}}}name"))
            if name:
                entry.authors.append(name)
            if entry.author_birth_year is None:
                entry.author_birth_year = _int_or_none(
                    _text(agent.find(f"{{{NS_PGTERMS}}}birthdate"))
                )
            if entry.author_death_year is None:
                entry.author_death_year = _int_or_none(
                    _text(agent.find(f"{{{NS_PGTERMS}}}deathdate"))
                )

    for language in ebook.findall(f"{{{NS_DCTERMS}}}language"):
        for description in language.findall(f"{{{NS_RDF}}}Description"):
            value = _text(description.find(f"{{{NS_RDF}}}value"))
            if value:
                entry.languages.append(value)

    for dc_type in ebook.findall(f"{{{NS_DCTERMS}}}type"):
        for description in dc_type.findall(f"{{{NS_RDF}}}Description"):
            value = _text(description.find(f"{{{NS_RDF}}}value"))
            if value:
                entry.dcmi_types.append(value)

    for subject in ebook.findall(f"{{{NS_DCTERMS}}}subject"):
        for description in subject.findall(f"{{{NS_RDF}}}Description"):
            member = description.find(f"{{{NS_DCAM}}}memberOf")
            scheme = (
                member.get(f"{{{NS_RDF}}}resource", "") if member is not None else ""
            )
            value = _text(description.find(f"{{{NS_RDF}}}value"))
            if not value:
                continue
            if scheme == MEMBER_OF_LCC:
                entry.lcc_subclasses.append(value.upper())
            elif scheme == MEMBER_OF_LCSH:
                entry.lcsh_headings.append(value)

    for shelf in ebook.findall(f"{{{NS_PGTERMS}}}bookshelf"):
        for description in shelf.findall(f"{{{NS_RDF}}}Description"):
            value = _text(description.find(f"{{{NS_RDF}}}value"))
            if value:
                entry.bookshelves.append(value)

    # Plain-text availability, and its size, so selection can skip works too
    # short to chunk and too long to be worth downloading.
    for has_format in ebook.findall(f"{{{NS_DCTERMS}}}hasFormat"):
        for pg_file in has_format.findall(f"{{{NS_PGTERMS}}}file"):
            imts = [
                _text(description.find(f"{{{NS_RDF}}}value"))
                for fmt in pg_file.findall(f"{{{NS_DCTERMS}}}format")
                for description in fmt.findall(f"{{{NS_RDF}}}Description")
            ]
            if not any(imt.startswith("text/plain") for imt in imts):
                continue
            extent = _int_or_none(_text(pg_file.find(f"{{{NS_DCTERMS}}}extent"))) or 0
            entry.text_bytes = max(entry.text_bytes, extent)

    return entry


def iter_catalog_archive(archive: Path) -> Iterator[CatalogEntry]:
    """Stream ``rdf-files.tar.bz2`` and yield one entry per ebook.

    Streamed rather than extracted: the archive holds ~75,000 tiny files, and
    unpacking them costs inodes and minutes for no benefit when a single pass
    produces the catalogue.
    """
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".rdf"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            entry = parse_ebook_rdf(handle.read())
            if entry is not None:
                yield entry


# =============================================================================
# HTTP: polite, cached, resumable
# =============================================================================


class PoliteFetcher:
    """Minimal rate-limited HTTP client that respects robots.txt.

    Gutenberg asks bulk users to be gentle and to identify themselves. This
    enforces a floor on the interval between requests, sends a contactable
    User-Agent, retries transient failures with backoff, and refuses any URL
    robots.txt disallows.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval: float = 1.0,
        timeout: float = 60.0,
        max_retries: int = 3,
        respect_robots: bool = True,
    ) -> None:
        self.user_agent = user_agent
        self.min_interval = max(0.0, min_interval)
        self.timeout = timeout
        self.max_retries = max_retries
        self.respect_robots = respect_robots
        self._last_request = 0.0
        self._robots: urllib.robotparser.RobotFileParser | None = None

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _load_robots(self) -> urllib.robotparser.RobotFileParser | None:
        if self._robots is not None or not self.respect_robots:
            return self._robots
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(ROBOTS_URL)
        try:
            self._wait()
            request = urllib.request.Request(
                ROBOTS_URL, headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                parser.parse(response.read().decode("utf-8", "replace").splitlines())
        except (urllib.error.URLError, OSError):
            # An unreachable robots.txt is not permission to ignore it, but it
            # is also not a reason to abort a resumable job; treat as allow-all
            # and keep the rate limit doing the protective work.
            parser.parse(["User-agent: *", "Allow: /"])
        self._robots = parser
        return parser

    def allowed(self, url: str) -> bool:
        parser = self._load_robots()
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def get(self, url: str) -> bytes:
        """GET ``url`` with rate limiting and retries.

        Raises:
            PermissionError: robots.txt disallows the URL.
            urllib.error.URLError: All retries failed.
        """
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._wait()
            request = urllib.request.Request(
                url, headers={"User-Agent": self.user_agent}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return bytes(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 403, 410):
                    raise
                last_error = exc
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
            time.sleep(2.0 * (attempt + 1))
        raise urllib.error.URLError(
            f"GET {url} failed after {self.max_retries} attempts: {last_error}"
        )

    def download(self, url: str, destination: Path) -> Path:
        """Download to ``destination`` atomically; skip if it already exists."""
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        self._wait()
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with (
            urllib.request.urlopen(request, timeout=self.timeout) as response,
            partial.open("wb") as handle,
        ):
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
        partial.replace(destination)
        return destination


# =============================================================================
# Text cleaning and chunking
# =============================================================================

#: Gutenberg's boilerplate fences. Wording drifted over 25 years, so match the
#: stable core ("START OF ... PROJECT GUTENBERG EBOOK") rather than any one form.
_START_MARKER = re.compile(
    r"^\s*\*\*\*\s*START OF (?:THE|THIS)\s+PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_END_MARKER = re.compile(
    r"^\s*\*\*\*\s*END OF (?:THE|THIS)\s+PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
#: Pre-2005 texts used a bare sentence instead of the starred fence.
_LEGACY_END = re.compile(
    r"^\s*End of (?:the )?(?:Project Gutenberg|The Project Gutenberg).*$",
    re.IGNORECASE | re.MULTILINE,
)
#: Transcription credits that sit inside the fences and are not part of the work.
_CREDIT_LINE = re.compile(
    r"^\s*(?:Produced by|E-text prepared by|Transcribed (?:from|by)|"
    r"Transcriber'?s? Note|Updated:|Credits:|Release date:|Language:|"
    r"Most recently updated:|Original publication:)",
    re.IGNORECASE,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return the work's own text, without PG licence headers or credits."""
    start = _START_MARKER.search(text)
    if start is not None:
        text = text[start.end() :]

    end = _END_MARKER.search(text)
    if end is not None:
        text = text[: end.start()]
    else:
        legacy = _LEGACY_END.search(text)
        if legacy is not None:
            text = text[: legacy.start()]

    # Drop leading transcription credits: any of the first few paragraphs that
    # open with a known credit marker.
    paragraphs = split_paragraphs(text)
    while paragraphs and _CREDIT_LINE.match(paragraphs[0]):
        paragraphs.pop(0)
    return "\n\n".join(paragraphs)


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; join hard-wrapped lines back into one paragraph."""
    normalized = unicodedata.normalize(
        "NFC", text.replace("\r\n", "\n").replace("\r", "\n")
    )
    out: list[str] = []
    for block in re.split(r"\n\s*\n+", normalized):
        joined = " ".join(line.strip() for line in block.split("\n"))
        joined = " ".join(joined.split())
        if joined:
            out.append(joined)
    return out


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()


def drop_title_lines(
    paragraphs: Sequence[str], title: str, authors: Sequence[str]
) -> list[str]:
    """Remove paragraphs that are just the title or a "by <author>" byline.

    SHELF's ``text`` field is body-only precisely so titles cannot leak the
    label; the natural slice has to match that or the two conditions are not
    comparable.
    """
    banned = {_normalize_for_match(title)} if title else set()
    for author in authors:
        # Gutenberg stores authors inverted ("Austen, Jane") while title pages
        # print them forward ("By Jane Austen"), so both orders are needed.
        surname, _, given = (part.strip() for part in author.partition(","))
        variants = {author, surname}
        if given:
            variants.add(f"{given} {surname}")
        for variant in variants:
            if variant:
                banned.add(_normalize_for_match(variant))
                banned.add(_normalize_for_match(f"by {variant}"))
    banned.discard("")
    return [p for p in paragraphs if _normalize_for_match(p) not in banned]


@dataclass
class ChunkConfig:
    """Passage-chunking policy.

    Defaults target SHELF's own length profile (median 323 words, mean 630) so
    that a transfer comparison is not silently a length comparison -- Finding 5
    of the data plan puts 17 macro-F1 points between 22-word and 1,900-word
    documents, which would swamp any real transfer effect.
    """

    target_words: int = 400
    min_words: int = 150
    max_words: int = 900
    skip_head_chunks: int = 1
    skip_tail_chunks: int = 1
    max_per_work: int = 3


def chunk_paragraphs(paragraphs: Sequence[str], config: ChunkConfig) -> list[str]:
    """Pack paragraphs into passages of roughly ``target_words``.

    Paragraph boundaries are preserved wherever possible; a single paragraph
    longer than ``max_words`` is hard-split on word windows rather than dropped.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    buffered_words = 0

    def flush() -> None:
        nonlocal buffer, buffered_words
        if buffered_words >= config.min_words:
            chunks.append("\n\n".join(buffer))
        buffer = []
        buffered_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > config.max_words:
            flush()
            for start in range(0, len(words), config.target_words):
                window = words[start : start + config.target_words]
                if len(window) >= config.min_words:
                    chunks.append(" ".join(window))
            continue

        # Close the current passage before it would exceed the hard ceiling, so
        # ``max_words`` is a real bound rather than a bound on paragraphs only.
        if buffered_words and buffered_words + len(words) > config.max_words:
            flush()

        buffer.append(paragraph)
        buffered_words += len(words)
        if buffered_words >= config.target_words:
            flush()

    flush()
    return chunks


def select_chunks(chunks: Sequence[str], config: ChunkConfig) -> list[tuple[int, str]]:
    """Pick up to ``max_per_work`` passages, evenly spaced across the work.

    Even spacing rather than "the first N": the opening of a book is front
    matter, dedications, and a table of contents, none of which is
    representative of the classified work. Head and tail chunks are dropped
    outright for the same reason.
    """
    if not chunks:
        return []

    head = config.skip_head_chunks
    tail = len(chunks) - config.skip_tail_chunks
    body = list(range(head, tail))
    if not body:
        body = list(range(len(chunks)))

    if len(body) <= config.max_per_work:
        picked = body
    else:
        step = len(body) / config.max_per_work
        picked = [
            body[min(len(body) - 1, int(i * step + step / 2))]
            for i in range(config.max_per_work)
        ]
        picked = sorted(set(picked))

    return [(i, chunks[i]) for i in picked]


# =============================================================================
# Selection
# =============================================================================


@dataclass
class SelectionConfig:
    """Which works are eligible, and how flat the sample is pushed."""

    target_passages: int = 3000
    languages: tuple[str, ...] = ("en",)
    min_text_bytes: int = 30_000
    max_text_bytes: int = 3_000_000
    max_works_per_author: int = 2
    require_single_lcc_letter: bool = True
    require_public_domain: bool = True
    seed: int = 42


def _normalized_work_key(entry: CatalogEntry) -> str:
    author = entry.authors[0] if entry.authors else ""
    return f"{_normalize_for_match(entry.title)}|{_normalize_for_match(author)}"


def eligible(entry: CatalogEntry, config: SelectionConfig) -> str | None:
    """Return a rejection reason, or ``None`` when the entry is usable."""
    if "Text" not in entry.dcmi_types:
        return "not_text"
    if not entry.title:
        return "no_title"
    if not entry.lcc_subclasses:
        return "no_lcc"
    letters = entry.lcc_letters
    if not letters:
        return "lcc_not_main_class"
    if config.require_single_lcc_letter and len(letters) > 1:
        # Multi-class works are genuinely ambiguous under a 21-way single-label
        # task. Dropping them keeps the labels as clean as the cataloguer left
        # them instead of picking a winner ourselves.
        return "multi_lcc_letter"
    if config.languages and not (set(entry.languages) & set(config.languages)):
        return "language"
    if config.require_public_domain and "public domain" not in entry.rights.lower():
        return "not_public_domain"
    if entry.text_bytes < config.min_text_bytes:
        return "too_short"
    if entry.text_bytes > config.max_text_bytes:
        return "too_long"
    return None


@dataclass
class Selection:
    """The chosen works plus the bookkeeping needed to explain the choice."""

    works: list[dict[str, Any]] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    eligible_per_class: dict[str, int] = field(default_factory=dict)
    selected_per_class: dict[str, int] = field(default_factory=dict)
    per_class_work_quota: int = 0
    config: dict[str, Any] = field(default_factory=dict)


def select_works(
    entries: Iterable[CatalogEntry],
    config: SelectionConfig,
    chunk_config: ChunkConfig,
) -> Selection:
    """Stratify toward the flattest achievable LCC distribution.

    Every class gets the *same* work quota. Classes that cannot fill it
    contribute what they have and the shortfall is **not** redistributed --
    redistributing would hand the slack straight back to P, undoing the
    flattening. The realized deficit is reported instead.
    """
    rng = random.Random(config.seed)
    rejections: Counter[str] = Counter()
    by_class: dict[str, list[CatalogEntry]] = defaultdict(list)
    seen_keys: set[str] = set()

    for entry in entries:
        reason = eligible(entry, config)
        if reason is not None:
            rejections[reason] += 1
            continue
        key = _normalized_work_key(entry)
        if key in seen_keys:
            rejections["duplicate_work"] += 1
            continue
        seen_keys.add(key)
        by_class[entry.lcc_letters[0]].append(entry)

    per_class_passages = max(1, config.target_passages // len(LCC_MAIN_CLASSES))
    quota = max(1, -(-per_class_passages // max(1, chunk_config.max_per_work)))

    selection = Selection(
        per_class_work_quota=quota,
        config={
            "selection": asdict(config),
            "chunking": asdict(chunk_config),
        },
    )
    selection.eligible_per_class = {
        letter: len(by_class.get(letter, [])) for letter in LCC_MAIN_CLASSES
    }

    for letter in LCC_MAIN_CLASSES:
        candidates = list(by_class.get(letter, []))
        # Shuffle rather than rank by download count: popularity correlates with
        # how heavily a text was memorized, and ranking on it would make the
        # contaminated condition maximally contaminated.
        rng.shuffle(candidates)

        per_author: Counter[str] = Counter()
        chosen: list[CatalogEntry] = []
        for entry in candidates:
            if len(chosen) >= quota:
                break
            author = _normalize_for_match(entry.authors[0] if entry.authors else "")
            if author and per_author[author] >= config.max_works_per_author:
                continue
            per_author[author] += 1
            chosen.append(entry)

        selection.selected_per_class[letter] = len(chosen)
        for entry in chosen:
            selection.works.append(entry.to_dict())

    selection.rejections = dict(rejections.most_common())
    return selection


# =============================================================================
# Record construction
# =============================================================================


def build_records(
    works: Sequence[dict[str, Any]],
    texts_dir: Path,
    chunk_config: ChunkConfig,
    *,
    retrieved_at: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Turn fetched texts into SHELF-shaped records with document-level labels."""
    records: list[dict[str, Any]] = []
    problems: Counter[str] = Counter()

    for work in works:
        gid = int(work["gutenberg_id"])
        path = texts_dir / f"{gid}.txt"
        if not path.exists():
            problems["text_missing"] += 1
            continue

        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = strip_gutenberg_boilerplate(raw)
        paragraphs = drop_title_lines(
            split_paragraphs(cleaned), work.get("title", ""), work.get("authors") or []
        )
        chunks = chunk_paragraphs(paragraphs, chunk_config)
        picked = select_chunks(chunks, chunk_config)
        if not picked:
            problems["no_usable_chunks"] += 1
            continue

        letters = [c[0] for c in work.get("lcc_subclasses", []) if c]
        letter = next((letter for letter in letters if letter in LCC_MAIN_CLASSES), "")
        if not letter:
            problems["no_lcc_letter"] += 1
            continue

        headings = work.get("lcsh_headings") or []
        form, category = derive_lcgft(headings)
        text_url = TEXT_URL_TEMPLATE.format(gid=gid)
        digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()

        for chunk_index, chunk_text in picked:
            record = {
                "id": f"pg-{gid}-{chunk_index:04d}",
                "text": chunk_text,
                "body": chunk_text,
                "title": work.get("title", ""),
                "word_count": len(chunk_text.split()),
                "lcc_code": letter,
                "lcc_name": LCC_MAIN_CLASSES[letter],
                "lcc_uri": f"{LCC_URI_BASE}{letter}",
                "lcgft_category": category,
                "lcgft_form": form,
                "topics": lcsh_main_headings(headings),
                "geographic": [],
                "audience": derive_audience(headings),
                "register": "",
                "language": (work.get("languages") or [""])[0],
                "author": (work.get("authors") or [""])[0],
                "source_type": SourceType.NATURAL.value,
                "source": NaturalSource.PROJECT_GUTENBERG.value,
                "contamination_status": ContaminationStatus.KNOWN_CONTAMINATED.value,
                # The label spaces of a natural record are NOT SHELF's. topics
                # here are raw LCSH headings, not the 112-value SHELF pool, and
                # lcgft_* is inferred from LCSH rather than catalogued. Anything
                # that trains or scores across both must read this field first.
                "label_space": {
                    "lcc_code": "lcc_main_class",
                    "topics": "lcsh_main_headings",
                    "lcgft": "shelf_v0.3.1_pool_derived_from_lcsh",
                    "audience": "shelf_v0.3.1_pool_derived_from_lcsh",
                },
                "schema_version": TRANSFER_SCHEMA_VERSION,
                "provenance": {
                    "gutenberg_id": gid,
                    "lcc_letter": letter,
                    "lcc_subclasses": work.get("lcc_subclasses") or [],
                    "lcsh_headings": headings,
                    "bookshelves": work.get("bookshelves") or [],
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "chunks_selected": len(picked),
                    "text_url": text_url,
                    "text_sha256": digest,
                    "text_bytes": len(raw.encode("utf-8", "replace")),
                    "gutenberg_issued": work.get("issued", ""),
                    "author_birth_year": work.get("author_birth_year"),
                    "author_death_year": work.get("author_death_year"),
                    "rights": work.get("rights", ""),
                    "language": (work.get("languages") or [""])[0],
                    "retrieved_at": retrieved_at,
                    "builder_version": BUILDER_VERSION,
                },
            }
            records.append(record)

    return records, problems


# =============================================================================
# Stages
# =============================================================================


def stage_download(args: argparse.Namespace, fetcher: PoliteFetcher) -> Path:
    archive = Path(args.cache_dir) / "rdf-files.tar.bz2"
    if args.dry_run:
        print(f"[dry-run] would download {RDF_ARCHIVE_URL} -> {archive}")
        return archive
    if archive.exists() and archive.stat().st_size > 0:
        print(f"[download] cached: {archive} ({archive.stat().st_size:,} bytes)")
        return archive
    print(f"[download] {RDF_ARCHIVE_URL}")
    fetcher.download(RDF_ARCHIVE_URL, archive)
    print(f"[download] wrote {archive} ({archive.stat().st_size:,} bytes)")
    return archive


def stage_catalog(args: argparse.Namespace) -> Path:
    archive = Path(args.cache_dir) / "rdf-files.tar.bz2"
    catalog_path = Path(args.cache_dir) / "catalog.jsonl"
    stats_path = Path(args.cache_dir) / "catalog_stats.json"

    if args.dry_run:
        print(f"[dry-run] would parse {archive} -> {catalog_path}")
        return catalog_path
    if catalog_path.exists() and not args.force:
        print(f"[catalog] cached: {catalog_path}")
        return catalog_path
    if not archive.exists():
        raise FileNotFoundError(
            f"Catalogue archive missing; run the download stage first: {archive}"
        )

    total = 0
    with_lcc = 0
    letters: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    types: Counter[str] = Counter()

    tmp = catalog_path.with_suffix(".jsonl.part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        for entry in iter_catalog_archive(archive):
            total += 1
            for dc_type in entry.dcmi_types:
                types[dc_type] += 1
            for language in entry.languages:
                languages[language] += 1
            if not entry.lcc_subclasses:
                continue
            with_lcc += 1
            for letter in entry.lcc_letters:
                letters[letter] += 1
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            if total % 10_000 == 0:
                print(
                    f"[catalog] parsed {total:,} ebooks ({with_lcc:,} with an LCC class)"
                )
    tmp.replace(catalog_path)

    stats = {
        "ebooks_total": total,
        "ebooks_with_lcc": with_lcc,
        "lcc_letter_counts": dict(letters.most_common()),
        "language_counts": dict(languages.most_common(25)),
        "dcmi_type_counts": dict(types.most_common()),
        "archive_sha256": _sha256_file(archive),
        "source_url": RDF_ARCHIVE_URL,
        "parsed_at": _now(),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"[catalog] {total:,} ebooks, {with_lcc:,} with an LCC class -> {catalog_path}"
    )
    return catalog_path


def iter_catalog(path: Path) -> Iterator[CatalogEntry]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield CatalogEntry(**json.loads(line))


def stage_select(args: argparse.Namespace) -> Path:
    catalog_path = Path(args.cache_dir) / "catalog.jsonl"
    selection_path = Path(args.output_dir) / "selection.json"
    chunk_config = _chunk_config(args)
    selection_config = _selection_config(args)

    if args.dry_run:
        print(
            f"[dry-run] would select ~{selection_config.target_passages} passages "
            f"from {catalog_path} -> {selection_path}"
        )
        return selection_path
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalogue missing; run the catalog stage first: {catalog_path}"
        )

    selection = select_works(iter_catalog(catalog_path), selection_config, chunk_config)

    if args.limit:
        # Round-robin across classes rather than truncating the (class-ordered)
        # list, so a smoke run still exercises every class it can reach instead
        # of only the alphabetically first few.
        by_letter: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for work in selection.works:
            letters = [c[0] for c in work.get("lcc_subclasses", []) if c]
            letter = next(
                (letter for letter in letters if letter in LCC_MAIN_CLASSES), "?"
            )
            by_letter[letter].append(work)

        limited: list[dict[str, Any]] = []
        depth = 0
        while len(limited) < args.limit and any(
            len(v) > depth for v in by_letter.values()
        ):
            for letter in LCC_MAIN_CLASSES:
                bucket = by_letter.get(letter, [])
                if depth < len(bucket) and len(limited) < args.limit:
                    limited.append(bucket[depth])
            depth += 1

        selection.works = limited
        kept: Counter[str] = Counter()
        for work in limited:
            letters = [c[0] for c in work.get("lcc_subclasses", []) if c]
            kept[
                next((letter for letter in letters if letter in LCC_MAIN_CLASSES), "?")
            ] += 1
        selection.selected_per_class = dict(sorted(kept.items()))

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(asdict(selection), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[select] {len(selection.works):,} works "
        f"(quota {selection.per_class_work_quota}/class) -> {selection_path}"
    )
    thin = [
        f"{letter}={selection.selected_per_class.get(letter, 0)}"
        for letter in LCC_MAIN_CLASSES
        if selection.selected_per_class.get(letter, 0) < selection.per_class_work_quota
    ]
    if thin:
        print(
            f"[select] classes below quota (shortfall NOT redistributed): {', '.join(thin)}"
        )
    return selection_path


def stage_fetch(args: argparse.Namespace, fetcher: PoliteFetcher) -> Path:
    selection_path = Path(args.output_dir) / "selection.json"
    texts_dir = Path(args.cache_dir) / "texts"

    if args.dry_run:
        print(
            f"[dry-run] would fetch texts for the works in {selection_path} -> {texts_dir}"
        )
        return texts_dir
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selection missing; run the select stage first: {selection_path}"
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    works = selection["works"]
    texts_dir.mkdir(parents=True, exist_ok=True)

    fetched = skipped = failed = 0
    for i, work in enumerate(works, start=1):
        gid = int(work["gutenberg_id"])
        destination = texts_dir / f"{gid}.txt"
        if destination.exists() and destination.stat().st_size > 0:
            skipped += 1
            continue
        url = TEXT_URL_TEMPLATE.format(gid=gid)
        try:
            payload = fetcher.get(url)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            PermissionError,
            OSError,
        ) as exc:
            failed += 1
            print(f"[fetch] {gid}: FAILED ({exc})")
            continue
        destination.write_text(payload.decode("utf-8", "replace"), encoding="utf-8")
        fetched += 1
        if i % 50 == 0:
            print(
                f"[fetch] {i:,}/{len(works):,} (new {fetched:,}, cached {skipped:,}, failed {failed:,})"
            )

    print(
        f"[fetch] done: {fetched:,} new, {skipped:,} cached, {failed:,} failed -> {texts_dir}"
    )
    return texts_dir


def stage_build(args: argparse.Namespace) -> Path:
    selection_path = Path(args.output_dir) / "selection.json"
    texts_dir = Path(args.cache_dir) / "texts"
    records_path = Path(args.output_dir) / "records.jsonl"

    if args.dry_run:
        print(f"[dry-run] would build records from {texts_dir} -> {records_path}")
        return records_path
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selection missing; run the select stage first: {selection_path}"
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chunk_config = _chunk_config(args)
    retrieved_at = _now()

    records, problems = build_records(
        selection["works"], texts_dir, chunk_config, retrieved_at=retrieved_at
    )
    if not records:
        raise RuntimeError(
            "No records produced. Run the fetch stage first, or relax the chunking limits."
        )

    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # The eligible-works pool is what makes the residual skew visible once the
    # passage counts themselves are flat, so it goes into the report.
    profile, comparison, report = build_report(
        records, pool_counts=selection.get("eligible_per_class") or None
    )

    (Path(args.output_dir) / "distribution_report.txt").write_text(
        report, encoding="utf-8"
    )
    (Path(args.output_dir) / "distribution_report.json").write_text(
        json.dumps(
            {"profile": profile.to_dict(), "comparison_to_shelf": comparison.to_dict()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "built_at": retrieved_at,
        "source": NaturalSource.PROJECT_GUTENBERG.value,
        "source_type": SourceType.NATURAL.value,
        "source_urls": {"catalog": RDF_ARCHIVE_URL, "text": TEXT_URL_TEMPLATE},
        "contamination_status": ContaminationStatus.KNOWN_CONTAMINATED.value,
        "contamination_notice": CONTAMINATION_NOTICE,
        "skew_notice": SKEW_NOTICE,
        "n_records": len(records),
        "n_works": profile.n_works,
        "build_problems": dict(problems.most_common()),
        "selection": selection.get("config", {}),
        "selection_rejections": selection.get("rejections", {}),
        "eligible_per_class": selection.get("eligible_per_class", {}),
        "selected_works_per_class": selection.get("selected_per_class", {}),
        "records_sha256": _sha256_file(records_path),
        "profile": profile.to_dict(),
        "comparison_to_shelf": comparison.to_dict(),
    }
    (Path(args.output_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(report)
    print(
        f"[build] {len(records):,} passages from {profile.n_works:,} works -> {records_path}"
    )
    if problems:
        print(f"[build] problems: {dict(problems.most_common())}")
    return records_path


# =============================================================================
# Helpers
# =============================================================================


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_config(args: argparse.Namespace) -> ChunkConfig:
    return ChunkConfig(
        target_words=args.chunk_words,
        min_words=args.min_chunk_words,
        max_words=args.max_chunk_words,
        skip_head_chunks=args.skip_head_chunks,
        skip_tail_chunks=args.skip_tail_chunks,
        max_per_work=args.max_chunks_per_work,
    )


def _selection_config(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        target_passages=args.target_passages,
        languages=tuple(args.languages),
        min_text_bytes=args.min_text_bytes,
        max_text_bytes=args.max_text_bytes,
        max_works_per_author=args.max_works_per_author,
        require_single_lcc_letter=not args.allow_multi_lcc,
        require_public_domain=not args.allow_non_public_domain,
        seed=args.seed,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 6 natural-data transfer slice from Project Gutenberg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--stage",
        type=Stage,
        choices=list(Stage),
        default=Stage.ALL,
        help="Run one stage only (default: all). Stages are individually resumable.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/transfer/cache"),
        help="Bulk downloads and intermediates (git-ignored)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/transfer/gutenberg"),
        help="Slice output: records.jsonl, manifest.json, distribution report",
    )

    parser.add_argument(
        "--target-passages", type=int, default=3000, help="Total passages to aim for"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of selected works (smoke tests; 0 = no cap)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Selection seed")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Accepted RFC4646 language codes (default: en)",
    )
    parser.add_argument("--min-text-bytes", type=int, default=30_000)
    parser.add_argument("--max-text-bytes", type=int, default=3_000_000)
    parser.add_argument("--max-works-per-author", type=int, default=2)
    parser.add_argument(
        "--allow-multi-lcc",
        action="store_true",
        help="Keep works catalogued under several LCC main classes (ambiguous labels)",
    )
    parser.add_argument(
        "--allow-non-public-domain",
        action="store_true",
        help="Keep works whose dcterms:rights is not 'Public domain in the USA.'",
    )

    parser.add_argument(
        "--chunk-words", type=int, default=400, help="Target passage length"
    )
    parser.add_argument("--min-chunk-words", type=int, default=150)
    parser.add_argument("--max-chunk-words", type=int, default=900)
    parser.add_argument(
        "--skip-head-chunks", type=int, default=1, help="Drop leading front matter"
    )
    parser.add_argument(
        "--skip-tail-chunks", type=int, default=1, help="Drop trailing back matter"
    )
    parser.add_argument("--max-chunks-per-work", type=int, default=3)

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Minimum seconds between HTTP requests (be gentle with the mirrors)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Skip the robots.txt check (do not use against gutenberg.org)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Rebuild cached intermediates"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan; make no network requests and write nothing",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 78)
    print("SHELF Phase 6 - natural-data transfer slice (Project Gutenberg)")
    print("=" * 78)
    print(CONTAMINATION_NOTICE)
    print()
    print(SKEW_NOTICE)
    print("=" * 78)

    if not args.dry_run:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fetcher = PoliteFetcher(
        user_agent=args.user_agent,
        min_interval=args.rate_limit,
        timeout=args.timeout,
        respect_robots=not args.ignore_robots,
    )

    stages = STAGE_ORDER if args.stage is Stage.ALL else (args.stage,)
    try:
        for stage in stages:
            if stage is Stage.DOWNLOAD:
                stage_download(args, fetcher)
            elif stage is Stage.CATALOG:
                stage_catalog(args)
            elif stage is Stage.SELECT:
                stage_select(args)
            elif stage is Stage.FETCH:
                stage_fetch(args, fetcher)
            elif stage is Stage.BUILD:
                stage_build(args)
    except (FileNotFoundError, RuntimeError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

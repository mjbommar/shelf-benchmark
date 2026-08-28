#!/usr/bin/env python3
"""Enrich SHELF taxonomy frequency tables with authoritative LC scope notes.

Background
----------
Every file under ``data/taxonomies/*.json`` is a pure MARC frequency-rank
table: ``id``, ``label``, ``frequency``, ``rank`` are populated and
``uri`` / ``description`` / ``alt_labels`` / ``broader`` / ``narrower`` are
empty for all 4,829 entries (verified 2026-08-26). See
``docs/data_plan_v0.4.md`` section 4.1.1 -- the blocker for expanding the
sampling pools is not pool size, it is the absence of semantic descriptions
for the labels the generation prompt has to describe *without naming*.

This script fetches those descriptions from the authoritative source
(id.loc.gov) rather than inventing them, and writes NEW files under
``data/taxonomies/enriched/``. The frequency tables are inputs and are never
modified.

Source strategy (chosen by measurement, see ``docs`` note in the coverage
report):

======================  ==========================================================
Taxonomy                Source
======================  ==========================================================
LCSH topical / geo      Bulk SKOS/RDF JSON-LD dump (``subjects.skosrdf.jsonld.gz``,
                        66 MB gzip, ~15 s download, ~5 s to stream-index
                        507,849 concepts). One HTTP request instead of the
                        ~5,000 that per-URI lookups for 2,500 labels would need.
LCGFT forms             Bulk SKOS/RDF JSON-LD dump (``genreForms.skosrdf.jsonld.gz``,
                        522 KB). Trivially cheaper than 554 per-URI lookups.
LCC subclasses          No bulk download exists. Uses the id.loc.gov
                        classification API. The per-subclass collection
                        endpoint 404s for all but a couple of law schedules
                        (measured: only KF and KZ of 80 tested return 200), so
                        subclass ranges are discovered through the *main class*
                        collection (<=21 requests) and then fetched per range.
LC Name Authority       NOT used by default. ``names.skosrdf.jsonld.gz`` is
(geographic fallback)   1.69 GB and NAF place records carry no definitions --
                        only variant names, citation sources and a GAC code
                        (verified on n79041717 "California"). Opt in with
                        ``--include-naf`` for URIs + variant labels only.
======================  ==========================================================

Both bulk dumps are streamed line-by-line (they are ND-JSON, one concept
record per line) and never loaded into memory whole.

Usage
-----
    uv run python scripts/enrich_taxonomies.py --dry-run
    uv run python scripts/enrich_taxonomies.py --taxonomy lcgft --limit 50
    uv run python scripts/enrich_taxonomies.py            # everything

Every HTTP response is cached on disk under
``data/taxonomies/enriched/.cache/`` so an interrupted run resumes without
re-fetching. Requests are rate limited (``--delay``, default 0.34 s ~ 3 rps).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.parse
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

BULK_DUMPS: dict[str, str] = {
    "lcsh": "https://id.loc.gov/download/authorities/subjects.skosrdf.jsonld.gz",
    "lcgft": "https://id.loc.gov/download/authorities/genreForms.skosrdf.jsonld.gz",
}

CLASSIFICATION_BASE = "https://id.loc.gov/authorities/classification/"
NAF_LABEL_BASE = "https://id.loc.gov/authorities/names/label/"
NAF_URI_BASE = "http://id.loc.gov/authorities/names/"

AUTHORITY_URI_BASE: dict[str, str] = {
    "lcsh": "http://id.loc.gov/authorities/subjects/",
    "lcgft": "http://id.loc.gov/authorities/genreForms/",
    "lcc": "http://id.loc.gov/authorities/classification/",
    "naf": NAF_URI_BASE,
}

# JSON-LD predicate IRIs used by the classification API (which serves
# expanded IRIs, unlike the bulk dumps which use the ``skos:`` prefix form).
MADS_AUTHORITATIVE_LABEL = "http://www.loc.gov/mads/rdf/v1#authoritativeLabel"
MADS_COLLECTION_MEMBER = "http://www.loc.gov/mads/rdf/v1#hasMADSCollectionMember"
MADS_VARIANT_LABEL = "http://www.loc.gov/mads/rdf/v1#variantLabel"
MADS_CODE = "http://www.loc.gov/mads/rdf/v1#code"
MADS_EDITORIAL_NOTE = "http://www.loc.gov/mads/rdf/v1#editorialNote"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"
SKOS_SCOPE_NOTE = "http://www.w3.org/2004/02/skos/core#scopeNote"
SKOS_NARROWER = "http://www.w3.org/2004/02/skos/core#narrower"

# ---------------------------------------------------------------------------
# What we enrich
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaxonomySpec:
    """One frequency table to enrich, and where its labels live upstream."""

    key: str
    source_file: str
    output_file: str
    # Bulk dumps to search, in priority order.
    dumps: tuple[str, ...]
    kind: str  # "topical" | "geographic" | "form"


TAXONOMIES: dict[str, TaxonomySpec] = {
    "lcsh_topical": TaxonomySpec(
        key="lcsh_topical",
        source_file="lcsh_topical_top2000.json",
        output_file="lcsh_topical_top2000.json",
        dumps=("lcsh", "lcgft"),
        kind="topical",
    ),
    "lcsh_geo": TaxonomySpec(
        key="lcsh_geo",
        source_file="lcsh_geo_top500.json",
        output_file="lcsh_geo_top500.json",
        dumps=("lcsh",),
        kind="geographic",
    ),
    "lcgft": TaxonomySpec(
        key="lcgft",
        source_file="lcgft.json",
        output_file="lcgft.json",
        dumps=("lcgft", "lcsh"),
        kind="form",
    ),
}

LCC_SPEC = TaxonomySpec(
    key="lcc_subclass",
    source_file="lcc_subclass_top100.json",
    output_file="lcc_subclass_top100.json",
    dumps=(),
    kind="classification",
)

# The hand-curated v0.3.1 pools are NOT subsets of the MARC frequency tables:
# 63 of the 133 curated LCGFT forms, 70 of the 112 curated topics and 14 of the
# 44 curated geographic areas do not appear in data/taxonomies/*.json at all
# (measured 2026-08-26). Enriching only the frequency tables would therefore
# leave the labels that actually produced the published corpus undescribed, so
# the curated pools are enriched as first-class targets from the same dumps.
CURATED_SOURCES: dict[str, str] = {
    "curated_forms": "LCGFT_DATA",
    "curated_topics": "TOPICS_BY_DOMAIN",
    "curated_geographic": "GEOGRAPHIC_AREAS",
}

CURATED_TAXONOMIES: dict[str, TaxonomySpec] = {
    "curated_forms": TaxonomySpec(
        key="curated_forms",
        source_file="<shelf.sampler.dimensions.LCGFT_DATA>",
        output_file="curated_lcgft_forms.json",
        dumps=("lcgft", "lcsh"),
        kind="form",
    ),
    "curated_topics": TaxonomySpec(
        key="curated_topics",
        source_file="<shelf.sampler.dimensions.TOPICS_BY_DOMAIN>",
        output_file="curated_lcsh_topics.json",
        dumps=("lcsh", "lcgft"),
        kind="topical",
    ),
    "curated_geographic": TaxonomySpec(
        key="curated_geographic",
        source_file="<shelf.sampler.dimensions.GEOGRAPHIC_AREAS>",
        output_file="curated_geographic.json",
        dumps=("lcsh",),
        kind="geographic",
    ),
}

TAXONOMIES.update(CURATED_TAXONOMIES)

ALL_KEYS = (*TAXONOMIES.keys(), LCC_SPEC.key)


def load_curated_labels(key: str) -> list[dict[str, Any]]:
    """Read one curated v0.3.1 pool out of ``shelf.sampler.dimensions``.

    Read-only: the pool constants are the frozen v0.3.1 label sets and are not
    modified here. Returned in the frequency-table shape so the rest of the
    pipeline treats them identically (``frequency`` is 0 because these pools
    are not frequency ranked).
    """
    from shelf.sampler import dimensions

    raw = getattr(dimensions, CURATED_SOURCES[key])
    values: list[str]
    if isinstance(raw, dict):
        seen: dict[str, None] = {}
        for group in raw.values():
            for item in group:
                seen.setdefault(str(item), None)
        values = list(seen)
    else:
        values = [str(item) for item in raw]
    return [
        {"id": value, "label": value, "frequency": 0, "rank": i + 1}
        for i, value in enumerate(sorted(values))
    ]


# ---------------------------------------------------------------------------
# JSON-LD helpers
# ---------------------------------------------------------------------------


def literals(node: dict[str, Any], key: str) -> list[str]:
    """Extract literal string values for ``key`` from a JSON-LD node.

    LC serves the same predicate as a bare string, a single ``{"@value": ...}``
    object, or a list of either, depending on cardinality -- all three occur in
    the published dumps.
    """
    raw = node.get(key)
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            value = item.get("@value")
            if isinstance(value, str):
                out.append(value)
    return out


def references(node: dict[str, Any], key: str) -> list[str]:
    """Extract the trailing path segment of every ``{"@id": ...}`` under ``key``."""
    raw = node.get(key)
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            ident = item.get("@id")
            if isinstance(ident, str) and not ident.startswith("_:"):
                out.append(ident.rstrip("/").split("/")[-1])
    return out


# ---------------------------------------------------------------------------
# SKOS concepts from the bulk dumps
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SkosConcept:
    """A single ``skos:Concept`` lifted out of a bulk SKOS/RDF dump."""

    authority_id: str
    scheme: str
    pref_label: str
    alt_labels: list[str] = field(default_factory=list)
    broader: list[str] = field(default_factory=list)
    narrower: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    history_notes: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    @property
    def uri(self) -> str:
        return AUTHORITY_URI_BASE[self.scheme] + self.authority_id

    def content_score(self) -> int:
        """Rank duplicate ``prefLabel`` records so the substantive one wins.

        LCSH really does publish two records with ``prefLabel`` "Globalization":
        sh2007000663 (a subdivision-use record whose only note is
        "Use as a topical subdivision under...") and sh99010179 (the topical
        heading with a definitional scope note, broader/narrower and variants).
        """
        score = 0
        for note in self.notes:
            score += 4 if classify_note(note) == "definitional" else -1
        if self.broader or self.narrower:
            score += 2
        if self.alt_labels:
            score += 1
        return score


def parse_skos_record(line: str, scheme: str) -> SkosConcept | None:
    """Parse one ND-JSON line of a bulk SKOS dump into a :class:`SkosConcept`.

    Returns ``None`` for lines whose primary node is not a labelled
    ``skos:Concept`` (deprecated ``skosxl:Label`` stubs, change-set only
    records, malformed lines).
    """
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None

    ident = str(record.get("@id", "")).rstrip("/").split("/")[-1]
    if not ident:
        return None

    for node in record.get("@graph", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("@id", ""))
        if not node_id.endswith("/" + ident):
            continue
        types = node.get("@type")
        type_list = types if isinstance(types, list) else [types]
        if "skos:Concept" not in type_list:
            continue
        pref = literals(node, "skos:prefLabel")
        if not pref:
            continue
        return SkosConcept(
            authority_id=ident,
            scheme=scheme,
            pref_label=pref[0],
            alt_labels=literals(node, "skos:altLabel"),
            broader=references(node, "skos:broader"),
            narrower=references(node, "skos:narrower"),
            related=references(node, "skos:related"),
            notes=literals(node, "skos:note"),
            history_notes=literals(node, "skos:historyNote"),
            examples=literals(node, "skos:example"),
        )
    return None


def iter_skos_concepts(path: Path, scheme: str) -> Iterator[SkosConcept]:
    """Stream a gzipped ND-JSON SKOS dump one concept at a time."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            concept = parse_skos_record(line, scheme)
            if concept is not None:
                yield concept


@dataclass(slots=True)
class DumpIndex:
    """Label -> concept lookups collected from a single streaming pass."""

    scheme: str
    by_pref: dict[str, SkosConcept] = field(default_factory=dict)
    by_alt: dict[str, SkosConcept] = field(default_factory=dict)
    id_labels: dict[str, str] = field(default_factory=dict)
    concepts_seen: int = 0

    def lookup(self, label: str) -> tuple[SkosConcept, str] | None:
        key = normalize_label(label)
        concept = self.by_pref.get(key)
        if concept is not None:
            return concept, "pref_label"
        concept = self.by_alt.get(key)
        if concept is not None:
            return concept, "alt_label"
        for variant in label_variants(label)[1:]:
            concept = self.by_pref.get(variant)
            if concept is not None:
                return concept, "pref_label_variant"
            concept = self.by_alt.get(variant)
            if concept is not None:
                return concept, "alt_label_variant"
        return None


def normalize_label(label: str) -> str:
    """Fold a label for matching: lowercase, collapse whitespace, drop trailing dots."""
    return re.sub(r"\s+", " ", label).strip().rstrip(".").lower()


def label_variants(label: str) -> list[str]:
    """Return the normalized label followed by conservative number variants.

    MARC 655 values in the frequency tables are frequently singular ("Index",
    "Map", "Directory") while LCGFT and LCSH establish the plural ("Indexes",
    "Maps", "Directories"). Measured on the 554-term form table: 26 of the 117
    labels that miss on an exact match resolve correctly through this rule, and
    matches found this way are tagged ``*_variant`` in the output so they stay
    auditable.
    """
    base = normalize_label(label)
    variants = [base]

    def add(value: str) -> None:
        if value and value != base and value not in variants:
            variants.append(value)

    if base.endswith("ies"):
        add(base[:-3] + "y")
    elif base.endswith("es") and base[:-2].endswith(("s", "x", "z", "ch", "sh")):
        add(base[:-2])
    elif base.endswith("s") and not base.endswith("ss"):
        add(base[:-1])
    elif base.endswith(("s", "x", "z", "ch", "sh")):
        add(base + "es")
    else:
        add(base + "s")
        if base.endswith("y") and not base.endswith(("ay", "ey", "oy", "uy")):
            add(base[:-1] + "ies")
    return variants


def build_dump_index(path: Path, scheme: str, wanted: set[str]) -> DumpIndex:
    """First pass: keep only concepts whose pref/alt label is in ``wanted``."""
    index = DumpIndex(scheme=scheme)
    for concept in iter_skos_concepts(path, scheme):
        index.concepts_seen += 1
        pref_key = normalize_label(concept.pref_label)
        if pref_key in wanted:
            incumbent = index.by_pref.get(pref_key)
            if incumbent is None or concept.content_score() > incumbent.content_score():
                index.by_pref[pref_key] = concept
        for alt in concept.alt_labels:
            alt_key = normalize_label(alt)
            if alt_key in wanted and alt_key not in index.by_alt:
                index.by_alt[alt_key] = concept
    return index


def resolve_reference_labels(path: Path, scheme: str, ids: set[str]) -> dict[str, str]:
    """Second pass: resolve broader/narrower authority ids to preferred labels.

    Done as a separate streaming pass rather than by holding a 507k-entry
    id->label map from pass one, so peak memory stays proportional to the
    labels we actually care about.
    """
    if not ids:
        return {}
    resolved: dict[str, str] = {}
    for concept in iter_skos_concepts(path, scheme):
        if concept.authority_id in ids:
            resolved[concept.authority_id] = concept.pref_label
            if len(resolved) == len(ids):
                break
    return resolved


# ---------------------------------------------------------------------------
# Scope-note handling
# ---------------------------------------------------------------------------

# LC mixes two very different things into skos:note. Only the definitional
# kind is usable as a semantic description; the usage kind is cataloguing
# instruction ("Use as a topical subdivision under...") and would be actively
# misleading in a generation prompt.
USAGE_NOTE_PREFIXES: tuple[str, ...] = (
    "use ",
    "used ",
    "this heading",
    "this authority record",
    "this record",
    "assign",
    "do not ",
    "subarrange",
    "for works",
    "for general works",
    "see ",
    "prefer ",
    "enter ",
    "may be subdivided",
    "duplicate entry",
    "when this",
)

# Boilerplate openings on genuinely definitional notes. Stripping them turns
# "Here are entered works on the process by which..." into a description that
# reads as a description.
DEFINITION_PREFIXES: tuple[str, ...] = (
    "here are entered works on ",
    "here are entered works about ",
    "here are entered works dealing with ",
    "here are entered individual works and collections of ",
    "here are entered collections of ",
    "here are entered works ",
    "here are entered ",
    "works consisting wholly or chiefly of ",
    "works consisting primarily of ",
    "works consisting of ",
    "works that consist of ",
    "works about ",
    "works on ",
    "this term is used for ",
)

# Sentences matching this are catalogue cross-references, not definition.
CROSS_REFERENCE_PATTERN = re.compile(
    r"\bare entered under\b|\bsee\b|\bfor (?:general |individual )?works\b"
    r"|^for\b|\bprefer\b|\bsearch also under\b",
    re.IGNORECASE,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")

BRACKET_PATTERN = re.compile(r"[\[\]{}]")


def classify_note(note: str) -> str:
    """Classify an LC ``skos:note`` as ``"definitional"`` or ``"usage"``."""
    text = note.strip().lower()
    if not text:
        return "usage"
    for prefix in USAGE_NOTE_PREFIXES:
        if text.startswith(prefix):
            return "usage"
    return "definitional"


def clean_scope_note(note: str) -> str:
    """Reduce an LC scope note to the descriptive clause it contains.

    Purely subtractive: boilerplate openings, cross-reference tails and MARC
    bracketing are removed. No wording is added, so the result stays verbatim
    LC prose.
    """
    text = BRACKET_PATTERN.sub("", note).strip()
    text = re.sub(r"\s+", " ", text)

    sentences = SENTENCE_SPLIT.split(text)
    kept = [
        sentence
        for sentence in sentences
        if sentence.strip()
        and not CROSS_REFERENCE_PATTERN.search(sentence)
        and classify_note(sentence) == "definitional"
    ]
    # A note made entirely of cross-references still beats nothing; keep its
    # opening sentence rather than returning an empty description.
    text = " ".join(kept) if kept else sentences[0]

    lowered = text.lower()
    for prefix in DEFINITION_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.strip().rstrip(".;,").strip()
    if text and text[0].isupper() and not text[1:2].isupper():
        text = text[0].lower() + text[1:]
    return text


def informative_alt_labels(label: str, alt_labels: Sequence[str]) -> list[str]:
    """Drop variant labels that are just re-orderings of the label itself.

    "Insurance, Flood" carries no information the label "Flood insurance" does
    not already carry, and re-using it as a description would reintroduce the
    self-labeling the generation prompt exists to avoid.
    """
    label_tokens = set(re.findall(r"[a-z0-9]+", label.lower()))
    out: list[str] = []
    for alt in alt_labels:
        alt_tokens = set(re.findall(r"[a-z0-9]+", alt.lower()))
        if alt_tokens and alt_tokens <= label_tokens:
            continue
        if contains_label(alt, label):
            continue
        out.append(alt)
    return out


def contains_label(text: str, label: str) -> bool:
    """True if ``label`` appears verbatim (word-bounded) inside ``text``."""
    if not text:
        return False
    return re.search(rf"\b{re.escape(label.lower())}\b", text.lower()) is not None


@dataclass(slots=True)
class Description:
    """A derived semantic description plus an honest provenance tag."""

    text: str | None
    source: str  # scope_note | hierarchy | variant_labels | none
    verbatim: bool


def build_description(
    label: str,
    kind: str,
    definitional_notes: Sequence[str],
    broader_labels: Sequence[str],
    narrower_labels: Sequence[str],
    alt_labels: Sequence[str],
) -> Description:
    """Derive a prompt-ready description, preferring real LC prose.

    Tier 1 (``scope_note``) is LC's own wording with boilerplate stripped --
    verbatim. Tiers 2 and 3 are deterministic templates over LC's own
    hierarchy and variant labels; nothing is authored here that is not present
    in the fetched record.
    """
    for note in definitional_notes:
        cleaned = clean_scope_note(note)
        if len(cleaned) >= 15:
            return Description(text=cleaned, source="scope_note", verbatim=True)

    broader = [b for b in broader_labels if not contains_label(b, label)]
    if broader:
        relation = {
            "form": "a kind of",
            "topical": "a topic within",
            "geographic": "a place within",
        }.get(kind, "a kind of")
        text = f"{relation} {join_phrases(broader[:3])}"
        narrower = [n for n in narrower_labels if not contains_label(n, label)]
        if narrower:
            text += f"; includes {join_phrases(narrower[:4])}"
        return Description(text=text, source="hierarchy", verbatim=False)

    informative = informative_alt_labels(label, alt_labels)
    if informative:
        return Description(
            text=f"also known as {join_phrases(informative[:3])}",
            source="variant_labels",
            verbatim=False,
        )

    return Description(text=None, source="none", verbatim=False)


def join_phrases(items: Iterable[str]) -> str:
    values = [item.strip().rstrip(".") for item in items if item.strip()]
    return ", ".join(values)


# ---------------------------------------------------------------------------
# HTTP client with a resumable on-disk cache
# ---------------------------------------------------------------------------


class LocClient:
    """Polite, cached HTTP access to id.loc.gov.

    Every response body is written under ``cache_dir`` keyed by URL hash, so a
    run interrupted halfway resumes without re-fetching. Requests that hit the
    cache do not sleep.
    """

    def __init__(
        self,
        cache_dir: Path,
        delay: float = 0.34,
        refresh: bool = False,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.delay = delay
        self.refresh = refresh
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={
                "User-Agent": f"shelf-benchmark/enrich_taxonomies {SCRIPT_VERSION}"
            },
        )
        self.requests_made = 0
        self.cache_hits = 0

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LocClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / "api" / f"{digest}.json"

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def fetch(self, url: str) -> dict[str, Any]:
        """Return ``{"status": int, "body": str, "location": str | None}``."""
        path = self._cache_path(url)
        if path.exists() and not self.refresh:
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))

        self._sleep()
        self.requests_made += 1
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:  # network flake -> record, do not cache
            return {"status": 0, "body": "", "location": None, "error": str(exc)}

        payload: dict[str, Any] = {
            "url": url,
            "status": response.status_code,
            "body": response.text if response.status_code == 200 else "",
            "location": response.headers.get("location"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fetch_json(self, url: str) -> list[dict[str, Any]] | None:
        payload = self.fetch(url)
        if payload.get("status") != 200 or not payload.get("body"):
            return None
        try:
            data = json.loads(payload["body"])
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [node for node in data if isinstance(node, dict)]
        return None

    def resolve_redirect(self, url: str) -> str | None:
        payload = self.fetch(url)
        if payload.get("status") in (301, 302, 303, 307, 308):
            location = payload.get("location")
            return location if isinstance(location, str) else None
        return None

    def download(self, url: str, destination: Path) -> Path:
        """Stream a bulk file to disk, skipping the download if already cached."""
        if destination.exists() and not self.refresh:
            self.cache_hits += 1
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".part")
        self.requests_made += 1
        with self._client.stream(
            "GET", url, follow_redirects=True, timeout=None
        ) as response:
            response.raise_for_status()
            with open(temp, "wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    handle.write(chunk)
        temp.replace(destination)
        return destination


# ---------------------------------------------------------------------------
# LCC classification API
# ---------------------------------------------------------------------------

RANGE_PATTERN = re.compile(r"^([A-Z]+)([0-9]+(?:\.[0-9]+)?)$")


def parse_range_code(code: str) -> tuple[str, float, float] | None:
    """Parse ``"KF1-KF9827"`` -> ``("KF", 1.0, 9827.0)``.

    Returns ``None`` for codes that are not a simple numeric range within a
    single alphabetic prefix (e.g. cutter-bearing codes like ``QE51.A1-QE51.Z``).
    """
    parts = code.split("-", 1)
    start = RANGE_PATTERN.match(parts[0])
    if start is None:
        return None
    prefix = start.group(1)
    try:
        low = float(re.sub(r"[^0-9.]", "", start.group(2)) or "0")
    except ValueError:
        return None
    high = low
    if len(parts) == 2:
        end = RANGE_PATTERN.match(parts[1])
        if end is None or end.group(1) != prefix:
            return None
        try:
            high = float(re.sub(r"[^0-9.]", "", end.group(2)) or "0")
        except ValueError:
            return None
    return prefix, low, high


def pick_primary_range(
    codes: Sequence[str], subclass: str
) -> tuple[str | None, list[str]]:
    """Choose the widest range whose alphabetic prefix is exactly ``subclass``.

    Returns ``(primary, siblings)``. The widest range is LC's "general" span
    for the subclass (PN1-PN6790 for PN), and the remaining same-prefix ranges
    are its major subdivisions.
    """
    candidates: list[tuple[str, float, float]] = []
    for code in codes:
        parsed = parse_range_code(code)
        if parsed is None:
            continue
        prefix, low, high = parsed
        if prefix == subclass:
            candidates.append((code, low, high))
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: (-(item[2] - item[1]), item[1]))
    primary = candidates[0][0]
    siblings = [code for code, _, _ in sorted(candidates[1:], key=lambda i: i[1])]
    return primary, siblings


@dataclass(slots=True)
class ClassRange:
    """One LCC range record from the classification API."""

    code: str
    caption: str
    full_caption: str
    scope_notes: list[str] = field(default_factory=list)
    narrower: list[str] = field(default_factory=list)

    @property
    def uri(self) -> str:
        return AUTHORITY_URI_BASE["lcc"] + self.code


def parse_class_range(nodes: Sequence[dict[str, Any]], code: str) -> ClassRange | None:
    for node in nodes:
        if str(node.get("@id", "")).rstrip("/").split("/")[-1] != code:
            continue
        captions = literals(node, MADS_AUTHORITATIVE_LABEL)
        if not captions:
            continue
        full = literals(node, RDFS_LABEL)
        return ClassRange(
            code=code,
            caption=captions[0],
            full_caption=full[0] if full else captions[0],
            scope_notes=literals(node, SKOS_SCOPE_NOTE),
            narrower=references(node, SKOS_NARROWER),
        )
    return None


def parse_class_collection(
    nodes: Sequence[dict[str, Any]], code: str
) -> tuple[str | None, list[str]]:
    """Return ``(class caption, member range codes)`` for a MADSCollection."""
    for node in nodes:
        if str(node.get("@id", "")).rstrip("/").split("/")[-1] != code:
            continue
        comment = literals(node, RDFS_COMMENT)
        caption = None
        if comment:
            text = re.sub(r"\s+", " ", comment[0]).strip()
            # "G -- GEOGRAPHY. ANTHROPOLOGY. RECREATION"
            _, _, tail = text.partition("--")
            caption = (tail or text).strip().title() or None
        members = references(node, MADS_COLLECTION_MEMBER)
        return caption, members
    return None, []


# Captions that every LCC schedule repeats for bibliographic apparatus rather
# than subject matter. They are kept in the record's `covers` list (they are
# real LC data) but skipped when composing the subject-area phrase, where they
# would crowd out the substantive subdivisions.
STRUCTURAL_CAPTIONS: frozenset[str] = frozenset(
    {
        "bibliography",
        "biography",
        "by region or country",
        "collected works",
        "collections",
        "communication of information",
        "congresses",
        "dictionaries and encyclopedias",
        "directories",
        "documents",
        "general works",
        "geographical divisions",
        "history",
        "history and conditions",
        "information services",
        "miscellany",
        "museums",
        "museums. exhibitions",
        "periodicals",
        "periodicals and societies",
        "periodicals. societies. congresses",
        "research",
        "societies",
        "special fields",
        "statistics",
        "study and teaching",
        "yearbooks",
    }
)


def lcc_prompt_description(caption: str, covers: Sequence[str]) -> str | None:
    """Compose the LCC subject-area phrase in the shape the prompt already uses.

    ``LCC_SEMANTIC_DESCRIPTIONS`` in ``sampler/generator.py`` uses comma-joined
    lowercase noun phrases ("law, legal systems, courts, legislation"), so the
    caption ("Industries. Land use. Labor") is split on its sentence dots and
    the widest subdivision captions are appended.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[.;]", caption):
        phrase = chunk.strip().lower()
        if phrase and phrase not in seen:
            seen.add(phrase)
            parts.append(phrase)
    for cover in covers:
        if cover.strip().lower() in STRUCTURAL_CAPTIONS:
            continue
        for chunk in re.split(r"[.;]", cover):
            phrase = chunk.strip().lower()
            if (
                phrase
                and phrase not in seen
                and phrase not in STRUCTURAL_CAPTIONS
                and len(parts) < 12
            ):
                seen.add(phrase)
                parts.append(phrase)
    return ", ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_frequency_table(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'labels' key")
    labels = raw.get("labels")
    if not isinstance(labels, list):
        raise ValueError(f"{path}: missing 'labels' list")
    meta = {k: v for k, v in raw.items() if k != "labels"}
    return meta, labels


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_from_dumps(
    spec: TaxonomySpec,
    entries: Sequence[dict[str, Any]],
    indexes: dict[str, DumpIndex],
    reference_labels: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Attach LC data to every entry of one frequency table."""
    enriched: list[dict[str, Any]] = []
    for entry in entries:
        label = str(entry.get("label", entry.get("id", "")))
        record: dict[str, Any] = {
            "id": entry.get("id", label),
            "label": label,
            "frequency": entry.get("frequency", 0),
            "rank": entry.get("rank"),
            "uri": None,
            "authority_id": None,
            "authority_scheme": None,
            "pref_label": None,
            "alt_labels": [],
            "broader": [],
            "narrower": [],
            "scope_notes": [],
            "usage_notes": [],
            "examples": [],
            "description": None,
            "description_source": "none",
            "description_verbatim": False,
            "description_contains_label": False,
            "match": "none",
        }

        hit: tuple[SkosConcept, str] | None = None
        for scheme in spec.dumps:
            index = indexes.get(scheme)
            if index is None:
                continue
            hit = index.lookup(label)
            if hit is not None:
                break

        if hit is not None:
            concept, match_kind = hit
            labels_for_scheme = reference_labels.get(concept.scheme, {})
            broader = [labels_for_scheme.get(i, i) for i in concept.broader]
            narrower = [labels_for_scheme.get(i, i) for i in concept.narrower]
            definitional = [
                n for n in concept.notes if classify_note(n) == "definitional"
            ]
            usage = [n for n in concept.notes if classify_note(n) == "usage"]
            description = build_description(
                label=label,
                kind=spec.kind,
                definitional_notes=definitional,
                broader_labels=broader,
                narrower_labels=narrower,
                alt_labels=concept.alt_labels,
            )
            record.update(
                {
                    "uri": concept.uri,
                    "authority_id": concept.authority_id,
                    "authority_scheme": concept.scheme,
                    "pref_label": concept.pref_label,
                    "alt_labels": concept.alt_labels,
                    "broader": broader,
                    "narrower": narrower,
                    "scope_notes": definitional,
                    "usage_notes": usage,
                    "examples": concept.examples,
                    "description": description.text,
                    "description_source": description.source,
                    "description_verbatim": description.verbatim,
                    "description_contains_label": contains_label(
                        description.text or "", label
                    ),
                    "match": f"{concept.scheme}:{match_kind}",
                }
            )
        enriched.append(record)
    return enriched


def enrich_geographic_with_naf(
    records: list[dict[str, Any]], client: LocClient, verbose: bool = True
) -> int:
    """Fill in URIs and variant names for places absent from LCSH.

    Jurisdictional place names live in the LC Name Authority File, not in
    LCSH. NAF records carry no definitions (verified on "California",
    n79041717) so this adds provenance and variants only -- never a
    description. ``names.skosrdf.jsonld.gz`` is 1.69 GB, so this uses targeted
    label lookups instead.
    """
    filled = 0
    for record in records:
        if record["match"] != "none":
            continue
        label = str(record["label"])
        quoted = urllib.parse.quote(label, safe="")
        location = client.resolve_redirect(NAF_LABEL_BASE + quoted)
        if not location:
            continue
        authority_id = location.rstrip("/").split("/")[-1]
        nodes = client.fetch_json(
            f"{NAF_URI_BASE}{authority_id}.json".replace("http://", "https://")
        )
        pref_label = label
        variants: list[str] = []
        gac: list[str] = []
        editorial: list[str] = []
        if nodes:
            for node in nodes:
                node_id = str(node.get("@id", ""))
                if node_id.endswith("/" + authority_id):
                    labels_ = literals(node, MADS_AUTHORITATIVE_LABEL)
                    if labels_:
                        pref_label = labels_[0]
                    editorial = literals(node, MADS_EDITORIAL_NOTE)
                    gac = literals(node, MADS_CODE)
                variants.extend(literals(node, MADS_VARIANT_LABEL))
        record.update(
            {
                "uri": NAF_URI_BASE + authority_id,
                "authority_id": authority_id,
                "authority_scheme": "naf",
                "pref_label": pref_label,
                "alt_labels": sorted(set(variants))[:20],
                "usage_notes": editorial,
                "gac_codes": sorted(set(gac)),
                "match": "naf:label_service",
            }
        )
        filled += 1
        if verbose and filled % 25 == 0:
            print(f"    NAF: resolved {filled} places", file=sys.stderr)
    return filled


def enrich_lcc(
    entries: Sequence[dict[str, Any]],
    client: LocClient,
    narrower_limit: int,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Enrich LCC subclass codes through the id.loc.gov classification API."""
    collections: dict[str, tuple[str | None, list[str]]] = {}
    range_cache: dict[str, ClassRange | None] = {}

    def get_collection(code: str) -> tuple[str | None, list[str]]:
        if code not in collections:
            nodes = client.fetch_json(f"{CLASSIFICATION_BASE}{code}.json")
            collections[code] = (
                parse_class_collection(nodes, code) if nodes else (None, [])
            )
        return collections[code]

    def get_range(code: str) -> ClassRange | None:
        if code not in range_cache:
            nodes = client.fetch_json(f"{CLASSIFICATION_BASE}{code}.json")
            range_cache[code] = parse_class_range(nodes, code) if nodes else None
        return range_cache[code]

    enriched: list[dict[str, Any]] = []
    for position, entry in enumerate(entries, start=1):
        code = str(entry.get("id", "")).strip()
        record: dict[str, Any] = {
            "id": code,
            "label": entry.get("label", code),
            "frequency": entry.get("frequency", 0),
            "rank": entry.get("rank"),
            "uri": None,
            "caption": None,
            "full_caption": None,
            "class_letter": code[:1] if code else None,
            "class_caption": None,
            "range": None,
            "scope_notes": [],
            "covers": [],
            "description": None,
            "description_source": "none",
            "description_verbatim": False,
            "match": "none",
        }

        if not re.fullmatch(r"[A-Z]{1,3}", code):
            record["match"] = "invalid_code"
            enriched.append(record)
            continue

        letter = code[0]
        class_caption, members = get_collection(letter)
        record["class_caption"] = class_caption

        if len(code) == 1:
            # A main class: its collection members are already the subclass
            # ranges, so they are the "covers" list directly.
            candidates = members
            primary_range = None
            siblings = members
        else:
            own_caption, own_members = get_collection(code)
            if own_members:
                # KF and KZ publish their own collection.
                primary_range, siblings = pick_primary_range(own_members, code)
                candidates = own_members
                if own_caption and not record["class_caption"]:
                    record["class_caption"] = own_caption
            else:
                primary_range, siblings = pick_primary_range(members, code)
                candidates = members

        covers_codes: list[str] = []
        if primary_range is not None:
            record["range"] = primary_range
            record["uri"] = AUTHORITY_URI_BASE["lcc"] + primary_range
            primary = get_range(primary_range)
            if primary is not None:
                record["caption"] = primary.caption
                record["full_caption"] = primary.full_caption
                record["scope_notes"] = primary.scope_notes
                record["match"] = "classification_api"
                ranked = rank_ranges_by_span(primary.narrower)
                covers_codes = [*siblings, *ranked]
        elif len(code) == 1:
            # A main class has no single "general" range; its member ranges are
            # the subclasses, so they serve as the covers list directly. A
            # multi-letter code with no matching range is simply not an LCC
            # subclass ("NOT", "PAR" and the obsolete "JX" all reach here) and
            # must stay unmatched rather than inherit its parent class.
            covers_codes = [c for c in candidates if parse_range_code(c) is not None]
            if covers_codes:
                record["match"] = "classification_api"
                record["uri"] = AUTHORITY_URI_BASE["lcc"] + code
                if class_caption:
                    record["caption"] = class_caption
                    record["full_caption"] = class_caption

        covers: list[str] = []
        for cover_code in covers_codes[:narrower_limit]:
            cover = get_range(cover_code)
            if cover is not None and cover.caption not in covers:
                covers.append(cover.caption)
        record["covers"] = covers

        notes = [n for n in record["scope_notes"] if classify_note(n) == "definitional"]
        caption = record["caption"]
        if notes and caption:
            record["description"] = (
                f"{lcc_prompt_description(caption, [])}; {clean_scope_note(notes[0])}"
            )
            record["description_source"] = "scope_note"
            record["description_verbatim"] = True
        elif caption:
            record["description"] = lcc_prompt_description(caption, covers)
            record["description_source"] = "caption_hierarchy"
        if verbose and position % 10 == 0:
            print(
                f"    LCC: {position}/{len(entries)} codes "
                f"({client.requests_made} requests, {client.cache_hits} cached)",
                file=sys.stderr,
            )
        enriched.append(record)
    return enriched


def rank_ranges_by_span(codes: Sequence[str]) -> list[str]:
    """Order range codes widest-first; the widest spans are the substantive ones."""
    scored: list[tuple[float, float, str]] = []
    for code in codes:
        parsed = parse_range_code(code)
        if parsed is None:
            continue
        _, low, high = parsed
        scored.append((-(high - low), low, code))
    scored.sort()
    return [code for _, _, code in scored]


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def coverage_for(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    matched = sum(1 for r in records if r["match"] != "none")
    by_source: dict[str, int] = {}
    for record in records:
        by_source[record["description_source"]] = (
            by_source.get(record["description_source"], 0) + 1
        )
    by_match: dict[str, int] = {}
    for record in records:
        by_match[record["match"]] = by_match.get(record["match"], 0) + 1
    return {
        "total": total,
        "matched_to_authority": matched,
        "unmatched": total - matched,
        "with_description": sum(1 for r in records if r["description"]),
        "verbatim_scope_note": sum(1 for r in records if r["description_verbatim"]),
        "description_by_source": dict(sorted(by_source.items())),
        "match_by_kind": dict(sorted(by_match.items())),
        "description_contains_own_label": sum(
            1 for r in records if r.get("description_contains_label")
        ),
        "unmatched_examples": [r["label"] for r in records if r["match"] == "none"][
            :20
        ],
        "no_description_examples": [
            r["label"] for r in records if not r["description"]
        ][:20],
    }


def render_report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Taxonomy enrichment coverage",
        "",
        f"Generated: {report['generated']}",
        f"Script version: {report['script_version']}",
        "",
        "Source strategy: bulk SKOS/RDF dumps for LCSH and LCGFT "
        "(one request each, streamed); the id.loc.gov classification API for "
        "LCC (no bulk download exists). The LC Name Authority File bulk dump "
        "(1.69 GB) is not used -- NAF place records carry no definitions.",
        "",
        "| Taxonomy | Labels | Matched | Any description | Verbatim LC scope note | No description |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, coverage in report["taxonomies"].items():
        lines.append(
            f"| {key} | {coverage['total']} | {coverage['matched_to_authority']} | "
            f"{coverage['with_description']} | {coverage['verbatim_scope_note']} | "
            f"{coverage['total'] - coverage['with_description']} |"
        )
    lines.extend(["", "## Fallback chain", ""])
    lines.extend(
        [
            "1. `scope_note` -- LC's own prose, boilerplate openings and",
            "   cross-reference tails stripped. Verbatim.",
            "2. `hierarchy` (LCSH/LCGFT) / `caption_hierarchy` (LCC) -- a",
            "   deterministic template over LC broader/narrower labels or over",
            "   the classification caption plus its widest subdivisions.",
            "3. `variant_labels` -- LC variant labels, excluding those that are",
            "   only re-orderings of the label itself.",
            "4. `none` -- no LC record, or a record with no usable content. The",
            "   consumer must keep its existing fallback for these.",
            "",
            "## Per-taxonomy detail",
            "",
        ]
    )
    for key, coverage in report["taxonomies"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"- description sources: `{coverage['description_by_source']}`")
        lines.append(f"- match kinds: `{coverage['match_by_kind']}`")
        lines.append(
            f"- descriptions containing their own label (self-labeling risk): "
            f"{coverage['description_contains_own_label']}"
        )
        if coverage["unmatched_examples"]:
            lines.append(
                f"- unmatched examples: {', '.join(coverage['unmatched_examples'][:10])}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Repository data directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write enriched files (default: <data-dir>/taxonomies/enriched)",
    )
    parser.add_argument(
        "--taxonomy",
        action="append",
        choices=[*ALL_KEYS, "all"],
        help="Taxonomy to enrich; repeatable (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N labels per taxonomy (for smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and cache state; make no network requests",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.34,
        help="Seconds to sleep between uncached HTTP requests (default: 0.34)",
    )
    parser.add_argument(
        "--lcc-narrower",
        type=int,
        default=12,
        help="Subdivision captions to fetch per LCC subclass (default: 12)",
    )
    parser.add_argument(
        "--include-naf",
        action="store_true",
        help="Resolve geographic labels missing from LCSH against the LC Name "
        "Authority File (URIs and variant names only; NAF has no definitions)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the on-disk cache and re-fetch everything",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return parser


def selected_keys(values: list[str] | None) -> list[str]:
    if not values or "all" in values:
        return list(ALL_KEYS)
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def print_plan(
    keys: Sequence[str],
    taxonomy_dir: Path,
    cache_dir: Path,
    limit: int | None,
    narrower_limit: int,
    include_naf: bool,
) -> None:
    print("Enrichment plan (dry run -- no network requests will be made)")
    print(f"  taxonomy source dir : {taxonomy_dir}")
    print(f"  cache dir           : {cache_dir}")
    print(f"  cached API responses: {len(list((cache_dir / 'api').glob('*.json')))}")
    for key in keys:
        spec = LCC_SPEC if key == LCC_SPEC.key else TAXONOMIES[key]
        if key in CURATED_TAXONOMIES:
            labels = load_curated_labels(key)
        else:
            path = taxonomy_dir / spec.source_file
            if not path.exists():
                print(f"  {key:<18} MISSING {path}")
                continue
            _, labels = load_frequency_table(path)
        count = len(labels) if limit is None else min(limit, len(labels))
        if key == LCC_SPEC.key:
            estimate = f"~{21 + count + count * narrower_limit} API requests"
        else:
            dumps = ", ".join(
                f"{d} ({human_bytes(bulk_size(cache_dir, d))})" for d in spec.dumps
            )
            estimate = f"bulk dumps: {dumps}"
            if spec.kind == "geographic" and include_naf:
                estimate += f"; plus up to {count * 2} NAF requests"
        print(f"  {key:<18} {count:>5} labels  <- {spec.source_file}  [{estimate}]")


def bulk_size(cache_dir: Path, scheme: str) -> int | None:
    path = cache_dir / "bulk" / Path(BULK_DUMPS[scheme]).name
    return path.stat().st_size if path.exists() else None


def human_bytes(size: int | None) -> str:
    if size is None:
        return "not cached"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size = int(size / 1024)
    return f"{size} B"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    data_dir: Path = args.data_dir
    taxonomy_dir = data_dir / "taxonomies"
    output_dir: Path = args.output_dir or (taxonomy_dir / "enriched")
    cache_dir = output_dir / ".cache"
    keys = selected_keys(args.taxonomy)

    if not taxonomy_dir.exists():
        print(f"error: {taxonomy_dir} does not exist", file=sys.stderr)
        return 2

    if args.dry_run:
        print_plan(
            keys,
            taxonomy_dir,
            cache_dir,
            args.limit,
            args.lcc_narrower,
            args.include_naf,
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    gitignore = output_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# Downloaded LC bulk dumps and cached API responses.\n.cache/\n",
            encoding="utf-8",
        )

    dump_keys = keys_needing_dumps(keys)
    started = datetime.now(UTC)
    report: dict[str, Any] = {
        "generated": started.isoformat(),
        "script_version": SCRIPT_VERSION,
        "sources": {
            "bulk_dumps": {k: BULK_DUMPS[k] for k in dump_keys},
            "classification_api": CLASSIFICATION_BASE,
            "naf_label_service": NAF_LABEL_BASE if args.include_naf else None,
        },
        "taxonomies": {},
    }

    with LocClient(
        cache_dir=cache_dir, delay=args.delay, refresh=args.refresh
    ) as client:
        indexes: dict[str, DumpIndex] = {}
        reference_labels: dict[str, dict[str, str]] = {}
        tables: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}

        for key in keys:
            spec = LCC_SPEC if key == LCC_SPEC.key else TAXONOMIES[key]
            meta: dict[str, Any]
            if key in CURATED_TAXONOMIES:
                meta = {
                    "type": key,
                    "name": f"Curated v0.3.1 {spec.kind} pool",
                    "description": f"Frozen v0.3.1 pool from {spec.source_file}",
                }
                labels = load_curated_labels(key)
            else:
                path = taxonomy_dir / spec.source_file
                if not path.exists():
                    print(f"warning: skipping {key}: {path} not found", file=sys.stderr)
                    continue
                meta, labels = load_frequency_table(path)
            if args.limit is not None:
                labels = labels[: args.limit]
            tables[key] = (meta, labels)

        if dump_keys:
            wanted: set[str] = set()
            for key, (_, labels) in tables.items():
                if key == LCC_SPEC.key:
                    continue
                for entry in labels:
                    label = str(entry.get("label", entry.get("id", "")))
                    wanted.update(label_variants(label))

            for scheme in dump_keys:
                url = BULK_DUMPS[scheme]
                destination = cache_dir / "bulk" / Path(url).name
                if verbose:
                    print(f"[{scheme}] downloading {url}", file=sys.stderr)
                client.download(url, destination)
                if verbose:
                    print(
                        f"[{scheme}] indexing {human_bytes(destination.stat().st_size)} "
                        "(streaming, pass 1/2)",
                        file=sys.stderr,
                    )
                index = build_dump_index(destination, scheme, wanted)
                indexes[scheme] = index
                needed_ids = {
                    ident
                    for concept in (*index.by_pref.values(), *index.by_alt.values())
                    for ident in (*concept.broader, *concept.narrower)
                }
                if verbose:
                    print(
                        f"[{scheme}] {index.concepts_seen} concepts scanned, "
                        f"{len(index.by_pref)} pref-label hits, "
                        f"resolving {len(needed_ids)} related terms (pass 2/2)",
                        file=sys.stderr,
                    )
                reference_labels[scheme] = resolve_reference_labels(
                    destination, scheme, needed_ids
                )

        for key in keys:
            if key not in tables:
                continue
            meta, labels = tables[key]
            if key == LCC_SPEC.key:
                spec = LCC_SPEC
                if verbose:
                    print(
                        f"[{key}] enriching {len(labels)} codes via the classification API",
                        file=sys.stderr,
                    )
                records = enrich_lcc(labels, client, args.lcc_narrower, verbose)
            else:
                spec = TAXONOMIES[key]
                records = enrich_from_dumps(spec, labels, indexes, reference_labels)
                if spec.kind == "geographic" and args.include_naf:
                    if verbose:
                        print(
                            f"[{key}] resolving LCSH misses against NAF",
                            file=sys.stderr,
                        )
                    enrich_geographic_with_naf(records, client, verbose)

            coverage = coverage_for(records)
            report["taxonomies"][key] = coverage
            payload = {
                "type": meta.get("type", key),
                "name": meta.get("name", key),
                "description": meta.get("description"),
                "source_file": spec.source_file,
                "enrichment": {
                    "script": Path(__file__).name,
                    "script_version": SCRIPT_VERSION,
                    "generated": started.isoformat(),
                    "sources": report["sources"],
                },
                "coverage": coverage,
                "labels": records,
            }
            write_json(output_dir / spec.output_file, payload)
            if verbose:
                print(
                    f"[{key}] {coverage['with_description']}/{coverage['total']} "
                    f"described ({coverage['verbatim_scope_note']} verbatim LC scope notes)"
                    f" -> {output_dir / spec.output_file}",
                    file=sys.stderr,
                )

        report["http"] = {
            "requests_made": client.requests_made,
            "cache_hits": client.cache_hits,
        }

    write_json(output_dir / "coverage_report.json", report)
    (output_dir / "coverage_report.md").write_text(
        render_report_markdown(report), encoding="utf-8"
    )
    if verbose:
        print(render_report_markdown(report))
    return 0


def keys_needing_dumps(keys: Sequence[str]) -> list[str]:
    needed: list[str] = []
    for key in keys:
        if key == LCC_SPEC.key:
            continue
        for scheme in TAXONOMIES[key].dumps:
            if scheme not in needed:
                needed.append(scheme)
    return needed


if __name__ == "__main__":
    raise SystemExit(main())

"""Load real LC taxonomy data with URIs from id.loc.gov.

Provides actual Library of Congress authority data:
- LCGFT (Genre/Form Terms) with URIs
- LCSH (Subject Headings) with URIs
- LCDGT (Demographic Group Terms) with URIs
- LCC (Classification) codes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path

# Base URIs for LC authorities
LC_BASE_URIS = {
    "lcgft": "http://id.loc.gov/authorities/genreForms/",
    "lcsh": "http://id.loc.gov/authorities/subjects/",
    "lcdgt": "http://id.loc.gov/authorities/demographicTerms/",
    "lcc": "http://id.loc.gov/authorities/classification/",
}


@dataclass
class LCTerm:
    """A Library of Congress authority term."""

    id: str
    label: str
    uri: str
    alt_labels: list[str]
    broader: list[str]
    narrower: list[str]
    scope_note: str | None = None

    @classmethod
    def from_parsed(cls, data: dict, term_type: str) -> LCTerm:
        """Create from parsed LC data."""
        term_id = data["id"]
        base_uri = LC_BASE_URIS.get(term_type, "")
        uri = f"{base_uri}{term_id}" if base_uri else term_id

        return cls(
            id=term_id,
            label=data["pref_label"],
            uri=uri,
            alt_labels=data.get("alt_labels", []),
            broader=data.get("broader", []),
            narrower=data.get("narrower", []),
            scope_note=data.get("scope_note"),
        )


@dataclass
class LCCClass:
    """Library of Congress Classification class."""

    code: str
    name: str
    uri: str

    @property
    def id(self) -> str:
        return self.code


# LCC main classes (these are fixed)
LCC_CLASSES = {
    "A": "General Works",
    "B": "Philosophy, Psychology, Religion",
    "C": "Auxiliary Sciences of History",
    "D": "World History (except Americas)",
    "E": "History of the Americas (general, US)",
    "F": "History of the Americas (local)",
    "G": "Geography, Anthropology, Recreation",
    "H": "Social Sciences",
    "J": "Political Science",
    "K": "Law",
    "L": "Education",
    "M": "Music",
    "N": "Fine Arts",
    "P": "Language and Literature",
    "Q": "Science",
    "R": "Medicine",
    "S": "Agriculture",
    "T": "Technology",
    "U": "Military Science",
    "V": "Naval Science",
    "Z": "Bibliography, Library Science",
}


def get_lcc_terms() -> list[LCCClass]:
    """Get all LCC main classes with URIs."""
    return [
        LCCClass(
            code=code,
            name=name,
            uri=f"{LC_BASE_URIS['lcc']}{code}",
        )
        for code, name in LCC_CLASSES.items()
    ]


@lru_cache(maxsize=1)
def _load_lcgft_data(data_dir: str) -> dict[str, LCTerm]:
    """Load LCGFT data from parsed JSON."""
    path = Path(data_dir) / "loc_parsed" / "lcgft.json"
    if not path.exists():
        return {}

    with open(path) as f:
        raw = json.load(f)

    return {uri: LCTerm.from_parsed(data, "lcgft") for uri, data in raw.items()}


@lru_cache(maxsize=1)
def _load_lcsh_data(data_dir: str) -> dict[str, LCTerm]:
    """Load LCSH data from parsed JSON."""
    path = Path(data_dir) / "loc_parsed" / "lcsh.json"
    if not path.exists():
        return {}

    with open(path) as f:
        raw = json.load(f)

    return {uri: LCTerm.from_parsed(data, "lcsh") for uri, data in raw.items()}


@lru_cache(maxsize=1)
def _load_lcdgt_data(data_dir: str) -> dict[str, LCTerm]:
    """Load LCDGT data from parsed JSON."""
    path = Path(data_dir) / "loc_parsed" / "lcdgt.json"
    if not path.exists():
        return {}

    with open(path) as f:
        raw = json.load(f)

    return {uri: LCTerm.from_parsed(data, "lcdgt") for uri, data in raw.items()}


def _load_frequency_data(data_dir: str, filename: str) -> list[tuple[str, int]]:
    """Load frequency data from MARC analysis."""
    path = Path(data_dir) / "frequencies" / filename
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    # Handle list of {term, count} dicts
    if isinstance(data, list):
        return [(item["term"], item["count"]) for item in data]
    # Handle dict of term -> count
    return list(data.items())


class LCDataLoader:
    """Load and query LC taxonomy data with URIs."""

    def __init__(self, data_dir: Path | str | None = None):
        if data_dir is None:
            # Default to package data directory
            data_dir = Path(__file__).parent.parent.parent.parent / "data"
        self.data_dir = Path(data_dir)

    @property
    def lcgft(self) -> dict[str, LCTerm]:
        """Get all LCGFT terms keyed by URI."""
        return _load_lcgft_data(str(self.data_dir))

    @property
    def lcsh(self) -> dict[str, LCTerm]:
        """Get all LCSH terms keyed by URI."""
        return _load_lcsh_data(str(self.data_dir))

    @property
    def lcdgt(self) -> dict[str, LCTerm]:
        """Get all LCDGT terms keyed by URI."""
        return _load_lcdgt_data(str(self.data_dir))

    @property
    def lcc(self) -> list[LCCClass]:
        """Get all LCC main classes."""
        return get_lcc_terms()

    def get_lcgft_by_label(self, label: str) -> LCTerm | None:
        """Find LCGFT term by preferred or alternate label."""
        label_lower = label.lower()
        for term in self.lcgft.values():
            if term.label.lower() == label_lower:
                return term
            if any(alt.lower() == label_lower for alt in term.alt_labels):
                return term
        return None

    def get_lcsh_by_label(self, label: str) -> LCTerm | None:
        """Find LCSH term by preferred or alternate label."""
        label_lower = label.lower()
        for term in self.lcsh.values():
            if term.label.lower() == label_lower:
                return term
            if any(alt.lower() == label_lower for alt in term.alt_labels):
                return term
        return None

    def get_lcdgt_by_label(self, label: str) -> LCTerm | None:
        """Find LCDGT term by preferred or alternate label."""
        label_lower = label.lower()
        for term in self.lcdgt.values():
            if term.label.lower() == label_lower:
                return term
            if any(alt.lower() == label_lower for alt in term.alt_labels):
                return term
        return None

    def get_top_lcgft(self, n: int = 50) -> list[tuple[LCTerm, int]]:
        """Get top N LCGFT terms by frequency in CGP MARC records."""
        freqs = _load_frequency_data(
            str(self.data_dir), "marc_lcgft_655_frequencies.json"
        )
        if not freqs:
            return []

        results = []
        for label, count in freqs[:n]:
            term = self.get_lcgft_by_label(label)
            if term:
                results.append((term, count))
        return results

    def get_top_lcsh(self, n: int = 100) -> list[tuple[LCTerm, int]]:
        """Get top N LCSH topical terms by frequency."""
        freqs = _load_frequency_data(
            str(self.data_dir), "marc_lcsh_650_frequencies.json"
        )
        if not freqs:
            return []

        results = []
        for label, count in freqs[:n]:
            term = self.get_lcsh_by_label(label)
            if term:
                results.append((term, count))
        return results

    def stats(self) -> dict:
        """Get stats about loaded data."""
        return {
            "lcgft_count": len(self.lcgft),
            "lcsh_count": len(self.lcsh),
            "lcdgt_count": len(self.lcdgt),
            "lcc_count": len(self.lcc),
        }


# Convenience function
def load_lc_data(data_dir: Path | str | None = None) -> LCDataLoader:
    """Load LC taxonomy data."""
    return LCDataLoader(data_dir)


# =============================================================================
# Expanded taxonomy pools (data/taxonomies/*.json)
#
# These are frequency-ranked label lists extracted from MARC bibliographic
# records (see data/taxonomies/extraction_summary.json) -- a different, and
# currently much better populated, source than the `_load_lcgft_data` /
# `_load_lcsh_data` / `_load_lcdgt_data` loaders above, which read from
# `data_dir/loc_parsed/*.json` (a directory that does not exist in this
# checkout, so those loaders always return an empty dict).
#
# dimensions.py uses these to build expandable sampling pools for topics,
# geographic areas, and genre/form terms (see TopicSampler, GeographicSampler,
# LCGFTSampler `pool_size` parameter).
# =============================================================================


@dataclass
class TaxonomyLabel:
    """One ranked label entry from a data/taxonomies/*.json frequency file.

    Note: as extracted, `broader`, `narrower`, `alt_labels`, `description`,
    and `uri` are empty/None for every entry checked across the topical,
    geographic, and lcgft frequency files (2,554 entries sampled) -- only
    `id`, `label`, `frequency`, and `rank` carry real content. Callers
    should not assume the richer fields are populated; they are kept here
    for forward compatibility if a future extraction pass fills them in.
    """

    id: str
    label: str
    frequency: int
    rank: int
    broader: list[str] = field(default_factory=list)
    narrower: list[str] = field(default_factory=list)
    alt_labels: list[str] = field(default_factory=list)
    description: str | None = None
    uri: str | None = None


def _default_data_dir() -> Path:
    """Default package data directory (shelf-benchmark/data)."""
    return Path(__file__).parent.parent.parent.parent / "data"


@cache
def _load_taxonomy_labels(data_dir: str, filename: str) -> tuple[TaxonomyLabel, ...]:
    """Load a ranked label list from `<data_dir>/taxonomies/<filename>`.

    Returns an empty tuple if the file is missing, so callers can raise a
    clear error rather than failing deep inside a cache lookup.
    """
    path = Path(data_dir) / "taxonomies" / filename
    if not path.exists():
        return ()

    with open(path) as f:
        raw = json.load(f)

    return tuple(
        TaxonomyLabel(
            id=item["id"],
            label=item["label"],
            frequency=item.get("frequency", 0),
            rank=item.get("rank", i + 1),
            broader=item.get("broader") or [],
            narrower=item.get("narrower") or [],
            alt_labels=item.get("alt_labels") or [],
            description=item.get("description"),
            uri=item.get("uri"),
        )
        for i, item in enumerate(raw.get("labels", []))
    )


# Source files and the maximum pool size each supports (see
# data/taxonomies/extraction_summary.json). Each file is pre-sorted by
# ascending rank, and a smaller cut is an exact rank-ordered prefix of every
# larger cut of the same taxonomy (verified: top500 topics == top1000
# topics[:500] == top2000 topics[:500]), so loading the largest available
# file and slicing is equivalent to loading the differently-sized file
# directly.
TOPIC_POOL_SOURCE = "lcsh_topical_top2000.json"
TOPIC_POOL_MAX = 2000
GEOGRAPHIC_POOL_SOURCE = "lcsh_geo_top500.json"
GEOGRAPHIC_POOL_MAX = 500
FORM_POOL_SOURCE = "lcgft.json"
FORM_POOL_MAX = 554


def _load_ranked_pool(
    filename: str,
    size: int,
    pool_max: int,
    data_dir: Path | str | None,
) -> list[TaxonomyLabel]:
    """Load the top `size` entries (by frequency rank) from `filename`."""
    if size < 1:
        raise ValueError(f"pool size must be >= 1, got {size}")
    if size > pool_max:
        raise ValueError(
            f"pool size {size} exceeds the {pool_max} labels available in {filename}"
        )

    resolved_dir = str(data_dir) if data_dir is not None else str(_default_data_dir())
    labels = _load_taxonomy_labels(resolved_dir, filename)
    if not labels:
        raise FileNotFoundError(
            f"Taxonomy source file not found or empty: "
            f"{Path(resolved_dir) / 'taxonomies' / filename}"
        )
    return list(labels[:size])


def load_topic_pool(
    size: int, data_dir: Path | str | None = None
) -> list[TaxonomyLabel]:
    """Load the top `size` LCSH topical subject headings by MARC corpus
    frequency (1 <= size <= TOPIC_POOL_MAX)."""
    return _load_ranked_pool(TOPIC_POOL_SOURCE, size, TOPIC_POOL_MAX, data_dir)


def load_geographic_pool(
    size: int, data_dir: Path | str | None = None
) -> list[TaxonomyLabel]:
    """Load the top `size` LCSH geographic headings by MARC corpus frequency
    (1 <= size <= GEOGRAPHIC_POOL_MAX)."""
    return _load_ranked_pool(
        GEOGRAPHIC_POOL_SOURCE, size, GEOGRAPHIC_POOL_MAX, data_dir
    )


def load_form_pool(
    size: int, data_dir: Path | str | None = None
) -> list[TaxonomyLabel]:
    """Load the top `size` LCGFT genre/form terms by MARC corpus frequency
    (1 <= size <= FORM_POOL_MAX)."""
    return _load_ranked_pool(FORM_POOL_SOURCE, size, FORM_POOL_MAX, data_dir)


@lru_cache(maxsize=1)
def _lcgft_category_lookup(data_dir: str) -> dict[str, tuple[str, ...]]:
    """Map a lowercased LCGFT form label to the curated (v0.3.1) category
    name(s) it belongs to under `data/taxonomies/lcgft_hierarchy.json`.

    Only the 14 curated categories from `dimensions.LCGFT_DATA` are
    considered -- the hierarchy file actually lists 23 categories; the other
    9 (e.g. "Dance", "Manuscripts", "Incunabula") are not part of the v0.3.1
    schema, so a term that matches only one of those is treated as
    unmatched here and left for the caller to bucket as "Uncategorized".
    """
    from .dimensions import LCGFT_DATA

    curated_categories = set(LCGFT_DATA.keys())
    path = Path(data_dir) / "taxonomies" / "lcgft_hierarchy.json"
    if not path.exists():
        return {}

    with open(path) as f:
        raw = json.load(f)

    lookup: dict[str, set[str]] = {}
    for entry in raw.get("categories", []):
        category = entry.get("category", "")
        if category not in curated_categories:
            continue
        for child in entry.get("children", []):
            lookup.setdefault(child.lower(), set()).add(category)

    return {term: tuple(sorted(cats)) for term, cats in lookup.items()}


def build_expanded_form_pool(
    size: int, data_dir: Path | str | None = None
) -> dict[str, list[str]]:
    """Build an expanded LCGFT category -> forms pool of `size` distinct
    forms, starting from the 133 curated v0.3.1 forms and adding the next
    most frequent forms from `lcgft.json` until `size` distinct forms are
    reached.

    The curated forms keep their hand-verified categories. Added forms are
    assigned a category via `_lcgft_category_lookup` when unambiguous
    (falling back to the alphabetically-first candidate category when a
    term matches more than one), or bucketed under "Uncategorized" when the
    hierarchy file has no entry for them. In practice, category coverage
    for added forms is poor: at size=300, only ~19% of the 167 added forms
    get a real category, because `lcgft_hierarchy.json` maps only 132 of
    the 554 ranked forms to any curated category at all.

    Raises ValueError if `size` is smaller than the 133 curated forms
    (use `pool_size="v0.3.1"` with a `categories` filter instead to work
    with a subset of the curated pool) or larger than FORM_POOL_MAX.
    """
    from .dimensions import LCGFT_DATA

    if size > FORM_POOL_MAX:
        raise ValueError(
            f"pool size {size} exceeds the {FORM_POOL_MAX} labels available"
        )

    data: dict[str, list[str]] = {
        category: list(forms) for category, forms in LCGFT_DATA.items()
    }
    existing_lower = {form.lower() for forms in data.values() for form in forms}

    if size < len(existing_lower):
        raise ValueError(
            f"pool size {size} is smaller than the {len(existing_lower)} curated "
            "v0.3.1 forms; use pool_size='v0.3.1' (optionally with a `categories` "
            "filter) for a pool at or below the curated size"
        )

    needed = size - len(existing_lower)
    if needed == 0:
        return data

    resolved_dir = str(data_dir) if data_dir is not None else str(_default_data_dir())
    category_lookup = _lcgft_category_lookup(resolved_dir)

    added = 0
    for entry in _load_ranked_pool(
        FORM_POOL_SOURCE, FORM_POOL_MAX, FORM_POOL_MAX, data_dir
    ):
        if added >= needed:
            break
        label_lower = entry.label.lower()
        if label_lower in existing_lower:
            continue
        candidates = category_lookup.get(label_lower)
        category = candidates[0] if candidates else "Uncategorized"
        data.setdefault(category, []).append(entry.label)
        existing_lower.add(label_lower)
        added += 1

    return data

"""Load real LC taxonomy data with URIs from id.loc.gov.

Provides actual Library of Congress authority data:
- LCGFT (Genre/Form Terms) with URIs
- LCSH (Subject Headings) with URIs
- LCDGT (Demographic Group Terms) with URIs
- LCC (Classification) codes
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache


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
    def from_parsed(cls, data: dict, term_type: str) -> "LCTerm":
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

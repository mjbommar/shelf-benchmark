"""
Library of Congress Taxonomies Downloader and Parser

Downloads and parses:
- LCSH (Subject Headings) - ~340,000+ headings
- LCGFT (Genre/Form Terms) - ~2,000+ terms
- LCDGT (Demographic Group Terms) - ~1,000+ terms
- LCC (Classification) - 21 main classes, ~200+ subclasses

Data sources:
- id.loc.gov - SKOS/RDF bulk downloads (free)
- classweb.org - MARC downloads for LCDGT (free)
"""

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, SKOS

# Namespaces
MADSRDF = Namespace("http://www.loc.gov/mads/rdf/v1#")
LC_AUTHORITIES = "http://id.loc.gov/authorities/"

# Download URLs for SKOS format (smaller, simpler structure)
DOWNLOAD_URLS = {
    "lcsh": {
        "skos_nt": "https://id.loc.gov/download/authorities/subjects.skosrdf.nt.gz",
        "skos_jsonld": "https://id.loc.gov/download/authorities/subjects.skosrdf.jsonld.gz",
        "mads_nt": "https://id.loc.gov/download/authorities/subjects.madsrdf.nt.gz",
    },
    "lcgft": {
        "skos_nt": "https://id.loc.gov/download/authorities/genreForms.skosrdf.nt.gz",
        "skos_jsonld": "https://id.loc.gov/download/authorities/genreForms.skosrdf.jsonld.gz",
        "mads_nt": "https://id.loc.gov/download/authorities/genreForms.madsrdf.nt.gz",
    },
    "lcdgt": {
        "skos_nt": "https://id.loc.gov/download/authorities/demographicTerms.skosrdf.nt.gz",
        "skos_jsonld": "https://id.loc.gov/download/authorities/demographicTerms.skosrdf.jsonld.gz",
        "marc": "https://classweb.org/LCDGT/LCDGTvocab.mrc.zip",
    },
    "classification": {
        # LCC is NOT available as bulk download - only via API or subscription
        # These are placeholder URLs for individual class lookups
        "api_base": "https://id.loc.gov/authorities/classification/",
    },
}

# LCC main classes (hardcoded since no bulk download available)
LCC_MAIN_CLASSES = {
    "A": "General Works",
    "B": "Philosophy, Psychology, Religion",
    "C": "Auxiliary Sciences of History",
    "D": "World History (except America)",
    "E": "History of the Americas (United States)",
    "F": "History of the Americas (Canada, Latin America)",
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


@dataclass
class Concept:
    """A SKOS concept from LC vocabularies."""

    uri: str
    pref_label: str
    alt_labels: list[str] = field(default_factory=list)
    broader: list[str] = field(default_factory=list)
    narrower: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    scope_note: str | None = None
    notation: str | None = None

    @property
    def id(self) -> str:
        """Extract the LC identifier from the URI."""
        return self.uri.split("/")[-1]


def download_file(url: str, cache_dir: Path | None = None) -> bytes:
    """Download a file, optionally caching it locally."""
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        cache_path = cache_dir / filename
        if cache_path.exists():
            print(f"Using cached: {cache_path}")
            return cache_path.read_bytes()

    print(f"Downloading: {url}")
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.content

    if cache_dir:
        cache_path.write_bytes(data)
        print(f"Cached to: {cache_path}")

    return data


def decompress_gzip(data: bytes) -> bytes:
    """Decompress gzip data."""
    return gzip.decompress(data)


def parse_skos_ntriples(data: bytes) -> Graph:
    """Parse N-Triples format into an RDF graph."""
    g = Graph()
    g.parse(data=data, format="nt")
    return g


def extract_concepts_from_graph(g: Graph) -> Iterator[Concept]:
    """Extract SKOS concepts from an RDF graph."""
    for subject in g.subjects(RDF.type, SKOS.Concept):
        uri = str(subject)

        # Get preferred label
        pref_label = None
        for label in g.objects(subject, SKOS.prefLabel):
            if hasattr(label, "language") and label.language == "en":
                pref_label = str(label)
                break
            elif pref_label is None:
                pref_label = str(label)

        if not pref_label:
            continue

        # Get alternate labels
        alt_labels = [str(label) for label in g.objects(subject, SKOS.altLabel)]

        # Get hierarchical relations
        broader = [str(b) for b in g.objects(subject, SKOS.broader)]
        narrower = [str(n) for n in g.objects(subject, SKOS.narrower)]
        related = [str(r) for r in g.objects(subject, SKOS.related)]

        # Get scope note
        scope_note = None
        for note in g.objects(subject, SKOS.scopeNote):
            scope_note = str(note)
            break

        # Get notation
        notation = None
        for n in g.objects(subject, SKOS.notation):
            notation = str(n)
            break

        yield Concept(
            uri=uri,
            pref_label=pref_label,
            alt_labels=alt_labels,
            broader=broader,
            narrower=narrower,
            related=related,
            scope_note=scope_note,
            notation=notation,
        )


def load_lcsh(cache_dir: Path | None = None) -> list[Concept]:
    """Load Library of Congress Subject Headings."""
    url = DOWNLOAD_URLS["lcsh"]["skos_nt"]
    data = download_file(url, cache_dir)
    data = decompress_gzip(data)
    print(f"Parsing LCSH ({len(data):,} bytes)...")
    g = parse_skos_ntriples(data)
    concepts = list(extract_concepts_from_graph(g))
    print(f"Loaded {len(concepts):,} LCSH concepts")
    return concepts


def load_lcgft(cache_dir: Path | None = None) -> list[Concept]:
    """Load Library of Congress Genre/Form Terms."""
    url = DOWNLOAD_URLS["lcgft"]["skos_nt"]
    data = download_file(url, cache_dir)
    data = decompress_gzip(data)
    print(f"Parsing LCGFT ({len(data):,} bytes)...")
    g = parse_skos_ntriples(data)
    concepts = list(extract_concepts_from_graph(g))
    print(f"Loaded {len(concepts):,} LCGFT concepts")
    return concepts


def load_lcdgt(cache_dir: Path | None = None) -> list[Concept]:
    """Load Library of Congress Demographic Group Terms."""
    url = DOWNLOAD_URLS["lcdgt"]["skos_nt"]
    data = download_file(url, cache_dir)
    data = decompress_gzip(data)
    print(f"Parsing LCDGT ({len(data):,} bytes)...")
    g = parse_skos_ntriples(data)
    concepts = list(extract_concepts_from_graph(g))
    print(f"Loaded {len(concepts):,} LCDGT concepts")
    return concepts


def get_lcc_classes() -> dict[str, str]:
    """
    Get LCC main classes.

    Note: Full LCC classification data requires a subscription to Classification Web
    or MARC Distribution Service. Only main classes are freely available.
    """
    return LCC_MAIN_CLASSES.copy()


def fetch_lcc_subclasses(main_class: str) -> dict:
    """
    Fetch subclasses for a main LCC class via the API.

    This uses content negotiation to get JSON-LD from id.loc.gov.
    Note: This is rate-limited and not suitable for bulk access.
    """
    url = f"{DOWNLOAD_URLS['classification']['api_base']}{main_class}"
    headers = {"Accept": "application/ld+json"}

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def concepts_to_dict(concepts: list[Concept]) -> dict[str, dict]:
    """Convert concepts to a dictionary keyed by URI."""
    return {
        c.uri: {
            "id": c.id,
            "pref_label": c.pref_label,
            "alt_labels": c.alt_labels,
            "broader": c.broader,
            "narrower": c.narrower,
            "related": c.related,
            "scope_note": c.scope_note,
            "notation": c.notation,
        }
        for c in concepts
    }


def save_concepts_json(concepts: list[Concept], path: Path):
    """Save concepts to a JSON file."""
    data = concepts_to_dict(concepts)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(concepts):,} concepts to {path}")


def get_top_level_concepts(concepts: list[Concept]) -> list[Concept]:
    """Get concepts that have no broader terms (top of hierarchy)."""
    return [c for c in concepts if not c.broader]


def filter_by_frequency(
    concepts: list[Concept], min_narrower: int = 0
) -> list[Concept]:
    """Filter to concepts that have at least min_narrower narrower terms."""
    return [c for c in concepts if len(c.narrower) >= min_narrower]


def print_summary(name: str, concepts: list[Concept]):
    """Print a summary of a concept list."""
    top_level = get_top_level_concepts(concepts)
    with_narrower = [c for c in concepts if c.narrower]

    print(f"\n=== {name} Summary ===")
    print(f"Total concepts: {len(concepts):,}")
    print(f"Top-level (no broader): {len(top_level):,}")
    print(f"With narrower terms: {len(with_narrower):,}")

    # Sample some top-level concepts
    print("\nSample top-level concepts:")
    for c in top_level[:10]:
        print(f"  - {c.pref_label}")


if __name__ == "__main__":
    import sys

    cache_dir = Path("./data/loc_cache")
    output_dir = Path("./data/loc_parsed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        vocab = sys.argv[1].lower()
    else:
        vocab = "all"

    if vocab in ("lcgft", "all"):
        print("\n" + "=" * 60)
        print("Loading LCGFT (Genre/Form Terms)...")
        print("=" * 60)
        lcgft = load_lcgft(cache_dir)
        print_summary("LCGFT", lcgft)
        save_concepts_json(lcgft, output_dir / "lcgft.json")

    if vocab in ("lcdgt", "all"):
        print("\n" + "=" * 60)
        print("Loading LCDGT (Demographic Group Terms)...")
        print("=" * 60)
        lcdgt = load_lcdgt(cache_dir)
        print_summary("LCDGT", lcdgt)
        save_concepts_json(lcdgt, output_dir / "lcdgt.json")

    if vocab in ("lcsh", "all"):
        print("\n" + "=" * 60)
        print("Loading LCSH (Subject Headings) - this may take a while...")
        print("=" * 60)
        lcsh = load_lcsh(cache_dir)
        print_summary("LCSH", lcsh)
        save_concepts_json(lcsh, output_dir / "lcsh.json")

    if vocab in ("lcc", "all"):
        print("\n" + "=" * 60)
        print("LCC (Classification) - Main Classes Only")
        print("=" * 60)
        print("\nNote: Full LCC requires Classification Web subscription.")
        print("Main classes available:")
        for code, name in LCC_MAIN_CLASSES.items():
            print(f"  {code}: {name}")

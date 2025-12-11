"""
Functions to load taxonomies from various sources.
"""

import gzip
from pathlib import Path
from typing import Iterator

import httpx
import orjson
from rdflib import Graph
from rdflib.namespace import RDF, SKOS

from .models import Label, Taxonomy, TaxonomyType


# ============================================================================
# LCC Main Classes (hardcoded - no bulk download available)
# ============================================================================

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


# ============================================================================
# id.loc.gov Download URLs
# ============================================================================

LOC_DOWNLOAD_URLS = {
    "lcsh": "https://id.loc.gov/download/authorities/subjects.skosrdf.nt.gz",
    "lcgft": "https://id.loc.gov/download/authorities/genreForms.skosrdf.nt.gz",
    "lcdgt": "https://id.loc.gov/download/authorities/demographicTerms.skosrdf.nt.gz",
}


# ============================================================================
# Download and Parse Functions
# ============================================================================

def download_file(url: str, cache_dir: Path | None = None) -> bytes:
    """Download a file, optionally caching it locally."""
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        cache_path = cache_dir / filename
        if cache_path.exists():
            return cache_path.read_bytes()

    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.content

    if cache_dir:
        cache_path.write_bytes(data)

    return data


def parse_skos_graph(data: bytes) -> Graph:
    """Parse N-Triples data into an RDF graph."""
    g = Graph()
    g.parse(data=data, format="nt")
    return g


def extract_concepts_from_graph(g: Graph) -> Iterator[Label]:
    """Extract SKOS concepts from an RDF graph as Label objects."""
    for subject in g.subjects(RDF.type, SKOS.Concept):
        uri = str(subject)
        label_id = uri.split("/")[-1]

        # Get preferred label (prefer English)
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
        broader = [str(b).split("/")[-1] for b in g.objects(subject, SKOS.broader)]
        narrower = [str(n).split("/")[-1] for n in g.objects(subject, SKOS.narrower)]

        # Get scope note
        scope_note = None
        for note in g.objects(subject, SKOS.scopeNote):
            scope_note = str(note)
            break

        yield Label(
            id=label_id,
            label=pref_label,
            uri=uri,
            alt_labels=alt_labels,
            broader=broader,
            narrower=narrower,
            description=scope_note,
        )


# ============================================================================
# Load from id.loc.gov
# ============================================================================

def load_taxonomy_from_loc(
    taxonomy_type: TaxonomyType,
    cache_dir: Path | None = None,
) -> Taxonomy:
    """
    Load a taxonomy directly from id.loc.gov SKOS downloads.

    Note: These don't include frequency data - use load_taxonomy_from_frequencies
    for frequency-ranked labels.
    """
    if taxonomy_type == TaxonomyType.LCC_MAIN:
        labels = [
            Label(id=code, label=name, rank=i + 1)
            for i, (code, name) in enumerate(LCC_MAIN_CLASSES.items())
        ]
        return Taxonomy(
            type=taxonomy_type,
            name="Library of Congress Classification (Main Classes)",
            description="21 main classes of the Library of Congress Classification system",
            source="hardcoded",
            labels=labels,
        )

    url_key = {
        TaxonomyType.LCSH_TOPICAL: "lcsh",
        TaxonomyType.LCSH_GEOGRAPHIC: "lcsh",
        TaxonomyType.LCSH_FULL: "lcsh",
        TaxonomyType.LCGFT: "lcgft",
        TaxonomyType.LCDGT: "lcdgt",
    }.get(taxonomy_type)

    if not url_key:
        raise ValueError(f"Cannot load {taxonomy_type} from id.loc.gov")

    url = LOC_DOWNLOAD_URLS[url_key]
    data = download_file(url, cache_dir)
    data = gzip.decompress(data)

    g = parse_skos_graph(data)
    labels = list(extract_concepts_from_graph(g))

    name_map = {
        TaxonomyType.LCSH_TOPICAL: "Library of Congress Subject Headings",
        TaxonomyType.LCSH_GEOGRAPHIC: "Library of Congress Subject Headings (Geographic)",
        TaxonomyType.LCSH_FULL: "Library of Congress Subject Headings (Full)",
        TaxonomyType.LCGFT: "Library of Congress Genre/Form Terms",
        TaxonomyType.LCDGT: "Library of Congress Demographic Group Terms",
    }

    return Taxonomy(
        type=taxonomy_type,
        name=name_map.get(taxonomy_type, str(taxonomy_type)),
        description=f"Loaded from {url}",
        source="id.loc.gov",
        labels=labels,
    )


# ============================================================================
# Load from pre-computed frequency files
# ============================================================================

def load_taxonomy_from_frequencies(
    taxonomy_type: TaxonomyType,
    frequencies_dir: Path,
    loc_labels_dir: Path | None = None,
) -> Taxonomy:
    """
    Load a taxonomy from pre-computed MARC frequency files.

    Args:
        taxonomy_type: Type of taxonomy to load
        frequencies_dir: Directory containing *_frequencies.json files
        loc_labels_dir: Optional directory with id.loc.gov label data for enrichment
    """
    # Map taxonomy types to frequency files
    file_map = {
        TaxonomyType.LCC_MAIN: "marc_lcc_main_class_frequencies.json",
        TaxonomyType.LCC_SUBCLASS: "marc_lcc_subclass_frequencies.json",
        TaxonomyType.LCSH_TOPICAL: "marc_lcsh_650_frequencies.json",
        TaxonomyType.LCSH_GEOGRAPHIC: "marc_lcsh_651_geo_frequencies.json",
        TaxonomyType.LCSH_FULL: "marc_lcsh_650_full_frequencies.json",
        TaxonomyType.LCGFT: "marc_lcgft_655_frequencies.json",
        TaxonomyType.SUDOC_AGENCY: "marc_sudoc_agency_frequencies.json",
        TaxonomyType.CORP_NAMES: "marc_corp_name_610_frequencies.json",
    }

    filename = file_map.get(taxonomy_type)
    if not filename:
        raise ValueError(f"No frequency file for {taxonomy_type}")

    freq_path = frequencies_dir / filename
    if not freq_path.exists():
        raise FileNotFoundError(f"Frequency file not found: {freq_path}")

    # Use orjson for faster parsing
    freq_data = orjson.loads(freq_path.read_bytes())

    # Load additional label info if available (skip for now - too slow for large files)
    loc_labels = {}

    # Build labels
    labels = []
    total_uses = 0

    for rank, item in enumerate(freq_data, 1):
        term = item["term"]
        count = item["count"]
        total_uses += count

        # For LCC main classes, use our hardcoded names
        if taxonomy_type == TaxonomyType.LCC_MAIN:
            label_text = LCC_MAIN_CLASSES.get(term, term)
            label_id = term
        else:
            label_text = term
            label_id = term

        # Try to enrich from LOC data
        uri = None
        alt_labels = []
        broader = []
        narrower = []
        description = None

        # For LCSH/LCGFT, try to find matching LOC entry
        if loc_labels:
            # Match by label text
            for loc_uri, loc_info in loc_labels.items():
                if loc_info.get("pref_label") == term:
                    uri = loc_uri
                    label_id = loc_info.get("id", term)
                    alt_labels = loc_info.get("alt_labels", [])
                    broader = loc_info.get("broader", [])
                    narrower = loc_info.get("narrower", [])
                    description = loc_info.get("scope_note")
                    break

        labels.append(Label(
            id=label_id,
            label=label_text,
            uri=uri,
            frequency=count,
            rank=rank,
            alt_labels=alt_labels,
            broader=broader,
            narrower=narrower,
            description=description,
        ))

    name_map = {
        TaxonomyType.LCC_MAIN: "LCC Main Classes",
        TaxonomyType.LCC_SUBCLASS: "LCC Subclasses",
        TaxonomyType.LCSH_TOPICAL: "LCSH Topical Subjects",
        TaxonomyType.LCSH_GEOGRAPHIC: "LCSH Geographic Subjects",
        TaxonomyType.LCSH_FULL: "LCSH Full Headings",
        TaxonomyType.LCGFT: "LCGFT Genre/Form Terms",
        TaxonomyType.SUDOC_AGENCY: "SuDoc Agency Codes",
        TaxonomyType.CORP_NAMES: "Corporate Name Subjects",
    }

    return Taxonomy(
        type=taxonomy_type,
        name=name_map.get(taxonomy_type, str(taxonomy_type)),
        description=f"Loaded from MARC frequency analysis of CGP records",
        source=str(freq_path),
        labels=labels,
        total_corpus_uses=total_uses,
        corpus_coverage=1.0,
    )

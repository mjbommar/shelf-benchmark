"""
Geographic Region Mapping for SHELF Benchmark

This module provides utilities for mapping the 44 unique geographic locations
in the SHELF corpus to 8 broad geographic regions for clustering tasks.

The mapping groups countries, regions, cities, and US states into meaningful
regional clusters that enable geographic-based document clustering evaluation.

Regions:
- North America: US, Canada, US states/cities
- South America: Brazil, South American countries
- Europe: European countries and cities
- East Asia: China, Japan, South Korea
- South/Southeast Asia: India, Southeast Asian countries
- Middle East & North Africa: Middle Eastern countries
- Sub-Saharan Africa: African countries south of Sahara
- Central America & Caribbean: Mexico, Central American nations

Multi-tag ground truth (v0.4)
------------------------------
33.4% of the corpus carries two geographic tags, and 76.4% of those tag pairs
map to *different* regions (e.g. ``["Paris", "Brazil"]``, ``["Beijing",
"Florida"]``). :func:`get_region_from_list` historically resolves this by
taking the first recognized tag, which means ~38.5% of geographically
labelled documents carry an arbitrary region label: the document is really
about two regions and the ground truth silently picks one.

:func:`get_region_from_list` keeps that behavior as the default for
reproducibility. :class:`GeographicLabelPolicy` adds two opt-in alternatives
(``unambiguous_only`` drops the ambiguous documents, ``all_regions`` returns
every region the tags span) via :func:`get_region_with_policy`, and
:func:`analyze_geographic_ambiguity` makes the scale of the defect visible
against a real corpus rather than leaving it as an assumption.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

# Complete mapping of all 44 geographic locations to 8 regions
GEOGRAPHIC_REGION_MAPPING: dict[str, str] = {
    # North America (15 locations)
    "United States": "North America",
    "Canada": "North America",
    "California": "North America",
    "New York": "North America",
    "Texas": "North America",
    "Florida": "North America",
    "Illinois": "North America",
    "Pennsylvania": "North America",
    "Ohio": "North America",
    "Georgia": "North America",
    "North Carolina": "North America",
    "Michigan": "North America",
    "New York City": "North America",
    "Los Angeles": "North America",
    "Chicago": "North America",
    "North America": "North America",
    # South America (3 locations)
    "Brazil": "South America",
    "São Paulo": "South America",
    "South America": "South America",
    # Europe (10 locations)
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Russia": "Europe",
    "Europe": "Europe",
    "London": "Europe",
    "Paris": "Europe",
    "Berlin": "Europe",
    # East Asia (6 locations)
    "China": "East Asia",
    "Japan": "East Asia",
    "South Korea": "East Asia",
    "Tokyo": "East Asia",
    "Beijing": "East Asia",
    "Asia": "East Asia",  # Default mapping - most Asia references are East Asian
    # South/Southeast Asia (3 locations)
    "India": "South/Southeast Asia",
    "Southeast Asia": "South/Southeast Asia",
    "Mumbai": "South/Southeast Asia",
    # Middle East & North Africa (1 location)
    "Middle East": "Middle East & North Africa",
    # Sub-Saharan Africa (1 location)
    "Africa": "Sub-Saharan Africa",
    # Central America & Caribbean (3 locations)
    "Mexico": "Central America & Caribbean",
    "Central America": "Central America & Caribbean",
    "Caribbean": "Central America & Caribbean",
}

# Reverse mapping: region -> list of locations
REGION_TO_LOCATIONS: dict[str, list[str]] = {}
for location, region in GEOGRAPHIC_REGION_MAPPING.items():
    if region not in REGION_TO_LOCATIONS:
        REGION_TO_LOCATIONS[region] = []
    REGION_TO_LOCATIONS[region].append(location)

# All unique regions
ALL_REGIONS: list[str] = sorted(set(GEOGRAPHIC_REGION_MAPPING.values()))

# Number of regions (for k-means clustering)
NUM_REGIONS: int = len(ALL_REGIONS)


def get_region(location: str) -> str | None:
    """Get the geographic region for a given location.

    Args:
        location: Geographic location string (country, city, state, or region)

    Returns:
        Geographic region name, or None if location not recognized

    Examples:
        >>> get_region("United States")
        'North America'
        >>> get_region("Tokyo")
        'East Asia'
        >>> get_region("Unknown Place")
        None
    """
    return GEOGRAPHIC_REGION_MAPPING.get(location)


def get_region_from_list(locations: list[str]) -> str | None:
    """Get the geographic region from a list of locations (uses first valid).

    This is the standard method for SHELF clustering tasks, which use the
    first geographic tag when documents have multiple tags.

    Args:
        locations: List of geographic location strings

    Returns:
        Geographic region for the first recognized location, or None if
        no recognized locations found

    Examples:
        >>> get_region_from_list(["Paris", "London"])
        'Europe'
        >>> get_region_from_list(["Unknown", "Tokyo", "Beijing"])
        'East Asia'
        >>> get_region_from_list([])
        None
    """
    for location in locations:
        region = get_region(location)
        if region is not None:
            return region
    return None


class GeographicLabelPolicy(str, Enum):
    """Policy for resolving a document's region from its geographic tags.

    Attributes:
        FIRST: Use the first recognized tag (the historical/default SHELF
            behavior). Kept as the default for reproducibility even though it
            silently assigns an arbitrary region to ambiguous documents.
        UNAMBIGUOUS_ONLY: Return a region only when every recognized tag on
            the document maps to the *same* region; ambiguous or unlabeled
            documents resolve to ``None`` and should be dropped by callers.
        ALL_REGIONS: Return the full set of distinct regions the document's
            tags span, for multi-label treatments.
    """

    FIRST = "first"
    UNAMBIGUOUS_ONLY = "unambiguous_only"
    ALL_REGIONS = "all_regions"


def get_regions_from_list(locations: list[str]) -> frozenset[str]:
    """Return the distinct recognized regions spanned by a list of tags.

    Args:
        locations: List of geographic location strings

    Returns:
        Frozenset of distinct region names among the recognized tags. Empty
        if no tag is recognized.

    Examples:
        >>> sorted(get_regions_from_list(["Paris", "Brazil"]))
        ['Europe', 'South America']
        >>> get_regions_from_list(["Unknown"])
        frozenset()
    """
    regions = {get_region(location) for location in locations}
    regions.discard(None)
    return frozenset(regions)  # type: ignore[arg-type]


def is_unambiguous_region(locations: list[str]) -> bool:
    """Return True if the tags map to zero or exactly one distinct region.

    Args:
        locations: List of geographic location strings

    Returns:
        True when the recognized tags agree on a single region (or there are
        none); False when they span two or more regions.

    Examples:
        >>> is_unambiguous_region(["Paris", "London"])
        True
        >>> is_unambiguous_region(["Paris", "Brazil"])
        False
    """
    return len(get_regions_from_list(locations)) <= 1


def get_region_with_policy(
    locations: list[str],
    policy: GeographicLabelPolicy = GeographicLabelPolicy.FIRST,
) -> str | frozenset[str] | None:
    """Resolve a document's region(s) from its tags under an explicit policy.

    This is the opt-in entry point for the multi-tag defect documented at
    module level. The default policy reproduces :func:`get_region_from_list`
    exactly; existing callers are unaffected unless they pass a different
    policy explicitly.

    Args:
        locations: List of geographic location strings
        policy: Resolution policy. ``FIRST`` (default) reproduces the
            historical behavior. ``UNAMBIGUOUS_ONLY`` returns ``None`` for
            any document whose tags span more than one region, so it can be
            filtered out by the caller. ``ALL_REGIONS`` returns the full set
            of regions spanned (empty documents/no recognized tags return
            ``None`` here too, for a uniform "excluded" signal).

    Returns:
        - ``FIRST``: a single region name, or ``None``.
        - ``UNAMBIGUOUS_ONLY``: a single region name, or ``None`` if
          ambiguous or unrecognized.
        - ``ALL_REGIONS``: a non-empty ``frozenset[str]``, or ``None`` if no
          tag is recognized.

    Raises:
        ValueError: If ``policy`` is not a recognized :class:`GeographicLabelPolicy`.

    Examples:
        >>> get_region_with_policy(["Paris", "Brazil"], GeographicLabelPolicy.FIRST)
        'Europe'
        >>> get_region_with_policy(["Paris", "Brazil"], GeographicLabelPolicy.UNAMBIGUOUS_ONLY) is None
        True
        >>> sorted(get_region_with_policy(["Paris", "Brazil"], GeographicLabelPolicy.ALL_REGIONS))
        ['Europe', 'South America']
    """
    if policy is GeographicLabelPolicy.FIRST:
        return get_region_from_list(locations)

    regions = get_regions_from_list(locations)

    if policy is GeographicLabelPolicy.UNAMBIGUOUS_ONLY:
        return next(iter(regions)) if len(regions) == 1 else None

    if policy is GeographicLabelPolicy.ALL_REGIONS:
        return regions if regions else None

    raise ValueError(f"Unknown GeographicLabelPolicy: {policy!r}")


@dataclass(frozen=True)
class GeographicAmbiguityReport:
    """Statistics on multi-tag region ambiguity across a corpus.

    "Ambiguous" means a document has two or more geographic tags whose
    recognized regions are not all the same -- e.g. ``["Paris", "Brazil"]``
    (Europe + South America). A document with multiple tags that all map to
    the same region (e.g. ``["Paris", "London"]``, both Europe) is not
    ambiguous.

    Attributes:
        total_documents: Total documents examined.
        documents_with_geo: Documents with at least one recognized geographic tag.
        documents_with_multiple_tags: Documents with 2+ raw geographic tags
            (recognized or not).
        documents_ambiguous: Documents whose recognized tags span 2+ distinct
            regions.
        region_pair_counts: Count of each unordered pair of regions
            co-occurring on an ambiguous document, keyed by a sorted
            ``(region_a, region_b)`` tuple.
        example_ambiguous_tags: Sample raw tag lists that were ambiguous,
            capped at the ``max_examples`` passed to
            :func:`analyze_geographic_ambiguity`.
    """

    total_documents: int
    documents_with_geo: int
    documents_with_multiple_tags: int
    documents_ambiguous: int
    region_pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    example_ambiguous_tags: list[list[str]] = field(default_factory=list)

    @property
    def multi_tag_fraction(self) -> float:
        """Share of all documents that carry 2+ geographic tags."""
        return (
            self.documents_with_multiple_tags / self.total_documents
            if self.total_documents
            else 0.0
        )

    @property
    def ambiguous_fraction_of_multi_tag(self) -> float:
        """Share of multi-tag documents whose tags span different regions."""
        return (
            self.documents_ambiguous / self.documents_with_multiple_tags
            if self.documents_with_multiple_tags
            else 0.0
        )

    @property
    def ambiguous_fraction_of_geo_labelled(self) -> float:
        """Share of geographically-labelled documents (>=1 recognized tag)
        that carry an arbitrary region label under the ``FIRST`` policy."""
        return (
            self.documents_ambiguous / self.documents_with_geo
            if self.documents_with_geo
            else 0.0
        )

    def summary(self) -> str:
        """Human-readable one-paragraph summary, mirroring the data plan's framing."""
        return (
            f"{self.documents_with_multiple_tags}/{self.total_documents} "
            f"({self.multi_tag_fraction:.1%}) documents carry multiple geographic "
            f"tags; {self.documents_ambiguous}/{self.documents_with_multiple_tags} "
            f"({self.ambiguous_fraction_of_multi_tag:.1%}) of those pairs map to "
            f"different regions. {self.documents_ambiguous}/{self.documents_with_geo} "
            f"({self.ambiguous_fraction_of_geo_labelled:.1%}) of all "
            f"geographically-labelled documents therefore carry an arbitrary "
            f"region label under the default FIRST-tag policy."
        )


def analyze_geographic_ambiguity(
    documents: list[dict],
    max_examples: int = 10,
) -> GeographicAmbiguityReport:
    """Measure how much of a corpus has an ambiguous multi-region geographic label.

    This makes the defect described at module level measurable against any
    corpus slice rather than assumed. It does not mutate ``documents`` or
    change any existing default behavior.

    Args:
        documents: List of document dictionaries with a ``'geographic'`` field
            (list of location tag strings).
        max_examples: Maximum number of example ambiguous tag lists to retain.

    Returns:
        A :class:`GeographicAmbiguityReport` with the ambiguity counts and
        the distribution of which region pairs co-occur.

    Examples:
        >>> docs = [
        ...     {"geographic": ["Paris", "Brazil"]},
        ...     {"geographic": ["Paris", "London"]},
        ...     {"geographic": ["Tokyo"]},
        ...     {"geographic": []},
        ... ]
        >>> report = analyze_geographic_ambiguity(docs)
        >>> report.documents_ambiguous
        1
        >>> report.documents_with_multiple_tags
        2
    """
    total = len(documents)
    with_geo = 0
    with_multiple = 0
    ambiguous = 0
    pair_counter: Counter[tuple[str, str]] = Counter()
    examples: list[list[str]] = []

    for doc in documents:
        geo_list = doc.get("geographic", []) or []

        regions = get_regions_from_list(geo_list)
        if regions:
            with_geo += 1

        if len(geo_list) > 1:
            with_multiple += 1

        if len(regions) > 1:
            ambiguous += 1
            sorted_regions = sorted(regions)
            for i, region_a in enumerate(sorted_regions):
                for region_b in sorted_regions[i + 1 :]:
                    pair_counter[(region_a, region_b)] += 1
            if len(examples) < max_examples:
                examples.append(list(geo_list))

    return GeographicAmbiguityReport(
        total_documents=total,
        documents_with_geo=with_geo,
        documents_with_multiple_tags=with_multiple,
        documents_ambiguous=ambiguous,
        region_pair_counts=dict(pair_counter),
        example_ambiguous_tags=examples,
    )


def get_locations_for_region(region: str) -> list[str]:
    """Get all locations that map to a given region.

    Args:
        region: Geographic region name

    Returns:
        List of location strings that map to this region

    Examples:
        >>> locations = get_locations_for_region("South America")
        >>> "Brazil" in locations
        True
        >>> len(locations)
        3
    """
    return REGION_TO_LOCATIONS.get(region, [])


def get_all_regions() -> list[str]:
    """Get list of all geographic regions.

    Returns:
        Sorted list of all region names

    Examples:
        >>> regions = get_all_regions()
        >>> len(regions)
        8
        >>> "North America" in regions
        True
    """
    return ALL_REGIONS.copy()


def get_all_locations() -> list[str]:
    """Get list of all geographic locations.

    Returns:
        Sorted list of all location names

    Examples:
        >>> locations = get_all_locations()
        >>> len(locations)
        44
        >>> "Tokyo" in locations
        True
    """
    return sorted(GEOGRAPHIC_REGION_MAPPING.keys())


def validate_geographic_data(documents: list[dict]) -> dict:
    """Validate geographic data in a corpus and return statistics.

    Args:
        documents: List of document dictionaries with 'geographic' field

    Returns:
        Dictionary with validation statistics:
        - total_documents: Total number of documents
        - documents_with_geo: Number with at least one geographic tag
        - documents_without_geo: Number without any geographic tags
        - documents_with_multiple_geo: Number with multiple tags
        - unrecognized_locations: Set of location strings not in mapping
        - region_distribution: Count of documents per region (first tag only)

    Examples:
        >>> docs = [
        ...     {"geographic": ["Tokyo", "Beijing"]},
        ...     {"geographic": ["Paris"]},
        ...     {"geographic": []},
        ...     {"geographic": ["Unknown"]}
        ... ]
        >>> stats = validate_geographic_data(docs)
        >>> stats['total_documents']
        4
        >>> stats['documents_with_geo']
        3
    """
    from collections import Counter

    total = len(documents)
    with_geo = 0
    without_geo = 0
    with_multiple = 0
    unrecognized: set[str] = set()
    region_counter: Counter = Counter()

    for doc in documents:
        geo_list = doc.get("geographic", [])

        if not geo_list:
            without_geo += 1
            continue

        with_geo += 1
        if len(geo_list) > 1:
            with_multiple += 1

        # Check first tag for region
        first_location = geo_list[0]
        region = get_region(first_location)
        if region:
            region_counter[region] += 1
        else:
            unrecognized.add(first_location)

        # Check all tags for unrecognized locations
        for location in geo_list:
            if get_region(location) is None:
                unrecognized.add(location)

    return {
        "total_documents": total,
        "documents_with_geo": with_geo,
        "documents_without_geo": without_geo,
        "documents_with_multiple_geo": with_multiple,
        "unrecognized_locations": unrecognized,
        "region_distribution": dict(region_counter),
    }


def filter_documents_for_clustering(
    documents: list[dict],
    policy: GeographicLabelPolicy = GeographicLabelPolicy.FIRST,
) -> list[dict]:
    """Filter documents for geographic clustering (keep only those with valid regions).

    Args:
        documents: List of document dictionaries with 'geographic' field
        policy: Region resolution policy (default: ``FIRST``, matching the
            historical behavior). Pass ``GeographicLabelPolicy.UNAMBIGUOUS_ONLY``
            to additionally drop documents whose tags span multiple regions.

    Returns:
        Filtered list of documents that resolve to a region under ``policy``

    Examples:
        >>> docs = [
        ...     {"id": "1", "geographic": ["Tokyo"]},
        ...     {"id": "2", "geographic": []},
        ...     {"id": "3", "geographic": ["Unknown"]},
        ...     {"id": "4", "geographic": ["Paris", "London"]}
        ... ]
        >>> filtered = filter_documents_for_clustering(docs)
        >>> len(filtered)
        2
        >>> filtered[0]['id']
        '1'
    """
    return [
        doc
        for doc in documents
        if get_region_with_policy(doc.get("geographic", []), policy) is not None
    ]


def add_geographic_region_field(
    documents: list[dict],
    policy: GeographicLabelPolicy = GeographicLabelPolicy.FIRST,
) -> list[dict]:
    """Add 'geographic_region' field to documents based on their geographic tags.

    Modifies documents in-place and returns the list.

    Args:
        documents: List of document dictionaries with 'geographic' field
        policy: Region resolution policy (default: ``FIRST``, matching the
            historical behavior, and producing a single region name per
            document as before). ``UNAMBIGUOUS_ONLY`` also produces a single
            region name or ``None``. ``ALL_REGIONS`` produces a
            ``frozenset[str]`` or ``None`` instead of a single string.

    Returns:
        Same list with 'geographic_region' field added (None if no valid
        region under ``policy``)

    Examples:
        >>> docs = [
        ...     {"id": "1", "geographic": ["Tokyo", "Beijing"]},
        ...     {"id": "2", "geographic": []}
        ... ]
        >>> updated = add_geographic_region_field(docs)
        >>> updated[0]['geographic_region']
        'East Asia'
        >>> updated[1]['geographic_region'] is None
        True
    """
    for doc in documents:
        geo_list = doc.get("geographic", [])
        doc["geographic_region"] = get_region_with_policy(geo_list, policy)
    return documents


# Export mapping for use in other modules
__all__ = [
    "GEOGRAPHIC_REGION_MAPPING",
    "REGION_TO_LOCATIONS",
    "ALL_REGIONS",
    "NUM_REGIONS",
    "GeographicLabelPolicy",
    "GeographicAmbiguityReport",
    "get_region",
    "get_region_from_list",
    "get_regions_from_list",
    "is_unambiguous_region",
    "get_region_with_policy",
    "analyze_geographic_ambiguity",
    "get_locations_for_region",
    "get_all_regions",
    "get_all_locations",
    "validate_geographic_data",
    "filter_documents_for_clustering",
    "add_geographic_region_field",
]

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
"""

from typing import Dict, List, Set

# Complete mapping of all 44 geographic locations to 8 regions
GEOGRAPHIC_REGION_MAPPING: Dict[str, str] = {
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
REGION_TO_LOCATIONS: Dict[str, List[str]] = {}
for location, region in GEOGRAPHIC_REGION_MAPPING.items():
    if region not in REGION_TO_LOCATIONS:
        REGION_TO_LOCATIONS[region] = []
    REGION_TO_LOCATIONS[region].append(location)

# All unique regions
ALL_REGIONS: List[str] = sorted(set(GEOGRAPHIC_REGION_MAPPING.values()))

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


def get_region_from_list(locations: List[str]) -> str | None:
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


def get_locations_for_region(region: str) -> List[str]:
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


def get_all_regions() -> List[str]:
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


def get_all_locations() -> List[str]:
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


def validate_geographic_data(documents: List[dict]) -> dict:
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
    unrecognized: Set[str] = set()
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


def filter_documents_for_clustering(documents: List[dict]) -> List[dict]:
    """Filter documents for geographic clustering (keep only those with valid regions).

    Args:
        documents: List of document dictionaries with 'geographic' field

    Returns:
        Filtered list of documents that have at least one recognized geographic tag

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
        if get_region_from_list(doc.get("geographic", [])) is not None
    ]


def add_geographic_region_field(documents: List[dict]) -> List[dict]:
    """Add 'geographic_region' field to documents based on first geographic tag.

    Modifies documents in-place and returns the list.

    Args:
        documents: List of document dictionaries with 'geographic' field

    Returns:
        Same list with 'geographic_region' field added (None if no valid region)

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
        doc["geographic_region"] = get_region_from_list(geo_list)
    return documents


# Export mapping for use in other modules
__all__ = [
    "GEOGRAPHIC_REGION_MAPPING",
    "REGION_TO_LOCATIONS",
    "ALL_REGIONS",
    "NUM_REGIONS",
    "get_region",
    "get_region_from_list",
    "get_locations_for_region",
    "get_all_regions",
    "get_all_locations",
    "validate_geographic_data",
    "filter_documents_for_clustering",
    "add_geographic_region_field",
]

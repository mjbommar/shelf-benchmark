#!/usr/bin/env python
"""
Analyze Geographic Clustering Distribution for SHELF

This script analyzes the geographic distribution in the SHELF corpus
and validates the regional mapping for the geographic clustering task.

Usage:
    python scripts/analyze_geographic_clustering.py
    python scripts/analyze_geographic_clustering.py --artifacts-dir data/artifacts
    python scripts/analyze_geographic_clustering.py --output results/geographic_analysis.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from shelf.taxonomies.geographic import (
    GEOGRAPHIC_REGION_MAPPING,
    REGION_TO_LOCATIONS,
    add_geographic_region_field,
    filter_documents_for_clustering,
    get_all_regions,
    validate_geographic_data,
)


def load_artifacts(artifacts_dir: Path) -> List[dict]:
    """Load all JSON artifacts from the artifacts directory."""
    documents = []
    for json_file in sorted(artifacts_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            doc = json.load(f)
            documents.append(doc)
    return documents


def analyze_geographic_distribution(documents: List[dict]) -> Dict:
    """Perform comprehensive analysis of geographic distribution."""
    # Validate geographic data
    validation = validate_geographic_data(documents)

    # Count locations (including multiple per document)
    location_counter = Counter()
    multi_label_distribution = Counter()

    for doc in documents:
        geo_list = doc.get("geographic", [])
        if geo_list:
            multi_label_distribution[len(geo_list)] += 1
            for location in geo_list:
                location_counter[location] += 1

    # Filter for clustering (documents with valid regions)
    clusterable_docs = filter_documents_for_clustering(documents)
    clusterable_with_regions = add_geographic_region_field(clusterable_docs)

    # Count documents per region
    region_counter = Counter()
    for doc in clusterable_with_regions:
        region = doc.get("geographic_region")
        if region:
            region_counter[region] += 1

    # Calculate statistics
    total_docs = len(documents)
    clusterable_count = len(clusterable_docs)

    analysis = {
        "total_documents": total_docs,
        "documents_with_geographic_tags": validation["documents_with_geo"],
        "documents_without_geographic_tags": validation["documents_without_geo"],
        "documents_clusterable": clusterable_count,
        "percentage_clusterable": round(
            100 * clusterable_count / total_docs if total_docs > 0 else 0, 2
        ),
        "documents_with_multiple_tags": validation["documents_with_multiple_geo"],
        "unique_locations_in_corpus": len(location_counter),
        "locations_in_mapping": len(GEOGRAPHIC_REGION_MAPPING),
        "unrecognized_locations": list(validation["unrecognized_locations"]),
        "multi_label_distribution": dict(multi_label_distribution),
        "location_frequencies": dict(location_counter.most_common()),
        "region_distribution": dict(region_counter.most_common()),
        "regions_in_mapping": get_all_regions(),
        "region_to_locations_mapping": REGION_TO_LOCATIONS,
    }

    return analysis


def print_analysis_report(analysis: Dict) -> None:
    """Print a human-readable analysis report."""
    print("=" * 80)
    print("SHELF Geographic Clustering Analysis")
    print("=" * 80)
    print()

    # Overview
    print("OVERVIEW")
    print("-" * 80)
    print(f"Total documents:                 {analysis['total_documents']:,}")
    print(
        f"Documents with geographic tags:  {analysis['documents_with_geographic_tags']:,}"
    )
    print(
        f"Documents without tags:          {analysis['documents_without_geographic_tags']:,}"
    )
    print(f"Documents clusterable:           {analysis['documents_clusterable']:,}")
    print(f"Percentage clusterable:          {analysis['percentage_clusterable']}%")
    print(
        f"Documents with multiple tags:    {analysis['documents_with_multiple_tags']:,}"
    )
    print()

    # Location coverage
    print("LOCATION COVERAGE")
    print("-" * 80)
    print(f"Unique locations in corpus:      {analysis['unique_locations_in_corpus']}")
    print(f"Locations in mapping:            {analysis['locations_in_mapping']}")
    if analysis["unrecognized_locations"]:
        print(f"\nUnrecognized locations ({len(analysis['unrecognized_locations'])}):")
        for loc in sorted(analysis["unrecognized_locations"]):
            print(f"  - {loc}")
    else:
        print("All locations recognized: OK")
    print()

    # Multi-label distribution
    print("MULTI-LABEL DISTRIBUTION")
    print("-" * 80)
    for num_tags, count in sorted(analysis["multi_label_distribution"].items()):
        print(f"Documents with {num_tags} tag(s):  {count:,}")
    print()

    # Top locations
    print("TOP 15 LOCATIONS BY FREQUENCY")
    print("-" * 80)
    location_freq = analysis["location_frequencies"]
    for i, (location, count) in enumerate(
        sorted(location_freq.items(), key=lambda x: x[1], reverse=True)[:15], 1
    ):
        region = GEOGRAPHIC_REGION_MAPPING.get(location, "Unknown")
        print(f"{i:2d}. {location:30s} {count:5d}  ({region})")
    print()

    # Regional distribution
    print("REGIONAL DISTRIBUTION (for clustering)")
    print("-" * 80)
    region_dist = analysis["region_distribution"]
    total_clusterable = sum(region_dist.values())

    for i, (region, count) in enumerate(
        sorted(region_dist.items(), key=lambda x: x[1], reverse=True), 1
    ):
        percentage = 100 * count / total_clusterable if total_clusterable > 0 else 0
        print(f"{i}. {region:35s} {count:5d}  ({percentage:5.2f}%)")

    print()
    print(f"Total clusterable documents: {total_clusterable:,}")
    print()

    # Region balance assessment
    print("BALANCE ASSESSMENT")
    print("-" * 80)
    num_regions = len(region_dist)
    if num_regions > 0:
        expected_per_region = total_clusterable / num_regions
        max_count = max(region_dist.values())
        min_count = min(region_dist.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        print(f"Number of regions:               {num_regions}")
        print(f"Expected per region:             {expected_per_region:.1f}")
        print(f"Largest region:                  {max_count}")
        print(f"Smallest region:                 {min_count}")
        print(f"Imbalance ratio:                 {imbalance_ratio:.2f}")
        print()

        if imbalance_ratio < 2.0:
            print("Distribution: WELL-BALANCED")
        elif imbalance_ratio < 4.0:
            print("Distribution: MODERATELY BALANCED")
        else:
            print("Distribution: IMBALANCED (consider rebalancing)")
    print()

    # Regional mapping summary
    print("REGIONAL MAPPING SUMMARY")
    print("-" * 80)
    for region in get_all_regions():
        locations = REGION_TO_LOCATIONS.get(region, [])
        print(f"\n{region} ({len(locations)} locations):")
        for loc in sorted(locations):
            count = analysis["location_frequencies"].get(loc, 0)
            print(f"  - {loc:30s} ({count} docs)")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze geographic distribution for SHELF clustering task"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/artifacts"),
        help="Directory containing JSON artifact files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for analysis results",
    )

    args = parser.parse_args()

    # Check artifacts directory exists
    if not args.artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {args.artifacts_dir}")
        return 1

    # Load documents
    print(f"Loading documents from {args.artifacts_dir}...")
    documents = load_artifacts(args.artifacts_dir)
    print(f"Loaded {len(documents):,} documents")
    print()

    # Analyze
    analysis = analyze_geographic_distribution(documents)

    # Print report
    print_analysis_report(analysis)

    # Save to JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Analyze geographic coverage in the SHELF benchmark dataset.

This script loads the SHELF dataset and analyzes:
1. Geographic distribution across regions and continents
2. LCC class distribution by geography
3. Topic distribution by geography
4. Cross-product diversity metrics
"""

from collections import Counter, defaultdict
from datasets import load_dataset
import json


def categorize_geography(geo_name):
    """Categorize geographic regions by continent/major region."""

    # Continents and major regions
    asia = {
        'Asia', 'Southeast Asia', 'Japan', 'Tokyo', 'Beijing', 'China',
        'India', 'Mumbai', 'South Korea', 'Middle East'
    }

    europe = {
        'Europe', 'United Kingdom', 'Germany', 'Italy', 'Russia',
        'London', 'Paris', 'Berlin', 'France', 'Spain'
    }

    north_america = {
        'United States', 'Canada', 'North America', 'Mexico',
        # US states and cities
        'Florida', 'Ohio', 'North Carolina', 'New York City', 'Pennsylvania',
        'Los Angeles', 'Chicago', 'Georgia', 'Michigan', 'Illinois',
        'California', 'Texas', 'New York'
    }

    south_america = {
        'South America', 'Brazil', 'São Paulo', 'Central America', 'Caribbean'
    }

    oceania = {
        'Australia'
    }

    africa = {
        'Africa'
    }

    if geo_name in asia:
        return 'Asia'
    elif geo_name in europe:
        return 'Europe'
    elif geo_name in north_america:
        return 'North America'
    elif geo_name in south_america:
        return 'South America'
    elif geo_name in oceania:
        return 'Oceania'
    elif geo_name in africa:
        return 'Africa'
    else:
        return 'Other'


def analyze_geographic_coverage():
    """Main analysis function."""

    print("Loading SHELF dataset...")
    ds = load_dataset('mjbommar/SHELF')

    # Combine all splits
    all_docs = []
    for split in ['train', 'validation', 'test']:
        all_docs.extend(ds[split])

    print(f"Total documents: {len(all_docs):,}")
    print()

    # Geographic distribution
    print("=" * 80)
    print("GEOGRAPHIC DISTRIBUTION")
    print("=" * 80)
    print()

    geo_counter = Counter()
    continent_counter = Counter()
    docs_with_geo = 0

    for doc in all_docs:
        if doc.get('geographic') and len(doc['geographic']) > 0:
            docs_with_geo += 1
            for geo in doc['geographic']:
                geo_counter[geo] += 1
                continent = categorize_geography(geo)
                continent_counter[continent] += 1

    print(f"Documents with geographic metadata: {docs_with_geo:,} ({docs_with_geo/len(all_docs)*100:.1f}%)")
    print(f"Unique geographic regions: {len(geo_counter)}")
    print()

    print("By Continent/Major Region:")
    for continent, count in sorted(continent_counter.items(), key=lambda x: x[1], reverse=True):
        pct = count / sum(continent_counter.values()) * 100
        print(f"  {continent:20s}: {count:6,} ({pct:5.1f}%)")
    print()

    print("Top 20 Specific Regions:")
    for geo, count in geo_counter.most_common(20):
        continent = categorize_geography(geo)
        print(f"  {geo:25s} [{continent:15s}]: {count:5,}")
    print()

    # LCC class by geography
    print("=" * 80)
    print("LCC CLASS BY GEOGRAPHY")
    print("=" * 80)
    print()

    lcc_geo_matrix = defaultdict(lambda: defaultdict(int))

    for doc in all_docs:
        lcc = doc.get('lcc_code', 'Unknown')
        if doc.get('geographic'):
            for geo in doc['geographic']:
                continent = categorize_geography(geo)
                lcc_geo_matrix[lcc][continent] += 1

    # Print matrix
    continents = sorted(continent_counter.keys())
    print(f"{'LCC':5s} | " + " | ".join(f"{c:15s}" for c in continents))
    print("-" * (7 + len(continents) * 18))

    for lcc in sorted(lcc_geo_matrix.keys()):
        row = f"{lcc:5s} | "
        for continent in continents:
            count = lcc_geo_matrix[lcc][continent]
            row += f"{count:15,} | "
        print(row)
    print()

    # Topic by geography
    print("=" * 80)
    print("TOPIC COVERAGE BY GEOGRAPHY")
    print("=" * 80)
    print()

    topic_geo_matrix = defaultdict(lambda: defaultdict(int))

    for doc in all_docs:
        if doc.get('topics') and doc.get('geographic'):
            for topic in doc['topics']:
                for geo in doc['geographic']:
                    continent = categorize_geography(geo)
                    topic_geo_matrix[topic][continent] += 1

    print(f"Unique topics: {len(topic_geo_matrix)}")
    print()

    # Show topics with best global distribution
    print("Topics with broadest geographic distribution (top 10):")
    topic_geo_diversity = []
    for topic, continents in topic_geo_matrix.items():
        # Calculate how many continents have this topic
        num_continents = len([c for c in continents.values() if c > 0])
        total_count = sum(continents.values())
        topic_geo_diversity.append((topic, num_continents, total_count))

    for topic, num_continents, total_count in sorted(topic_geo_diversity, key=lambda x: (-x[1], -x[2]))[:10]:
        print(f"  {topic:30s}: {num_continents}/7 continents, {total_count:5,} total docs")
    print()

    # Non-Western topic examples
    print("=" * 80)
    print("NON-WESTERN TOPIC COVERAGE")
    print("=" * 80)
    print()

    print("Topics appearing in Asian contexts:")
    asia_topics = [(t, c['Asia']) for t, c in topic_geo_matrix.items() if c['Asia'] > 0]
    for topic, count in sorted(asia_topics, key=lambda x: -x[1])[:15]:
        print(f"  {topic:30s}: {count:5,} docs")
    print()

    print("Topics appearing in African contexts:")
    africa_topics = [(t, c['Africa']) for t, c in topic_geo_matrix.items() if c['Africa'] > 0]
    for topic, count in sorted(africa_topics, key=lambda x: -x[1])[:15]:
        print(f"  {topic:30s}: {count:5,} docs")
    print()

    # Cross-product diversity
    print("=" * 80)
    print("CROSS-PRODUCT DIVERSITY EXAMPLES")
    print("=" * 80)
    print()

    print("Unexpected LCC + Geography combinations:")

    # Find interesting cross-products
    interesting_combos = []

    for doc in all_docs:
        lcc_name = doc.get('lcc_name', '')
        form = doc.get('lcgft_form', '')
        geos = doc.get('geographic', [])

        # Look for unusual combinations
        if lcc_name and form and geos:
            for geo in geos:
                combo = (lcc_name, form, geo)
                interesting_combos.append(combo)

    # Sample some interesting ones
    combo_counter = Counter(interesting_combos)

    print("\nSample diversity examples (LCC Class + Form + Geography):")
    sampled = []
    seen_patterns = set()

    for (lcc, form, geo), count in combo_counter.most_common(1000):
        pattern = (lcc, form)
        if pattern not in seen_patterns and len(sampled) < 20:
            seen_patterns.add(pattern)
            continent = categorize_geography(geo)
            sampled.append((lcc, form, geo, continent, count))

    for lcc, form, geo, continent, count in sampled:
        print(f"  {lcc:30s} + {form:20s} + {geo:20s} [{continent}]: {count:3} docs")
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    print(f"Total documents:                 {len(all_docs):,}")
    print(f"Documents with geography:        {docs_with_geo:,} ({docs_with_geo/len(all_docs)*100:.1f}%)")
    print(f"Unique geographic regions:       {len(geo_counter)}")
    print(f"Continents/major regions:        {len(continent_counter)}")
    print(f"Unique LCC classes:              {len(lcc_geo_matrix)}")
    print(f"Unique topics:                   {len(topic_geo_matrix)}")
    print()

    # Calculate balance metrics
    geo_counts = list(geo_counter.values())
    avg_geo_count = sum(geo_counts) / len(geo_counts)
    min_geo_count = min(geo_counts)
    max_geo_count = max(geo_counts)

    print(f"Geographic balance:")
    print(f"  Average docs per region:       {avg_geo_count:.0f}")
    print(f"  Min docs per region:           {min_geo_count}")
    print(f"  Max docs per region:           {max_geo_count}")
    print(f"  Balance ratio (min/max):       {min_geo_count/max_geo_count:.2f}")
    print()

    continent_counts = list(continent_counter.values())
    avg_continent_count = sum(continent_counts) / len(continent_counts)

    print(f"Continental balance:")
    print(f"  Average docs per continent:    {avg_continent_count:.0f}")
    print(f"  Min docs per continent:        {min(continent_counts)}")
    print(f"  Max docs per continent:        {max(continent_counts)}")
    print()


if __name__ == '__main__':
    analyze_geographic_coverage()

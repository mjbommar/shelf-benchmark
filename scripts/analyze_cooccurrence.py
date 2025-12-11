"""
Analyze co-occurrence patterns between taxonomy dimensions in MARC records.

This helps us understand which combinations are realistic for synthetic data.
"""

import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import json

from pymarc import MARCReader


def extract_lc_class(call_number: str) -> str | None:
    """Extract main class letter from LC call number."""
    if not call_number:
        return None
    letters = ""
    for char in call_number:
        if char.isalpha():
            letters += char.upper()
        else:
            break
    if letters and len(letters) <= 3:
        return letters[0]
    return None


def extract_sudoc_agency(sudoc: str) -> str | None:
    """Extract agency code from SuDoc number."""
    if not sudoc:
        return None
    parts = sudoc.split()
    if parts:
        return parts[0].rstrip(':').split('.')[0]
    return None


def analyze_marc_cooccurrence(record_sets_dir: Path, max_records: int = 100000):
    """Analyze co-occurrence patterns in MARC records."""

    # Co-occurrence counters
    lcc_genre = Counter()       # LCC class + genre
    lcc_topic = Counter()       # LCC class + topic
    genre_topic = Counter()     # genre + topic
    agency_lcc = Counter()      # agency + LCC class
    agency_genre = Counter()    # agency + genre
    agency_topic = Counter()    # agency + topic

    # Document profiles (all dimensions together)
    profiles = Counter()

    # Individual dimension frequencies
    lcc_freq = Counter()
    genre_freq = Counter()
    topic_freq = Counter()
    agency_freq = Counter()
    geo_freq = Counter()

    record_count = 0
    zip_files = sorted(record_sets_dir.glob("cataloging-records-all-cgp-utf8-*.zip"))

    print(f"Analyzing {len(zip_files)} zip files (max {max_records:,} records)...")

    for zip_path in zip_files:
        if record_count >= max_records:
            break

        with zipfile.ZipFile(zip_path, 'r') as zf:
            mrc_names = [n for n in zf.namelist() if n.endswith('.mrc')]
            if not mrc_names:
                continue

            with zf.open(mrc_names[0]) as mrc_file:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mrc', delete=False) as tmp:
                    tmp.write(mrc_file.read())
                    tmp_path = Path(tmp.name)

                try:
                    with open(tmp_path, 'rb') as f:
                        reader = MARCReader(f, to_unicode=True, force_utf8=True)

                        for record in reader:
                            if record is None:
                                continue
                            if record_count >= max_records:
                                break

                            record_count += 1

                            # Extract dimensions
                            lcc = None
                            for field in record.get_fields('050'):
                                for sub in field.get_subfields('a'):
                                    lcc = extract_lc_class(sub)
                                    if lcc:
                                        break
                                if lcc:
                                    break

                            agency = None
                            for field in record.get_fields('086'):
                                for sub in field.get_subfields('a'):
                                    agency = extract_sudoc_agency(sub)
                                    if agency:
                                        break
                                if agency:
                                    break

                            genres = set()
                            for field in record.get_fields('655'):
                                for sub in field.get_subfields('a'):
                                    genres.add(sub.rstrip('.'))

                            topics = set()
                            for field in record.get_fields('650'):
                                for sub in field.get_subfields('a'):
                                    topics.add(sub.rstrip('.'))

                            geos = set()
                            for field in record.get_fields('651'):
                                for sub in field.get_subfields('a'):
                                    geos.add(sub.rstrip('.'))

                            # Count individual frequencies
                            if lcc:
                                lcc_freq[lcc] += 1
                            if agency:
                                agency_freq[agency] += 1
                            for g in genres:
                                genre_freq[g] += 1
                            for t in topics:
                                topic_freq[t] += 1
                            for geo in geos:
                                geo_freq[geo] += 1

                            # Count co-occurrences
                            for g in genres:
                                if lcc:
                                    lcc_genre[(lcc, g)] += 1
                                if agency:
                                    agency_genre[(agency, g)] += 1
                                for t in topics:
                                    genre_topic[(g, t)] += 1

                            for t in topics:
                                if lcc:
                                    lcc_topic[(lcc, t)] += 1
                                if agency:
                                    agency_topic[(agency, t)] += 1

                            if lcc and agency:
                                agency_lcc[(agency, lcc)] += 1

                finally:
                    tmp_path.unlink()

        print(f"  Processed {record_count:,} records...")

    return {
        "record_count": record_count,
        "lcc_freq": lcc_freq,
        "genre_freq": genre_freq,
        "topic_freq": topic_freq,
        "agency_freq": agency_freq,
        "geo_freq": geo_freq,
        "lcc_genre": lcc_genre,
        "lcc_topic": lcc_topic,
        "genre_topic": genre_topic,
        "agency_lcc": agency_lcc,
        "agency_genre": agency_genre,
        "agency_topic": agency_topic,
    }


def print_top_cooccurrences(name: str, counter: Counter, n: int = 20):
    """Print top co-occurrences."""
    print(f"\n{'='*60}")
    print(f"TOP {n} {name}")
    print('='*60)
    for pair, count in counter.most_common(n):
        print(f"  {count:>6,}  {pair}")


def main():
    record_sets_dir = Path("/nas3/data/legal/us/cataloging-records-all-cgp-utf8/Record_sets")

    results = analyze_marc_cooccurrence(record_sets_dir, max_records=200000)

    print(f"\n\nAnalyzed {results['record_count']:,} records")

    # Print summaries
    print(f"\n{'='*60}")
    print("DIMENSION CARDINALITIES (in sample)")
    print('='*60)
    print(f"  LCC Classes:    {len(results['lcc_freq']):>6,}")
    print(f"  Agencies:       {len(results['agency_freq']):>6,}")
    print(f"  Genres:         {len(results['genre_freq']):>6,}")
    print(f"  Topics:         {len(results['topic_freq']):>6,}")
    print(f"  Geographic:     {len(results['geo_freq']):>6,}")

    print_top_cooccurrences("LCC + GENRE", results['lcc_genre'], 30)
    print_top_cooccurrences("AGENCY + LCC", results['agency_lcc'], 30)
    print_top_cooccurrences("AGENCY + GENRE", results['agency_genre'], 30)
    print_top_cooccurrences("GENRE + TOPIC", results['genre_topic'], 30)

    # Analyze sparsity - what % of possible combinations exist?
    top_lcc = [x[0] for x in results['lcc_freq'].most_common(10)]
    top_genres = [x[0] for x in results['genre_freq'].most_common(10)]
    top_agencies = [x[0] for x in results['agency_freq'].most_common(10)]

    print(f"\n{'='*60}")
    print("SPARSITY ANALYSIS (Top 10 x Top 10)")
    print('='*60)

    # LCC x Genre matrix
    print("\nLCC x GENRE co-occurrence matrix (top 10 each):")
    possible = len(top_lcc) * len(top_genres)
    observed = sum(1 for lcc in top_lcc for g in top_genres if results['lcc_genre'].get((lcc, g), 0) > 0)
    print(f"  Possible cells: {possible}")
    print(f"  Observed cells: {observed} ({observed/possible*100:.1f}%)")

    # Agency x LCC matrix
    print("\nAGENCY x LCC co-occurrence matrix (top 10 each):")
    possible = len(top_agencies) * len(top_lcc)
    observed = sum(1 for a in top_agencies for lcc in top_lcc if results['agency_lcc'].get((a, lcc), 0) > 0)
    print(f"  Possible cells: {possible}")
    print(f"  Observed cells: {observed} ({observed/possible*100:.1f}%)")

    # Save for later use
    output = {
        "record_count": results['record_count'],
        "top_lcc": results['lcc_freq'].most_common(20),
        "top_genres": results['genre_freq'].most_common(20),
        "top_agencies": results['agency_freq'].most_common(20),
        "top_topics": results['topic_freq'].most_common(50),
        "top_geo": results['geo_freq'].most_common(20),
        "lcc_genre_top": results['lcc_genre'].most_common(100),
        "agency_lcc_top": results['agency_lcc'].most_common(100),
        "agency_genre_top": results['agency_genre'].most_common(100),
    }

    output_path = Path("data/cooccurrence_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()

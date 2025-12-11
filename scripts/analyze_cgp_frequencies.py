"""
Analyze frequency of LC authority labels in the CGP MARC records.

This extracts and counts:
- LCSH (subjects) - http://id.loc.gov/authorities/subjects/
- LCGFT (genreForms) - http://id.loc.gov/authorities/genreForms/
- LCDGT (demographicTerms) - http://id.loc.gov/authorities/demographicTerms/
- LCNAF (names) - http://id.loc.gov/authorities/names/
- LCC (classification) - http://id.loc.gov/authorities/classification/
"""

import gzip
import json
from collections import Counter
from pathlib import Path


def extract_authority_urls(jsonl_path: Path) -> dict[str, Counter]:
    """Extract and count authority URLs by type."""
    counters = {
        "subjects": Counter(),      # LCSH
        "genreForms": Counter(),    # LCGFT
        "demographicTerms": Counter(),  # LCDGT
        "names": Counter(),         # LCNAF
        "classification": Counter(),  # LCC
        "other": Counter(),
    }

    total_records = 0

    open_fn = gzip.open if str(jsonl_path).endswith('.gz') else open

    with open_fn(jsonl_path, 'rt', encoding='utf-8') as f:
        for line in f:
            total_records += 1
            if total_records % 100000 == 0:
                print(f"Processed {total_records:,} records...")

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            urls = record.get("urls", [])
            for url in urls:
                if not url.startswith("http://id.loc.gov/authorities/"):
                    continue

                # Extract the authority type and ID
                path = url.replace("http://id.loc.gov/authorities/", "")
                parts = path.split("/")
                if len(parts) >= 2:
                    auth_type = parts[0]
                    auth_id = parts[1]

                    if auth_type in counters:
                        counters[auth_type][url] += 1
                    else:
                        counters["other"][url] += 1

    print(f"\nTotal records processed: {total_records:,}")
    return counters


def print_top_n(counter: Counter, n: int = 50, label_lookup: dict = None):
    """Print top N items from counter with optional label lookup."""
    for url, count in counter.most_common(n):
        if label_lookup and url in label_lookup:
            label = label_lookup[url]
            print(f"  {count:>6,}  {label}")
        else:
            # Extract ID from URL
            id_part = url.split("/")[-1]
            print(f"  {count:>6,}  {id_part} ({url})")


def load_label_lookup(json_path: Path) -> dict[str, str]:
    """Load URI -> prefLabel mapping from parsed JSON."""
    if not json_path.exists():
        return {}

    data = json.loads(json_path.read_text())
    return {uri: info["pref_label"] for uri, info in data.items()}


def main():
    jsonl_path = Path("/nas3/data/legal/us/cgp-purls/marc-records-urls.jsonl.gz")
    parsed_dir = Path("data/loc_parsed")

    print("="*70)
    print("Analyzing LC Authority Frequencies in CGP MARC Records")
    print("="*70)

    # Extract frequencies
    counters = extract_authority_urls(jsonl_path)

    # Load label lookups
    lcsh_labels = load_label_lookup(parsed_dir / "lcsh.json")
    lcgft_labels = load_label_lookup(parsed_dir / "lcgft.json")
    lcdgt_labels = load_label_lookup(parsed_dir / "lcdgt.json")

    # Print summaries
    print("\n" + "="*70)
    print("SUMMARY BY AUTHORITY TYPE")
    print("="*70)

    for auth_type, counter in counters.items():
        unique = len(counter)
        total = sum(counter.values())
        print(f"\n{auth_type.upper()}: {unique:,} unique, {total:,} total uses")

    # Top LCSH
    print("\n" + "="*70)
    print("TOP 100 LCSH (Subject Headings)")
    print("="*70)
    print_top_n(counters["subjects"], 100, lcsh_labels)

    # Top LCGFT
    print("\n" + "="*70)
    print("TOP 50 LCGFT (Genre/Form Terms)")
    print("="*70)
    print_top_n(counters["genreForms"], 50, lcgft_labels)

    # Top LCDGT
    print("\n" + "="*70)
    print("TOP 50 LCDGT (Demographic Terms)")
    print("="*70)
    print_top_n(counters["demographicTerms"], 50, lcdgt_labels)

    # Top Names
    print("\n" + "="*70)
    print("TOP 50 LCNAF (Name Authorities)")
    print("="*70)
    print_top_n(counters["names"], 50)

    # Distribution analysis
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)

    for auth_type in ["subjects", "genreForms", "demographicTerms"]:
        counter = counters[auth_type]
        if not counter:
            continue

        counts = list(counter.values())
        counts.sort(reverse=True)

        total = sum(counts)
        cumsum = 0

        thresholds = [10, 50, 100, 500, 1000, 5000]
        print(f"\n{auth_type.upper()} coverage:")

        for i, count in enumerate(counts, 1):
            cumsum += count
            pct = cumsum / total * 100

            if i in thresholds or (i == len(counts)):
                print(f"  Top {i:>5,} terms cover {pct:>5.1f}% of uses ({cumsum:,}/{total:,})")

    # Save frequency data
    output_dir = Path("data/frequencies")
    output_dir.mkdir(parents=True, exist_ok=True)

    for auth_type, counter in counters.items():
        if counter:
            # Save as list of [url, count, label] tuples
            label_lookup = {
                "subjects": lcsh_labels,
                "genreForms": lcgft_labels,
                "demographicTerms": lcdgt_labels,
            }.get(auth_type, {})

            data = [
                {
                    "url": url,
                    "count": count,
                    "label": label_lookup.get(url, url.split("/")[-1])
                }
                for url, count in counter.most_common()
            ]

            out_path = output_dir / f"cgp_{auth_type}_frequencies.json"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nSaved {len(data):,} {auth_type} frequencies to {out_path}")


if __name__ == "__main__":
    main()

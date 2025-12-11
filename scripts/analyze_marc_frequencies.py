"""
Analyze frequency of LC classifications and subject headings from raw MARC records.

MARC fields of interest:
- 050: LC Classification Number
- 055: Classification Numbers Assigned in Canada
- 060: NLM Call Number
- 086: Government Document Classification Number (SuDoc)
- 6XX: Subject Access Fields
  - 600: Personal Name
  - 610: Corporate Name
  - 611: Meeting Name
  - 630: Uniform Title
  - 648: Chronological Term
  - 650: Topical Term (LCSH)
  - 651: Geographic Name
  - 655: Genre/Form (LCGFT)
  - 656: Occupation
  - 657: Function
"""

import zipfile
from collections import Counter
from pathlib import Path

from pymarc import MARCReader


def extract_lc_class(call_number: str) -> tuple[str, str] | None:
    """Extract main class and subclass from LC call number."""
    if not call_number:
        return None

    # LC call numbers start with 1-3 letters
    letters = ""
    for char in call_number:
        if char.isalpha():
            letters += char.upper()
        else:
            break

    if not letters or len(letters) > 3:
        return None

    main_class = letters[0]
    subclass = letters

    return main_class, subclass


def analyze_marc_file(marc_path: Path, counters: dict):
    """Analyze a single MARC file."""
    record_count = 0

    with open(marc_path, 'rb') as f:
        reader = MARCReader(f, to_unicode=True, force_utf8=True)

        for record in reader:
            if record is None:
                continue

            record_count += 1

            # 050: LC Classification
            for field in record.get_fields('050'):
                for subfield in field.get_subfields('a'):
                    result = extract_lc_class(subfield)
                    if result:
                        main_class, subclass = result
                        counters["lcc_main_class"][main_class] += 1
                        counters["lcc_subclass"][subclass] += 1

            # 086: SuDoc Classification
            for field in record.get_fields('086'):
                for subfield in field.get_subfields('a'):
                    if subfield:
                        # Extract agency code (first part before space or number)
                        parts = subfield.split()
                        if parts:
                            agency = parts[0].rstrip(':').split('.')[0]
                            counters["sudoc_agency"][agency] += 1

            # 650: Topical Subject (LCSH)
            for field in record.get_fields('650'):
                # Check indicator 2 for source (0=LCSH, 7=other specified)
                subfield_a = field.get_subfields('a')
                if subfield_a:
                    heading = subfield_a[0].rstrip('.')
                    counters["lcsh_650"][heading] += 1

                    # Also get subdivisions
                    full_heading = heading
                    for sub in field.get_subfields('x', 'y', 'z', 'v'):
                        full_heading += f"--{sub.rstrip('.')}"
                    counters["lcsh_650_full"][full_heading] += 1

            # 651: Geographic Subject
            for field in record.get_fields('651'):
                subfield_a = field.get_subfields('a')
                if subfield_a:
                    heading = subfield_a[0].rstrip('.')
                    counters["lcsh_651_geo"][heading] += 1

            # 655: Genre/Form (LCGFT)
            for field in record.get_fields('655'):
                subfield_a = field.get_subfields('a')
                if subfield_a:
                    heading = subfield_a[0].rstrip('.')
                    counters["lcgft_655"][heading] += 1

            # 610: Corporate Name Subject
            for field in record.get_fields('610'):
                subfield_a = field.get_subfields('a')
                if subfield_a:
                    heading = subfield_a[0].rstrip('.')
                    counters["corp_name_610"][heading] += 1

    return record_count


def main():
    import json

    record_sets_dir = Path("/nas3/data/legal/us/cataloging-records-all-cgp-utf8/Record_sets")
    output_dir = Path("data/frequencies")
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = {
        "lcc_main_class": Counter(),
        "lcc_subclass": Counter(),
        "sudoc_agency": Counter(),
        "lcsh_650": Counter(),
        "lcsh_650_full": Counter(),
        "lcsh_651_geo": Counter(),
        "lcgft_655": Counter(),
        "corp_name_610": Counter(),
    }

    total_records = 0

    # Process all zip files
    zip_files = sorted(record_sets_dir.glob("cataloging-records-all-cgp-utf8-*.zip"))

    print(f"Found {len(zip_files)} zip files to process")
    print("="*70)

    for i, zip_path in enumerate(zip_files):
        print(f"\nProcessing {zip_path.name} ({i+1}/{len(zip_files)})...")

        # Extract to temp location
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get the .mrc filename inside
            mrc_names = [n for n in zf.namelist() if n.endswith('.mrc')]
            if not mrc_names:
                print(f"  No .mrc file found in {zip_path.name}")
                continue

            mrc_name = mrc_names[0]

            # Extract and process
            with zf.open(mrc_name) as mrc_file:
                # Write to temp file since MARCReader needs seekable file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mrc', delete=False) as tmp:
                    tmp.write(mrc_file.read())
                    tmp_path = Path(tmp.name)

                try:
                    count = analyze_marc_file(tmp_path, counters)
                    total_records += count
                    print(f"  Processed {count:,} records (total: {total_records:,})")
                finally:
                    tmp_path.unlink()

    # Print summaries
    print("\n" + "="*70)
    print(f"TOTAL RECORDS: {total_records:,}")
    print("="*70)

    # LCC Main Classes
    print("\n\nLCC MAIN CLASSES:")
    print("-"*50)
    for cls, count in counters["lcc_main_class"].most_common():
        from loc_taxonomies import LCC_MAIN_CLASSES
        name = LCC_MAIN_CLASSES.get(cls, "Unknown")
        print(f"  {cls}: {count:>8,}  {name}")

    # LCC Subclasses (top 30)
    print("\n\nLCC SUBCLASSES (Top 30):")
    print("-"*50)
    for cls, count in counters["lcc_subclass"].most_common(30):
        print(f"  {cls:>4}: {count:>8,}")

    # SuDoc Agencies (top 30)
    print("\n\nSUDOC AGENCIES (Top 30):")
    print("-"*50)
    for agency, count in counters["sudoc_agency"].most_common(30):
        print(f"  {agency:>12}: {count:>8,}")

    # LCSH Topics (top 50)
    print("\n\nLCSH TOPICAL SUBJECTS (Top 50 main headings):")
    print("-"*50)
    for heading, count in counters["lcsh_650"].most_common(50):
        print(f"  {count:>8,}  {heading[:60]}")

    # LCSH Geographic (top 30)
    print("\n\nLCSH GEOGRAPHIC SUBJECTS (Top 30):")
    print("-"*50)
    for heading, count in counters["lcsh_651_geo"].most_common(30):
        print(f"  {count:>8,}  {heading[:60]}")

    # LCGFT Genre/Form (top 30)
    print("\n\nLCGFT GENRE/FORM (Top 30):")
    print("-"*50)
    for heading, count in counters["lcgft_655"].most_common(30):
        print(f"  {count:>8,}  {heading[:60]}")

    # Corporate Names (top 30)
    print("\n\nCORPORATE NAME SUBJECTS (Top 30):")
    print("-"*50)
    for heading, count in counters["corp_name_610"].most_common(30):
        print(f"  {count:>8,}  {heading[:60]}")

    # Save all frequencies to JSON
    for name, counter in counters.items():
        data = [{"term": term, "count": count} for term, count in counter.most_common()]
        out_path = output_dir / f"marc_{name}_frequencies.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved {len(data):,} {name} to {out_path}")

    # Distribution analysis
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)

    for name in ["lcsh_650", "lcgft_655", "lcsh_651_geo"]:
        counter = counters[name]
        if not counter:
            continue

        counts = list(counter.values())
        counts.sort(reverse=True)
        total = sum(counts)

        print(f"\n{name}:")
        print(f"  Unique terms: {len(counts):,}")
        print(f"  Total uses: {total:,}")

        cumsum = 0
        for threshold in [10, 50, 100, 500, 1000, 2000, 5000]:
            if threshold > len(counts):
                break
            cumsum = sum(counts[:threshold])
            pct = cumsum / total * 100
            print(f"  Top {threshold:>5,} cover {pct:>5.1f}% ({cumsum:,}/{total:,})")


if __name__ == "__main__":
    main()

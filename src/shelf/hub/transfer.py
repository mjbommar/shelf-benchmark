"""
Natural-data transfer slice: loading, validation, and distribution reporting.

SHELF v0.3.1 is 100% LLM-generated. Phase 4's "holdout" is still synthetic --
newer generators, same generation process -- so nothing in the benchmark tests
whether SHELF performance transfers to documents no LLM wrote. Phase 6 of
``docs/data_plan_v0.4.md`` closes that gap with a *natural* slice: human-authored,
human-catalogued text carrying real Library of Congress class letters.

The reference build is Project Gutenberg (see ``scripts/build_transfer_slice.py``),
whose RDF catalogue records a cataloguer-assigned LCC subclass per work::

    <dcterms:subject>
      <rdf:Description rdf:nodeID="N6c38e2...">
        <dcam:memberOf rdf:resource="http://purl.org/dc/terms/LCC"/>
        <rdf:value>PR</rdf:value>
      </rdf:Description>
    </dcterms:subject>

Two properties of this slice govern every function in this module.

**1. It is not contamination-free, and must never be described as such.**
Project Gutenberg is in the pretraining corpus of essentially every model this
benchmark evaluates. That is not a defect to be apologised for -- it is the point.
SHELF is the *clean-synthetic* condition; the transfer slice is the
*contaminated-natural* condition; the **gap between them** is the measurement.
Consequently a metric computed over a pool of synthetic *and* natural records is
uninterpretable: it averages the two conditions whose difference is the result.
``assert_single_source_type`` exists to make that mistake raise rather than
silently produce a number, and ``split_by_source_type`` is the supported way to
evaluate both.

**2. Its label distribution is skewed, and the skew is published, not hidden.**
Gutenberg is heavily P (Language and Literature) and pre-1929. The builder
stratifies and subsamples toward the flattest achievable LCC distribution, and
this module reports what was actually realized -- against SHELF's near-uniform
distribution -- so that a per-class reading of any transfer result is possible.
A transfer slice is allowed to be unbalanced. It is not allowed to be unbalanced
*quietly*.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Schema
# =============================================================================

#: Bumped whenever the record layout emitted by ``scripts/build_transfer_slice.py``
#: changes in a way that older readers cannot handle.
TRANSFER_SCHEMA_VERSION = "1.0"


class SourceType(str, Enum):
    """Whether a record's text was written by a model or by a person.

    This is the ``source_type`` field of the v0.4 schema (§13 of the data plan).
    It is the partition key for reporting: never pool across it.
    """

    SYNTHETIC = "synthetic"
    NATURAL = "natural"


class ContaminationStatus(str, Enum):
    """Known pretraining-contamination status of a slice.

    Deliberately has no ``CLEAN`` member. Nothing here can be *proved* clean;
    the strongest honest claim for generated-after-cutoff synthetic text is
    "not known to be contaminated".
    """

    #: Synthetic text generated for this benchmark. Not known to be in any
    #: pretraining corpus, which is weaker than "known absent from one".
    NOT_KNOWN_CONTAMINATED = "not_known_contaminated"

    #: Public-domain text that is demonstrably in widely used pretraining
    #: corpora (Project Gutenberg, Wikipedia, Common Crawl snapshots).
    KNOWN_CONTAMINATED = "known_contaminated"

    #: Provenance not established.
    UNKNOWN = "unknown"


class NaturalSource(str, Enum):
    """Corpora the transfer slice may be built from."""

    PROJECT_GUTENBERG = "project_gutenberg"


#: Per-source contamination status. Project Gutenberg is not a borderline case:
#: it predates and is included in essentially every public pretraining corpus.
SOURCE_CONTAMINATION: dict[str, ContaminationStatus] = {
    NaturalSource.PROJECT_GUTENBERG.value: ContaminationStatus.KNOWN_CONTAMINATED,
}


#: Verbatim text that must accompany any published result on this slice.
#: ``scripts/build_transfer_slice.py`` copies it into every manifest and every
#: distribution report it writes.
CONTAMINATION_NOTICE = (
    "CONTAMINATION: This slice is NOT contamination-free and must never be "
    "presented as such. Project Gutenberg is in the pretraining data of "
    "essentially every model SHELF evaluates. Its value is comparative: SHELF "
    "is the clean-synthetic condition, this slice is the contaminated-natural "
    "condition, and the GAP between them is the measurement. Report the two as "
    "separate conditions; never pool them into a single metric."
)

#: Verbatim text that must accompany any published LCC distribution for this
#: slice.
SKEW_NOTICE = (
    "SKEW: The Project Gutenberg pool is heavily P (Language and Literature), "
    "pre-1929, English, and book-length. The builder subsamples toward the "
    "flattest achievable LCC distribution and publishes what it actually "
    "realized -- never an assumed outcome -- next to the source-pool coverage. "
    "Read both: an even passage count can still rest on a near-exhausted class, "
    "so a class drawn from 100 candidates and one drawn from 30,000 are not "
    "equally diverse even at identical N. This is a transfer slice, not a "
    "balanced corpus; report per-class results, not only the macro average."
)


#: Fields every record in the slice must carry. Deliberately a subset of the
#: SHELF schema plus the Phase 6 provenance fields -- a natural record cannot
#: supply ``temperature`` or ``model``, and pretending otherwise would make the
#: two conditions look more alike than they are.
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "text",
    "title",
    "lcc_code",
    "lcc_name",
    "source_type",
    "source",
    "contamination_status",
    "provenance",
)

#: Provenance keys required on a Project Gutenberg record.
REQUIRED_GUTENBERG_PROVENANCE: tuple[str, ...] = (
    "gutenberg_id",
    "lcc_letter",
    "lcc_subclasses",
    "chunk_index",
    "chunk_count",
    "text_url",
)


#: The 21 LCC main classes, as SHELF uses them. Mirrors
#: ``shelf.sampler.lc_data.LCC_CLASSES``; duplicated here so that validating a
#: transfer slice does not drag in the generation-side sampler.
LCC_MAIN_CLASSES: dict[str, str] = {
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


#: Realized ``lcc_code`` counts of the SHELF v0.3.1 **test** split (8,507 docs),
#: read from the HuggingFace datasets-server statistics endpoint for
#: ``mjbommar/SHELF`` config ``default`` on 2026-08-26.
#:
#: The test split is used rather than the full 42,532 because splits are
#: stratified on ``lcc_code``, so split shares match corpus shares to within
#: sampling noise, and the endpoint returns exact per-label frequencies for a
#: split without downloading the parquet. Shares run 4.5%-5.0% -- the
#: "near-uniform" distribution of Finding 1 in the data plan.
SHELF_V031_TEST_LCC_COUNTS: dict[str, int] = {
    "A": 412,
    "B": 424,
    "C": 400,
    "D": 418,
    "E": 411,
    "F": 398,
    "G": 389,
    "H": 404,
    "J": 403,
    "K": 394,
    "L": 401,
    "M": 403,
    "N": 415,
    "P": 401,
    "Q": 407,
    "R": 404,
    "S": 403,
    "T": 417,
    "U": 406,
    "V": 413,
    "Z": 384,
}


# =============================================================================
# Errors
# =============================================================================


class TransferSliceError(Exception):
    """Base class for transfer-slice problems."""


class TransferValidationError(TransferSliceError):
    """A slice failed validation in strict mode."""


class SourceTypePoolingError(TransferSliceError):
    """Synthetic and natural records were about to be pooled into one metric.

    Raised by :func:`assert_single_source_type`. The fix is never to relax the
    check -- it is to evaluate each condition separately and report the gap.
    """


# =============================================================================
# Loading
# =============================================================================


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise TransferSliceError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise TransferSliceError(f"{p}:{lineno}: expected an object")
            yield obj


def load_transfer_slice(
    path: str | Path,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Load a transfer slice from JSONL.

    Args:
        path: Path to ``records.jsonl`` produced by ``build_transfer_slice.py``.
        strict: Raise :class:`TransferValidationError` if validation finds any
            error. Warnings never raise.

    Returns:
        The records, in file order.

    Raises:
        FileNotFoundError: The path does not exist.
        TransferValidationError: ``strict`` and the slice has errors.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transfer slice not found: {p}")

    records = list(iter_jsonl(p))
    report = validate_records(records)
    if strict and not report.ok:
        raise TransferValidationError(
            f"{p}: {len(report.errors)} validation error(s); first: {report.errors[0]}"
        )
    return records


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a slice manifest written alongside ``records.jsonl``."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TransferSliceError(f"{p}: manifest must be a JSON object")
    return data


# =============================================================================
# Validation
# =============================================================================


@dataclass
class ValidationReport:
    """Outcome of validating a slice.

    ``errors`` block use of the slice; ``warnings`` are facts worth surfacing
    (an empty ``lcgft_form``, say) that do not make the record unusable.
    """

    n_records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        state = "PASS" if self.ok else "FAIL"
        return (
            f"{state}: {self.n_records} record(s), "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


def validate_record(record: dict[str, Any], *, index: int = 0) -> list[str]:
    """Return a list of error strings for one record (empty means valid)."""
    errors: list[str] = []
    prefix = f"record[{index}]"

    for key in REQUIRED_FIELDS:
        if key not in record:
            errors.append(f"{prefix}: missing required field {key!r}")

    rec_id = record.get("id")
    if not isinstance(rec_id, str) or not rec_id:
        errors.append(f"{prefix}: 'id' must be a non-empty string")
    else:
        prefix = f"record[{index}] id={rec_id}"

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append(f"{prefix}: 'text' must be a non-empty string")

    lcc = record.get("lcc_code")
    if not isinstance(lcc, str) or lcc not in LCC_MAIN_CLASSES:
        errors.append(
            f"{prefix}: 'lcc_code' must be one of the 21 LCC main classes, got {lcc!r}"
        )

    source_type = record.get("source_type")
    if source_type != SourceType.NATURAL.value:
        errors.append(
            f"{prefix}: 'source_type' must be {SourceType.NATURAL.value!r} in a "
            f"transfer slice, got {source_type!r}"
        )

    status = record.get("contamination_status")
    source = record.get("source")
    if isinstance(source, str) and source in SOURCE_CONTAMINATION:
        expected = SOURCE_CONTAMINATION[source].value
        if status != expected:
            errors.append(
                f"{prefix}: source {source!r} is {expected!r}; record claims {status!r}. "
                "Contamination status is a property of the source, not a choice."
            )
    elif status not in {s.value for s in ContaminationStatus}:
        errors.append(f"{prefix}: unknown 'contamination_status' {status!r}")

    if status == ContaminationStatus.NOT_KNOWN_CONTAMINATED.value:
        errors.append(
            f"{prefix}: a natural slice may not claim "
            f"{ContaminationStatus.NOT_KNOWN_CONTAMINATED.value!r}"
        )

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        errors.append(f"{prefix}: 'provenance' must be an object")
    elif source == NaturalSource.PROJECT_GUTENBERG.value:
        for key in REQUIRED_GUTENBERG_PROVENANCE:
            if key not in provenance:
                errors.append(f"{prefix}: provenance missing {key!r}")

    return errors


def validate_records(records: Sequence[dict[str, Any]]) -> ValidationReport:
    """Validate a whole slice, including cross-record checks."""
    report = ValidationReport(n_records=len(records))

    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    missing_form = 0

    for i, record in enumerate(records):
        report.errors.extend(validate_record(record, index=i))

        rec_id = record.get("id")
        if isinstance(rec_id, str):
            if rec_id in seen_ids:
                report.errors.append(f"duplicate id {rec_id!r}")
            seen_ids.add(rec_id)

        text = record.get("text")
        if isinstance(text, str):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_texts:
                report.warnings.append(
                    f"record {rec_id!r}: text duplicates an earlier record"
                )
            seen_texts.add(digest)

        if not record.get("lcgft_form"):
            missing_form += 1

    if missing_form:
        report.warnings.append(
            f"{missing_form}/{len(records)} record(s) have no derivable 'lcgft_form'. "
            "Gutenberg RDF carries no LCGFT; form is inferred from LCSH form "
            "subdivisions only where unambiguous."
        )

    return report


# =============================================================================
# Source-type separation
# =============================================================================


def split_by_source_type(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Partition records by ``source_type``.

    The supported way to evaluate a mixed pool: run each partition through the
    metric separately, then report both numbers and their difference.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("source_type") or SourceType.SYNTHETIC.value)
        buckets.setdefault(key, []).append(record)
    return buckets


def assert_single_source_type(records: Sequence[dict[str, Any]]) -> str:
    """Assert that every record shares one ``source_type``; return it.

    Call this at the top of anything that reduces records to a single number.

    Raises:
        SourceTypePoolingError: More than one ``source_type`` is present.
    """
    kinds = {str(r.get("source_type") or SourceType.SYNTHETIC.value) for r in records}
    if len(kinds) > 1:
        raise SourceTypePoolingError(
            f"Refusing to compute one metric over mixed source types {sorted(kinds)}. "
            "Synthetic and natural records are two conditions, and the gap between "
            "them is the result -- pooling them averages it away. Use "
            "split_by_source_type() and report each condition."
        )
    return next(iter(kinds), SourceType.SYNTHETIC.value)


# =============================================================================
# Distribution reporting
# =============================================================================


def _shares(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return dict.fromkeys(counts, 0.0)
    return {k: v / total for k, v in counts.items()}


def _normalized_entropy(counts: dict[str, int], *, support: int) -> float:
    """Shannon entropy over ``support`` categories, scaled to [0, 1].

    1.0 is a perfectly flat distribution over ``support`` categories; 0.0 puts
    all mass on one. Used as the single-number summary of "how flat did the
    subsampling actually get".
    """
    total = sum(counts.values())
    if total == 0 or support <= 1:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log(p)
    return entropy / math.log(support)


def total_variation_distance(a: dict[str, int], b: dict[str, int]) -> float:
    """Total variation distance between two count distributions (0 = identical)."""
    sa, sb = _shares(a), _shares(b)
    keys = set(sa) | set(sb)
    return 0.5 * sum(abs(sa.get(k, 0.0) - sb.get(k, 0.0)) for k in keys)


@dataclass
class LengthStats:
    """Word-count summary of a set of records."""

    n: int = 0
    minimum: int = 0
    p10: int = 0
    median: int = 0
    mean: float = 0.0
    p90: int = 0
    maximum: int = 0

    @classmethod
    def from_counts(cls, counts: Sequence[int]) -> LengthStats:
        if not counts:
            return cls()
        ordered = sorted(counts)
        n = len(ordered)

        def pct(q: float) -> int:
            idx = min(n - 1, max(0, int(round(q * (n - 1)))))
            return ordered[idx]

        return cls(
            n=n,
            minimum=ordered[0],
            p10=pct(0.10),
            median=int(statistics.median(ordered)),
            mean=sum(ordered) / n,
            p90=pct(0.90),
            maximum=ordered[-1],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "min": self.minimum,
            "p10": self.p10,
            "median": self.median,
            "mean": round(self.mean, 1),
            "p90": self.p90,
            "max": self.maximum,
        }


@dataclass
class SliceProfile:
    """The realized shape of a transfer slice.

    Everything a reader needs to judge how far the slice departs from SHELF,
    in one object: label distribution, passage lengths, language mix, and the
    date range of the underlying works.
    """

    n_records: int = 0
    n_works: int = 0
    source: str = ""
    source_type: str = SourceType.NATURAL.value
    contamination_status: str = ContaminationStatus.UNKNOWN.value
    lcc_counts: dict[str, int] = field(default_factory=dict)
    lcc_subclass_counts: dict[str, int] = field(default_factory=dict)
    works_per_lcc: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    lcgft_category_counts: dict[str, int] = field(default_factory=dict)
    lcgft_form_counts: dict[str, int] = field(default_factory=dict)
    length: LengthStats = field(default_factory=LengthStats)
    chunks_per_work: LengthStats = field(default_factory=LengthStats)
    author_year_range: tuple[int | None, int | None] = (None, None)
    author_century_counts: dict[str, int] = field(default_factory=dict)
    issued_year_range: tuple[int | None, int | None] = (None, None)
    lcc_normalized_entropy: float = 0.0

    @property
    def lcc_shares(self) -> dict[str, float]:
        return _shares(self.lcc_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_works": self.n_works,
            "source": self.source,
            "source_type": self.source_type,
            "contamination_status": self.contamination_status,
            "contamination_notice": CONTAMINATION_NOTICE,
            "skew_notice": SKEW_NOTICE,
            "lcc_counts": self.lcc_counts,
            "lcc_shares": {k: round(v, 5) for k, v in self.lcc_shares.items()},
            "lcc_subclass_counts": self.lcc_subclass_counts,
            "works_per_lcc": self.works_per_lcc,
            "lcc_normalized_entropy": round(self.lcc_normalized_entropy, 4),
            "language_counts": self.language_counts,
            "lcgft_category_counts": self.lcgft_category_counts,
            "lcgft_form_counts": self.lcgft_form_counts,
            "length_words": self.length.to_dict(),
            "chunks_per_work": self.chunks_per_work.to_dict(),
            "author_year_range": list(self.author_year_range),
            "author_century_counts": self.author_century_counts,
            "issued_year_range": list(self.issued_year_range),
        }


def _year_of(value: Any) -> int | None:
    """Best-effort year from an int, ``"1897"``, or ``"1998-06-01"``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    return None


def _century_label(year: int) -> str:
    """Label a year by its century as an inclusive range, e.g. ``"1801-1900"``.

    Ranges rather than ordinals ("19th c.") because the ordinal is a reliable
    source of off-by-one confusion in exactly this kind of report.
    """
    if year <= 0:
        return "<=0"
    start = (year - 1) // 100 * 100 + 1
    return f"{start}-{start + 99}"


def describe_slice(records: Sequence[dict[str, Any]]) -> SliceProfile:
    """Compute the realized distribution of a loaded transfer slice."""
    profile = SliceProfile(n_records=len(records))
    if not records:
        return profile

    lcc: Counter[str] = Counter()
    subclass: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    forms: Counter[str] = Counter()
    lengths: list[int] = []
    work_ids: set[str] = set()
    work_lcc: dict[str, str] = {}
    chunk_counts: dict[str, int] = {}
    author_years: list[int] = []
    century: Counter[str] = Counter()
    issued_years: list[int] = []
    sources: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    source_types: Counter[str] = Counter()

    for record in records:
        provenance = record.get("provenance") or {}

        lcc[str(record.get("lcc_code", ""))] += 1
        sources[str(record.get("source", ""))] += 1
        statuses[str(record.get("contamination_status", ""))] += 1
        source_types[str(record.get("source_type", ""))] += 1

        for code in provenance.get("lcc_subclasses") or []:
            subclass[str(code)] += 1

        language = record.get("language") or provenance.get("language") or ""
        if language:
            languages[str(language)] += 1

        if record.get("lcgft_category"):
            categories[str(record["lcgft_category"])] += 1
        if record.get("lcgft_form"):
            forms[str(record["lcgft_form"])] += 1

        word_count = record.get("word_count")
        if not isinstance(word_count, int):
            word_count = len(str(record.get("text", "")).split())
        lengths.append(word_count)

        work_key = str(
            provenance.get("gutenberg_id") or record.get("work_id") or record.get("id")
        )
        work_ids.add(work_key)
        work_lcc.setdefault(work_key, str(record.get("lcc_code", "")))
        chunk_counts[work_key] = chunk_counts.get(work_key, 0) + 1

        year = _year_of(provenance.get("author_death_year")) or _year_of(
            provenance.get("author_birth_year")
        )
        if year is not None:
            author_years.append(year)
            century[_century_label(year)] += 1

        issued = _year_of(provenance.get("gutenberg_issued"))
        if issued is not None:
            issued_years.append(issued)

    profile.lcc_counts = dict(sorted(lcc.items()))
    profile.lcc_subclass_counts = dict(subclass.most_common())
    profile.language_counts = dict(languages.most_common())
    profile.lcgft_category_counts = dict(categories.most_common())
    profile.lcgft_form_counts = dict(forms.most_common())
    profile.length = LengthStats.from_counts(lengths)
    profile.n_works = len(work_ids)
    profile.chunks_per_work = LengthStats.from_counts(list(chunk_counts.values()))
    profile.works_per_lcc = dict(sorted(Counter(work_lcc.values()).items()))
    profile.lcc_normalized_entropy = _normalized_entropy(
        profile.lcc_counts, support=len(LCC_MAIN_CLASSES)
    )
    profile.author_century_counts = dict(
        sorted(century.items(), key=lambda item: int(item[0].split("-")[0]))
    )
    if author_years:
        profile.author_year_range = (min(author_years), max(author_years))
    if issued_years:
        profile.issued_year_range = (min(issued_years), max(issued_years))
    profile.source = sources.most_common(1)[0][0] if sources else ""
    profile.contamination_status = statuses.most_common(1)[0][0] if statuses else ""
    profile.source_type = source_types.most_common(1)[0][0] if source_types else ""
    return profile


@dataclass
class DistributionComparison:
    """Realized natural distribution set against the synthetic reference."""

    natural_counts: dict[str, int] = field(default_factory=dict)
    synthetic_counts: dict[str, int] = field(default_factory=dict)
    natural_entropy: float = 0.0
    synthetic_entropy: float = 0.0
    total_variation: float = 0.0
    missing_classes: list[str] = field(default_factory=list)
    over_represented: list[tuple[str, float]] = field(default_factory=list)
    under_represented: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "natural_counts": self.natural_counts,
            "synthetic_counts": self.synthetic_counts,
            "natural_normalized_entropy": round(self.natural_entropy, 4),
            "synthetic_normalized_entropy": round(self.synthetic_entropy, 4),
            "total_variation_distance": round(self.total_variation, 4),
            "missing_classes": self.missing_classes,
            "over_represented": [[c, round(d, 5)] for c, d in self.over_represented],
            "under_represented": [[c, round(d, 5)] for c, d in self.under_represented],
        }


def compare_to_synthetic(
    profile: SliceProfile,
    synthetic_counts: dict[str, int] | None = None,
) -> DistributionComparison:
    """Compare a slice's LCC distribution with SHELF's near-uniform one.

    Args:
        profile: Output of :func:`describe_slice`.
        synthetic_counts: ``lcc_code`` counts for the synthetic reference.
            Defaults to :data:`SHELF_V031_TEST_LCC_COUNTS`.
    """
    reference = dict(synthetic_counts or SHELF_V031_TEST_LCC_COUNTS)
    natural = {code: profile.lcc_counts.get(code, 0) for code in LCC_MAIN_CLASSES}

    nat_shares = _shares(natural)
    ref_shares = _shares(reference)
    deltas = sorted(
        (
            (code, nat_shares.get(code, 0.0) - ref_shares.get(code, 0.0))
            for code in LCC_MAIN_CLASSES
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    return DistributionComparison(
        natural_counts=natural,
        synthetic_counts=reference,
        natural_entropy=_normalized_entropy(natural, support=len(LCC_MAIN_CLASSES)),
        synthetic_entropy=_normalized_entropy(reference, support=len(LCC_MAIN_CLASSES)),
        total_variation=total_variation_distance(natural, reference),
        missing_classes=[code for code, count in natural.items() if count == 0],
        over_represented=[item for item in deltas[:5] if item[1] > 0],
        under_represented=[item for item in reversed(deltas[-5:]) if item[1] < 0],
    )


def format_distribution_report(
    profile: SliceProfile,
    comparison: DistributionComparison | None = None,
    pool_counts: dict[str, int] | None = None,
) -> str:
    """Render a human-readable realized-distribution report.

    The contamination and skew notices are emitted first and unconditionally:
    the report is the artifact most likely to be pasted into a paper, so the
    caveats travel with the numbers.

    Args:
        profile: Output of :func:`describe_slice`.
        comparison: Output of :func:`compare_to_synthetic`; computed if omitted.
        pool_counts: Eligible *works* per LCC class in the source corpus, before
            subsampling. Supplying it adds the pool-coverage section, which is
            where the residual skew lives once the sample itself is flat: a
            class drawn from 100 candidates and a class drawn from 30,000 can
            reach identical passage counts with wildly different internal
            diversity. Omitting it hides that.
    """
    comparison = comparison or compare_to_synthetic(profile)
    nat_shares = _shares(comparison.natural_counts)
    ref_shares = _shares(comparison.synthetic_counts)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("NATURAL TRANSFER SLICE - REALIZED DISTRIBUTION")
    lines.append("=" * 78)
    lines.append("")
    lines.append(CONTAMINATION_NOTICE)
    lines.append("")
    lines.append(SKEW_NOTICE)
    lines.append("")
    lines.append("-" * 78)
    lines.append(
        f"source={profile.source or 'unknown'}  "
        f"source_type={profile.source_type}  "
        f"contamination={profile.contamination_status}"
    )
    lines.append(f"passages={profile.n_records:,}  works={profile.n_works:,}")
    lines.append("")

    lines.append("LCC MAIN CLASS: natural vs SHELF synthetic")
    lines.append(
        f"  {'cls':<4} {'name':<40} {'passages':>9} {'nat%':>7} {'shelf%':>7} {'delta':>7}"
    )
    for code, name in LCC_MAIN_CLASSES.items():
        count = comparison.natural_counts.get(code, 0)
        nat = nat_shares.get(code, 0.0) * 100
        ref = ref_shares.get(code, 0.0) * 100
        lines.append(
            f"  {code:<4} {name[:40]:<40} {count:>9,} {nat:>6.2f}% {ref:>6.2f}% {nat - ref:>+6.2f}"
        )
    lines.append("")
    lines.append(
        f"  normalized entropy (1.0 = flat over 21 classes): "
        f"natural {comparison.natural_entropy:.4f}  synthetic {comparison.synthetic_entropy:.4f}"
    )
    lines.append(
        f"  total variation distance from SHELF: {comparison.total_variation:.4f}"
    )
    if comparison.missing_classes:
        lines.append(
            f"  classes with ZERO passages: {', '.join(comparison.missing_classes)}"
        )
    if comparison.over_represented:
        top = ", ".join(f"{c} +{d * 100:.2f}pp" for c, d in comparison.over_represented)
        lines.append(f"  most over-represented: {top}")
    if comparison.under_represented:
        bottom = ", ".join(
            f"{c} {d * 100:.2f}pp" for c, d in comparison.under_represented
        )
        lines.append(f"  most under-represented: {bottom}")
    lines.append("")

    lines.append("WORKS PER LCC CLASS (label independence check)")
    per_class = ", ".join(f"{k}={v}" for k, v in sorted(profile.works_per_lcc.items()))
    lines.append(f"  {per_class or '(none)'}")
    lines.append("")

    lines.append("PASSAGE LENGTH (words)")
    stats = profile.length.to_dict()
    lines.append(
        f"  min {stats['min']:,}  p10 {stats['p10']:,}  median {stats['median']:,}  "
        f"mean {stats['mean']:,}  p90 {stats['p90']:,}  max {stats['max']:,}"
    )
    chunk_stats = profile.chunks_per_work.to_dict()
    lines.append(
        f"  chunks per work: min {chunk_stats['min']}  median {chunk_stats['median']}  "
        f"max {chunk_stats['max']}"
    )
    lines.append("")

    lines.append("LANGUAGE MIX")
    total = max(1, profile.n_records)
    for language, count in list(profile.language_counts.items())[:10]:
        lines.append(f"  {language:<8} {count:>8,}  {count / total * 100:>6.2f}%")
    lines.append("")

    lines.append("DATE RANGE")
    lo, hi = profile.author_year_range
    lines.append(
        f"  author birth/death years: {lo if lo is not None else 'n/a'} .. {hi if hi is not None else 'n/a'}"
    )
    if profile.author_century_counts:
        by_century = ", ".join(
            f"{k}={v}" for k, v in profile.author_century_counts.items()
        )
        lines.append(f"  author era: {by_century}")
    ilo, ihi = profile.issued_year_range
    lines.append(
        f"  Project Gutenberg release years: "
        f"{ilo if ilo is not None else 'n/a'} .. {ihi if ihi is not None else 'n/a'}"
    )
    lines.append("")

    if pool_counts:
        lines.append(
            "SOURCE POOL COVERAGE (residual skew after the sample is flattened)"
        )
        lines.append(
            "  A flat passage count is not a flat corpus. These are the eligible works"
        )
        lines.append(
            "  available per class before subsampling; a class sampled at a high fraction"
        )
        lines.append(
            "  is near-exhausted and carries far less variety than its count suggests."
        )
        lines.append(f"  {'cls':<4} {'eligible':>10} {'sampled':>9} {'sampled %':>10}")
        for code in LCC_MAIN_CLASSES:
            available = pool_counts.get(code, 0)
            sampled = profile.works_per_lcc.get(code, 0)
            fraction = (sampled / available * 100) if available else 0.0
            lines.append(
                f"  {code:<4} {available:>10,} {sampled:>9,} {fraction:>9.2f}%"
            )
        # Only classes that actually contributed works; a class sampled zero
        # times has no sampling fraction to speak of and would zero the span.
        fractions = [
            profile.works_per_lcc[code] / pool_counts[code]
            for code in LCC_MAIN_CLASSES
            if pool_counts.get(code) and profile.works_per_lcc.get(code)
        ]
        if fractions:
            lines.append(
                f"  sampling fraction spans {min(fractions) * 100:.2f}% to "
                f"{max(fractions) * 100:.2f}% -- a "
                f"{max(fractions) / min(fractions):.0f}x range"
            )
        lines.append("")

    if profile.lcgft_category_counts:
        derived = sum(profile.lcgft_category_counts.values())
        coverage = derived / max(1, profile.n_records) * 100
        lines.append(
            f"DERIVED LCGFT CATEGORY (inferred from LCSH form subdivisions; "
            f"{derived:,}/{profile.n_records:,} = {coverage:.1f}% of passages)"
        )
        lines.append(
            "  Gutenberg RDF carries no LCGFT. Passages with no unambiguous LCSH form"
        )
        lines.append("  marker are left unlabelled rather than guessed.")
        for name, count in list(profile.lcgft_category_counts.items())[:14]:
            lines.append(f"  {name:<40} {count:>8,}")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


def build_report(
    records: Sequence[dict[str, Any]],
    synthetic_counts: dict[str, int] | None = None,
    pool_counts: dict[str, int] | None = None,
) -> tuple[SliceProfile, DistributionComparison, str]:
    """One-call convenience: profile, comparison, and rendered report.

    Pass ``pool_counts`` (eligible works per class in the source corpus)
    whenever it is known -- see :func:`format_distribution_report`.
    """
    profile = describe_slice(records)
    comparison = compare_to_synthetic(profile, synthetic_counts)
    return (
        profile,
        comparison,
        format_distribution_report(profile, comparison, pool_counts),
    )


__all__ = [
    "CONTAMINATION_NOTICE",
    "ContaminationStatus",
    "DistributionComparison",
    "LCC_MAIN_CLASSES",
    "LengthStats",
    "NaturalSource",
    "REQUIRED_FIELDS",
    "REQUIRED_GUTENBERG_PROVENANCE",
    "SHELF_V031_TEST_LCC_COUNTS",
    "SKEW_NOTICE",
    "SOURCE_CONTAMINATION",
    "SliceProfile",
    "SourceType",
    "SourceTypePoolingError",
    "TRANSFER_SCHEMA_VERSION",
    "TransferSliceError",
    "TransferValidationError",
    "ValidationReport",
    "assert_single_source_type",
    "build_report",
    "compare_to_synthetic",
    "describe_slice",
    "format_distribution_report",
    "iter_jsonl",
    "load_manifest",
    "load_transfer_slice",
    "split_by_source_type",
    "total_variation_distance",
    "validate_record",
    "validate_records",
]

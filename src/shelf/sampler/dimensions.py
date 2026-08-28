"""
Individual dimension samplers for LC taxonomies.

Each sampler is independent and can be used standalone or composed.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final, Generic, Literal, TypeVar

T = TypeVar("T")

# Sentinel selecting the frozen, hand-curated v0.3.1 label pools (below) for
# TopicSampler / GeographicSampler / LCGFTSampler, instead of the larger
# frequency-ranked pools loaded from data/taxonomies/*.json. This is the
# default for all three samplers, so existing callers -- and the published
# v0.3.1 corpus -- are reproduced exactly unless a numeric `pool_size` is
# passed explicitly. See docs/data_plan_v0.4.md section 4.1.
PRESET_V0_3_1: Final = "v0.3.1"


# =============================================================================
# Base Sampler
# =============================================================================


class Sampler(ABC, Generic[T]):
    """Abstract base for all samplers."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._seed = seed

    def reseed(self, seed: int) -> Sampler[T]:
        """Reset the random state with a new seed."""
        self._rng = random.Random(seed)
        self._seed = seed
        return self

    @abstractmethod
    def sample(self) -> T:
        """Draw a single sample."""
        ...

    def sample_n(self, n: int) -> list[T]:
        """Draw n samples."""
        return [self.sample() for _ in range(n)]

    @abstractmethod
    def values(self) -> list:
        """Return all possible values."""
        ...

    def __len__(self) -> int:
        return len(self.values())


class WeightedSampler(Sampler[T]):
    """Sampler with optional weights."""

    def __init__(
        self,
        items: Sequence[T],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)
        self._items = list(items)
        self._weights = list(weights) if weights else None

    def sample(self) -> T:
        if self._weights:
            return self._rng.choices(self._items, weights=self._weights, k=1)[0]
        return self._rng.choice(self._items)

    def values(self) -> list[T]:
        return self._items


# =============================================================================
# LCC Sampler (21 main classes)
# =============================================================================

LCC_DATA: dict[str, str] = {
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


@dataclass
class LCCClass:
    """An LCC main class, optionally narrowed to one of its subclasses.

    ``subclass`` carries the v0.4 difficulty tier (docs/data_plan_v0.4.md
    section 6): ``code`` stays the single-letter main class so every existing
    label, topic-domain lookup and task keeps working, while ``subclass`` adds
    the finer LC code ("QA" under "Q") that demands within-domain
    discrimination. It defaults to ``None``, so a v0.3.1 draw is unchanged in
    every observable way -- including ``__str__`` and the derived ``uri``.
    """

    code: str
    name: str
    uri: str | None = None
    subclass: str | None = None
    subclass_name: str | None = None

    def __str__(self) -> str:
        if self.subclass:
            return f"{self.subclass}: {self.subclass_name or self.name}"
        return f"{self.code}: {self.name}"

    def __post_init__(self):
        if self.uri is None:
            self.uri = f"http://id.loc.gov/authorities/classification/{self.code}"


class LCCSampler(Sampler[LCCClass]):
    """Sample from LCC main classes."""

    def __init__(
        self,
        classes: list[str] | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)
        if classes:
            self._classes = [LCCClass(c, LCC_DATA[c]) for c in classes if c in LCC_DATA]
        else:
            self._classes = [LCCClass(c, n) for c, n in LCC_DATA.items()]

    def sample(self) -> LCCClass:
        return self._rng.choice(self._classes)

    def values(self) -> list[LCCClass]:
        return self._classes

    @staticmethod
    def all_codes() -> list[str]:
        """Get all LCC codes."""
        return list(LCC_DATA.keys())


# =============================================================================
# LCC Subclass Sampler (v0.4 Phase 2 difficulty tier)
# =============================================================================
#
# The flagship `lcc_classification` task is lexically saturated: TF-IDF+LR
# reaches 0.892 macro-F1 and still scores 0.754 on 22-word documents, because
# the 21 top-level classes are decodable from domain vocabulary alone. LCC
# *subclasses* are not: QA/QC/QH all speak the vocabulary of science, KF/KJ/KZ
# all speak the vocabulary of law. See docs/data_plan_v0.4.md section 6.

# Ranked MARC frequency table (ids, frequencies, ranks) and the LC enrichment
# pass over it (captions, parent class letters, URIs, scope notes).
LCC_SUBCLASS_SOURCE: Final = "lcc_subclass_top100.json"
LCC_SUBCLASS_ENRICHED_SOURCE: Final = "enriched/lcc_subclass_top100.json"

# Not LCC codes. `IN`, `PAR` and `NOT` are extraction artifacts in the MARC
# frequency table -- almost certainly truncated call-number text, not
# classification -- and `IN` is even assigned parent letter "I", which LCC does
# not use. data_plan_v0.4.md section 4.1 flags all three explicitly. They are
# excluded from every pool this module builds, unconditionally.
LCC_SUBCLASS_EXTRACTION_ARTIFACTS: Final[frozenset[str]] = frozenset(
    {"IN", "PAR", "NOT"}
)


@dataclass(frozen=True)
class LCCSubclass:
    """One LCC subclass with its parent main class and LC caption."""

    code: str
    caption: str | None
    class_code: str
    class_name: str
    frequency: int = 0
    rank: int = 0
    uri: str | None = None
    description: str | None = None

    @property
    def is_main_class(self) -> bool:
        """Whether this 'subclass' is really the bare main class letter.

        The frequency table lists 16 single-letter codes (works classed at the
        top level, e.g. bare "Q"). They are a poor difficulty-tier label: "Q"
        versus "QA" is genuinely ambiguous for a generated document, so the
        default pool leaves them out.
        """
        return len(self.code) == 1

    def to_lcc_class(self) -> LCCClass:
        """Materialize as an ``LCCClass`` carrying this subclass."""
        return LCCClass(
            code=self.class_code,
            name=self.class_name,
            subclass=self.code,
            subclass_name=self.caption,
        )

    def __str__(self) -> str:
        return f"{self.code}: {self.caption or self.class_name}"


@lru_cache(maxsize=4)
def _load_lcc_subclass_table(data_dir: str) -> tuple[LCCSubclass, ...]:
    """Read the ranked subclass table, overlaid with the LC enrichment pass.

    Rank order from the frequency table is preserved. Extraction artifacts and
    codes whose parent letter is not a real LCC main class are dropped here, so
    no caller can reintroduce them.
    """
    base = Path(data_dir) / "taxonomies" / LCC_SUBCLASS_SOURCE
    if not base.exists():
        return ()

    with open(base, encoding="utf-8") as handle:
        rows = json.load(handle).get("labels", [])

    enriched_path = Path(data_dir) / "taxonomies" / LCC_SUBCLASS_ENRICHED_SOURCE
    enriched: dict[str, dict] = {}
    if enriched_path.exists():
        with open(enriched_path, encoding="utf-8") as handle:
            enriched = {
                str(row.get("id")): row
                for row in json.load(handle).get("labels", [])
                if row.get("id")
            }

    pool: list[LCCSubclass] = []
    for index, row in enumerate(rows):
        code = str(row.get("id") or row.get("label") or "").strip()
        if not code or code in LCC_SUBCLASS_EXTRACTION_ARTIFACTS:
            continue
        extra = enriched.get(code, {})
        class_code = str(extra.get("class_letter") or code[0]).strip()
        if class_code not in LCC_DATA:
            # Belt-and-braces against a future artifact that is not on the
            # hard-coded list: LCC has no I, O, W, X or Y main class.
            continue
        pool.append(
            LCCSubclass(
                code=code,
                caption=(str(extra["caption"]) if extra.get("caption") else None),
                class_code=class_code,
                class_name=LCC_DATA[class_code],
                frequency=int(row.get("frequency", 0)),
                rank=int(row.get("rank", index + 1)),
                uri=(str(extra["uri"]) if extra.get("uri") else None),
                description=(
                    str(extra["description"]) if extra.get("description") else None
                ),
            )
        )
    return tuple(pool)


def load_lcc_subclass_pool(
    size: int | None = None,
    data_dir: str | None = None,
    require_description: bool = True,
    include_main_classes: bool = False,
) -> list[LCCSubclass]:
    """Load the LCC subclass label pool, in MARC frequency-rank order.

    Args:
        size: Keep only the top `size` entries by rank after filtering. ``None``
            keeps the whole filtered pool.
        data_dir: Data directory containing ``taxonomies/``. Defaults to the
            repository's ``data/``.
        require_description: Drop subclasses with no LC description. This is the
            default because an undescribed code cannot be prompted for -- the
            generator would fall back to the bare code name, which is exactly
            the self-labeling the prompt forbids. Only `JX` (a discontinued
            subclass that the classification API no longer resolves) is lost.
        include_main_classes: Include the 16 single-letter codes. Off by
            default; see :attr:`LCCSubclass.is_main_class`.

    Returns:
        Subclasses in rank order.

    Raises:
        FileNotFoundError: If the frequency table is missing.
        ValueError: If `size` is not positive, or exceeds the filtered pool.
    """
    resolved = data_dir if data_dir is not None else str(_lcc_default_data_dir())
    table = _load_lcc_subclass_table(resolved)
    if not table:
        raise FileNotFoundError(
            f"LCC subclass table not found or empty: "
            f"{Path(resolved) / 'taxonomies' / LCC_SUBCLASS_SOURCE}"
        )

    pool = [
        entry
        for entry in table
        if (include_main_classes or not entry.is_main_class)
        and (not require_description or entry.description)
    ]

    if size is None:
        return pool
    if size < 1:
        raise ValueError(f"pool size must be >= 1, got {size}")
    if size > len(pool):
        raise ValueError(
            f"pool size {size} exceeds the {len(pool)} usable LCC subclasses "
            f"(require_description={require_description}, "
            f"include_main_classes={include_main_classes})"
        )
    return pool[:size]


def _lcc_default_data_dir() -> Path:
    """Repository ``data/`` directory (shared with `lc_data`)."""
    from .lc_data import _default_data_dir

    return _default_data_dir()


class LCCSubclassSampler(Sampler[LCCClass]):
    """Sample LCC subclasses, uniformly by default.

    **Uniform, not frequency-weighted.** The MARC table is wildly skewed --
    `KF` alone is 122,484 of the reference corpus, 12x the next code and more
    than the other 99 combined -- so frequency sampling would produce a corpus
    that is mostly American law and would leave most subclasses with too few
    documents to score. Uniform sampling is what the top-level `LCCSampler`
    already does, and section 6 of the data plan requires it here. Frequency
    weighting is reachable via ``weighting="frequency"`` for anyone who wants
    to measure the skew, but it is not the corpus-building setting.

    Every sample is an :class:`LCCClass` whose ``code`` is the parent main
    class and whose ``subclass`` is the finer code, so a subclass-bearing
    document still carries a valid top-level `lcc_code` label.

    Composing it with `DocumentSampler` requires care about draw order.
    `DocumentSampler` draws topics *for the class it drew*, so the subclass has
    to be chosen first and the document constrained to its parent -- swapping
    the class in afterwards would leave a document with, say, literature topics
    under a mathematics label::

        subclasses = LCCSubclassSampler(seed=seed)
        lcc = subclasses.sample()                       # e.g. Q + subclass QA
        doc = sampler.with_lcc_classes([lcc.code]).sample()
        doc.lcc = lcc                                   # same code/name, plus
                                                        # the finer label
        spec = DocumentSpec.from_document(doc, length, register)
    """

    def __init__(
        self,
        codes: list[str] | None = None,
        seed: int | None = None,
        pool_size: int | None = None,
        data_dir: str | None = None,
        require_description: bool = True,
        include_main_classes: bool = False,
        weighting: Literal["uniform", "frequency"] = "uniform",
    ):
        super().__init__(seed)
        self._weighting = weighting

        pool = load_lcc_subclass_pool(
            size=pool_size,
            data_dir=data_dir,
            require_description=require_description,
            include_main_classes=include_main_classes,
        )
        if codes:
            wanted = set(codes)
            pool = [entry for entry in pool if entry.code in wanted]
        if not pool:
            raise ValueError("LCC subclass pool is empty after filtering")

        self._pool = pool
        self._classes = [entry.to_lcc_class() for entry in pool]
        self._weights = (
            [float(entry.frequency) or 1.0 for entry in pool]
            if weighting == "frequency"
            else None
        )

    def sample(self) -> LCCClass:
        if self._weights:
            return self._rng.choices(self._classes, weights=self._weights, k=1)[0]
        return self._rng.choice(self._classes)

    def values(self) -> list[LCCClass]:
        return self._classes

    def subclasses(self) -> list[LCCSubclass]:
        """The underlying pool entries, in rank order."""
        return list(self._pool)

    def codes(self) -> list[str]:
        """Subclass codes in the pool, in rank order."""
        return [entry.code for entry in self._pool]


# =============================================================================
# LCGFT Sampler (Genre/Form Terms)
# =============================================================================

LCGFT_DATA: dict[str, list[str]] = {
    "Informational works": [
        "Abstracts",
        "Academic theses",
        "Administrative decisions",
        "Administrative regulations",
        "Biographies",
        "Blogs",
        "Case studies",
        "Conference papers and proceedings",
        "Data sets",
        "Databases",
        "Essays",
        "Infographics",
        "Maps",
        "News articles",
        "Personal narratives",
        "Policy briefs",
        "Press releases",
        "Reference works",
        "Reviews",
        "Statistics",
        "Technical reports",
        "Surveys",
    ],
    "Law materials": [
        "Administrative regulations",
        "Casebooks (Law)",
        "Constitutions",
        "Court decisions and opinions",
        "Legal forms",
        "Legislative materials",
        "Statutes and codes",
        "Treaties",
        "Contracts",
        "Legal briefs",
    ],
    "Instructional and educational works": [
        "Cookbooks",
        "Course materials",
        "Educational films",
        "Examinations",
        "FAQs",
        "Handbooks and manuals",
        "Lectures",
        "Lesson plans",
        "Textbooks",
        "Tutorials",
        "Study guides",
        "How-to guides",
        "Workbooks",
    ],
    "Literature": [
        "Drama",
        "Fiction",
        "Novels",
        "Poetry",
        "Short stories",
        "Plays",
        "Screenplays",
        "Comics (Graphic works)",
        "Folk literature",
        "Sagas",
        "Satire",
        "Mystery fiction",
        "Science fiction",
        "Fantasy fiction",
    ],
    "Discursive works": [
        "Commentaries",
        "Debates",
        "Editorials",
        "Interviews",
        "Lectures",
        "Letters to the editor",
        "Panel discussions",
        "Speeches",
        "Criticism",
        "Opinion pieces",
        "Analysis",
    ],
    "Creative nonfiction": [
        "Biographies",
        "Diaries",
        "Essays",
        "Memoirs",
        "Personal narratives",
        "Travel writing",
        "True crime stories",
        "Journalism",
        "Profiles",
    ],
    "Sound recordings": [
        "Audiobooks",
        "Podcasts",
        "Radio programs",
        "Music recordings",
        "Interviews",
        "Lectures",
        "Field recordings",
        "Oral histories",
    ],
    "Visual works": [
        "Art",
        "Drawings",
        "Photographs",
        "Motion pictures",
        "Television programs",
        "Video recordings",
        "Illustrations",
        "Infographics",
        "Charts",
        "Diagrams",
    ],
    "Music": [
        "Art music",
        "Folk music",
        "Popular music",
        "Sacred music",
        "Songs",
        "Instrumental music",
        "Orchestral music",
        "Chamber music",
        "Opera",
    ],
    "Religious materials": [
        "Sacred works",
        "Sermons",
        "Prayers",
        "Devotional literature",
        "Theological works",
        "Liturgical texts",
        "Commentaries",
    ],
    "Ephemera": [
        "Broadsides",
        "Calendars",
        "Greeting cards",
        "Menus",
        "Postcards",
        "Posters",
        "Pamphlets",
        "Flyers",
        "Brochures",
    ],
    "Commemorative works": [
        "Eulogies",
        "Festschriften",
        "Memorial books",
        "Obituaries",
        "Yearbooks",
        "Anniversary publications",
        "Tributes",
    ],
    "Cartographic materials": [
        "Maps",
        "Atlases",
        "Globes",
        "Nautical charts",
        "Aerial photographs",
        "Satellite imagery",
        "Floor plans",
        "Architectural drawings",
    ],
    "Recreational works": [
        "Games",
        "Puzzles",
        "Activity books",
        "Humor",
        "Jokes",
        "Riddles",
    ],
}


@dataclass
class LCGFTTerm:
    """An LCGFT genre/form term."""

    category: str
    form: str
    uri: str | None = None
    id: str | None = None

    def __str__(self) -> str:
        return f"{self.category} > {self.form}"


class LCGFTSampler(Sampler[LCGFTTerm]):
    """Sample from LCGFT genre/form terms.

    Two pool sources, selected by `pool_size`:

    - ``pool_size="v0.3.1"`` (default): the original 133-form hand-curated
      pool grouped into 14 categories (`LCGFT_DATA`). Frozen so the v0.3.1
      corpus stays exactly reproducible.
    - ``pool_size=<int>``: the 133 curated forms extended with the next most
      frequent forms from `data/taxonomies/lcgft.json` (554 available; see
      `lc_data.build_expanded_form_pool`) until `pool_size` distinct forms
      are reached. Added forms keep a category where the LCGFT hierarchy
      file resolves one unambiguously; most do not (category coverage for
      added forms is ~19% at pool_size=300) and fall under "Uncategorized".
      Must be >= 133 (the curated count) and <= 554.

    `categories` filters the resulting pool to the named categories, in
    either mode.

    Sampling is two-stage in both modes: a category is chosen uniformly,
    then a form is chosen from within it (uniformly, or by MARC frequency
    if ``weighting="frequency"`` -- see the module-level docs on
    `TopicSampler` for why uniform is the default).
    """

    def __init__(
        self,
        categories: list[str] | None = None,
        seed: int | None = None,
        pool_size: int | Literal["v0.3.1"] = PRESET_V0_3_1,
        weighting: Literal["uniform", "frequency"] = "uniform",
        data_dir: str | None = None,
    ):
        super().__init__(seed)
        self._pool_size = pool_size
        self._weighting = weighting
        self._weights_by_form: dict[str, float] | None = None

        if pool_size == PRESET_V0_3_1:
            if categories:
                self._data = {k: v for k, v in LCGFT_DATA.items() if k in categories}
            else:
                self._data = LCGFT_DATA
        else:
            from .lc_data import build_expanded_form_pool, load_form_pool

            expanded = build_expanded_form_pool(pool_size, data_dir=data_dir)
            if categories:
                expanded = {k: v for k, v in expanded.items() if k in categories}
            self._data = expanded

            if weighting == "frequency":
                freqs = {
                    e.label.lower(): float(e.frequency)
                    for e in load_form_pool(pool_size, data_dir=data_dir)
                }
                # Forms added beyond the ranked slice used to build the
                # curated+extended pool (i.e. curated forms not already in
                # the top `pool_size` ranked forms) get the lowest observed
                # frequency as a conservative weight.
                fallback = min(freqs.values()) if freqs else 1.0
                self._weights_by_form = {
                    form: freqs.get(form.lower(), fallback)
                    for forms in self._data.values()
                    for form in forms
                }

        self._categories = list(self._data.keys())

    def sample(self) -> LCGFTTerm:
        category = self._rng.choice(self._categories)
        forms = self._data[category]
        if self._weights_by_form:
            weights = [self._weights_by_form.get(f, 1.0) for f in forms]
            form = (
                self._rng.choices(forms, weights=weights, k=1)[0] if forms else category
            )
        else:
            form = self._rng.choice(forms) if forms else category
        return LCGFTTerm(category, form)

    def sample_from_category(self, category: str) -> LCGFTTerm:
        """Sample a form from a specific category."""
        forms = self._data.get(category, [category])
        form = self._rng.choice(forms) if forms else category
        return LCGFTTerm(category, form)

    def values(self) -> list[str]:
        return self._categories

    def all_forms(self) -> list[LCGFTTerm]:
        """Get all category/form combinations."""
        return [
            LCGFTTerm(cat, form) for cat, forms in self._data.items() for form in forms
        ]

    @staticmethod
    def all_categories() -> list[str]:
        """Get all category names."""
        return list(LCGFT_DATA.keys())


# =============================================================================
# Topic Sampler (LCSH-style topics by domain)
# =============================================================================

TOPICS_BY_DOMAIN: dict[str, list[str]] = {
    "general": [
        "Information",
        "Knowledge",
        "Research",
        "Analysis",
        "Methodology",
    ],
    "social_sciences": [
        "Economics",
        "Sociology",
        "Psychology",
        "Anthropology",
        "Demographics",
        "Public policy",
        "Social welfare",
        "Labor",
        "Commerce",
        "Finance",
        "Statistics",
        "Surveys",
        "Population",
        "Immigration",
        "Poverty",
    ],
    "law": [
        "Constitutional law",
        "Criminal law",
        "Civil law",
        "Contracts",
        "Property",
        "Torts",
        "Administrative law",
        "International law",
        "Human rights",
        "Intellectual property",
        "Environmental law",
    ],
    "science": [
        "Biology",
        "Chemistry",
        "Physics",
        "Mathematics",
        "Geology",
        "Astronomy",
        "Ecology",
        "Genetics",
        "Climate",
        "Evolution",
        "Neuroscience",
        "Quantum mechanics",
        "Thermodynamics",
    ],
    "technology": [
        "Engineering",
        "Computer science",
        "Software",
        "Artificial intelligence",
        "Machine learning",
        "Robotics",
        "Biotechnology",
        "Nanotechnology",
        "Cybersecurity",
        "Data science",
        "Cloud computing",
        "Networks",
    ],
    "medicine": [
        "Public health",
        "Epidemiology",
        "Diseases",
        "Therapeutics",
        "Surgery",
        "Pharmacology",
        "Mental health",
        "Nutrition",
        "Pediatrics",
        "Oncology",
        "Cardiology",
        "Immunology",
    ],
    "humanities": [
        "Philosophy",
        "Ethics",
        "History",
        "Literature",
        "Languages",
        "Art",
        "Music",
        "Religion",
        "Culture",
        "Aesthetics",
    ],
    "business": [
        "Management",
        "Marketing",
        "Accounting",
        "Finance",
        "Strategy",
        "Entrepreneurship",
        "Operations",
        "Human resources",
        "Leadership",
        "Innovation",
        "Supply chain",
        "E-commerce",
    ],
    "politics": [
        "Government",
        "Elections",
        "Political parties",
        "Public administration",
        "International relations",
        "Diplomacy",
        "Security",
        "Defense",
        "Democracy",
        "Authoritarianism",
        "Nationalism",
        "Globalization",
    ],
    "environment": [
        "Climate change",
        "Conservation",
        "Sustainability",
        "Pollution",
        "Biodiversity",
        "Renewable energy",
        "Ecosystems",
        "Wildlife",
        "Deforestation",
        "Ocean conservation",
        "Carbon emissions",
    ],
}

# Map LCC classes to topic domains
LCC_TO_DOMAIN: dict[str, list[str]] = {
    "A": ["general"],
    "B": ["humanities"],
    "C": ["humanities", "social_sciences"],
    "D": ["humanities", "politics"],
    "E": ["humanities", "politics"],
    "F": ["humanities"],
    "G": ["social_sciences", "environment"],
    "H": ["social_sciences", "business"],
    "J": ["politics", "law"],
    "K": ["law"],
    "L": ["social_sciences", "humanities"],
    "M": ["humanities"],
    "N": ["humanities"],
    "P": ["humanities"],
    "Q": ["science"],
    "R": ["medicine", "science"],
    "S": ["environment", "science"],
    "T": ["technology", "science"],
    "U": ["politics"],
    "V": ["politics"],
    "Z": ["general", "technology"],
}


class TopicSampler(Sampler[str]):
    """Sample LCSH-style topics, optionally filtered by domain.

    Two pool sources, selected by `pool_size`:

    - ``pool_size="v0.3.1"`` (default): the original 112-topic hand-curated
      pool grouped into 10 domains (`TOPICS_BY_DOMAIN`). Several of these,
      especially in the "humanities" domain (e.g. "Art", "History",
      "Culture", "Religion"), are broad top-level abstractions rather than
      specific subject headings -- see docs/data_plan_v0.4.md section 4.1.
      Frozen so the v0.3.1 corpus stays exactly reproducible. `domains` /
      `lcc_class` filtering only applies to this pool.
    - ``pool_size=<int>``: the top-N specific LCSH topical subject headings
      by MARC corpus frequency (e.g. "Flood insurance", "Groundwater"),
      loaded from `data/taxonomies/lcsh_topical_top*.json` via
      `lc_data.load_topic_pool` (1 <= N <= `lc_data.TOPIC_POOL_MAX` == 2000).
      This pool carries no domain grouping, so `domains` / `lcc_class` are
      ignored when it is used.

    Sampling within either pool is **uniform** by default
    (``weighting="uniform"``). Pass ``weighting="frequency"`` to sample
    proportional to raw MARC frequency instead. Uniform is the default
    deliberately: raw frequency is extremely skewed even within these
    curated top-N lists (e.g. rank-1 "Flood insurance" is ~22x the
    rank-500 cutoff frequency in the topical list, and the skew is far
    worse for geographic headings), so frequency weighting would recreate
    the "corpus collapses onto a handful of head terms" problem that this
    pool expansion exists to fix. It exists as an option for callers who
    deliberately want a naturalistic, imbalanced label distribution.
    """

    def __init__(
        self,
        domains: list[str] | None = None,
        lcc_class: str | None = None,
        seed: int | None = None,
        pool_size: int | Literal["v0.3.1"] = PRESET_V0_3_1,
        weighting: Literal["uniform", "frequency"] = "uniform",
        data_dir: str | None = None,
    ):
        super().__init__(seed)
        self._pool_size = pool_size
        self._weighting = weighting
        self._weights: list[float] | None = None

        if pool_size == PRESET_V0_3_1:
            # Determine which domains to use
            if domains:
                self._domains = domains
            elif lcc_class and lcc_class in LCC_TO_DOMAIN:
                self._domains = LCC_TO_DOMAIN[lcc_class]
            else:
                self._domains = list(TOPICS_BY_DOMAIN.keys())

            # Collect topics from those domains
            self._topics = []
            for domain in self._domains:
                self._topics.extend(TOPICS_BY_DOMAIN.get(domain, []))

            if not self._topics:
                self._topics = TOPICS_BY_DOMAIN["general"]
        else:
            from .lc_data import load_topic_pool

            self._domains = []
            entries = load_topic_pool(pool_size, data_dir=data_dir)
            self._topics = [e.label for e in entries]
            if weighting == "frequency":
                self._weights = [float(e.frequency) for e in entries]

    def sample(self) -> str:
        if self._weights:
            return self._rng.choices(self._topics, weights=self._weights, k=1)[0]
        return self._rng.choice(self._topics)

    def sample_n_unique(self, n: int) -> list[str]:
        """Sample n unique topics (weighting, if any, is ignored -- this
        draws a uniform simple random sample without replacement)."""
        n = min(n, len(self._topics))
        return self._rng.sample(self._topics, n)

    def values(self) -> list[str]:
        return self._topics

    @staticmethod
    def all_domains() -> list[str]:
        """Get all domain names."""
        return list(TOPICS_BY_DOMAIN.keys())

    @staticmethod
    def for_lcc(lcc_code: str, seed: int | None = None) -> TopicSampler:
        """Create a topic sampler for a specific LCC class."""
        return TopicSampler(lcc_class=lcc_code, seed=seed)


# =============================================================================
# Audience Sampler (LCDGT-style demographics)
# =============================================================================

AUDIENCES: list[str] = [
    # Age groups
    "Children",
    "Adolescents",
    "Young adults",
    "Adults",
    "Older adults",
    # Educational
    "Students",
    "Graduate students",
    "Researchers",
    "Scholars",
    # Professional
    "Professionals",
    "Practitioners",
    "Specialists",
    "Experts",
    "Scientists",
    "Engineers",
    "Physicians",
    "Lawyers",
    "Educators",
    "Business professionals",
    "Policy makers",
    # General
    "General public",
    "Beginners",
    "Non-specialists",
    "Lay readers",
]


class AudienceSampler(Sampler[str | None]):
    """Sample target audiences (LCDGT-style)."""

    def __init__(
        self,
        audiences: list[str] | None = None,
        include_none: bool = True,
        none_probability: float = 0.3,
        seed: int | None = None,
    ):
        super().__init__(seed)
        self._audiences = audiences or AUDIENCES
        self._include_none = include_none
        self._none_prob = none_probability

    def sample(self) -> str | None:
        if self._include_none and self._rng.random() < self._none_prob:
            return None
        return self._rng.choice(self._audiences)

    def values(self) -> list[str]:
        return self._audiences


# =============================================================================
# Geographic Sampler
# =============================================================================

GEOGRAPHIC_AREAS: dict[str, list[str]] = {
    "countries": [
        "United States",
        "United Kingdom",
        "Canada",
        "Australia",
        "Germany",
        "France",
        "Japan",
        "China",
        "India",
        "Brazil",
        "Mexico",
        "Italy",
        "Spain",
        "South Korea",
        "Russia",
    ],
    "us_states": [
        "California",
        "New York",
        "Texas",
        "Florida",
        "Illinois",
        "Pennsylvania",
        "Ohio",
        "Georgia",
        "North Carolina",
        "Michigan",
    ],
    "regions": [
        "North America",
        "South America",
        "Europe",
        "Asia",
        "Africa",
        "Middle East",
        "Southeast Asia",
        "Central America",
        "Caribbean",
    ],
    "cities": [
        "New York City",
        "Los Angeles",
        "Chicago",
        "London",
        "Paris",
        "Tokyo",
        "Beijing",
        "Mumbai",
        "São Paulo",
        "Berlin",
    ],
}


class GeographicSampler(Sampler[str | None]):
    """Sample geographic areas.

    Two pool sources, selected by `pool_size`:

    - ``pool_size="v0.3.1"`` (default): the original 44-area hand-curated
      pool grouped into 4 area types (`GEOGRAPHIC_AREAS`). Frozen so the
      v0.3.1 corpus stays exactly reproducible. `area_types` filtering only
      applies to this pool.
    - ``pool_size=<int>``: the top-N LCSH geographic headings by MARC corpus
      frequency, loaded from `data/taxonomies/lcsh_geo_top500.json` via
      `lc_data.load_geographic_pool` (1 <= N <= `lc_data.GEOGRAPHIC_POOL_MAX`
      == 500). This pool carries no area-type grouping, so `area_types` is
      ignored when it is used.

    Sampling is **uniform** by default (see `TopicSampler` for the reasoning
    -- it applies here even more strongly: "United States" alone accounts
    for ~45% of total MARC frequency mass in the top-500 geographic list
    (143,648 of 318,405; the #2 entry has only 6,172), so frequency
    weighting would make most of the other 499 headings vanishingly rare).
    Pass ``weighting="frequency"`` to opt into
    frequency-proportional sampling instead.
    """

    def __init__(
        self,
        area_types: list[str] | None = None,
        include_none: bool = True,
        none_probability: float = 0.4,
        seed: int | None = None,
        pool_size: int | Literal["v0.3.1"] = PRESET_V0_3_1,
        weighting: Literal["uniform", "frequency"] = "uniform",
        data_dir: str | None = None,
    ):
        super().__init__(seed)
        self._pool_size = pool_size
        self._weighting = weighting
        self._weights: list[float] | None = None

        if pool_size == PRESET_V0_3_1:
            if area_types:
                self._areas = []
                for t in area_types:
                    self._areas.extend(GEOGRAPHIC_AREAS.get(t, []))
            else:
                self._areas = [
                    area for areas in GEOGRAPHIC_AREAS.values() for area in areas
                ]
        else:
            from .lc_data import load_geographic_pool

            entries = load_geographic_pool(pool_size, data_dir=data_dir)
            self._areas = [e.label for e in entries]
            if weighting == "frequency":
                self._weights = [float(e.frequency) for e in entries]

        self._include_none = include_none
        self._none_prob = none_probability

    def sample(self) -> str | None:
        if self._include_none and self._rng.random() < self._none_prob:
            return None
        if self._weights:
            return self._rng.choices(self._areas, weights=self._weights, k=1)[0]
        return self._rng.choice(self._areas)

    def sample_n(self, n: int) -> list[str | None]:
        """Sample n geographic areas."""
        return [self.sample() for _ in range(n)]

    def sample_n_non_null(self, n: int) -> list[str]:
        """Sample n geographic areas (no None values)."""
        n = min(n, len(self._areas))
        return self._rng.sample(self._areas, n)

    def values(self) -> list[str]:
        return self._areas

    @staticmethod
    def area_types() -> list[str]:
        """Get available area type categories."""
        return list(GEOGRAPHIC_AREAS.keys())


# =============================================================================
# Real LC Data Samplers (with URIs from id.loc.gov)
# =============================================================================


@dataclass
class LCTerm:
    """A term from any LC vocabulary with URI."""

    id: str
    label: str
    uri: str
    alt_labels: list[str] = field(default_factory=list)
    broader: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.label


class RealLCGFTSampler(Sampler[LCTerm]):
    """Sample from real LCGFT data with URIs.

    Uses actual LC Genre/Form Terms from id.loc.gov.
    Can be filtered to top-N by frequency in government documents.
    """

    _terms: list[LCTerm]

    def __init__(
        self,
        data_dir: str | None = None,
        top_n: int | None = 50,
        seed: int | None = None,
    ):
        super().__init__(seed)
        from .lc_data import LCTerm as LCDataTerm
        from .lc_data import load_lc_data

        loader = load_lc_data(data_dir)

        if top_n:
            # Use frequency-ranked terms
            raw_terms = [term for term, _ in loader.get_top_lcgft(top_n)]
        else:
            # Use all terms
            raw_terms = list(loader.lcgft.values())

        # Convert to LCTerm dataclass (use existing terms if already LCTerm)
        converted_terms: list[LCTerm] = []
        for t in raw_terms:
            if isinstance(t, LCDataTerm):
                converted_terms.append(
                    LCTerm(
                        id=t.id,
                        label=t.label,
                        uri=t.uri,
                        alt_labels=t.alt_labels,
                        broader=t.broader,
                    )
                )
        self._terms = converted_terms

    def sample(self) -> LCTerm:
        return self._rng.choice(self._terms)

    def values(self) -> list[LCTerm]:
        return self._terms


class RealLCSHSampler(Sampler[LCTerm]):
    """Sample from real LCSH data with URIs.

    Uses actual LC Subject Headings from id.loc.gov.
    Can be filtered to top-N by frequency in government documents.
    """

    _terms: list[LCTerm]

    def __init__(
        self,
        data_dir: str | None = None,
        top_n: int | None = 100,
        seed: int | None = None,
    ):
        super().__init__(seed)
        from .lc_data import LCTerm as LCDataTerm
        from .lc_data import load_lc_data

        loader = load_lc_data(data_dir)

        if top_n:
            raw_terms = [term for term, _ in loader.get_top_lcsh(top_n)]
        else:
            # Warning: LCSH has 500k+ terms
            raw_terms = list(loader.lcsh.values())[:1000]

        converted_terms: list[LCTerm] = []
        for t in raw_terms:
            if isinstance(t, LCDataTerm):
                converted_terms.append(
                    LCTerm(
                        id=t.id,
                        label=t.label,
                        uri=t.uri,
                        alt_labels=t.alt_labels,
                        broader=t.broader,
                    )
                )
        self._terms = converted_terms

    def sample(self) -> LCTerm:
        return self._rng.choice(self._terms)

    def sample_n_unique(self, n: int) -> list[LCTerm]:
        """Sample n unique terms."""
        n = min(n, len(self._terms))
        return self._rng.sample(self._terms, n)

    def values(self) -> list[LCTerm]:
        return self._terms


class RealLCDGTSampler(Sampler[LCTerm]):
    """Sample from real LCDGT data with URIs.

    Uses actual LC Demographic Group Terms from id.loc.gov.
    """

    _terms: list[LCTerm]

    def __init__(
        self,
        data_dir: str | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)
        from .lc_data import LCTerm as LCDataTerm
        from .lc_data import load_lc_data

        loader = load_lc_data(data_dir)
        converted_terms: list[LCTerm] = []
        for t in loader.lcdgt.values():
            if isinstance(t, LCDataTerm):
                converted_terms.append(
                    LCTerm(
                        id=t.id,
                        label=t.label,
                        uri=t.uri,
                        alt_labels=t.alt_labels,
                        broader=t.broader,
                    )
                )
        self._terms = converted_terms

    def sample(self) -> LCTerm:
        return self._rng.choice(self._terms)

    def values(self) -> list[LCTerm]:
        return self._terms

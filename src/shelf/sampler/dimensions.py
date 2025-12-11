"""
Individual dimension samplers for LC taxonomies.

Each sampler is independent and can be used standalone or composed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import TypeVar, Generic, Sequence

T = TypeVar("T")


# =============================================================================
# Base Sampler
# =============================================================================

class Sampler(ABC, Generic[T]):
    """Abstract base for all samplers."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._seed = seed

    def reseed(self, seed: int) -> "Sampler[T]":
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
    """An LCC main class."""
    code: str
    name: str
    uri: str | None = None

    def __str__(self) -> str:
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
# LCGFT Sampler (Genre/Form Terms)
# =============================================================================

LCGFT_DATA: dict[str, list[str]] = {
    "Informational works": [
        "Abstracts", "Academic theses", "Administrative decisions", "Administrative regulations",
        "Biographies", "Blogs", "Case studies", "Conference papers and proceedings",
        "Data sets", "Databases", "Essays", "Infographics", "Maps", "News articles",
        "Personal narratives", "Policy briefs", "Press releases", "Reference works",
        "Reviews", "Statistics", "Technical reports", "Surveys",
    ],
    "Law materials": [
        "Administrative regulations", "Casebooks (Law)", "Constitutions",
        "Court decisions and opinions", "Legal forms", "Legislative materials",
        "Statutes and codes", "Treaties", "Contracts", "Legal briefs",
    ],
    "Instructional and educational works": [
        "Cookbooks", "Course materials", "Educational films", "Examinations",
        "FAQs", "Handbooks and manuals", "Lectures", "Lesson plans", "Textbooks",
        "Tutorials", "Study guides", "How-to guides", "Workbooks",
    ],
    "Literature": [
        "Drama", "Fiction", "Novels", "Poetry", "Short stories", "Plays",
        "Screenplays", "Comics (Graphic works)", "Folk literature", "Sagas",
        "Satire", "Mystery fiction", "Science fiction", "Fantasy fiction",
    ],
    "Discursive works": [
        "Commentaries", "Debates", "Editorials", "Interviews", "Lectures",
        "Letters to the editor", "Panel discussions", "Speeches", "Criticism",
        "Opinion pieces", "Analysis",
    ],
    "Creative nonfiction": [
        "Biographies", "Diaries", "Essays", "Memoirs", "Personal narratives",
        "Travel writing", "True crime stories", "Journalism", "Profiles",
    ],
    "Sound recordings": [
        "Audiobooks", "Podcasts", "Radio programs", "Music recordings",
        "Interviews", "Lectures", "Field recordings", "Oral histories",
    ],
    "Visual works": [
        "Art", "Drawings", "Photographs", "Motion pictures", "Television programs",
        "Video recordings", "Illustrations", "Infographics", "Charts", "Diagrams",
    ],
    "Music": [
        "Art music", "Folk music", "Popular music", "Sacred music", "Songs",
        "Instrumental music", "Orchestral music", "Chamber music", "Opera",
    ],
    "Religious materials": [
        "Sacred works", "Sermons", "Prayers", "Devotional literature",
        "Theological works", "Liturgical texts", "Commentaries",
    ],
    "Ephemera": [
        "Broadsides", "Calendars", "Greeting cards", "Menus", "Postcards",
        "Posters", "Pamphlets", "Flyers", "Brochures",
    ],
    "Commemorative works": [
        "Eulogies", "Festschriften", "Memorial books", "Obituaries", "Yearbooks",
        "Anniversary publications", "Tributes",
    ],
    "Cartographic materials": [
        "Maps", "Atlases", "Globes", "Nautical charts", "Aerial photographs",
        "Satellite imagery", "Floor plans", "Architectural drawings",
    ],
    "Recreational works": [
        "Games", "Puzzles", "Activity books", "Humor", "Jokes", "Riddles",
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
    """Sample from LCGFT genre/form terms."""

    def __init__(
        self,
        categories: list[str] | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)
        if categories:
            self._data = {k: v for k, v in LCGFT_DATA.items() if k in categories}
        else:
            self._data = LCGFT_DATA
        self._categories = list(self._data.keys())

    def sample(self) -> LCGFTTerm:
        category = self._rng.choice(self._categories)
        forms = self._data[category]
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
            LCGFTTerm(cat, form)
            for cat, forms in self._data.items()
            for form in forms
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
        "Information", "Knowledge", "Research", "Analysis", "Methodology",
    ],
    "social_sciences": [
        "Economics", "Sociology", "Psychology", "Anthropology", "Demographics",
        "Public policy", "Social welfare", "Labor", "Commerce", "Finance",
        "Statistics", "Surveys", "Population", "Immigration", "Poverty",
    ],
    "law": [
        "Constitutional law", "Criminal law", "Civil law", "Contracts",
        "Property", "Torts", "Administrative law", "International law",
        "Human rights", "Intellectual property", "Environmental law",
    ],
    "science": [
        "Biology", "Chemistry", "Physics", "Mathematics", "Geology",
        "Astronomy", "Ecology", "Genetics", "Climate", "Evolution",
        "Neuroscience", "Quantum mechanics", "Thermodynamics",
    ],
    "technology": [
        "Engineering", "Computer science", "Software", "Artificial intelligence",
        "Machine learning", "Robotics", "Biotechnology", "Nanotechnology",
        "Cybersecurity", "Data science", "Cloud computing", "Networks",
    ],
    "medicine": [
        "Public health", "Epidemiology", "Diseases", "Therapeutics",
        "Surgery", "Pharmacology", "Mental health", "Nutrition",
        "Pediatrics", "Oncology", "Cardiology", "Immunology",
    ],
    "humanities": [
        "Philosophy", "Ethics", "History", "Literature", "Languages",
        "Art", "Music", "Religion", "Culture", "Aesthetics",
    ],
    "business": [
        "Management", "Marketing", "Accounting", "Finance", "Strategy",
        "Entrepreneurship", "Operations", "Human resources", "Leadership",
        "Innovation", "Supply chain", "E-commerce",
    ],
    "politics": [
        "Government", "Elections", "Political parties", "Public administration",
        "International relations", "Diplomacy", "Security", "Defense",
        "Democracy", "Authoritarianism", "Nationalism", "Globalization",
    ],
    "environment": [
        "Climate change", "Conservation", "Sustainability", "Pollution",
        "Biodiversity", "Renewable energy", "Ecosystems", "Wildlife",
        "Deforestation", "Ocean conservation", "Carbon emissions",
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
    """Sample LCSH-style topics, optionally filtered by domain."""

    def __init__(
        self,
        domains: list[str] | None = None,
        lcc_class: str | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)

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

    def sample(self) -> str:
        return self._rng.choice(self._topics)

    def sample_n_unique(self, n: int) -> list[str]:
        """Sample n unique topics."""
        n = min(n, len(self._topics))
        return self._rng.sample(self._topics, n)

    def values(self) -> list[str]:
        return self._topics

    @staticmethod
    def all_domains() -> list[str]:
        """Get all domain names."""
        return list(TOPICS_BY_DOMAIN.keys())

    @staticmethod
    def for_lcc(lcc_code: str, seed: int | None = None) -> "TopicSampler":
        """Create a topic sampler for a specific LCC class."""
        return TopicSampler(lcc_class=lcc_code, seed=seed)


# =============================================================================
# Audience Sampler (LCDGT-style demographics)
# =============================================================================

AUDIENCES: list[str] = [
    # Age groups
    "Children", "Adolescents", "Young adults", "Adults", "Older adults",
    # Educational
    "Students", "Graduate students", "Researchers", "Scholars",
    # Professional
    "Professionals", "Practitioners", "Specialists", "Experts",
    "Scientists", "Engineers", "Physicians", "Lawyers", "Educators",
    "Business professionals", "Policy makers",
    # General
    "General public", "Beginners", "Non-specialists", "Lay readers",
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
        "United States", "United Kingdom", "Canada", "Australia",
        "Germany", "France", "Japan", "China", "India", "Brazil",
        "Mexico", "Italy", "Spain", "South Korea", "Russia",
    ],
    "us_states": [
        "California", "New York", "Texas", "Florida", "Illinois",
        "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan",
    ],
    "regions": [
        "North America", "South America", "Europe", "Asia", "Africa",
        "Middle East", "Southeast Asia", "Central America", "Caribbean",
    ],
    "cities": [
        "New York City", "Los Angeles", "Chicago", "London", "Paris",
        "Tokyo", "Beijing", "Mumbai", "São Paulo", "Berlin",
    ],
}


class GeographicSampler(Sampler[str | None]):
    """Sample geographic areas."""

    def __init__(
        self,
        area_types: list[str] | None = None,
        include_none: bool = True,
        none_probability: float = 0.4,
        seed: int | None = None,
    ):
        super().__init__(seed)

        if area_types:
            self._areas = []
            for t in area_types:
                self._areas.extend(GEOGRAPHIC_AREAS.get(t, []))
        else:
            self._areas = [
                area for areas in GEOGRAPHIC_AREAS.values() for area in areas
            ]

        self._include_none = include_none
        self._none_prob = none_probability

    def sample(self) -> str | None:
        if self._include_none and self._rng.random() < self._none_prob:
            return None
        return self._rng.choice(self._areas)

    def sample_n(self, n: int) -> list[str]:
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

    def __init__(
        self,
        data_dir: str | None = None,
        top_n: int | None = 50,
        seed: int | None = None,
    ):
        super().__init__(seed)
        from .lc_data import load_lc_data

        loader = load_lc_data(data_dir)

        if top_n:
            # Use frequency-ranked terms
            self._terms = [term for term, _ in loader.get_top_lcgft(top_n)]
        else:
            # Use all terms
            self._terms = list(loader.lcgft.values())

        # Convert to LCTerm dataclass
        self._terms = [
            LCTerm(
                id=t.id,
                label=t.label,
                uri=t.uri,
                alt_labels=t.alt_labels,
                broader=t.broader,
            )
            for t in self._terms
        ]

    def sample(self) -> LCTerm:
        return self._rng.choice(self._terms)

    def values(self) -> list[LCTerm]:
        return self._terms


class RealLCSHSampler(Sampler[LCTerm]):
    """Sample from real LCSH data with URIs.

    Uses actual LC Subject Headings from id.loc.gov.
    Can be filtered to top-N by frequency in government documents.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        top_n: int | None = 100,
        seed: int | None = None,
    ):
        super().__init__(seed)
        from .lc_data import load_lc_data

        loader = load_lc_data(data_dir)

        if top_n:
            self._terms = [term for term, _ in loader.get_top_lcsh(top_n)]
        else:
            # Warning: LCSH has 500k+ terms
            self._terms = list(loader.lcsh.values())[:1000]

        self._terms = [
            LCTerm(
                id=t.id,
                label=t.label,
                uri=t.uri,
                alt_labels=t.alt_labels,
                broader=t.broader,
            )
            for t in self._terms
        ]

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

    def __init__(
        self,
        data_dir: str | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed)
        from .lc_data import load_lc_data

        loader = load_lc_data(data_dir)
        self._terms = [
            LCTerm(
                id=t.id,
                label=t.label,
                uri=t.uri,
                alt_labels=t.alt_labels,
                broader=t.broader,
            )
            for t in loader.lcdgt.values()
        ]

    def sample(self) -> LCTerm:
        return self._rng.choice(self._terms)

    def values(self) -> list[LCTerm]:
        return self._terms

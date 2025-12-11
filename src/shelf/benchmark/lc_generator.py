"""
LC-Based Benchmark Generator

Generates synthetic documents by sampling from pure LC taxonomies:
- LCC main class (subject domain)
- LCGFT category and specific form (document type)
- LCSH topics (subject matter)
- LCDGT audience (optional)
"""

import random
from pydantic import BaseModel, Field

from .lc_taxonomy import (
    LCCClass,
    LCC_NAMES,
    LCGFT_CHILDREN,
    LCSH_TOPICS_BY_LCC,
    LCDGT_GROUPS,
    BenchmarkProfile,
)


class LCDocument(BaseModel):
    """A synthetic document with LC taxonomy labels."""

    id: str = Field(description="Unique document ID")

    # LC Classification dimensions
    lcc_class: str = Field(description="LCC main class letter")
    lcc_name: str = Field(description="LCC class name")

    # LCGFT dimensions
    lcgft_category: str = Field(description="LCGFT top-level category")
    lcgft_form: str = Field(description="Specific LCGFT form term")

    # LCSH topics
    topics: list[str] = Field(description="LCSH topic terms")

    # Optional dimensions
    lcdgt_audience: str | None = Field(default=None, description="Target audience")
    geographic: list[str] = Field(default_factory=list, description="Geographic subjects")

    # Generated content (placeholders for LLM generation)
    title: str = Field(default="", description="Document title")
    abstract: str = Field(default="", description="Document abstract")


class LCGeneratorConfig(BaseModel):
    """Configuration for LC-based benchmark generation."""

    # Sampling parameters
    seed: int = Field(default=42, description="Random seed")
    total_documents: int = Field(default=1000, description="Total documents to generate")

    # Dimension coverage
    min_topics: int = Field(default=1, description="Minimum LCSH topics per doc")
    max_topics: int = Field(default=4, description="Maximum LCSH topics per doc")
    include_audience: bool = Field(default=True, description="Include LCDGT audience")
    include_geographic: bool = Field(default=True, description="Include geographic subjects")

    # Category filtering (None = use all)
    lcc_classes: list[str] | None = Field(default=None, description="LCC classes to include")
    lcgft_categories: list[str] | None = Field(default=None, description="LCGFT categories to include")

    # Weighting
    uniform_lcc: bool = Field(default=True, description="Uniform sampling across LCC classes")
    uniform_lcgft: bool = Field(default=True, description="Uniform sampling across LCGFT categories")


# Define natural affinities between LCGFT categories and LCC classes
# This helps generate more realistic document combinations
LCGFT_LCC_AFFINITIES: dict[str, list[str]] = {
    "Informational works": ["H", "Q", "T", "J", "R", "S"],  # Social sci, science, tech, political sci
    "Law materials": ["K"],  # Law
    "Instructional and educational works": ["L", "Q", "T", "R"],  # Education, science, tech, medicine
    "Sound recordings": ["M", "P"],  # Music, language/lit
    "Discursive works": ["B", "P", "H", "J"],  # Philosophy, lit, social sci, political sci
    "Ephemera": ["A", "H", "N"],  # General, social sci, fine arts
    "Literature": ["P"],  # Language and literature
    "Religious materials": ["B"],  # Philosophy/religion
    "Music": ["M"],  # Music
    "Visual works": ["N", "G", "T"],  # Fine arts, geography, technology
    "Derivative works": ["P", "Z"],  # Literature, bibliography
    "Commemorative works": ["C", "D", "E"],  # History related
    "Cartographic materials": ["G"],  # Geography
    "Creative nonfiction": ["P", "D", "E", "G"],  # Literature, history, geography
    "Dance": ["M", "G"],  # Music, recreation
    "Tactile works": ["A", "L"],  # General, education
    "Recreational works": ["G"],  # Recreation
    "Radio interviews": ["P"],  # Media
    "Elegies": ["P"],  # Literature
    "Large print books": ["A", "P"],  # General, literature
    "Manuscripts": ["Z", "C"],  # Bibliography, history
    "Incunabula": ["Z"],  # Bibliography
}

# Common geographic subjects
GEOGRAPHIC_SUBJECTS = [
    "United States", "Europe", "Asia", "Africa", "North America", "South America",
    "California", "New York (State)", "Texas", "Florida", "Illinois",
    "Great Britain", "France", "Germany", "China", "Japan", "India",
    "Canada", "Mexico", "Brazil", "Australia",
]


class LCBenchmarkGenerator:
    """Generate benchmark documents from LC taxonomies."""

    def __init__(self, config: LCGeneratorConfig | None = None):
        self.config = config or LCGeneratorConfig()
        self._rng = random.Random(self.config.seed)

        # Filter available categories based on config
        self._lcc_classes = self._get_lcc_classes()
        self._lcgft_categories = self._get_lcgft_categories()

    def _get_lcc_classes(self) -> list[LCCClass]:
        """Get available LCC classes based on config."""
        if self.config.lcc_classes:
            return [LCCClass(c) for c in self.config.lcc_classes if c in LCCClass.__members__]
        return list(LCCClass)

    def _get_lcgft_categories(self) -> list[str]:
        """Get available LCGFT categories based on config."""
        all_cats = [cat for cat in LCGFT_CHILDREN.keys() if LCGFT_CHILDREN[cat]]  # Only non-empty
        if self.config.lcgft_categories:
            return [c for c in self.config.lcgft_categories if c in all_cats]
        return all_cats

    def sample_lcc_class(self) -> LCCClass:
        """Sample an LCC class."""
        return self._rng.choice(self._lcc_classes)

    def sample_lcgft_category(self, lcc_class: LCCClass | None = None) -> str:
        """Sample an LCGFT category, optionally biased by LCC class affinity."""
        if lcc_class and not self.config.uniform_lcgft:
            # Find categories with affinity to this LCC class
            affine_cats = [
                cat for cat, lcc_list in LCGFT_LCC_AFFINITIES.items()
                if lcc_class.value in lcc_list and cat in self._lcgft_categories
            ]
            if affine_cats and self._rng.random() < 0.7:  # 70% chance to use affinity
                return self._rng.choice(affine_cats)

        return self._rng.choice(self._lcgft_categories)

    def sample_lcgft_form(self, category: str) -> str:
        """Sample a specific LCGFT form from a category."""
        children = LCGFT_CHILDREN.get(category, [])
        if not children:
            return category  # Fallback to category name
        return self._rng.choice(children)

    def sample_topics(self, lcc_class: LCCClass, n: int | None = None) -> list[str]:
        """Sample LCSH topics appropriate for an LCC class."""
        if n is None:
            n = self._rng.randint(self.config.min_topics, self.config.max_topics)

        topics = LCSH_TOPICS_BY_LCC.get(lcc_class.value, [])
        if not topics:
            # Fallback to general topics
            topics = LCSH_TOPICS_BY_LCC.get("A", [])

        n = min(n, len(topics))
        return self._rng.sample(topics, n) if topics else []

    def sample_audience(self) -> str | None:
        """Sample a LCDGT audience group."""
        if not self.config.include_audience:
            return None
        if self._rng.random() < 0.7:  # 70% of docs have audience
            return self._rng.choice(LCDGT_GROUPS)
        return None

    def sample_geographic(self) -> list[str]:
        """Sample geographic subjects."""
        if not self.config.include_geographic:
            return []
        if self._rng.random() < 0.5:  # 50% of docs have geographic
            n = self._rng.randint(1, 2)
            return self._rng.sample(GEOGRAPHIC_SUBJECTS, n)
        return []

    def generate_document(self, doc_id: str) -> LCDocument:
        """Generate a single document with LC taxonomy labels."""
        # Sample primary dimensions
        lcc_class = self.sample_lcc_class()
        lcgft_category = self.sample_lcgft_category(lcc_class)
        lcgft_form = self.sample_lcgft_form(lcgft_category)

        # Sample secondary dimensions
        topics = self.sample_topics(lcc_class)
        audience = self.sample_audience()
        geographic = self.sample_geographic()

        return LCDocument(
            id=doc_id,
            lcc_class=lcc_class.value,
            lcc_name=LCC_NAMES[lcc_class],
            lcgft_category=lcgft_category,
            lcgft_form=lcgft_form,
            topics=topics,
            lcdgt_audience=audience,
            geographic=geographic,
        )

    def generate_batch(self, n: int | None = None) -> list[LCDocument]:
        """Generate a batch of documents."""
        n = n or self.config.total_documents
        return [self.generate_document(f"doc_{i:05d}") for i in range(n)]

    def generate_stratified(
        self,
        docs_per_lcc: int = 10,
        docs_per_lcgft: int = 5,
    ) -> list[LCDocument]:
        """Generate documents with stratified coverage across dimensions.

        This ensures representation of all LCC classes and LCGFT categories.
        """
        documents = []
        doc_idx = 0

        # Strategy: For each LCC class, sample from multiple LCGFT categories
        for lcc_class in self._lcc_classes:
            # Get LCGFT categories with affinity to this LCC class
            affine_cats = [
                cat for cat, lcc_list in LCGFT_LCC_AFFINITIES.items()
                if lcc_class.value in lcc_list and cat in self._lcgft_categories
            ]

            # Also include some random categories for variety
            other_cats = [c for c in self._lcgft_categories if c not in affine_cats]
            sampled_other = self._rng.sample(other_cats, min(2, len(other_cats)))

            categories_to_use = affine_cats + sampled_other

            for category in categories_to_use:
                for _ in range(docs_per_lcgft):
                    lcgft_form = self.sample_lcgft_form(category)
                    topics = self.sample_topics(lcc_class)
                    audience = self.sample_audience()
                    geographic = self.sample_geographic()

                    doc = LCDocument(
                        id=f"strat_{doc_idx:05d}",
                        lcc_class=lcc_class.value,
                        lcc_name=LCC_NAMES[lcc_class],
                        lcgft_category=category,
                        lcgft_form=lcgft_form,
                        topics=topics,
                        lcdgt_audience=audience,
                        geographic=geographic,
                    )
                    documents.append(doc)
                    doc_idx += 1

        self._rng.shuffle(documents)
        return documents

    def get_distribution(self, documents: list[LCDocument]) -> dict:
        """Analyze the distribution of taxonomy labels in generated documents."""
        from collections import Counter

        lcc_dist = Counter(doc.lcc_class for doc in documents)
        lcgft_cat_dist = Counter(doc.lcgft_category for doc in documents)
        lcgft_form_dist = Counter(doc.lcgft_form for doc in documents)
        topic_dist = Counter(t for doc in documents for t in doc.topics)
        audience_dist = Counter(doc.lcdgt_audience for doc in documents if doc.lcdgt_audience)
        geo_dist = Counter(g for doc in documents for g in doc.geographic)

        return {
            "total_documents": len(documents),
            "lcc_distribution": dict(lcc_dist.most_common()),
            "lcgft_category_distribution": dict(lcgft_cat_dist.most_common()),
            "lcgft_form_distribution": dict(lcgft_form_dist.most_common(30)),
            "topic_distribution": dict(topic_dist.most_common(30)),
            "audience_distribution": dict(audience_dist.most_common()),
            "geographic_distribution": dict(geo_dist.most_common()),
            "unique_lcc_classes": len(lcc_dist),
            "unique_lcgft_categories": len(lcgft_cat_dist),
            "unique_lcgft_forms": len(lcgft_form_dist),
            "unique_topics": len(topic_dist),
        }

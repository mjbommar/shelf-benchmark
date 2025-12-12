"""
Benchmark document generator using archetypes and conditional sampling.
"""

import random

from pydantic import BaseModel, Field

from .archetypes import ARCHETYPES, Archetype, normalize_weights


class SyntheticDocument(BaseModel):
    """A synthetic document with all taxonomy labels."""

    id: str = Field(description="Unique document ID")
    archetype_id: str = Field(description="Source archetype ID")

    # Generated content
    title: str = Field(description="Document title")
    abstract: str = Field(default="", description="Document abstract/description")

    # Labels
    agency: str = Field(description="SuDoc agency code")
    agency_name: str = Field(description="Full agency name")
    genres: list[str] = Field(description="LCGFT genre/form terms")
    lcc_class: str = Field(description="LCC main class")
    lcc_name: str = Field(description="LCC class name")
    topics: list[str] = Field(description="LCSH topical subjects")
    geographic: list[str] = Field(
        default_factory=list, description="LCSH geographic subjects"
    )

    # Metadata
    year: int | None = Field(default=None, description="Publication year")


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark generation."""

    total_documents: int = Field(
        default=1000, description="Total documents to generate"
    )
    min_topics: int = Field(default=1, description="Minimum topics per document")
    max_topics: int = Field(default=4, description="Maximum topics per document")
    include_geographic: bool = Field(
        default=True, description="Include geographic labels"
    )
    year_range: tuple[int, int] = Field(
        default=(1990, 2023), description="Year range for documents"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class BenchmarkGenerator:
    """Generate synthetic benchmark documents based on archetypes."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.archetypes = ARCHETYPES
        self.weights = normalize_weights()
        self._rng = random.Random(self.config.seed)

    def sample_archetype(self) -> Archetype:
        """Sample an archetype based on weights."""
        return self._rng.choices(self.archetypes, weights=self.weights, k=1)[0]

    def sample_topics(self, archetype: Archetype) -> list[str]:
        """Sample topics from an archetype's topic list."""
        n_topics = self._rng.randint(self.config.min_topics, self.config.max_topics)
        n_topics = min(n_topics, len(archetype.topics))
        return self._rng.sample(archetype.topics, n_topics)

    def sample_geographic(self, archetype: Archetype) -> list[str]:
        """Sample geographic areas from an archetype."""
        if not self.config.include_geographic or not archetype.geographic:
            return []

        # Usually 1-2 geographic areas, always include United States if present
        areas = []
        if "United States" in archetype.geographic:
            areas.append("United States")
            # Sometimes add a state
            other = [g for g in archetype.geographic if g != "United States"]
            if other and self._rng.random() < 0.5:
                areas.append(self._rng.choice(other))
        else:
            # Pick 1-2 areas
            n = self._rng.randint(1, min(2, len(archetype.geographic)))
            areas = self._rng.sample(archetype.geographic, n)

        return areas

    def generate_title_placeholder(
        self, archetype: Archetype, topics: list[str], geo: list[str]
    ) -> str:
        """Generate a placeholder title (to be replaced by LLM)."""
        # Simple template-based title for now
        topic = topics[0] if topics else "General"
        location = geo[0] if geo else "United States"

        templates = [
            f"{topic} in the {location}",
            f"Report on {topic}",
            f"{archetype.agency_name}: {topic}",
            f"Analysis of {topic}",
            f"{topic}: a study",
        ]

        if archetype.title_patterns:
            # Use archetype-specific patterns
            pattern = self._rng.choice(archetype.title_patterns)
            # Simple substitution
            title = pattern.replace("{topic}", topic)
            title = title.replace("{location}", location)
            title = title.replace(
                "{year}", str(self._rng.randint(*self.config.year_range))
            )
            return title

        return self._rng.choice(templates)

    def generate_document(self, doc_id: str) -> SyntheticDocument:
        """Generate a single synthetic document."""
        archetype = self.sample_archetype()
        topics = self.sample_topics(archetype)
        geographic = self.sample_geographic(archetype)

        title = self.generate_title_placeholder(archetype, topics, geographic)
        year = self._rng.randint(*self.config.year_range)

        return SyntheticDocument(
            id=doc_id,
            archetype_id=archetype.id,
            title=title,
            abstract="",  # To be filled by LLM
            agency=archetype.agency,
            agency_name=archetype.agency_name,
            genres=archetype.genres,
            lcc_class=archetype.lcc_class,
            lcc_name=archetype.lcc_name,
            topics=topics,
            geographic=geographic,
            year=year,
        )

    def generate_batch(self, n: int | None = None) -> list[SyntheticDocument]:
        """Generate a batch of synthetic documents."""
        n = n or self.config.total_documents
        return [self.generate_document(f"doc_{i:05d}") for i in range(n)]

    def generate_stratified(
        self, docs_per_archetype: int = 50
    ) -> list[SyntheticDocument]:
        """Generate documents with equal representation of each archetype."""
        documents = []
        for archetype in self.archetypes:
            for i in range(docs_per_archetype):
                doc_id = f"{archetype.id}_{i:03d}"
                topics = self.sample_topics(archetype)
                geographic = self.sample_geographic(archetype)
                title = self.generate_title_placeholder(archetype, topics, geographic)
                year = self._rng.randint(*self.config.year_range)

                doc = SyntheticDocument(
                    id=doc_id,
                    archetype_id=archetype.id,
                    title=title,
                    abstract="",
                    agency=archetype.agency,
                    agency_name=archetype.agency_name,
                    genres=archetype.genres,
                    lcc_class=archetype.lcc_class,
                    lcc_name=archetype.lcc_name,
                    topics=topics,
                    geographic=geographic,
                    year=year,
                )
                documents.append(doc)

        # Shuffle to mix archetypes
        self._rng.shuffle(documents)
        return documents

    def get_label_distribution(self, documents: list[SyntheticDocument]) -> dict:
        """Analyze the label distribution of generated documents."""
        from collections import Counter

        agency_dist = Counter(doc.agency for doc in documents)
        lcc_dist = Counter(doc.lcc_class for doc in documents)
        genre_dist = Counter(g for doc in documents for g in doc.genres)
        topic_dist = Counter(t for doc in documents for t in doc.topics)
        archetype_dist = Counter(doc.archetype_id for doc in documents)

        return {
            "total_documents": len(documents),
            "agency_distribution": dict(agency_dist.most_common()),
            "lcc_distribution": dict(lcc_dist.most_common()),
            "genre_distribution": dict(genre_dist.most_common(20)),
            "topic_distribution": dict(topic_dist.most_common(30)),
            "archetype_distribution": dict(archetype_dist.most_common()),
            "unique_agencies": len(agency_dist),
            "unique_lcc_classes": len(lcc_dist),
            "unique_genres": len(genre_dist),
            "unique_topics": len(topic_dist),
        }

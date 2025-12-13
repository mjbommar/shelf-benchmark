"""
Document sampler with fluent API for composing dimension samplers.
"""

from dataclasses import dataclass, field
from typing import Self, Iterator
import random

from .dimensions import (
    LCCSampler,
    LCGFTSampler,
    TopicSampler,
    AudienceSampler,
    GeographicSampler,
    LCCClass,
    LCGFTTerm,
)


@dataclass
class Document:
    """A sampled document with all taxonomy labels."""

    # Core dimensions
    lcc: LCCClass
    lcgft: LCGFTTerm
    topics: list[str]

    # Optional dimensions
    audience: str | None = None
    geographic: list[str] = field(default_factory=list)

    # Metadata
    id: str | None = None

    def __str__(self) -> str:
        lines = [
            f"LCC:      {self.lcc}",
            f"LCGFT:    {self.lcgft}",
            f"Topics:   {', '.join(self.topics)}",
        ]
        if self.audience:
            lines.append(f"Audience: {self.audience}")
        if self.geographic:
            lines.append(f"Geographic: {', '.join(self.geographic)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "lcc_code": self.lcc.code,
            "lcc_name": self.lcc.name,
            "lcgft_category": self.lcgft.category,
            "lcgft_form": self.lcgft.form,
            "topics": self.topics,
            "audience": self.audience,
            "geographic": self.geographic,
        }

    @property
    def label_vector(self) -> dict[str, str | list[str] | None]:
        """Get just the labels as a flat dict (for ML)."""
        return {
            "lcc": self.lcc.code,
            "lcgft_category": self.lcgft.category,
            "lcgft_form": self.lcgft.form,
            "topics": self.topics,
            "audience": self.audience,
            "geographic": self.geographic,
        }


class DocumentSampler:
    """
    Composable document sampler with fluent API.

    Example:
        sampler = (
            DocumentSampler()
            .with_seed(42)
            .with_lcc_classes(["H", "K", "P"])
            .with_topics_per_doc(2, 4)
        )

        doc = sampler.sample()
        docs = sampler.sample_batch(100)
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._rng = random.Random(seed)

        # Dimension configurations
        self._lcc_classes: list[str] | None = None
        self._lcgft_categories: list[str] | None = None
        self._topic_domains: list[str] | None = None
        self._min_topics = 1
        self._max_topics = 4
        self._include_audience = True
        self._include_geographic = True
        self._geographic_count = (0, 2)

        # Lazy-initialized samplers
        self._lcc_sampler: LCCSampler | None = None
        self._lcgft_sampler: LCGFTSampler | None = None
        self._audience_sampler: AudienceSampler | None = None
        self._geographic_sampler: GeographicSampler | None = None

    # =========================================================================
    # Fluent Configuration
    # =========================================================================

    def with_seed(self, seed: int) -> Self:
        """Set random seed for reproducibility."""
        self._seed = seed
        self._rng = random.Random(seed)
        self._reset_samplers()
        return self

    def with_lcc_classes(self, classes: list[str]) -> Self:
        """Restrict to specific LCC classes."""
        self._lcc_classes = classes
        self._lcc_sampler = None  # Reset
        return self

    def with_lcgft_categories(self, categories: list[str]) -> Self:
        """Restrict to specific LCGFT categories."""
        self._lcgft_categories = categories
        self._lcgft_sampler = None  # Reset
        return self

    def with_topic_domains(self, domains: list[str]) -> Self:
        """Restrict topics to specific domains."""
        self._topic_domains = domains
        return self

    def with_topics_per_doc(self, min_topics: int = 1, max_topics: int = 4) -> Self:
        """Set the range of topics per document."""
        self._min_topics = min_topics
        self._max_topics = max_topics
        return self

    def with_audience(self, include: bool = True) -> Self:
        """Include or exclude audience dimension."""
        self._include_audience = include
        return self

    def with_geographic(
        self, include: bool = True, count: tuple[int, int] = (0, 2)
    ) -> Self:
        """Include or exclude geographic dimension."""
        self._include_geographic = include
        self._geographic_count = count
        return self

    # =========================================================================
    # Sampler Access
    # =========================================================================

    def _reset_samplers(self):
        """Reset all lazy samplers."""
        self._lcc_sampler = None
        self._lcgft_sampler = None
        self._audience_sampler = None
        self._geographic_sampler = None

    @property
    def lcc(self) -> LCCSampler:
        """Get the LCC sampler."""
        if self._lcc_sampler is None:
            self._lcc_sampler = LCCSampler(
                classes=self._lcc_classes,
                seed=self._rng.randint(0, 2**31),
            )
        return self._lcc_sampler

    @property
    def lcgft(self) -> LCGFTSampler:
        """Get the LCGFT sampler."""
        if self._lcgft_sampler is None:
            self._lcgft_sampler = LCGFTSampler(
                categories=self._lcgft_categories,
                seed=self._rng.randint(0, 2**31),
            )
        return self._lcgft_sampler

    @property
    def audience(self) -> AudienceSampler:
        """Get the audience sampler."""
        if self._audience_sampler is None:
            self._audience_sampler = AudienceSampler(
                seed=self._rng.randint(0, 2**31),
            )
        return self._audience_sampler

    @property
    def geographic(self) -> GeographicSampler:
        """Get the geographic sampler."""
        if self._geographic_sampler is None:
            self._geographic_sampler = GeographicSampler(
                seed=self._rng.randint(0, 2**31),
            )
        return self._geographic_sampler

    def topic_sampler(self, lcc_code: str | None = None) -> TopicSampler:
        """Get a topic sampler, optionally for a specific LCC class."""
        return TopicSampler(
            domains=self._topic_domains,
            lcc_class=lcc_code,
            seed=self._rng.randint(0, 2**31),
        )

    # =========================================================================
    # Sampling Methods
    # =========================================================================

    def sample(self, doc_id: str | None = None) -> Document:
        """Sample a single document."""
        # Sample LCC class
        lcc = self.lcc.sample()

        # Sample LCGFT term
        lcgft = self.lcgft.sample()

        # Sample topics (using LCC-appropriate domains)
        topic_sampler = self.topic_sampler(lcc.code)
        n_topics = self._rng.randint(self._min_topics, self._max_topics)
        topics = topic_sampler.sample_n_unique(n_topics)

        # Sample optional dimensions
        aud = self.audience.sample() if self._include_audience else None

        geo: list[str] = []
        if self._include_geographic:
            n_geo = self._rng.randint(*self._geographic_count)
            if n_geo > 0:
                geo = self.geographic.sample_n_non_null(n_geo)

        return Document(
            id=doc_id,
            lcc=lcc,
            lcgft=lcgft,
            topics=topics,
            audience=aud,
            geographic=geo,
        )

    def sample_batch(self, n: int, id_prefix: str = "doc") -> list[Document]:
        """Sample a batch of documents."""
        return [self.sample(doc_id=f"{id_prefix}_{i:05d}") for i in range(n)]

    def stream(self, id_prefix: str = "doc") -> Iterator[Document]:
        """Infinite stream of sampled documents."""
        i = 0
        while True:
            yield self.sample(doc_id=f"{id_prefix}_{i:05d}")
            i += 1

    # =========================================================================
    # Stratified Sampling
    # =========================================================================

    def sample_stratified_by_lcc(self, docs_per_class: int = 10) -> list[Document]:
        """Sample with equal representation per LCC class."""
        docs = []
        for i, lcc in enumerate(self.lcc.values()):
            for j in range(docs_per_class):
                lcgft = self.lcgft.sample()
                topic_sampler = self.topic_sampler(lcc.code)
                n_topics = self._rng.randint(self._min_topics, self._max_topics)
                topics = topic_sampler.sample_n_unique(n_topics)

                aud = self.audience.sample() if self._include_audience else None
                geo: list[str] = []
                if self._include_geographic:
                    n_geo = self._rng.randint(*self._geographic_count)
                    if n_geo > 0:
                        geo = self.geographic.sample_n_non_null(n_geo)

                docs.append(
                    Document(
                        id=f"strat_{lcc.code}_{j:03d}",
                        lcc=lcc,
                        lcgft=lcgft,
                        topics=topics,
                        audience=aud,
                        geographic=geo,
                    )
                )

        self._rng.shuffle(docs)
        return docs

    def sample_stratified_by_lcgft(self, docs_per_category: int = 10) -> list[Document]:
        """Sample with equal representation per LCGFT category."""
        docs = []
        for cat in self.lcgft.values():
            for j in range(docs_per_category):
                lcc = self.lcc.sample()
                lcgft = self.lcgft.sample_from_category(cat)
                topic_sampler = self.topic_sampler(lcc.code)
                n_topics = self._rng.randint(self._min_topics, self._max_topics)
                topics = topic_sampler.sample_n_unique(n_topics)

                aud = self.audience.sample() if self._include_audience else None
                geo: list[str] = []
                if self._include_geographic:
                    n_geo = self._rng.randint(*self._geographic_count)
                    if n_geo > 0:
                        geo = self.geographic.sample_n_non_null(n_geo)

                docs.append(
                    Document(
                        id=f"strat_{cat[:10]}_{j:03d}",
                        lcc=lcc,
                        lcgft=lcgft,
                        topics=topics,
                        audience=aud,
                        geographic=geo,
                    )
                )

        self._rng.shuffle(docs)
        return docs

    # =========================================================================
    # Analysis
    # =========================================================================

    def coverage_report(self) -> dict:
        """Report on the configured sampling space."""
        return {
            "lcc_classes": len(self.lcc),
            "lcgft_categories": len(self.lcgft),
            "lcgft_forms": len(self.lcgft.all_forms()),
            "audiences": len(self.audience.values()),
            "geographic_areas": len(self.geographic.values()),
            "topic_domains": self._topic_domains or "all",
            "topics_per_doc": (self._min_topics, self._max_topics),
        }

    def __repr__(self) -> str:
        return (
            f"DocumentSampler("
            f"lcc={len(self.lcc)} classes, "
            f"lcgft={len(self.lcgft)} categories, "
            f"seed={self._seed})"
        )

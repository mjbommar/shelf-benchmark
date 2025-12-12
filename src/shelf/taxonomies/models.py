"""
Pydantic models for taxonomy labels and structures.
"""

from enum import Enum
from pydantic import BaseModel, Field


class TaxonomyType(str, Enum):
    """Supported taxonomy types."""

    # Library of Congress Classification
    LCC_MAIN = "lcc_main"  # 21 main classes (A-Z)
    LCC_SUBCLASS = "lcc_subclass"  # ~200-350 subclasses (e.g., KF, HD)

    # Subject Headings
    LCSH_TOPICAL = "lcsh_topical"  # Topical subjects (MARC 650)
    LCSH_GEOGRAPHIC = "lcsh_geo"  # Geographic subjects (MARC 651)
    LCSH_FULL = "lcsh_full"  # Full headings with subdivisions

    # Genre/Form Terms
    LCGFT = "lcgft"  # Genre/Form terms (MARC 655)

    # Demographic Group Terms
    LCDGT = "lcdgt"  # Demographic terms

    # Government Document Classification
    SUDOC_AGENCY = "sudoc_agency"  # SuDoc agency codes

    # Corporate Names
    CORP_NAMES = "corp_names"  # Corporate name subjects (MARC 610)


class Label(BaseModel):
    """A single taxonomy label with metadata."""

    id: str = Field(description="Unique identifier (e.g., LC authority ID or code)")
    label: str = Field(description="Human-readable label/name")
    uri: str | None = Field(default=None, description="Full URI if available")
    frequency: int = Field(default=0, description="Frequency count in corpus")
    rank: int = Field(default=0, description="Rank by frequency (1 = most frequent)")
    broader: list[str] = Field(default_factory=list, description="Broader term IDs")
    narrower: list[str] = Field(default_factory=list, description="Narrower term IDs")
    alt_labels: list[str] = Field(
        default_factory=list, description="Alternative labels"
    )
    description: str | None = Field(
        default=None, description="Scope note or description"
    )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Label):
            return self.id == other.id
        return False


class Taxonomy(BaseModel):
    """A collection of labels forming a taxonomy."""

    type: TaxonomyType = Field(description="Type of taxonomy")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="Description of the taxonomy")
    source: str = Field(default="", description="Data source (e.g., id.loc.gov, MARC)")
    labels: list[Label] = Field(
        default_factory=list, description="Labels in this taxonomy"
    )
    total_corpus_uses: int = Field(default=0, description="Total uses in source corpus")
    corpus_coverage: float = Field(default=0.0, description="Coverage of corpus (0-1)")

    @property
    def size(self) -> int:
        """Number of labels in this taxonomy."""
        return len(self.labels)

    def top_n(self, n: int) -> "Taxonomy":
        """Return a new taxonomy with only the top N labels by frequency."""
        sorted_labels = sorted(self.labels, key=lambda x: x.frequency, reverse=True)[:n]

        # Recalculate coverage
        top_n_uses = sum(label.frequency for label in sorted_labels)
        coverage = (
            top_n_uses / self.total_corpus_uses if self.total_corpus_uses > 0 else 0
        )

        # Re-rank
        for i, label in enumerate(sorted_labels, 1):
            label.rank = i

        return Taxonomy(
            type=self.type,
            name=f"{self.name} (Top {n})",
            description=f"Top {n} labels from {self.name}",
            source=self.source,
            labels=sorted_labels,
            total_corpus_uses=self.total_corpus_uses,
            corpus_coverage=coverage,
        )

    def filter_by_min_frequency(self, min_freq: int) -> "Taxonomy":
        """Return a new taxonomy with only labels meeting minimum frequency."""
        filtered = [label for label in self.labels if label.frequency >= min_freq]
        filtered_uses = sum(label.frequency for label in filtered)
        coverage = (
            filtered_uses / self.total_corpus_uses if self.total_corpus_uses > 0 else 0
        )

        return Taxonomy(
            type=self.type,
            name=f"{self.name} (freq >= {min_freq})",
            description=f"Labels with frequency >= {min_freq} from {self.name}",
            source=self.source,
            labels=filtered,
            total_corpus_uses=self.total_corpus_uses,
            corpus_coverage=coverage,
        )

    def get_label_by_id(self, label_id: str) -> Label | None:
        """Look up a label by ID."""
        for label in self.labels:
            if label.id == label_id:
                return label
        return None

    def get_labels_by_ids(self, label_ids: list[str]) -> list[Label]:
        """Look up multiple labels by ID."""
        id_set = set(label_ids)
        return [label for label in self.labels if label.id in id_set]


class TaxonomySet(BaseModel):
    """A collection of taxonomies for benchmarking."""

    name: str = Field(description="Name of this taxonomy set")
    description: str = Field(default="", description="Description")
    taxonomies: dict[TaxonomyType, Taxonomy] = Field(
        default_factory=dict, description="Taxonomies indexed by type"
    )

    def add_taxonomy(self, taxonomy: Taxonomy) -> None:
        """Add a taxonomy to the set."""
        self.taxonomies[taxonomy.type] = taxonomy

    def get_taxonomy(self, taxonomy_type: TaxonomyType) -> Taxonomy | None:
        """Get a taxonomy by type."""
        return self.taxonomies.get(taxonomy_type)

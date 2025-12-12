"""
Pydantic artifact models for generated benchmark documents.

Each artifact includes:
- Full LC taxonomy labels (LCC, LCGFT, LCSH topics, LCDGT audience)
- Generated content (title, body)
- The exact prompt used to generate it
- Generation parameters (length, register, model)
- Git commit info for reproducibility
"""

import hashlib
import subprocess
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import orjson
from pydantic import BaseModel, Field

from .generator import GeneratedDocument


@lru_cache(maxsize=1)
def get_git_version() -> dict[str, str | bool | None]:
    """Get git commit info for versioning and reproducibility.

    Returns:
        Dict with:
        - commit: Short commit hash (e.g., "ee52a05")
        - dirty: True if working directory has uncommitted changes
        - branch: Current branch name
        - version_string: Commit + "*" if dirty (e.g., "ee52a05*")

    If git is not available or fails, returns dict with error key.
    """
    try:
        # Get short commit hash
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        # Check if working directory is dirty
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            != 0
        )

        # Get branch name
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        return {
            "commit": commit,
            "dirty": dirty,
            "branch": branch,
            "version_string": f"{commit}{'*' if dirty else ''}",
        }
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {
            "commit": None,
            "dirty": None,
            "branch": None,
            "version_string": "unknown",
        }


def generate_artifact_id() -> str:
    """Generate a unique artifact ID with timestamp + uuid."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{ts}_{uid}"


# =============================================================================
# Pydantic Artifact Models
# =============================================================================


class LCCLabel(BaseModel):
    """Library of Congress Classification label."""

    code: str = Field(description="LCC main class letter (A-Z)")
    name: str = Field(description="Full class name")


class LCGFTLabel(BaseModel):
    """Library of Congress Genre/Form Terms label."""

    category: str = Field(description="Top-level LCGFT category")
    form: str = Field(description="Specific genre/form term")


class GenerationParams(BaseModel):
    """Parameters used to generate the document."""

    target_length: str | None = Field(description="Target length category")
    target_word_range: tuple[int, int] | None = Field(
        description="Target word count range"
    )
    writing_register: str | None = Field(description="Writing register/tone")
    writing_register_description: str | None = Field(
        description="Full register description"
    )
    temperature: float | None = Field(default=None, description="LLM temperature")
    top_p: float | None = Field(
        default=None, description="LLM top_p (nucleus sampling)"
    )


class GenerationMetadata(BaseModel):
    """Metadata about how the document was generated."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str | None = Field(default=None, description="LLM model used")
    generation_id: str | None = Field(default=None, description="Batch generation ID")
    prompt: str | None = Field(default=None, description="Exact prompt sent to LLM")

    # Git versioning for reproducibility
    git_commit: str | None = Field(default=None, description="Git commit hash (short)")
    git_dirty: bool | None = Field(
        default=None, description="True if uncommitted changes existed"
    )
    git_branch: str | None = Field(default=None, description="Git branch name")
    code_version: str | None = Field(
        default=None,
        description="Version string: commit + '*' if dirty (e.g., 'ee52a05*')",
    )


def _build_generation_metadata(
    model: str | None = None,
    generation_id: str | None = None,
    prompt: str | None = None,
) -> GenerationMetadata:
    """Build GenerationMetadata with git version info."""
    git_info = get_git_version()
    return GenerationMetadata(
        model=model,
        generation_id=generation_id,
        prompt=prompt,
        git_commit=git_info.get("commit"),
        git_dirty=git_info.get("dirty"),
        git_branch=git_info.get("branch"),
        code_version=git_info.get("version_string"),
    )


class BenchmarkArtifact(BaseModel):
    """
    Complete benchmark document artifact with all LC metadata.

    This is the canonical format for benchmark documents - everything
    needed to understand, reproduce, and evaluate the document.
    """

    # Identifiers
    id: str = Field(description="Unique document ID")
    hash: str = Field(description="Content hash for deduplication")

    # Generated content
    title: str = Field(description="Document title")
    body: str = Field(description="Document body text")
    word_count: int = Field(description="Actual word count")

    # LC Classification (subject domain)
    lcc: LCCLabel = Field(description="LCC subject classification")

    # LCGFT Genre/Form (document type)
    lcgft: LCGFTLabel = Field(description="LCGFT genre/form classification")

    # LCSH Topics (subject headings)
    topics: list[str] = Field(description="LCSH-style topic terms")

    # LCDGT Audience
    audience: str | None = Field(
        default=None, description="Target audience (LCDGT-style)"
    )

    # Geographic
    geographic: list[str] = Field(
        default_factory=list, description="Geographic focus areas"
    )

    # Generation parameters
    generation: GenerationParams = Field(description="Generation parameters")

    # Metadata
    metadata: GenerationMetadata = Field(description="Generation metadata")

    @classmethod
    def from_generated(
        cls,
        generated: GeneratedDocument,
        generation_id: str | None = None,
        model: str | None = None,
    ) -> "BenchmarkArtifact":
        """Create artifact from a GeneratedDocument."""
        from .generator import LENGTH_WORD_RANGES, REGISTER_DESCRIPTIONS

        doc = generated.document

        # Compute hash
        label_str = f"{doc.lcc.code}_{doc.lcgft.category}_{doc.lcgft.form}_{','.join(doc.topics)}"
        content_hash = hashlib.sha256(label_str.encode()).hexdigest()[:12]

        # Get word range and register description
        word_range = None
        if generated.target_length:
            word_range = LENGTH_WORD_RANGES.get(generated.target_length)

        register_desc = None
        if generated.register:
            register_desc = REGISTER_DESCRIPTIONS.get(generated.register)

        return cls(
            id=doc.id or generate_artifact_id(),
            hash=content_hash,
            title=generated.title,
            body=generated.body,
            word_count=generated.word_count or len(generated.body.split()),
            lcc=LCCLabel(code=doc.lcc.code, name=doc.lcc.name),
            lcgft=LCGFTLabel(category=doc.lcgft.category, form=doc.lcgft.form),
            topics=doc.topics,
            audience=doc.audience,
            geographic=doc.geographic,
            generation=GenerationParams(
                target_length=generated.target_length.value
                if generated.target_length
                else None,
                target_word_range=word_range,
                writing_register=generated.register.value
                if generated.register
                else None,
                writing_register_description=register_desc,
                temperature=generated.sampling_params.temperature
                if generated.sampling_params
                else None,
                top_p=generated.sampling_params.top_p
                if generated.sampling_params
                else None,
            ),
            metadata=_build_generation_metadata(
                model=model,
                generation_id=generation_id,
                prompt=generated.prompt,
            ),
        )

    def to_training_item(self) -> dict:
        """Format for ML training (text + flat labels)."""
        return {
            "id": self.id,
            "title": self.title,
            "text": self.body,
            "word_count": self.word_count,
            "labels": {
                "lcc_code": self.lcc.code,
                "lcc_name": self.lcc.name,
                "lcgft_category": self.lcgft.category,
                "lcgft_form": self.lcgft.form,
                "topics": self.topics,
                "audience": self.audience,
                "geographic": self.geographic,
                "register": self.generation.writing_register,
            },
        }


class DatasetMetadata(BaseModel):
    """Metadata for a complete benchmark dataset."""

    generation_id: str
    model: str | None
    created_at: datetime
    total_documents: int
    train_size: int
    test_size: int
    train_ratio: float

    # Distribution summaries
    lcc_distribution: dict[str, int]
    lcgft_category_distribution: dict[str, int]
    register_distribution: dict[str, int]
    length_distribution: dict[str, int]

    word_count_stats: dict[str, int]


# =============================================================================
# Artifact Storage
# =============================================================================


class ArtifactStore:
    """
    Store generated documents as Pydantic JSON artifacts.

    Directory structure:
        output_dir/
            index.jsonl          # Line-delimited index
            metadata.json        # Dataset metadata
            artifacts/
                doc_00001.json   # Full artifact with prompt
                doc_00002.json
                ...
    """

    def __init__(
        self,
        output_dir: Path | str,
        generation_id: str | None = None,
        model: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.artifacts_dir = self.output_dir / "artifacts"
        self.index_path = self.output_dir / "index.jsonl"

        self.generation_id = generation_id or datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        self.model = model

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

        self._artifacts: list[BenchmarkArtifact] = []

    def save(self, generated: GeneratedDocument) -> BenchmarkArtifact:
        """Save a single generated document as Pydantic JSON artifact."""
        artifact = BenchmarkArtifact.from_generated(
            generated,
            generation_id=self.generation_id,
            model=self.model,
        )

        # Save individual artifact (full Pydantic model as JSON)
        artifact_path = self.artifacts_dir / f"{artifact.id}.json"
        artifact_path.write_bytes(
            orjson.dumps(artifact.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )

        # Append to index
        index_entry = {
            "id": artifact.id,
            "hash": artifact.hash,
            "path": str(artifact_path.relative_to(self.output_dir)),
            "lcc": artifact.lcc.code,
            "lcgft_category": artifact.lcgft.category,
            "lcgft_form": artifact.lcgft.form,
            "word_count": artifact.word_count,
            "register": artifact.generation.writing_register,
            "target_length": artifact.generation.target_length,
        }
        with open(self.index_path, "ab") as f:
            f.write(orjson.dumps(index_entry))
            f.write(b"\n")

        self._artifacts.append(artifact)
        return artifact

    def save_batch(
        self,
        documents: list[GeneratedDocument],
        show_progress: bool = True,
    ) -> list[BenchmarkArtifact]:
        """Save a batch of generated documents."""
        artifacts = []
        for i, doc in enumerate(documents):
            if show_progress and (i + 1) % 100 == 0:
                print(f"Saved {i + 1}/{len(documents)}")
            artifacts.append(self.save(doc))
        return artifacts

    def load(self, doc_id: str) -> BenchmarkArtifact:
        """Load a single artifact by ID."""
        artifact_path = self.artifacts_dir / f"{doc_id}.json"
        data = orjson.loads(artifact_path.read_bytes())
        return BenchmarkArtifact(**data)

    def load_all(self) -> Iterator[BenchmarkArtifact]:
        """Iterate over all artifacts."""
        for path in sorted(self.artifacts_dir.glob("*.json")):
            data = orjson.loads(path.read_bytes())
            yield BenchmarkArtifact(**data)

    def load_index(self) -> list[dict]:
        """Load the index file."""
        if not self.index_path.exists():
            return []
        entries = []
        with open(self.index_path, "rb") as f:
            for line in f:
                if line.strip():
                    entries.append(orjson.loads(line))
        return entries

    @property
    def count(self) -> int:
        """Number of artifacts saved in this session."""
        return len(self._artifacts)

    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata from stored artifacts."""
        from collections import Counter

        index = self.load_index()

        lcc_dist = Counter(e["lcc"] for e in index)
        category_dist = Counter(e["lcgft_category"] for e in index)
        register_dist = Counter(e.get("register") for e in index if e.get("register"))
        length_dist = Counter(
            e.get("target_length") for e in index if e.get("target_length")
        )
        word_counts = [e["word_count"] for e in index if e.get("word_count")]

        return DatasetMetadata(
            generation_id=self.generation_id,
            model=self.model,
            created_at=datetime.now(timezone.utc),
            total_documents=len(index),
            train_size=0,  # Set by export_dataset
            test_size=0,
            train_ratio=0.0,
            lcc_distribution=dict(lcc_dist.most_common()),
            lcgft_category_distribution=dict(category_dist.most_common()),
            register_distribution=dict(register_dist.most_common()),
            length_distribution=dict(length_dist.most_common()),
            word_count_stats={
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0,
                "avg": sum(word_counts) // len(word_counts) if word_counts else 0,
            },
        )


def export_dataset(
    output_dir: Path | str,
    documents: list[GeneratedDocument],
    generation_id: str | None = None,
    model: str | None = None,
    train_ratio: float = 0.8,
) -> DatasetMetadata:
    """
    Export a complete benchmark dataset with train/test split.

    Creates:
        output_dir/
            train.jsonl      # Training set (text + labels)
            test.jsonl       # Test set (text + labels)
            metadata.json    # Dataset metadata (Pydantic)
            artifacts/       # Individual JSON files with full data + prompts
    """
    import random

    output_dir = Path(output_dir)

    # Save all artifacts
    store = ArtifactStore(output_dir, generation_id, model)
    artifacts = store.save_batch(documents, show_progress=True)

    # Shuffle and split
    random.shuffle(artifacts)
    split_idx = int(len(artifacts) * train_ratio)
    train_artifacts = artifacts[:split_idx]
    test_artifacts = artifacts[split_idx:]

    # Write train.jsonl (training format)
    train_path = output_dir / "train.jsonl"
    with open(train_path, "wb") as f:
        for artifact in train_artifacts:
            f.write(orjson.dumps(artifact.to_training_item()))
            f.write(b"\n")

    # Write test.jsonl
    test_path = output_dir / "test.jsonl"
    with open(test_path, "wb") as f:
        for artifact in test_artifacts:
            f.write(orjson.dumps(artifact.to_training_item()))
            f.write(b"\n")

    # Get and update metadata
    metadata = store.get_metadata()
    metadata.train_size = len(train_artifacts)
    metadata.test_size = len(test_artifacts)
    metadata.train_ratio = train_ratio

    # Write metadata.json
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_bytes(
        orjson.dumps(metadata.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )

    print(f"\nDataset exported to: {output_dir}")
    print(f"  Train: {len(train_artifacts)} documents")
    print(f"  Test:  {len(test_artifacts)} documents")
    print(f"  Artifacts: {store.count} JSON files (with prompts)")

    return metadata

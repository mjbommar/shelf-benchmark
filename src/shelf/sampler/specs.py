"""Immutable generation specs and spec blocks.

Phase 1 of the v0.4 data plan gives **the same spec to every generator**, so that
generator identity becomes the only varying factor. That buys paired
cross-generator comparison, a same-content/different-generator pair task, and a
train-on-family-A / test-on-family-B split, at no extra generation cost.

Two consequences drive this module's design.

**Realizations of one spec are near-duplicates.** Fifteen generators writing to
the same spec produce fifteen documents about the same thing. If they are split
independently, the corpus manufactures exactly the train/test leakage v0.3.1
currently avoids. Every realization therefore carries a stable ``spec_id``, and
splitting must be done on that id, never on the document id.

**One spec draw cannot estimate its own idiosyncrasy.** A single shared block
gives maximum paired control but no way to tell whether a result reflects the
corpus or the particular sample that produced it. Independent blocks drawn under
different seeds preserve every paired comparison *and* add a between-block
variance estimate at identical cost, so :func:`draw_spec_blocks` returns several
smaller blocks rather than one large one.

Spec ids are content hashes, so they are stable across runs, machines, and
Python versions -- a spec drawn today and redrawn next year gets the same id if
and only if its content is identical.

Example:
    from shelf.sampler.specs import draw_spec_blocks

    blocks = draw_spec_blocks(n_blocks=3, specs_per_block=500, base_seed=42)
    for block in blocks:
        print(block.block_id, block.checksum, len(block))

    # The same block is handed to every generator.
    for spec in blocks[0]:
        document = spec.to_document()
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shelf.sampler.document import Document, DocumentSampler
from shelf.sampler.generator import DocumentLength, PromptVariant, Register

__all__ = [
    "DocumentSpec",
    "SpecBlock",
    "draw_spec_blocks",
    "load_spec_block",
    "save_spec_block",
]

# Version tag embedded in every spec hash. Bumping it deliberately invalidates
# every previously drawn spec id, which is what we want if the spec schema
# changes meaning -- silently reusing ids across incompatible schemas would make
# two different things look like the same spec.
SPEC_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class DocumentSpec:
    """An immutable, content-addressed description of one document to generate.

    A spec is everything the generator needs *except* which model writes it.
    That separation is the whole point: hand one spec to fifteen generators and
    the only thing that varies is the generator.
    """

    lcc_code: str
    lcc_name: str
    lcgft_form: str
    lcgft_category: str
    topics: tuple[str, ...]
    target_length: DocumentLength
    register: Register
    audience: str | None = None
    geographic: tuple[str, ...] = ()
    prompt_variant: PromptVariant | None = None
    block_id: str = ""
    # v0.4 Phase 2 difficulty tier (docs/data_plan_v0.4.md section 6). Optional
    # and hashed in only when set, so every spec block drawn before this field
    # existed still loads and verifies against its recorded id.
    lcc_subclass: str | None = None
    lcc_subclass_name: str | None = None
    # Minimal-pair metadata (Phase 3). Like block_id these describe *grouping*,
    # not content, so they stay out of the content hash: the two members of a
    # pair already differ in the single facet the pair varies.
    pair_id: str = ""
    pair_role: str = ""
    pair_axis: str = ""

    @property
    def spec_id(self) -> str:
        """Stable content hash. Identical content always yields the same id."""
        return self._digest()[:16]

    @property
    def checksum(self) -> str:
        """Full SHA-256 of the spec content."""
        return self._digest()

    def _digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.content_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def content_dict(self) -> dict[str, Any]:
        """The fields that define spec identity.

        ``block_id`` is deliberately excluded: the same content drawn into two
        different blocks is the same spec, and hashing the block in would hide
        that collision instead of surfacing it.

        ``lcc_subclass`` is included **only when set**. Adding a key with a
        ``None`` value would have changed the hash of every spec ever drawn,
        breaking the id of the Phase 1 blocks already on disk; omitting it
        keeps those bytes identical while still making a subclass-bearing spec
        a distinct spec.
        """
        payload: dict[str, Any] = {
            "schema": SPEC_SCHEMA_VERSION,
            "lcc_code": self.lcc_code,
            "lcc_name": self.lcc_name,
            "lcgft_form": self.lcgft_form,
            "lcgft_category": self.lcgft_category,
            "topics": list(self.topics),
            "target_length": self.target_length.value,
            "register": self.register.value,
            "audience": self.audience,
            "geographic": list(self.geographic),
            "prompt_variant": (
                self.prompt_variant.value if self.prompt_variant else None
            ),
        }
        if self.lcc_subclass:
            payload["lcc_subclass"] = self.lcc_subclass
            payload["lcc_subclass_name"] = self.lcc_subclass_name
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Full serialization, including derived id and block membership."""
        payload = self.content_dict()
        payload["spec_id"] = self.spec_id
        payload["block_id"] = self.block_id
        if self.pair_id:
            payload["pair_id"] = self.pair_id
            payload["pair_role"] = self.pair_role
            payload["pair_axis"] = self.pair_axis
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentSpec:
        """Rebuild a spec from :meth:`to_dict` output.

        Raises:
            ValueError: If the stored ``spec_id`` does not match the rebuilt
                content, which means the file was edited or the schema moved.
        """
        spec = cls(
            lcc_code=data["lcc_code"],
            lcc_name=data["lcc_name"],
            lcgft_form=data["lcgft_form"],
            lcgft_category=data["lcgft_category"],
            topics=tuple(data.get("topics") or ()),
            target_length=DocumentLength(data["target_length"]),
            register=Register(data["register"]),
            audience=data.get("audience"),
            geographic=tuple(data.get("geographic") or ()),
            prompt_variant=(
                PromptVariant(data["prompt_variant"])
                if data.get("prompt_variant")
                else None
            ),
            block_id=data.get("block_id", ""),
            lcc_subclass=data.get("lcc_subclass"),
            lcc_subclass_name=data.get("lcc_subclass_name"),
            pair_id=data.get("pair_id", ""),
            pair_role=data.get("pair_role", ""),
            pair_axis=data.get("pair_axis", ""),
        )
        stored = data.get("spec_id")
        if stored and stored != spec.spec_id:
            raise ValueError(
                f"spec_id mismatch: file says {stored}, content hashes to "
                f"{spec.spec_id}. The spec file was modified or the schema changed."
            )
        return spec

    def to_document(self) -> Document:
        """Materialize the taxonomy half of this spec as a ``Document``."""
        from shelf.sampler.dimensions import LCCClass, LCGFTTerm

        return Document(
            lcc=LCCClass(
                code=self.lcc_code,
                name=self.lcc_name,
                subclass=self.lcc_subclass,
                subclass_name=self.lcc_subclass_name,
            ),
            lcgft=LCGFTTerm(category=self.lcgft_category, form=self.lcgft_form),
            topics=list(self.topics),
            audience=self.audience,
            geographic=list(self.geographic),
            id=self.spec_id,
        )

    @classmethod
    def from_document(
        cls,
        document: Document,
        target_length: DocumentLength,
        register: Register,
        prompt_variant: PromptVariant | None = None,
        block_id: str = "",
    ) -> DocumentSpec:
        """Build a spec from a sampled ``Document`` plus generation settings."""
        return cls(
            lcc_code=document.lcc.code,
            lcc_name=document.lcc.name,
            lcgft_form=document.lcgft.form,
            lcgft_category=document.lcgft.category,
            topics=tuple(document.topics),
            target_length=target_length,
            register=register,
            audience=document.audience,
            geographic=tuple(document.geographic or ()),
            prompt_variant=prompt_variant,
            block_id=block_id,
            lcc_subclass=document.lcc.subclass,
            lcc_subclass_name=document.lcc.subclass_name,
        )


@dataclass
class SpecBlock:
    """An immutable, independently-drawn set of specs.

    A block is the unit of replication *and* the unit of splitting. Because
    every generator writes every spec in the block, all realizations of a spec
    are near-duplicates, so the block records enough provenance to make a
    spec-level split auditable.
    """

    block_id: str
    seed: int
    specs: tuple[DocumentSpec, ...]
    sampler_config: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[DocumentSpec]:
        return iter(self.specs)

    @property
    def spec_ids(self) -> tuple[str, ...]:
        """Ids of every spec in the block, in draw order."""
        return tuple(spec.spec_id for spec in self.specs)

    @property
    def checksum(self) -> str:
        """SHA-256 over the block's spec ids, for the run manifest."""
        return hashlib.sha256("".join(self.spec_ids).encode()).hexdigest()

    def duplicate_spec_ids(self) -> dict[str, int]:
        """Spec ids appearing more than once, with their counts.

        A non-empty result means the sampler drew the same content twice. That
        is not fatal, but it silently reduces effective block size and must be
        visible rather than discovered later as a near-duplicate cluster.
        """
        counts: dict[str, int] = {}
        for spec_id in self.spec_ids:
            counts[spec_id] = counts.get(spec_id, 0) + 1
        return {k: v for k, v in counts.items() if v > 1}

    def to_manifest(self) -> dict[str, Any]:
        """Provenance record for publication alongside the dataset."""
        return {
            "block_id": self.block_id,
            "seed": self.seed,
            "n_specs": len(self.specs),
            "checksum": self.checksum,
            "schema_version": SPEC_SCHEMA_VERSION,
            "duplicate_spec_ids": self.duplicate_spec_ids(),
            "sampler_config": self.sampler_config,
        }


def draw_spec_blocks(
    n_blocks: int = 3,
    specs_per_block: int = 500,
    base_seed: int = 42,
    length_weights: dict[DocumentLength, float] | None = None,
    register_weights: dict[Register, float] | None = None,
    sampler_factory: Any = None,
) -> list[SpecBlock]:
    """Draw independent spec blocks.

    Independence is what makes between-block variance estimable: each block gets
    its own seed, so a result that holds in only one block is visibly a property
    of that draw rather than of the corpus.

    Args:
        n_blocks: Number of independent blocks.
        specs_per_block: Specs in each block.
        base_seed: Block ``i`` is drawn with seed ``base_seed + i``.
        length_weights: Optional length distribution override.
        register_weights: Optional register distribution override.
        sampler_factory: Callable taking a seed and returning a
            ``DocumentSampler``. Defaults to a plain seeded sampler.

    Returns:
        A list of ``n_blocks`` blocks.

    Raises:
        ValueError: If ``n_blocks`` or ``specs_per_block`` is not positive.
    """
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    if specs_per_block <= 0:
        raise ValueError(f"specs_per_block must be positive, got {specs_per_block}")

    from shelf.sampler.generator import LengthSampler, RegisterSampler

    blocks: list[SpecBlock] = []
    for index in range(n_blocks):
        seed = base_seed + index
        block_id = f"block-{index:02d}-seed{seed}"

        sampler = (
            sampler_factory(seed) if sampler_factory else DocumentSampler(seed=seed)
        )
        lengths = LengthSampler(weights=length_weights, seed=seed)
        registers = RegisterSampler(weights=register_weights, seed=seed)

        specs = tuple(
            DocumentSpec.from_document(
                sampler.sample(),
                target_length=lengths.sample(),
                register=registers.sample(),
                block_id=block_id,
            )
            for _ in range(specs_per_block)
        )

        blocks.append(
            SpecBlock(
                block_id=block_id,
                seed=seed,
                specs=specs,
                sampler_config={
                    "specs_per_block": specs_per_block,
                    "length_weights": (
                        {k.value: v for k, v in length_weights.items()}
                        if length_weights
                        else None
                    ),
                    "register_weights": (
                        {k.value: v for k, v in register_weights.items()}
                        if register_weights
                        else None
                    ),
                },
            )
        )

    return blocks


def save_spec_block(block: SpecBlock, path: Path | str) -> Path:
    """Write a block to JSONL, manifest first, then one spec per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"__manifest__": block.to_manifest()}) + "\n")
        for spec in block.specs:
            handle.write(json.dumps(spec.to_dict(), sort_keys=True) + "\n")
    return path


def load_spec_block(path: Path | str) -> SpecBlock:
    """Read a block written by :func:`save_spec_block`.

    Every spec id is re-verified against its content on load, so an edited or
    truncated block file fails loudly instead of quietly changing the split.

    Raises:
        ValueError: If the file is empty, lacks a manifest, or the recorded
            checksum does not match the specs present.
    """
    path = Path(path)
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise ValueError(f"spec block file is empty: {path}")

    header = json.loads(lines[0])
    if "__manifest__" not in header:
        raise ValueError(f"spec block file missing manifest header: {path}")
    manifest = header["__manifest__"]

    specs = tuple(DocumentSpec.from_dict(json.loads(line)) for line in lines[1:])
    block = SpecBlock(
        block_id=manifest["block_id"],
        seed=manifest["seed"],
        specs=specs,
        sampler_config=manifest.get("sampler_config", {}),
    )

    if manifest.get("checksum") and manifest["checksum"] != block.checksum:
        raise ValueError(
            f"spec block checksum mismatch for {path}: manifest says "
            f"{manifest['checksum'][:16]}..., specs hash to {block.checksum[:16]}..."
        )
    return block


def assign_blocks_to_splits(
    blocks: Sequence[SpecBlock],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[str]]:
    """Split spec ids into train/validation/test **within** each block.

    Splitting happens at spec level, so every generator's realization of a spec
    lands in the same split. Splitting within each block rather than assigning
    whole blocks to splits keeps all three splits representative of all blocks;
    assigning block 0 to train and block 1 to test would confound split with
    draw.

    Args:
        blocks: Blocks to split.
        ratios: Train/validation/test proportions; must sum to 1.

    Returns:
        Mapping of split name to spec ids.

    Raises:
        ValueError: If ratios do not sum to 1.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(
            f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}"
        )

    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    train_ratio, val_ratio, _ = ratios

    for block in blocks:
        ids = list(dict.fromkeys(block.spec_ids))  # de-duplicate, keep draw order
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(ids[:n_train])
        splits["validation"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])

    return splits

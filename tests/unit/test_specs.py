"""Tests for immutable generation specs and spec blocks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from shelf.sampler.generator import DocumentLength, PromptVariant, Register
from shelf.sampler.specs import (
    SPEC_SCHEMA_VERSION,
    DocumentSpec,
    SpecBlock,
    assign_blocks_to_splits,
    draw_spec_blocks,
    load_spec_block,
    save_spec_block,
)


def spec(**overrides) -> DocumentSpec:
    # Annotated so the **splat below is checked as a mapping rather than
    # field-by-field; without it every optional DocumentSpec field reports a
    # spurious mismatch against the heterogeneous literal.
    base: dict[str, Any] = dict(
        lcc_code="Q",
        lcc_name="Science",
        lcgft_form="Lectures",
        lcgft_category="Instructional and educational works",
        topics=("Mathematics", "Education"),
        target_length=DocumentLength.MEDIUM,
        register=Register.ACADEMIC,
        audience="Students",
        geographic=("France",),
    )
    base.update(overrides)
    return DocumentSpec(**base)


class TestSpecIdentity:
    def test_identical_content_gives_identical_id(self):
        assert spec().spec_id == spec().spec_id

    def test_id_is_stable_across_field_order(self):
        a = DocumentSpec(
            lcc_code="Q",
            lcc_name="Science",
            lcgft_form="Lectures",
            lcgft_category="Cat",
            topics=("A", "B"),
            target_length=DocumentLength.SHORT,
            register=Register.FORMAL,
        )
        b = DocumentSpec(
            register=Register.FORMAL,
            target_length=DocumentLength.SHORT,
            topics=("A", "B"),
            lcgft_category="Cat",
            lcgft_form="Lectures",
            lcc_name="Science",
            lcc_code="Q",
        )
        assert a.spec_id == b.spec_id

    @pytest.mark.parametrize(
        "field,value",
        [
            ("lcc_code", "K"),
            ("lcgft_form", "Poetry"),
            ("topics", ("Different",)),
            ("target_length", DocumentLength.LONG),
            ("register", Register.CASUAL),
            ("audience", "Experts"),
            ("geographic", ("Japan",)),
        ],
    )
    def test_any_content_change_changes_the_id(self, field, value):
        assert spec().spec_id != spec(**{field: value}).spec_id

    def test_block_id_does_not_affect_identity(self):
        """Same content in two blocks is the same spec — that must be visible."""
        assert spec(block_id="a").spec_id == spec(block_id="b").spec_id

    def test_topic_order_is_significant(self):
        assert spec(topics=("A", "B")).spec_id != spec(topics=("B", "A")).spec_id

    def test_schema_version_is_hashed_in(self):
        assert spec().content_dict()["schema"] == SPEC_SCHEMA_VERSION

    def test_spec_is_frozen(self):
        with pytest.raises(Exception):
            spec().lcc_code = "Z"  # type: ignore[misc]

    def test_id_is_hex_and_stable_length(self):
        sid = spec().spec_id
        assert len(sid) == 16
        int(sid, 16)


class TestSerialization:
    def test_roundtrip_preserves_everything(self):
        original = spec(prompt_variant=PromptVariant.ARCHIVAL, block_id="blk")
        restored = DocumentSpec.from_dict(original.to_dict())
        assert restored == original
        assert restored.spec_id == original.spec_id

    def test_roundtrip_without_optional_fields(self):
        original = spec(audience=None, geographic=(), prompt_variant=None)
        assert DocumentSpec.from_dict(original.to_dict()) == original

    def test_tampered_content_is_rejected(self):
        payload = spec().to_dict()
        payload["lcc_code"] = "Z"
        with pytest.raises(ValueError, match="spec_id mismatch"):
            DocumentSpec.from_dict(payload)

    def test_missing_spec_id_is_tolerated(self):
        payload = spec().to_dict()
        del payload["spec_id"]
        assert DocumentSpec.from_dict(payload).spec_id == spec().spec_id

    def test_to_document_carries_taxonomy(self):
        doc = spec().to_document()
        assert doc.lcc.code == "Q"
        assert doc.lcgft.form == "Lectures"
        assert doc.topics == ["Mathematics", "Education"]
        assert doc.geographic == ["France"]


class TestDrawSpecBlocks:
    def test_shape(self):
        blocks = draw_spec_blocks(n_blocks=3, specs_per_block=20, base_seed=1)
        assert len(blocks) == 3
        assert all(len(b) == 20 for b in blocks)

    def test_blocks_are_independent(self):
        blocks = draw_spec_blocks(n_blocks=3, specs_per_block=40, base_seed=1)
        ids = [set(b.spec_ids) for b in blocks]
        assert not (ids[0] & ids[1])
        assert not (ids[0] & ids[2])
        assert not (ids[1] & ids[2])

    def test_each_block_has_its_own_seed(self):
        blocks = draw_spec_blocks(n_blocks=3, specs_per_block=5, base_seed=100)
        assert [b.seed for b in blocks] == [100, 101, 102]

    def test_deterministic_under_seed(self):
        a = draw_spec_blocks(n_blocks=2, specs_per_block=25, base_seed=7)
        b = draw_spec_blocks(n_blocks=2, specs_per_block=25, base_seed=7)
        assert [x.checksum for x in a] == [x.checksum for x in b]

    def test_different_seed_gives_different_content(self):
        a = draw_spec_blocks(n_blocks=1, specs_per_block=25, base_seed=7)
        b = draw_spec_blocks(n_blocks=1, specs_per_block=25, base_seed=8)
        assert a[0].checksum != b[0].checksum

    def test_specs_carry_their_block_id(self):
        blocks = draw_spec_blocks(n_blocks=2, specs_per_block=5, base_seed=3)
        for block in blocks:
            assert all(s.block_id == block.block_id for s in block.specs)

    def test_rejects_non_positive_arguments(self):
        with pytest.raises(ValueError, match="n_blocks"):
            draw_spec_blocks(n_blocks=0)
        with pytest.raises(ValueError, match="specs_per_block"):
            draw_spec_blocks(specs_per_block=0)

    def test_block_is_iterable_and_sized(self):
        block = draw_spec_blocks(n_blocks=1, specs_per_block=6, base_seed=2)[0]
        assert len(list(block)) == len(block) == 6


class TestSpecBlockIntegrity:
    def test_manifest_contents(self):
        block = draw_spec_blocks(n_blocks=1, specs_per_block=10, base_seed=5)[0]
        m = block.to_manifest()
        assert m["n_specs"] == 10
        assert m["schema_version"] == SPEC_SCHEMA_VERSION
        assert len(m["checksum"]) == 64

    def test_duplicate_detection(self):
        s = spec()
        block = SpecBlock(block_id="b", seed=1, specs=(s, s, spec(lcc_code="K")))
        assert block.duplicate_spec_ids() == {s.spec_id: 2}

    def test_no_duplicates_in_a_clean_block(self):
        block = draw_spec_blocks(n_blocks=1, specs_per_block=50, base_seed=9)[0]
        assert block.duplicate_spec_ids() == {}

    def test_checksum_changes_with_content(self):
        a = SpecBlock(block_id="b", seed=1, specs=(spec(),))
        b = SpecBlock(block_id="b", seed=1, specs=(spec(lcc_code="K"),))
        assert a.checksum != b.checksum


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        block = draw_spec_blocks(n_blocks=1, specs_per_block=15, base_seed=4)[0]
        path = save_spec_block(block, tmp_path / "b.jsonl")
        restored = load_spec_block(path)
        assert restored.checksum == block.checksum
        assert restored.spec_ids == block.spec_ids
        assert restored.seed == block.seed

    def test_edited_spec_is_detected(self, tmp_path):
        block = draw_spec_blocks(n_blocks=1, specs_per_block=5, base_seed=4)[0]
        path = save_spec_block(block, tmp_path / "b.jsonl")
        lines = path.read_text().splitlines()
        payload = json.loads(lines[1])
        payload["lcc_code"] = "ZZ"
        lines[1] = json.dumps(payload)
        path.write_text("\n".join(lines))
        with pytest.raises(ValueError, match="spec_id mismatch"):
            load_spec_block(path)

    def test_truncated_block_is_detected(self, tmp_path):
        block = draw_spec_blocks(n_blocks=1, specs_per_block=10, base_seed=4)[0]
        path = save_spec_block(block, tmp_path / "b.jsonl")
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-3]))
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_spec_block(path)

    def test_empty_file_rejected(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_spec_block(p)

    def test_missing_manifest_rejected(self, tmp_path):
        p = tmp_path / "nomanifest.jsonl"
        p.write_text(json.dumps(spec().to_dict()) + "\n")
        with pytest.raises(ValueError, match="missing manifest"):
            load_spec_block(p)


class TestPreSubclassSpecBlocksStillLoad:
    """The Phase 1 blocks on disk were drawn before `lcc_subclass` existed.

    Their recorded `spec_id`s were computed from a content dict with no
    subclass key, so the field can only be hashed in when it is set. If this
    breaks, every generated document already tied to those ids is orphaned and
    the spec-level split cannot be reconstructed.
    """

    BLOCK_DIR = Path("data/artifacts/spec_blocks")

    def _blocks(self):
        if not self.BLOCK_DIR.is_dir():
            pytest.skip("no spec blocks on disk in this checkout")
        paths = sorted(self.BLOCK_DIR.glob("*.jsonl"))
        if not paths:
            pytest.skip("no spec blocks on disk in this checkout")
        return paths

    def test_every_block_on_disk_loads_and_verifies(self):
        for path in self._blocks():
            block = load_spec_block(path)
            assert len(block) > 0
            assert all(spec.lcc_subclass is None for spec in block)

    def test_recorded_spec_ids_are_unchanged(self):
        for path in self._blocks():
            lines = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ][1:]
            for row in lines:
                assert DocumentSpec.from_dict(row).spec_id == row["spec_id"]


class TestSubclassIsOptionalInIdentity:
    def test_absent_subclass_is_not_hashed_in(self):
        assert "lcc_subclass" not in spec().content_dict()

    def test_present_subclass_changes_identity(self):
        assert spec().spec_id != spec(lcc_subclass="QA").spec_id


class TestSplitAssignment:
    def test_splits_are_disjoint(self):
        blocks = draw_spec_blocks(n_blocks=3, specs_per_block=50, base_seed=6)
        s = assign_blocks_to_splits(blocks)
        assert not (set(s["train"]) & set(s["validation"]))
        assert not (set(s["train"]) & set(s["test"]))
        assert not (set(s["validation"]) & set(s["test"]))

    def test_splits_cover_every_spec(self):
        blocks = draw_spec_blocks(n_blocks=2, specs_per_block=50, base_seed=6)
        s = assign_blocks_to_splits(blocks)
        covered = set(s["train"]) | set(s["validation"]) | set(s["test"])
        expected = {sid for b in blocks for sid in b.spec_ids}
        assert covered == expected

    def test_every_block_contributes_to_every_split(self):
        """Blocks must be split within, not assigned whole to one split."""
        blocks = draw_spec_blocks(n_blocks=3, specs_per_block=50, base_seed=6)
        s = assign_blocks_to_splits(blocks)
        for block in blocks:
            ids = set(block.spec_ids)
            assert ids & set(s["train"])
            assert ids & set(s["validation"])
            assert ids & set(s["test"])

    def test_approximate_ratios(self):
        blocks = draw_spec_blocks(n_blocks=2, specs_per_block=100, base_seed=6)
        s = assign_blocks_to_splits(blocks, ratios=(0.6, 0.2, 0.2))
        assert len(s["train"]) == 120
        assert len(s["validation"]) == 40
        assert len(s["test"]) == 40

    def test_rejects_bad_ratios(self):
        blocks = draw_spec_blocks(n_blocks=1, specs_per_block=10, base_seed=6)
        with pytest.raises(ValueError, match="sum to 1"):
            assign_blocks_to_splits(blocks, ratios=(0.5, 0.2, 0.2))

"""Unit tests for the v0.4 Phase 2 LCC subclass difficulty tier.

Tests cover:
- Pool construction from `data/taxonomies/lcc_subclass_top100.json`: the three
  extraction artifacts (`IN`, `PAR`, `NOT`) are excluded, bare main-class
  letters are excluded by default, and every pooled code has a description the
  prompt can actually reach through `EnrichedDescriptions`.
- Uniform sampling as the default, and the measurable skew that frequency
  weighting would introduce (`KF` alone is 122,484 of the reference corpus).
- `LCCClass` carries a subclass without changing any v0.3.1-observable
  behaviour when it is absent.
- `DocumentSpec` identity: the subclass is hashed in when present and the
  content dict is byte-identical to the pre-subclass one when it is not, so
  spec blocks drawn before the field existed still load and verify.
- `_get_domain_description` resolves a subclass-bearing document to the
  subclass description rather than the parent class description, and prompts
  for subclass-free documents are unchanged.
- The `lcc_subclass_classification` task registration matches the sampler pool.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from shelf.evaluate.registry import (
    LCC_SUBCLASS_CODES,
    get_task,
    list_classification_tasks,
)
from shelf.evaluate.tasks import TaskType
from shelf.sampler.dimensions import (
    LCC_DATA,
    LCC_SUBCLASS_EXTRACTION_ARTIFACTS,
    LCCClass,
    LCCSubclassSampler,
    load_lcc_subclass_pool,
)
from shelf.sampler.document import Document
from shelf.sampler.enriched import EnrichedDescriptions
from shelf.sampler.generator import (
    LCC_SEMANTIC_DESCRIPTIONS,
    DocumentLength,
    Register,
    _get_domain_description,
    build_generation_prompt,
    build_title_prompt,
)
from shelf.sampler.specs import DocumentSpec

pytestmark = pytest.mark.unit


# Size of the default pool. Pinned so a change to the taxonomy files or the
# filtering rules is a deliberate, visible edit rather than a silent drift in
# the label space of a published task.
EXPECTED_POOL_SIZE = 80


@pytest.fixture(scope="module")
def pool():
    return load_lcc_subclass_pool()


@pytest.fixture(scope="module")
def enriched():
    return EnrichedDescriptions.load()


def _doc(lcc: LCCClass, form: str = "Lectures", category: str = "Informational works"):
    return Document(
        lcc=lcc,
        lcgft=__import__("shelf.sampler.dimensions", fromlist=["LCGFTTerm"]).LCGFTTerm(
            category=category, form=form
        ),
        topics=["Research"],
        audience="Experts",
        geographic=[],
        id="doc-1",
    )


# =============================================================================
# Pool construction
# =============================================================================


class TestSubclassPool:
    def test_pool_size(self, pool):
        assert len(pool) == EXPECTED_POOL_SIZE

    def test_pool_is_in_the_planned_60_to_100_band(self, pool):
        # docs/data_plan_v0.4.md section 6 targets 60-100 subclasses.
        assert 60 <= len(pool) <= 100

    def test_extraction_artifacts_are_excluded(self, pool):
        """`IN`, `PAR` and `NOT` are not LCC codes -- see data plan section 4.1."""
        codes = {entry.code for entry in pool}
        assert codes.isdisjoint(LCC_SUBCLASS_EXTRACTION_ARTIFACTS)

    def test_artifacts_excluded_even_with_every_filter_relaxed(self):
        codes = {
            entry.code
            for entry in load_lcc_subclass_pool(
                require_description=False, include_main_classes=True
            )
        }
        assert codes.isdisjoint(LCC_SUBCLASS_EXTRACTION_ARTIFACTS)

    def test_no_subclass_maps_to_a_nonexistent_main_class(self, pool):
        """`IN` claims parent letter I, which LCC does not use."""
        assert all(entry.class_code in LCC_DATA for entry in pool)

    def test_codes_are_unique(self, pool):
        codes = [entry.code for entry in pool]
        assert len(codes) == len(set(codes))

    def test_default_pool_excludes_bare_main_classes(self, pool):
        assert all(len(entry.code) == 2 for entry in pool)

    def test_main_classes_can_be_opted_back_in(self, pool):
        with_main = load_lcc_subclass_pool(include_main_classes=True)
        assert len(with_main) == len(pool) + 16
        assert any(entry.is_main_class for entry in with_main)

    def test_require_description_drops_only_jx(self, pool):
        loose = load_lcc_subclass_pool(require_description=False)
        dropped = {e.code for e in loose} - {e.code for e in pool}
        assert dropped == {"JX"}

    def test_pool_is_rank_ordered(self, pool):
        ranks = [entry.rank for entry in pool]
        assert ranks == sorted(ranks)

    def test_kf_is_the_head_of_the_frequency_table(self, pool):
        assert pool[0].code == "KF"
        assert pool[0].frequency == 122484

    def test_size_slices_the_rank_prefix(self, pool):
        assert [e.code for e in load_lcc_subclass_pool(size=10)] == [
            e.code for e in pool[:10]
        ]

    def test_size_is_validated(self, pool):
        with pytest.raises(ValueError, match="pool size must be >= 1"):
            load_lcc_subclass_pool(size=0)
        with pytest.raises(ValueError, match="exceeds"):
            load_lcc_subclass_pool(size=len(pool) + 1)

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lcc_subclass_pool(data_dir=str(tmp_path))

    def test_parent_names_come_from_lcc_data(self, pool):
        assert all(entry.class_name == LCC_DATA[entry.class_code] for entry in pool)


class TestDescriptionCoverage:
    """A pooled subclass with no reachable description would prompt as a bare
    code -- exactly the self-labeling the generation prompt forbids."""

    def test_every_pooled_code_has_a_raw_description(self, pool):
        assert all(entry.description for entry in pool)

    def test_every_pooled_code_survives_sanitization(self, pool, enriched):
        missing = [
            entry.code for entry in pool if not enriched.for_lcc_subclass(entry.code)
        ]
        assert missing == []

    def test_within_domain_neighbours_get_different_descriptions(self, enriched):
        science = {
            code: enriched.for_lcc_subclass(code) for code in ("QA", "QC", "QH", "QD")
        }
        assert len(set(science.values())) == 4
        law = {code: enriched.for_lcc_subclass(code) for code in ("KF", "KZ")}
        assert len(set(law.values())) == 2


# =============================================================================
# Sampling
# =============================================================================


class TestSubclassSampler:
    def test_sample_returns_lcc_class_with_parent_and_subclass(self):
        drawn = LCCSubclassSampler(seed=42).sample()
        assert drawn.subclass is not None
        assert drawn.code in LCC_DATA
        assert drawn.subclass.startswith(drawn.code)
        assert drawn.name == LCC_DATA[drawn.code]

    def test_deterministic_under_seed(self):
        a = [c.subclass for c in LCCSubclassSampler(seed=7).sample_n(50)]
        b = [c.subclass for c in LCCSubclassSampler(seed=7).sample_n(50)]
        assert a == b

    def test_different_seeds_diverge(self):
        a = [c.subclass for c in LCCSubclassSampler(seed=7).sample_n(50)]
        b = [c.subclass for c in LCCSubclassSampler(seed=8).sample_n(50)]
        assert a != b

    def test_uniform_is_the_default(self, pool):
        sampler = LCCSubclassSampler(seed=42)
        assert sampler._weighting == "uniform"
        counts = Counter(c.subclass for c in sampler.sample_n(8000))
        # Every subclass drawn, and none dominant: uniform over 80 codes gives
        # an expected share of 1.25%, so a 5% ceiling is a very loose bound
        # that frequency weighting would blow straight through.
        assert len(counts) == len(pool)
        assert max(counts.values()) / 8000 < 0.05

    def test_frequency_weighting_is_the_skew_uniform_exists_to_avoid(self):
        counts = Counter(
            c.subclass
            for c in LCCSubclassSampler(seed=42, weighting="frequency").sample_n(4000)
        )
        # KF is 122,484 of the reference corpus, more than the other 79 codes
        # combined, so frequency weighting makes the corpus mostly US law.
        assert counts.most_common(1)[0][0] == "KF"
        assert counts["KF"] / 4000 > 0.5

    def test_codes_can_be_restricted(self):
        sampler = LCCSubclassSampler(codes=["QA", "QC"], seed=1)
        assert set(sampler.codes()) == {"QA", "QC"}
        assert {c.subclass for c in sampler.sample_n(30)} == {"QA", "QC"}

    def test_empty_filter_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            LCCSubclassSampler(codes=["ZZZ"])

    def test_values_and_len(self, pool):
        sampler = LCCSubclassSampler(seed=1)
        assert len(sampler) == len(pool)
        assert all(isinstance(v, LCCClass) for v in sampler.values())

    def test_pool_matches_registry_label_space(self):
        assert sorted(LCCSubclassSampler(seed=1).codes()) == list(LCC_SUBCLASS_CODES)


# =============================================================================
# LCCClass: v0.3.1 behaviour is untouched
# =============================================================================


class TestLCCClassCompatibility:
    def test_subclass_defaults_to_none(self):
        assert LCCClass("Q", "Science").subclass is None
        assert LCCClass("Q", "Science").subclass_name is None

    def test_str_unchanged_without_subclass(self):
        assert str(LCCClass("Q", "Science")) == "Q: Science"

    def test_uri_unchanged_without_subclass(self):
        assert (
            LCCClass("Q", "Science").uri
            == "http://id.loc.gov/authorities/classification/Q"
        )

    def test_uri_still_derives_from_the_main_class_with_a_subclass(self):
        """The `lcc_uri` published in the dataset is a main-class URI; the
        subclass is a separate field, so this must not silently change."""
        lcc = LCCClass("Q", "Science", subclass="QA", subclass_name="Mathematics")
        assert lcc.uri == "http://id.loc.gov/authorities/classification/Q"

    def test_str_shows_the_subclass_when_present(self):
        lcc = LCCClass("Q", "Science", subclass="QA", subclass_name="Mathematics")
        assert str(lcc) == "QA: Mathematics"

    def test_equality_unaffected(self):
        assert LCCClass("Q", "Science") == LCCClass("Q", "Science")
        assert LCCClass("Q", "Science") != LCCClass("Q", "Science", subclass="QA")


# =============================================================================
# Spec identity
# =============================================================================


def _spec(**overrides) -> DocumentSpec:
    base: dict[str, Any] = {
        "lcc_code": "Q",
        "lcc_name": "Science",
        "lcgft_form": "Lectures",
        "lcgft_category": "Instructional and educational works",
        "topics": ("Mathematics",),
        "target_length": DocumentLength.MEDIUM,
        "register": Register.ACADEMIC,
    }
    base.update(overrides)
    return DocumentSpec(**base)


class TestSpecIdentityWithSubclass:
    def test_spec_id_changes_when_the_subclass_is_added(self):
        assert _spec().spec_id != _spec(lcc_subclass="QA").spec_id

    def test_spec_id_changes_between_subclasses(self):
        assert _spec(lcc_subclass="QA").spec_id != _spec(lcc_subclass="QC").spec_id

    def test_subclass_name_is_part_of_identity(self):
        a = _spec(lcc_subclass="QA", lcc_subclass_name="Mathematics")
        b = _spec(lcc_subclass="QA", lcc_subclass_name="Maths")
        assert a.spec_id != b.spec_id

    def test_content_dict_is_unchanged_without_a_subclass(self):
        """The hash of every spec drawn before this field existed must be
        untouched, or the spec blocks already on disk stop verifying."""
        keys = set(_spec().content_dict())
        assert "lcc_subclass" not in keys
        assert "lcc_subclass_name" not in keys

    def test_pinned_spec_id_for_a_subclass_free_spec(self):
        # Pinned against the pre-subclass implementation. A change here means
        # v0.3.1/Phase-1 spec identity moved.
        assert _spec().spec_id == "25be2cfe50f93256"

    def test_roundtrip_with_subclass(self):
        original = _spec(lcc_subclass="QA", lcc_subclass_name="Mathematics")
        restored = DocumentSpec.from_dict(original.to_dict())
        assert restored == original
        assert restored.spec_id == original.spec_id

    def test_a_spec_block_written_without_the_field_still_loads(self):
        """Simulates a spec file predating the subclass field."""
        payload = _spec().to_dict()
        assert "lcc_subclass" not in payload
        assert DocumentSpec.from_dict(payload) == _spec()

    def test_to_document_carries_the_subclass(self):
        doc = _spec(lcc_subclass="QA", lcc_subclass_name="Mathematics").to_document()
        assert doc.lcc.code == "Q"
        assert doc.lcc.subclass == "QA"
        assert doc.lcc.subclass_name == "Mathematics"

    def test_from_document_reads_the_subclass_back(self):
        lcc = LCCSubclassSampler(seed=3).sample()
        spec = DocumentSpec.from_document(
            _doc(lcc), target_length=DocumentLength.SHORT, register=Register.ACADEMIC
        )
        assert spec.lcc_subclass == lcc.subclass
        assert spec.lcc_subclass_name == lcc.subclass_name

    def test_spec_roundtrip_through_document(self):
        original = _spec(lcc_subclass="QA", lcc_subclass_name="Mathematics")
        rebuilt = DocumentSpec.from_document(
            original.to_document(),
            target_length=original.target_length,
            register=original.register,
        )
        assert rebuilt.lcc_subclass == "QA"


# =============================================================================
# Prompt conditioning
# =============================================================================


class TestSubclassPrompting:
    def test_subclass_description_replaces_the_parent(self, enriched):
        parent = _get_domain_description("Q", "Science", enriched)
        child = _get_domain_description("Q", "Science", enriched, subclass="QA")
        assert parent == LCC_SEMANTIC_DESCRIPTIONS["Q"]
        assert child == enriched.for_lcc_subclass("QA")
        assert child != parent

    def test_sibling_subclasses_get_different_conditioning(self, enriched):
        descriptions = {
            code: _get_domain_description("Q", "Science", enriched, subclass=code)
            for code in ("QA", "QC", "QH", "QD", "QK")
        }
        assert len(set(descriptions.values())) == 5

    def test_no_enrichment_means_no_change(self):
        """Without an `enriched` argument there is nothing to resolve, so the
        subclass cannot alter the v0.3.1 output."""
        assert _get_domain_description(
            "Q", "Science", None, subclass="QA"
        ) == _get_domain_description("Q", "Science")

    def test_unknown_subclass_falls_back_to_the_parent(self, enriched):
        assert (
            _get_domain_description("Q", "Science", enriched, subclass="QZZ")
            == LCC_SEMANTIC_DESCRIPTIONS["Q"]
        )

    def test_generation_prompt_uses_the_subclass(self, enriched):
        lcc = LCCClass("Q", "Science", subclass="QA", subclass_name="Mathematics")
        prompt = build_generation_prompt(
            _doc(lcc), DocumentLength.SHORT, Register.ACADEMIC, enriched
        )
        assert f"subject area: {enriched.for_lcc_subclass('QA')}" in prompt
        assert LCC_SEMANTIC_DESCRIPTIONS["Q"] not in prompt

    def test_title_prompt_uses_the_subclass(self, enriched):
        lcc = LCCClass("Q", "Science", subclass="QA", subclass_name="Mathematics")
        assert enriched.for_lcc_subclass("QA") in build_title_prompt(
            _doc(lcc), enriched
        )

    def test_prompt_does_not_name_the_subclass_code(self, enriched):
        """The prediction target must not appear in the prompt."""
        for code in ("QA", "QC", "KF", "KZ", "TK", "RA"):
            entry = next(e for e in load_lcc_subclass_pool() if e.code == code)
            lcc = LCCClass(
                entry.class_code,
                entry.class_name,
                subclass=entry.code,
                subclass_name=entry.caption,
            )
            prompt = build_generation_prompt(
                _doc(lcc), DocumentLength.SHORT, Register.ACADEMIC, enriched
            )
            assert code not in prompt

    def test_subclass_free_prompt_is_unchanged_by_enrichment_argument(self, enriched):
        lcc = LCCClass("Q", "Science")
        with_enrichment = build_generation_prompt(
            _doc(lcc), DocumentLength.SHORT, Register.ACADEMIC, enriched
        )
        without = build_generation_prompt(
            _doc(lcc), DocumentLength.SHORT, Register.ACADEMIC
        )
        assert "subject area: " + LCC_SEMANTIC_DESCRIPTIONS["Q"] in without
        assert with_enrichment == without


# =============================================================================
# Task registration
# =============================================================================


class TestTaskRegistration:
    def test_task_is_registered(self):
        assert "lcc_subclass_classification" in list_classification_tasks()

    def test_task_spec(self):
        task = get_task("lcc_subclass_classification")
        assert task.task_type is TaskType.CLASSIFICATION
        assert task.label_field == "lcc_subclass"
        assert task.primary_metric == "macro_f1"
        assert task.label_space == LCC_SUBCLASS_CODES

    def test_label_space_size_and_uniqueness(self):
        assert len(LCC_SUBCLASS_CODES) == EXPECTED_POOL_SIZE
        assert len(set(LCC_SUBCLASS_CODES)) == len(LCC_SUBCLASS_CODES)

    def test_label_space_is_sorted(self):
        assert list(LCC_SUBCLASS_CODES) == sorted(LCC_SUBCLASS_CODES)

    def test_label_space_holds_no_artifacts(self):
        assert set(LCC_SUBCLASS_CODES).isdisjoint(LCC_SUBCLASS_EXTRACTION_ARTIFACTS)

    def test_every_label_is_a_real_two_letter_code(self):
        assert all(
            len(code) == 2 and code[0] in LCC_DATA for code in LCC_SUBCLASS_CODES
        )

    def test_label_space_is_disjoint_from_the_main_class_task(self):
        main = get_task("lcc_classification")
        assert set(LCC_SUBCLASS_CODES).isdisjoint(main.label_space or ())


# =============================================================================
# Dataset schema
# =============================================================================


class TestDatasetSchema:
    def test_subclass_fields_default_to_empty(self):
        from shelf.hub.dataset import _normalize_document

        row = _normalize_document({"id": "a", "body": "text"})
        assert row["lcc_subclass"] == ""
        assert row["lcc_subclass_name"] == ""

    def test_subclass_fields_are_carried_through(self):
        from shelf.hub.dataset import _normalize_document

        row = _normalize_document(
            {
                "id": "a",
                "body": "text",
                "lcc_subclass": "QA",
                "lcc_subclass_name": "Mathematics",
            }
        )
        assert row["lcc_subclass"] == "QA"
        assert row["lcc_subclass_name"] == "Mathematics"

    def test_null_subclass_normalizes_to_empty(self):
        from shelf.hub.dataset import _normalize_document

        row = _normalize_document(
            {"id": "a", "body": "t", "lcc_subclass": None, "lcc_subclass_name": None}
        )
        assert row["lcc_subclass"] == ""


class TestBaselineConfig:
    def test_task_is_wired_into_the_baseline_suite(self):
        from pathlib import Path

        import yaml

        config = yaml.safe_load(
            Path("scripts/baselines/config.yaml").read_text(encoding="utf-8")
        )
        assert "lcc_subclass_classification" in config["tasks"]["classification"]

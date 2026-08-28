"""Tests for enriched LC description loading and sanitization."""

from __future__ import annotations

import json

import pytest
from shelf.sampler.enriched import EnrichedDescriptions, EnrichedEntry
from shelf.sampler.leakage import find_leaked_labels

CATEGORIES = ("Cartographic materials", "Informational works", "Visual works")


def write_export(path, labels, key="label"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels}), encoding="utf-8")


@pytest.fixture
def enriched_dir(tmp_path):
    write_export(
        tmp_path / "lcgft.json",
        [
            {
                "label": "Maps",
                "description": "a kind of Cartographic materials, Visual works; includes Atlases",
                "description_source": "hierarchy",
                "description_verbatim": "False",
                "uri": "http://id.loc.gov/x/1",
            },
            {
                "label": "Legal briefs",
                "description": "documents representing the arguments of one or more parties",
                "description_source": "scope_note",
                "description_verbatim": "True",
            },
            {
                "label": "Prayers",
                "description": "Prayers and Litanies",
                "description_source": "hierarchy",
                "description_verbatim": "False",
            },
            {"label": "NoDesc", "description": "", "description_verbatim": "False"},
        ],
    )
    write_export(
        tmp_path / "lcc_subclass_top100.json",
        [
            {
                "id": "QA",
                "description": "mathematics, geometry, algebra",
                "description_source": "caption_hierarchy",
                "description_verbatim": "False",
            }
        ],
        key="id",
    )
    return tmp_path


class TestLoading:
    def test_loads_forms_and_subclasses(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        assert e.for_form("Legal briefs")
        assert e.for_lcc_subclass("QA") == "mathematics, geometry, algebra"

    def test_missing_directory_is_not_an_error(self, tmp_path):
        e = EnrichedDescriptions.load(tmp_path / "nope", forbidden_labels=CATEGORIES)
        assert e.for_form("Maps") is None
        assert e.coverage()["forms"]["total"] == 0

    def test_entries_without_description_are_skipped(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        assert e.for_form("NoDesc") is None

    def test_unknown_label_returns_none(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        assert e.for_form("Nonexistent") is None
        assert e.for_topic("Nonexistent") is None
        assert e.for_geographic("Nonexistent") is None
        assert e.for_lcc_subclass("ZZ") is None


class TestSanitization:
    def test_category_names_are_stripped(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        desc = e.for_form("Maps")
        assert desc == "Atlases"
        assert find_leaked_labels(desc, CATEGORIES) == ()

    def test_own_label_is_stripped(self, enriched_dir):
        """A form's description must not name the form itself."""
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        desc = e.for_form("Prayers")
        assert desc is not None
        assert find_leaked_labels(desc, ("Prayers",)) == ()

    def test_clean_description_is_untouched(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        assert e.for_form("Legal briefs").startswith("documents representing")

    def test_entry_records_what_was_removed(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        entry = e.forms["Maps"]
        assert entry.sanitized
        assert "Cartographic materials" in entry.removed_labels

    def test_audit_is_empty_after_load(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        assert e.audit_leakage(CATEGORIES) == {}

    def test_description_reduced_to_nothing_is_dropped(self, tmp_path):
        write_export(
            tmp_path / "lcgft.json",
            [
                {
                    "label": "X",
                    "description": "a kind of Visual works",
                    "description_verbatim": "False",
                }
            ],
        )
        e = EnrichedDescriptions.load(tmp_path, forbidden_labels=CATEGORIES)
        assert e.for_form("X") is None


class TestVerbatimOnly:
    def test_keeps_only_scope_notes(self, enriched_dir):
        e = EnrichedDescriptions.load(
            enriched_dir, forbidden_labels=CATEGORIES, verbatim_only=True
        )
        assert e.for_form("Legal briefs")
        assert e.for_form("Maps") is None
        assert e.for_lcc_subclass("QA") is None

    def test_reduces_coverage(self, enriched_dir):
        full = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        verb = EnrichedDescriptions.load(
            enriched_dir, forbidden_labels=CATEGORIES, verbatim_only=True
        )
        assert verb.coverage()["forms"]["total"] < full.coverage()["forms"]["total"]


class TestCoverageReport:
    def test_counts_are_consistent(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        forms = e.coverage()["forms"]
        assert forms["total"] == forms["verbatim_scope_notes"] + forms["derived"]

    def test_reports_every_taxonomy(self, enriched_dir):
        e = EnrichedDescriptions.load(enriched_dir, forbidden_labels=CATEGORIES)
        assert set(e.coverage()) == {
            "forms",
            "topics",
            "geographic",
            "lcc_subclasses",
        }


class TestEnrichedEntry:
    def test_is_derived_is_the_inverse_of_verbatim(self):
        assert EnrichedEntry("a", "d", "scope_note", verbatim=True).is_derived is False
        assert EnrichedEntry("a", "d", "hierarchy", verbatim=False).is_derived is True


class TestRealExport:
    """Guards against regressions in the checked-in enriched data."""

    def test_real_export_has_no_leakage(self):
        from shelf.evaluate.registry import LCGFT_CATEGORIES

        e = EnrichedDescriptions.load()
        if e.coverage()["forms"]["total"] == 0:
            pytest.skip("enriched export not present")
        assert e.audit_leakage(tuple(LCGFT_CATEGORIES)) == {}

    def test_no_form_description_contains_its_own_label(self):
        e = EnrichedDescriptions.load()
        if e.coverage()["forms"]["total"] == 0:
            pytest.skip("enriched export not present")
        offenders = [
            entry.label
            for entry in e.forms.values()
            if find_leaked_labels(entry.description, (entry.label,))
        ]
        assert offenders == []

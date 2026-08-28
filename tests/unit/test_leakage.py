"""Tests for taxonomy label-leakage detection and sanitization."""

from __future__ import annotations

import pytest
from shelf.sampler.leakage import (
    find_leaked_labels,
    has_self_labeling,
    sanitize_description,
    scan_document,
)

CATEGORIES = (
    "Cartographic materials",
    "Commemorative works",
    "Creative nonfiction",
    "Discursive works",
    "Ephemera",
    "Informational works",
    "Instructional and educational works",
    "Law materials",
    "Literature",
    "Music",
    "Recreational works",
    "Religious materials",
    "Sound recordings",
    "Visual works",
)


class TestFindLeakedLabels:
    def test_detects_a_single_label(self):
        assert find_leaked_labels("a kind of Law materials", CATEGORIES) == (
            "Law materials",
        )

    def test_detects_multiple_labels(self):
        found = find_leaked_labels(
            "a kind of Cartographic materials, Informational works, Visual works",
            CATEGORIES,
        )
        assert set(found) == {
            "Cartographic materials",
            "Informational works",
            "Visual works",
        }

    def test_is_case_insensitive(self):
        assert find_leaked_labels("about LAW MATERIALS here", CATEGORIES) == (
            "Law materials",
        )

    def test_respects_word_boundaries(self):
        """'Music' must not fire on 'musicology'."""
        assert find_leaked_labels("musicology is a discipline", CATEGORIES) == ()

    def test_matches_label_followed_by_punctuation(self):
        assert "Music" in find_leaked_labels("the subject is Music.", CATEGORIES)

    def test_clean_text_returns_nothing(self):
        assert find_leaked_labels("mining engineering and metallurgy", CATEGORIES) == ()

    def test_empty_text(self):
        assert find_leaked_labels("", CATEGORIES) == ()

    def test_empty_labels(self):
        assert find_leaked_labels("anything at all", []) == ()

    def test_short_labels_are_ignored_by_default(self):
        """Single letters and codes appear constantly in ordinary prose."""
        assert find_leaked_labels("the G force and KF value", ("G", "KF")) == ()

    def test_min_length_is_configurable(self):
        assert find_leaked_labels("the KF value", ("KF",), min_length=2) == ("KF",)

    def test_results_are_longest_first(self):
        found = find_leaked_labels(
            "Informational works and Music", ("Music", "Informational works")
        )
        assert found[0] == "Informational works"

    def test_deduplicates_repeats(self):
        assert find_leaked_labels("Music, Music, Music", CATEGORIES) == ("Music",)


class TestSelfLabeling:
    @pytest.mark.parametrize(
        "text",
        [
            "Document Type: Lecture",
            "Subject Area: Medicine",
            "Category: Poetry",
            "LCGFT: Maps",
        ],
    )
    def test_detects_classification_headers(self, text):
        assert has_self_labeling(text)

    def test_detects_field_announcement(self):
        assert has_self_labeling("In the field of medicine, practitioners...")

    def test_detects_meta_commentary(self):
        assert has_self_labeling("This satire explores the folly of empire.")

    def test_does_not_flag_legitimate_domain_vocabulary(self):
        """Domain words are expected and must not trip the gate."""
        assert has_self_labeling("Civil law systems trace their origins to Rome.") == ()
        assert has_self_labeling("The patient presented with acute symptoms.") == ()
        assert has_self_labeling("The court ruled in favor of the appellant.") == ()

    def test_empty_text(self):
        assert has_self_labeling("") == ()

    def test_generic_header_must_be_at_line_start(self):
        assert has_self_labeling("we discussed the category: of things") == ()

    @pytest.mark.parametrize(
        "text",
        [
            "**Disciplina (LCC: Language and Literature)**",
            "1. **Folk Music (LCGFT: Folk music)**",
            "Tipo (LCGFT): Field recordings (Sound recordings)",
            "see also LCSH: Globalization",
        ],
    )
    def test_taxonomy_codes_are_caught_mid_line(self, text):
        """Real v0.3.1 leakage never appeared at a line start.

        202 published documents carry one of these markers, inside parentheses
        or YAML blocks; a line-anchored pattern missed them.
        """
        assert has_self_labeling(text)

    def test_taxonomy_codes_do_not_fire_on_ordinary_words(self):
        assert has_self_labeling("The lcc value was computed later.") == ()
        assert has_self_labeling("This genre of music emerged in the 1970s.") == ()


class TestScanDocument:
    def test_clean_document(self):
        report = scan_document("Roman legal codes shaped later systems.", CATEGORIES)
        assert report.is_clean
        assert report.as_dict()["is_clean"] is True

    def test_flags_both_kinds(self):
        report = scan_document(
            "Document Type: essay\nLaw materials: contracts", CATEGORIES
        )
        assert not report.is_clean
        assert report.leaked_labels == ("Law materials",)
        assert report.self_labeling

    def test_bare_vocabulary_is_not_leakage_in_a_document(self):
        """ "Music" and "Literature" are ordinary English words.

        Applying the bare word-boundary check to documents flagged 15-20% of
        real generations, essentially all on legitimate domain vocabulary that
        GENERATION_INSTRUCTIONS explicitly permits.
        """
        for text in [
            "the director of music at a parish in Over-the-Rhine",
            "English literature of the Victorian period",
            "IN THE SUPERIOR COURT OF THE STATE OF CALIFORNIA",
        ]:
            assert scan_document(text, CATEGORIES).is_clean, text

    def test_announced_label_is_still_leakage(self):
        for text in ["Category: Music", "(LCGFT: Music)", "Genre: Literature"]:
            assert not scan_document(text, CATEGORIES).is_clean, text

    def test_bare_check_is_available_for_prompt_text(self):
        """Descriptions bound for a prompt use the stricter bare-word check."""
        report = scan_document(
            "a kind of Law materials", CATEGORIES, require_labelling_context=False
        )
        assert report.leaked_labels == ("Law materials",)

    def test_serializes(self):
        payload = scan_document("Genre: Music", CATEGORIES).as_dict()
        assert payload["leaked_labels"] == ["Music"]
        assert payload["is_clean"] is False

    def test_works_without_labels(self):
        assert scan_document("plain text").is_clean


class TestSanitizeDescription:
    def test_strips_leaked_categories_and_keeps_content(self):
        out = sanitize_description(
            "a kind of Cartographic materials, Informational works, Visual works; "
            "includes Aeronautical charts, Bottle-charts",
            CATEGORIES,
        )
        assert out == "Aeronautical charts, Bottle-charts"
        assert find_leaked_labels(out, CATEGORIES) == ()

    def test_leaves_clean_text_untouched(self):
        text = "mining engineering, metallurgy; including mineral industries"
        assert sanitize_description(text, CATEGORIES) == text

    def test_longest_label_removed_first(self):
        """Stripping 'works' must not damage 'Informational works'."""
        out = sanitize_description(
            "a kind of Informational works; includes Reports",
            ("works", "Informational works"),
        )
        assert "Informational" not in out
        assert "Reports" in out

    def test_can_return_empty(self):
        assert sanitize_description("a kind of Music", CATEGORIES) == ""

    def test_empty_input(self):
        assert sanitize_description("", CATEGORIES) == ""

    def test_no_dangling_punctuation(self):
        out = sanitize_description(
            "a kind of Law materials, Informational works; includes Bills", CATEGORIES
        )
        assert out == "Bills"
        assert not out.startswith((",", ";", " "))
        assert not out.endswith((",", ";"))

    def test_output_is_always_leak_free(self):
        samples = [
            "a kind of Music, Sound recordings",
            "a kind of Literature; includes Sagas, Epics",
            "a kind of Visual works, Ephemera; includes Posters",
            "Instructional and educational works and Recreational works",
        ]
        for text in samples:
            assert (
                find_leaked_labels(sanitize_description(text, CATEGORIES), CATEGORIES)
                == ()
            )

    def test_short_labels_are_preserved(self):
        text = "the G class covers geography"
        assert sanitize_description(text, ("G",)) == text

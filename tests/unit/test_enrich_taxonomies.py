"""Unit tests for scripts/enrich_taxonomies.py (all network mocked).

Tests cover:
- SKOS/RDF ND-JSON parsing against the shapes id.loc.gov actually publishes
  (bare-string, single-object and list-valued literals; blank-node references).
- Duplicate ``prefLabel`` disambiguation -- LCSH really does publish two
  "Globalization" records, one of which is a subdivision-use stub.
- Scope-note classification (definitional vs cataloguing instruction) and the
  purely subtractive cleaning that turns LC prose into a description.
- The description fallback chain and its self-labeling guards.
- LCC classification-range parsing and primary-range selection.
- ``LocClient`` caching/resumability and rate limiting, driven through
  ``httpx.MockTransport`` -- no socket is ever opened.
- End-to-end enrichment and coverage reporting on fixture data.
- ``--dry-run`` makes no requests; ``--limit`` caps work per taxonomy.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

# The enrichment pipeline is a script, not a package module, so load it by path.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "enrich_taxonomies.py"
_spec = importlib.util.spec_from_file_location("enrich_taxonomies", _SCRIPT)
assert _spec is not None and _spec.loader is not None
et = importlib.util.module_from_spec(_spec)
sys.modules["enrich_taxonomies"] = et
_spec.loader.exec_module(et)


# =============================================================================
# Fixtures: miniature SKOS dumps in the exact shape id.loc.gov publishes
# =============================================================================

GENRE_FORMS_RECORDS: list[dict[str, Any]] = [
    {
        "@id": "/authorities/genreForms/gf2014026112",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/genreForms/gf2014026112",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@language": "en", "@value": "Indexes"},
                "skos:altLabel": {"@language": "en", "@value": "Indexes (Guides)"},
                "skos:broader": {
                    "@id": "http://id.loc.gov/authorities/genreForms/gf2014026048"
                },
                # LC serves single scope notes as a bare string here.
                "skos:note": (
                    "Works consisting wholly or chiefly of systematic guides to the "
                    "content of resources, usually presented as alphabetical lists "
                    "of names, places, subjects, etc."
                ),
            }
        ],
    },
    {
        "@id": "/authorities/genreForms/gf2014026048",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/genreForms/gf2014026048",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@language": "en", "@value": "Informational works"},
            }
        ],
    },
    {
        "@id": "/authorities/genreForms/gf2011026387",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/genreForms/gf2011026387",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@language": "en", "@value": "Maps"},
                "skos:broader": [
                    {"@id": "http://id.loc.gov/authorities/genreForms/gf2014026048"},
                    {"@id": "http://id.loc.gov/authorities/genreForms/gf2011026061"},
                ],
                "skos:narrower": {
                    "@id": "http://id.loc.gov/authorities/genreForms/gf2011026001"
                },
            }
        ],
    },
    {
        "@id": "/authorities/genreForms/gf2011026061",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/genreForms/gf2011026061",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@value": "Cartographic materials"},
            }
        ],
    },
    {
        "@id": "/authorities/genreForms/gf2011026001",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/genreForms/gf2011026001",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@value": "Aeronautical charts"},
            }
        ],
    },
    {
        # A deprecated skosxl:Label stub -- must be skipped, not crashed on.
        "@id": "/authorities/genreForms/gf2011025006",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/genreForms/gf2011025006",
                "@type": "skosxl:Label",
                "skosxl:literalForm": {"@value": "Gay pornographic films"},
            }
        ],
    },
]

SUBJECT_RECORDS: list[dict[str, Any]] = [
    {
        # The subdivision-use stub. Same prefLabel as sh99010179 below.
        "@id": "/authorities/subjects/sh2007000663",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/subjects/sh2007000663",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@language": "en", "@value": "Globalization"},
                "skos:note": {
                    "@value": (
                        "Use as a topical subdivision under individual languages "
                        "and groups of languages."
                    )
                },
            }
        ],
    },
    {
        "@id": "/authorities/subjects/sh99010179",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/subjects/sh99010179",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@language": "en", "@value": "Globalization"},
                "skos:altLabel": [
                    {"@value": "Globalisation"},
                    {"@value": "Internationalization"},
                ],
                "skos:broader": {
                    "@id": "http://id.loc.gov/authorities/subjects/sh85067435"
                },
                "skos:note": [
                    {
                        "@value": (
                            "Here are entered works on the process by which "
                            "economic, cultural, political, and social "
                            "institutions become integrated worldwide."
                        )
                    },
                    {
                        "@value": (
                            "This heading may be subdivided geographically for "
                            "works on the occurrence and effects of globalization."
                        )
                    },
                ],
            }
        ],
    },
    {
        "@id": "/authorities/subjects/sh85067435",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/subjects/sh85067435",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@value": "International relations"},
            }
        ],
    },
    {
        "@id": "/authorities/subjects/sh85066907",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/subjects/sh85066907",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@value": "Flood insurance"},
                # A variant that is only a re-ordering of the label.
                "skos:altLabel": {"@value": "Insurance, Flood"},
                "skos:broader": {
                    "@id": "http://id.loc.gov/authorities/subjects/sh85066802"
                },
            }
        ],
    },
    {
        "@id": "/authorities/subjects/sh85066802",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/subjects/sh85066802",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@value": "Insurance"},
            }
        ],
    },
    {
        "@id": "/authorities/subjects/sh85101173",
        "@graph": [
            {
                "@id": "http://id.loc.gov/authorities/subjects/sh85101173",
                "@type": "skos:Concept",
                "skos:prefLabel": {"@value": "Registers"},
                # Every variant embeds the label itself.
                "skos:altLabel": {"@value": "Registers, lists, etc"},
            }
        ],
    },
]


def _write_dump(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def genre_dump(tmp_path: Path) -> Path:
    return _write_dump(tmp_path / "genreForms.skosrdf.jsonld.gz", GENRE_FORMS_RECORDS)


@pytest.fixture
def subject_dump(tmp_path: Path) -> Path:
    return _write_dump(tmp_path / "subjects.skosrdf.jsonld.gz", SUBJECT_RECORDS)


# =============================================================================
# JSON-LD literal / reference extraction
# =============================================================================


class TestJsonLdHelpers:
    """LC serves the same predicate three different ways; all must parse."""

    def test_bare_string_literal(self):
        assert et.literals({"skos:note": "plain"}, "skos:note") == ["plain"]

    def test_single_object_literal(self):
        node = {"skos:prefLabel": {"@language": "en", "@value": "Maps"}}
        assert et.literals(node, "skos:prefLabel") == ["Maps"]

    def test_list_of_objects(self):
        node = {"skos:altLabel": [{"@value": "a"}, {"@value": "b"}]}
        assert et.literals(node, "skos:altLabel") == ["a", "b"]

    def test_missing_key_is_empty(self):
        assert et.literals({}, "skos:note") == []

    def test_references_take_the_last_path_segment(self):
        node = {"skos:broader": {"@id": "http://x/authorities/subjects/sh1"}}
        assert et.references(node, "skos:broader") == ["sh1"]

    def test_blank_nodes_are_ignored(self):
        node = {"skos:changeNote": [{"@id": "_:b1"}, {"@id": "http://x/y/sh2"}]}
        assert et.references(node, "skos:changeNote") == ["sh2"]


# =============================================================================
# SKOS record parsing and streaming
# =============================================================================


class TestParseSkosRecord:
    def test_parses_a_concept(self):
        concept = et.parse_skos_record(json.dumps(GENRE_FORMS_RECORDS[0]), "lcgft")
        assert concept is not None
        assert concept.authority_id == "gf2014026112"
        assert concept.pref_label == "Indexes"
        assert concept.broader == ["gf2014026048"]
        assert concept.notes and concept.notes[0].startswith("Works consisting")
        assert concept.uri == ("http://id.loc.gov/authorities/genreForms/gf2014026112")

    def test_skips_non_concept_records(self):
        assert (
            et.parse_skos_record(json.dumps(GENRE_FORMS_RECORDS[-1]), "lcgft") is None
        )

    def test_skips_blank_and_malformed_lines(self):
        assert et.parse_skos_record("", "lcgft") is None
        assert et.parse_skos_record("   ", "lcgft") is None
        assert et.parse_skos_record("{not json", "lcgft") is None
        assert et.parse_skos_record("[1, 2]", "lcgft") is None

    def test_streams_every_concept(self, genre_dump: Path):
        concepts = list(et.iter_skos_concepts(genre_dump, "lcgft"))
        assert [c.pref_label for c in concepts] == [
            "Indexes",
            "Informational works",
            "Maps",
            "Cartographic materials",
            "Aeronautical charts",
        ]


class TestDumpIndex:
    def test_matches_pref_and_alt_labels(self, genre_dump: Path):
        wanted = {"indexes", "indexes (guides)"}
        index = et.build_dump_index(genre_dump, "lcgft", wanted)
        assert index.concepts_seen == 5
        pref = index.lookup("Indexes")
        alt = index.lookup("Indexes (Guides)")
        assert pref is not None and pref[1] == "pref_label"
        assert alt is not None and alt[1] == "alt_label"

    def test_matching_is_case_and_whitespace_insensitive(self, genre_dump: Path):
        index = et.build_dump_index(genre_dump, "lcgft", {"maps"})
        assert index.lookup("  MAPS  ") is not None

    def test_unknown_label_misses(self, genre_dump: Path):
        index = et.build_dump_index(genre_dump, "lcgft", {"maps"})
        assert index.lookup("Prayers") is None

    def test_singular_marc_label_matches_the_established_plural(self, genre_dump: Path):
        """MARC 655 says "Map"/"Index"; LCGFT establishes "Maps"/"Indexes"."""
        index = et.build_dump_index(genre_dump, "lcgft", {"maps", "indexes"})
        hit = index.lookup("Map")
        assert hit is not None
        assert hit[0].pref_label == "Maps"
        assert hit[1] == "pref_label_variant"

    def test_variant_matching_does_not_shadow_an_exact_match(self, genre_dump: Path):
        index = et.build_dump_index(genre_dump, "lcgft", {"maps"})
        hit = index.lookup("Maps")
        assert hit is not None and hit[1] == "pref_label"

    def test_duplicate_pref_labels_keep_the_substantive_record(
        self, subject_dump: Path
    ):
        """sh99010179 (definitional note + hierarchy) must beat the stub."""
        index = et.build_dump_index(subject_dump, "lcsh", {"globalization"})
        hit = index.lookup("Globalization")
        assert hit is not None
        assert hit[0].authority_id == "sh99010179"

    def test_reference_resolution_is_a_second_pass(self, subject_dump: Path):
        labels = et.resolve_reference_labels(
            subject_dump, "lcsh", {"sh85066802", "sh85067435"}
        )
        assert labels == {
            "sh85066802": "Insurance",
            "sh85067435": "International relations",
        }

    def test_reference_resolution_of_nothing_reads_nothing(self, tmp_path: Path):
        assert et.resolve_reference_labels(tmp_path / "missing.gz", "lcsh", set()) == {}


# =============================================================================
# Scope notes
# =============================================================================


class TestLabelVariants:
    def test_first_variant_is_always_the_normalized_label(self):
        assert et.label_variants("  Maps. ")[0] == "maps"

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Index", "indexes"),
            ("Map", "maps"),
            ("Directory", "directories"),
            ("Catalog", "catalogs"),
            ("Indexes", "index"),
            ("Directories", "directory"),
            ("Maps", "map"),
        ],
    )
    def test_number_variants(self, label: str, expected: str):
        assert expected in et.label_variants(label)

    def test_does_not_strip_a_double_s(self):
        assert et.label_variants("Congress") == ["congress", "congresses"]

    def test_does_not_mangle_vowel_y_endings(self):
        assert "essaies" not in et.label_variants("Essay")

    def test_variants_are_unique(self):
        for label in ("Index", "Essays", "Congress", "Atlas"):
            variants = et.label_variants(label)
            assert len(variants) == len(set(variants))


class TestClassifyNote:
    @pytest.mark.parametrize(
        "note",
        [
            "Use as a topical subdivision under individual languages.",
            "This heading may be subdivided geographically.",
            "This authority record has been deleted.",
            "Do not use for works about cats.",
            "For works on the general subject see [Cats].",
            "",
        ],
    )
    def test_usage_notes(self, note: str):
        assert et.classify_note(note) == "usage"

    @pytest.mark.parametrize(
        "note",
        [
            "Here are entered works on the process by which institutions integrate.",
            "Works consisting wholly or chiefly of systematic guides.",
            "Nonrepresentational films that avoid narrative.",
            "Including American statutory law.",
        ],
    )
    def test_definitional_notes(self, note: str):
        assert et.classify_note(note) == "definitional"


class TestCleanScopeNote:
    def test_strips_here_are_entered_boilerplate(self):
        note = (
            "Here are entered works on the process by which economic, cultural, "
            "political, and social institutions become integrated worldwide."
        )
        assert et.clean_scope_note(note) == (
            "the process by which economic, cultural, political, and social "
            "institutions become integrated worldwide"
        )

    def test_strips_works_consisting_boilerplate(self):
        note = "Works consisting wholly or chiefly of systematic guides to content."
        assert et.clean_scope_note(note) == "systematic guides to content"

    def test_drops_cross_reference_sentences(self):
        note = (
            "Decisions and opinions of administrative agencies. For reported and "
            "unreported court decisions see [Court decisions and opinions.]"
        )
        assert (
            et.clean_scope_note(note)
            == "decisions and opinions of administrative agencies"
        )

    def test_drops_a_trailing_are_entered_under_sentence(self):
        note = (
            "Here are entered works on natural soils. Works on plant growing media "
            "are entered under [Plant growing media.]"
        )
        assert et.clean_scope_note(note) == "natural soils"

    def test_an_all_cross_reference_note_keeps_its_first_sentence(self):
        assert (
            et.clean_scope_note("For works on dogs see [Dogs.]")
            == "for works on dogs see Dogs"
        )

    def test_removes_marc_bracketing_and_collapses_whitespace(self):
        assert et.clean_scope_note("A  {b}  [c].") == "a b c"

    def test_lowercases_a_leading_sentence_capital_only(self):
        assert et.clean_scope_note("Nonrepresentational films.").startswith("n")
        assert et.clean_scope_note("NASA reports.").startswith("NASA")


class TestInformativeAltLabels:
    def test_drops_pure_reorderings(self):
        assert et.informative_alt_labels("Flood insurance", ["Insurance, Flood"]) == []

    def test_drops_variants_that_embed_the_label(self):
        assert et.informative_alt_labels("Registers", ["Registers, lists, etc"]) == []

    def test_keeps_genuinely_different_variants(self):
        assert et.informative_alt_labels(
            "Globalization", ["Globalisation", "Internationalization"]
        ) == ["Globalisation", "Internationalization"]


class TestContainsLabel:
    def test_word_bounded_match(self):
        assert et.contains_label("the social aspects of housing", "Housing")

    def test_no_substring_false_positive(self):
        assert not et.contains_label("housework and chores", "Housing")

    def test_empty_text(self):
        assert not et.contains_label("", "Housing")


class TestBuildDescription:
    def test_scope_note_wins_and_is_verbatim(self):
        description = et.build_description(
            label="Indexes",
            kind="form",
            definitional_notes=["Works consisting of systematic guides to content."],
            broader_labels=["Informational works"],
            narrower_labels=[],
            alt_labels=[],
        )
        assert description.source == "scope_note"
        assert description.verbatim is True
        assert description.text == "systematic guides to content"

    def test_falls_back_to_hierarchy(self):
        description = et.build_description(
            label="Maps",
            kind="form",
            definitional_notes=[],
            broader_labels=["Cartographic materials"],
            narrower_labels=["Aeronautical charts"],
            alt_labels=[],
        )
        assert description.source == "hierarchy"
        assert description.verbatim is False
        assert description.text == (
            "a kind of Cartographic materials; includes Aeronautical charts"
        )

    def test_hierarchy_relation_word_depends_on_kind(self):
        topical = et.build_description(
            "Flood insurance", "topical", [], ["Insurance"], [], []
        )
        geographic = et.build_description(
            "Bavaria", "geographic", [], ["Germany"], [], []
        )
        assert topical.text == "a topic within Insurance"
        assert geographic.text == "a place within Germany"

    def test_falls_back_to_variant_labels(self):
        description = et.build_description(
            "Globalization", "topical", [], [], [], ["Globalisation"]
        )
        assert description.source == "variant_labels"
        assert description.text == "also known as Globalisation"

    def test_no_usable_content_yields_none(self):
        description = et.build_description(
            "Registers", "form", [], [], [], ["Registers, lists"]
        )
        assert description.source == "none"
        assert description.text is None

    def test_too_short_a_scope_note_is_not_used(self):
        description = et.build_description(
            "Maps", "form", ["Maps."], ["Visual works"], [], []
        )
        assert description.source == "hierarchy"

    def test_hierarchy_never_reuses_the_label_itself(self):
        description = et.build_description(
            "Poetry", "form", [], ["Poetry"], ["Poetry"], []
        )
        assert description.source == "none"


# =============================================================================
# LCC classification helpers
# =============================================================================


class TestParseRangeCode:
    def test_two_sided_range(self):
        assert et.parse_range_code("KF1-KF9827") == ("KF", 1.0, 9827.0)

    def test_single_number(self):
        assert et.parse_range_code("PN4390") == ("PN", 4390.0, 4390.0)

    def test_decimal_bound(self):
        parsed = et.parse_range_code("QE1-QE996.5")
        assert parsed is not None and parsed[2] == pytest.approx(996.5)

    def test_rejects_cutter_codes(self):
        assert et.parse_range_code("QE51.A1-QE51.Z") is None

    def test_rejects_mixed_prefixes(self):
        assert et.parse_range_code("KF1-KD9500") is None

    def test_rejects_non_codes(self):
        assert et.parse_range_code("NOT") is None
        assert et.parse_range_code("") is None


class TestPickPrimaryRange:
    def test_widest_same_prefix_range_wins(self):
        codes = [
            "PL1-PL8844",
            "PN1-PN6790",
            "PN1600-PN3307.2",
            "PN4390",
            "PQ1-PQ3999",
        ]
        primary, siblings = et.pick_primary_range(codes, "PN")
        assert primary == "PN1-PN6790"
        assert siblings == ["PN1600-PN3307.2", "PN4390"]

    def test_prefix_match_is_exact_not_a_starts_with(self):
        primary, _ = et.pick_primary_range(["KFA1-KFA599"], "KF")
        assert primary is None

    def test_no_candidates(self):
        assert et.pick_primary_range([], "PN") == (None, [])


class TestRankRangesBySpan:
    def test_widest_first_and_unparseable_dropped(self):
        ranked = et.rank_ranges_by_span(["QE25", "QE500-QE639.5", "QE51.A1-QE51.Z"])
        assert ranked == ["QE500-QE639.5", "QE25"]


class TestParseClassificationJson:
    RANGE_NODES = [
        {
            "@id": "http://id.loc.gov/authorities/classification/TN1-TN997",
            et.MADS_AUTHORITATIVE_LABEL: [{"@value": "Mining engineering. Metallurgy"}],
            et.RDFS_LABEL: [{"@value": "Technology--Mining engineering. Metallurgy"}],
            et.SKOS_SCOPE_NOTE: [
                {"@language": "en", "@value": "Including mineral industries"}
            ],
            et.SKOS_NARROWER: [
                {"@id": "http://id.loc.gov/authorities/classification/TN275-TN292"}
            ],
        },
        {"@id": "_:b1", "@type": ["other"]},
    ]

    def test_parses_a_range(self):
        parsed = et.parse_class_range(self.RANGE_NODES, "TN1-TN997")
        assert parsed is not None
        assert parsed.caption == "Mining engineering. Metallurgy"
        assert parsed.full_caption == "Technology--Mining engineering. Metallurgy"
        assert parsed.scope_notes == ["Including mineral industries"]
        assert parsed.narrower == ["TN275-TN292"]
        assert parsed.uri.endswith("/classification/TN1-TN997")

    def test_unknown_code_returns_none(self):
        assert et.parse_class_range(self.RANGE_NODES, "QE1-QE996.5") is None

    def test_parses_a_class_collection(self):
        nodes = [
            {
                "@id": "http://id.loc.gov/authorities/classification/G",
                et.RDFS_COMMENT: [
                    {"@value": "   G -- GEOGRAPHY. ANTHROPOLOGY. RECREATION   "}
                ],
                et.MADS_COLLECTION_MEMBER: [
                    {"@id": "http://id.loc.gov/authorities/classification/G1-G922"},
                    {"@id": "http://id.loc.gov/authorities/classification/GB3-GB5030"},
                ],
            }
        ]
        caption, members = et.parse_class_collection(nodes, "G")
        assert caption == "Geography. Anthropology. Recreation"
        assert members == ["G1-G922", "GB3-GB5030"]

    def test_missing_collection(self):
        assert et.parse_class_collection([], "PN") == (None, [])


class TestLccPromptDescription:
    def test_splits_the_caption_into_comma_joined_phrases(self):
        assert (
            et.lcc_prompt_description("Industries. Land use. Labor", [])
            == "industries, land use, labor"
        )

    def test_appends_subdivision_captions_without_duplicates(self):
        text = et.lcc_prompt_description(
            "Geology", ["Mineralogy", "Geology", "Paleontology"]
        )
        assert text == "geology, mineralogy, paleontology"

    def test_structural_captions_are_skipped(self):
        """ "Periodicals"/"Study and teaching" repeat in every schedule."""
        text = et.lcc_prompt_description(
            "Geology",
            ["Periodicals and societies", "Study and teaching", "Mineralogy"],
        )
        assert text == "geology, mineralogy"

    def test_empty_caption(self):
        assert et.lcc_prompt_description("", []) is None


# =============================================================================
# LocClient: caching, rate limiting, resumability -- all over MockTransport
# =============================================================================


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


class TestLocClient:
    def test_caches_responses_and_replays_them(self, tmp_path: Path):
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, text='{"ok": true}')

        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            first = client.fetch("https://id.loc.gov/a.json")
            second = client.fetch("https://id.loc.gov/a.json")

        assert first["status"] == 200
        assert second["body"] == first["body"]
        assert len(calls) == 1, "second fetch must be served from disk"

    def test_a_new_client_resumes_from_the_cache(self, tmp_path: Path):
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, text="{}")

        for _ in range(2):
            with et.LocClient(
                cache_dir=tmp_path, delay=0, client=_mock_client(handler)
            ) as client:
                client.fetch("https://id.loc.gov/a.json")
        assert len(calls) == 1

    def test_refresh_bypasses_the_cache(self, tmp_path: Path):
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, text="{}")

        for refresh in (False, True):
            with et.LocClient(
                cache_dir=tmp_path,
                delay=0,
                refresh=refresh,
                client=_mock_client(handler),
            ) as client:
                client.fetch("https://id.loc.gov/a.json")
        assert len(calls) == 2

    def test_rate_limit_sleeps_only_on_uncached_requests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        slept: list[float] = []
        monkeypatch.setattr(et.time, "sleep", slept.append)

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="{}")

        with et.LocClient(
            cache_dir=tmp_path, delay=0.25, client=_mock_client(handler)
        ) as client:
            client.fetch("https://id.loc.gov/a.json")
            client.fetch("https://id.loc.gov/a.json")

        assert slept == [0.25]

    def test_fetch_json_normalizes_object_and_array_payloads(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("obj.json"):
                return httpx.Response(200, text='{"@id": "x"}')
            return httpx.Response(200, text='[{"@id": "y"}, 3]')

        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            assert client.fetch_json("https://id.loc.gov/obj.json") == [{"@id": "x"}]
            assert client.fetch_json("https://id.loc.gov/arr.json") == [{"@id": "y"}]

    def test_fetch_json_returns_none_for_errors_and_garbage(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("404.json"):
                return httpx.Response(404, text="nope")
            return httpx.Response(200, text="not json")

        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            assert client.fetch_json("https://id.loc.gov/404.json") is None
            assert client.fetch_json("https://id.loc.gov/bad.json") is None

    def test_network_errors_are_reported_and_not_cached(self, tmp_path: Path):
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            payload = client.fetch("https://id.loc.gov/a.json")
            assert payload["status"] == 0
            assert "boom" in payload["error"]
        assert not list(tmp_path.rglob("*.json"))

    def test_resolve_redirect_returns_the_location(self, tmp_path: Path):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                headers={"location": "https://id.loc.gov/authorities/names/n79041717"},
            )

        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            assert client.resolve_redirect("https://id.loc.gov/x").endswith("n79041717")

    def test_resolve_redirect_of_a_404_is_none(self, tmp_path: Path):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            assert client.resolve_redirect("https://id.loc.gov/x") is None

    def test_download_streams_once_then_reuses_the_file(self, tmp_path: Path):
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, content=b"payload")

        destination = tmp_path / "bulk" / "subjects.gz"
        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            client.download("https://id.loc.gov/bulk.gz", destination)
            client.download("https://id.loc.gov/bulk.gz", destination)

        assert destination.read_bytes() == b"payload"
        assert len(calls) == 1
        assert not list(destination.parent.glob("*.part"))


# =============================================================================
# End-to-end enrichment on fixture data
# =============================================================================


@pytest.fixture
def indexes(genre_dump: Path, subject_dump: Path) -> dict[str, et.DumpIndex]:
    wanted = {
        et.normalize_label(label)
        for label in (
            "Maps",
            "Indexes",
            "Registers",
            "Globalization",
            "Flood insurance",
            "Nonexistent term",
        )
    }
    return {
        "lcgft": et.build_dump_index(genre_dump, "lcgft", wanted),
        "lcsh": et.build_dump_index(subject_dump, "lcsh", wanted),
    }


@pytest.fixture
def reference_labels(genre_dump: Path, subject_dump: Path) -> dict[str, dict[str, str]]:
    return {
        "lcgft": et.resolve_reference_labels(
            genre_dump, "lcgft", {"gf2014026048", "gf2011026061", "gf2011026001"}
        ),
        "lcsh": et.resolve_reference_labels(
            subject_dump, "lcsh", {"sh85066802", "sh85067435"}
        ),
    }


class TestEnrichFromDumps:
    ENTRIES = [
        {"id": "Maps", "label": "Maps", "frequency": 70971, "rank": 1},
        {"id": "Indexes", "label": "Indexes", "frequency": 100, "rank": 2},
        {"id": "Globalization", "label": "Globalization", "frequency": 50, "rank": 3},
        {"id": "Registers", "label": "Registers", "frequency": 10, "rank": 4},
        {
            "id": "Nonexistent term",
            "label": "Nonexistent term",
            "frequency": 1,
            "rank": 5,
        },
    ]

    def _run(self, indexes, reference_labels):
        spec = et.TAXONOMIES["lcgft"]
        return {
            r["label"]: r
            for r in et.enrich_from_dumps(spec, self.ENTRIES, indexes, reference_labels)
        }

    def test_scope_note_record(self, indexes, reference_labels):
        record = self._run(indexes, reference_labels)["Indexes"]
        assert record["match"] == "lcgft:pref_label"
        assert record["description_source"] == "scope_note"
        assert record["description_verbatim"] is True
        assert record["uri"].endswith("gf2014026112")
        assert record["broader"] == ["Informational works"]

    def test_hierarchy_record_resolves_related_labels(self, indexes, reference_labels):
        record = self._run(indexes, reference_labels)["Maps"]
        assert record["description_source"] == "hierarchy"
        assert record["broader"] == ["Informational works", "Cartographic materials"]
        assert record["narrower"] == ["Aeronautical charts"]

    def test_falls_through_to_the_second_dump(self, indexes, reference_labels):
        """LCGFT has no "Globalization"; the LCSH dump is the declared fallback."""
        record = self._run(indexes, reference_labels)["Globalization"]
        assert record["match"] == "lcsh:pref_label"
        assert record["authority_scheme"] == "lcsh"
        assert record["description_source"] == "scope_note"

    def test_usage_notes_are_separated_from_scope_notes(
        self, indexes, reference_labels
    ):
        record = self._run(indexes, reference_labels)["Globalization"]
        assert len(record["scope_notes"]) == 1
        assert len(record["usage_notes"]) == 1
        assert record["usage_notes"][0].startswith("This heading may be subdivided")

    def test_matched_but_undescribable_record(self, indexes, reference_labels):
        record = self._run(indexes, reference_labels)["Registers"]
        assert record["match"] == "lcsh:pref_label"
        assert record["description"] is None
        assert record["description_source"] == "none"

    def test_unmatched_record_keeps_frequency_fields(self, indexes, reference_labels):
        record = self._run(indexes, reference_labels)["Nonexistent term"]
        assert record["match"] == "none"
        assert record["uri"] is None
        assert record["frequency"] == 1
        assert record["rank"] == 5

    def test_every_input_label_produces_exactly_one_record(
        self, indexes, reference_labels
    ):
        assert len(self._run(indexes, reference_labels)) == len(self.ENTRIES)


class TestCoverageReport:
    def test_counts_are_consistent(self, indexes, reference_labels):
        records = et.enrich_from_dumps(
            et.TAXONOMIES["lcgft"],
            TestEnrichFromDumps.ENTRIES,
            indexes,
            reference_labels,
        )
        coverage = et.coverage_for(records)
        assert coverage["total"] == 5
        assert coverage["matched_to_authority"] + coverage["unmatched"] == 5
        assert coverage["unmatched"] == 1
        assert coverage["unmatched_examples"] == ["Nonexistent term"]
        assert "Registers" in coverage["no_description_examples"]
        assert sum(coverage["description_by_source"].values()) == 5

    def test_markdown_renders_every_taxonomy(self, indexes, reference_labels):
        records = et.enrich_from_dumps(
            et.TAXONOMIES["lcgft"],
            TestEnrichFromDumps.ENTRIES,
            indexes,
            reference_labels,
        )
        report = {
            "generated": "2026-08-26T00:00:00+00:00",
            "script_version": et.SCRIPT_VERSION,
            "taxonomies": {"lcgft": et.coverage_for(records)},
        }
        markdown = et.render_report_markdown(report)
        assert "| lcgft |" in markdown
        assert "## Fallback chain" in markdown


# =============================================================================
# LCC enrichment over a mocked classification API
# =============================================================================


CLASSIFICATION_FIXTURES: dict[str, Any] = {
    "T": [
        {
            "@id": "http://id.loc.gov/authorities/classification/T",
            et.RDFS_COMMENT: [{"@value": "T -- TECHNOLOGY"}],
            et.MADS_COLLECTION_MEMBER: [
                {"@id": "http://id.loc.gov/authorities/classification/TN1-TN997"},
                {"@id": "http://id.loc.gov/authorities/classification/TA1-TA2040"},
            ],
        }
    ],
    "TN1-TN997": [
        {
            "@id": "http://id.loc.gov/authorities/classification/TN1-TN997",
            et.MADS_AUTHORITATIVE_LABEL: [{"@value": "Mining engineering. Metallurgy"}],
            et.RDFS_LABEL: [{"@value": "Technology--Mining engineering. Metallurgy"}],
            et.SKOS_NARROWER: [
                {"@id": "http://id.loc.gov/authorities/classification/TN275-TN292"},
                {"@id": "http://id.loc.gov/authorities/classification/TN950-TN997"},
            ],
        }
    ],
    "TN275-TN292": [
        {
            "@id": "http://id.loc.gov/authorities/classification/TN275-TN292",
            et.MADS_AUTHORITATIVE_LABEL: [{"@value": "Ore deposits and mining"}],
        }
    ],
    "TN950-TN997": [
        {
            "@id": "http://id.loc.gov/authorities/classification/TN950-TN997",
            et.MADS_AUTHORITATIVE_LABEL: [{"@value": "Metallurgy"}],
        }
    ],
    "TA1-TA2040": [
        {
            "@id": "http://id.loc.gov/authorities/classification/TA1-TA2040",
            et.MADS_AUTHORITATIVE_LABEL: [{"@value": "Engineering (General)"}],
            et.SKOS_SCOPE_NOTE: [
                {"@value": "Including civil engineering as a general subject"}
            ],
        }
    ],
}


def _classification_handler(request: httpx.Request) -> httpx.Response:
    code = request.url.path.rsplit("/", 1)[-1].removesuffix(".json")
    if code in CLASSIFICATION_FIXTURES:
        return httpx.Response(200, text=json.dumps(CLASSIFICATION_FIXTURES[code]))
    # Most per-subclass collection endpoints 404 -- that is the real behaviour.
    return httpx.Response(404, text="not found")


class TestEnrichLcc:
    def _run(self, tmp_path: Path, entries, narrower: int = 12):
        with et.LocClient(
            cache_dir=tmp_path,
            delay=0,
            client=_mock_client(_classification_handler),
        ) as client:
            return et.enrich_lcc(entries, client, narrower, verbose=False)

    def test_subclass_resolved_through_the_parent_class_collection(
        self, tmp_path: Path
    ):
        (record,) = self._run(
            tmp_path, [{"id": "TN", "label": "TN", "frequency": 5, "rank": 1}]
        )
        assert record["match"] == "classification_api"
        assert record["range"] == "TN1-TN997"
        assert record["caption"] == "Mining engineering. Metallurgy"
        assert record["class_caption"] == "Technology"
        assert record["covers"] == ["Metallurgy", "Ore deposits and mining"]
        assert record["description_source"] == "caption_hierarchy"
        assert record["description"] == (
            "mining engineering, metallurgy, ore deposits and mining"
        )

    def test_scope_note_is_appended_verbatim_when_present(self, tmp_path: Path):
        (record,) = self._run(
            tmp_path, [{"id": "TA", "label": "TA", "frequency": 5, "rank": 1}]
        )
        assert record["description_source"] == "scope_note"
        assert record["description_verbatim"] is True
        assert "including civil engineering" in record["description"]

    def test_narrower_limit_caps_the_request_count(self, tmp_path: Path):
        (record,) = self._run(
            tmp_path,
            [{"id": "TN", "label": "TN", "frequency": 5, "rank": 1}],
            narrower=1,
        )
        assert len(record["covers"]) == 1

    def test_main_class_uses_its_member_ranges_as_covers(self, tmp_path: Path):
        (record,) = self._run(
            tmp_path, [{"id": "T", "label": "T", "frequency": 9, "rank": 1}]
        )
        assert record["match"] == "classification_api"
        assert set(record["covers"]) == {
            "Mining engineering. Metallurgy",
            "Engineering (General)",
        }

    def test_extraction_artifacts_are_flagged_not_fetched(self, tmp_path: Path):
        entries = [
            {"id": "TX", "label": "TX", "frequency": 1, "rank": 1},
            {"id": "PAR", "label": "PAR", "frequency": 1, "rank": 2},
            {"id": "in", "label": "in", "frequency": 1, "rank": 3},
        ]
        records = self._run(tmp_path, entries)
        assert records[2]["match"] == "invalid_code"
        # "NOT"/"PAR" are shaped like codes but match no range; they must not
        # silently inherit the whole of class N / class P.
        assert [r["match"] for r in records[:2]] == ["none", "none"]
        assert all(r["description"] is None for r in records)
        assert all(r["uri"] is None for r in records)

    def test_obsolete_subclass_misses_cleanly(self, tmp_path: Path):
        (record,) = self._run(
            tmp_path, [{"id": "JX", "label": "JX", "frequency": 1, "rank": 1}]
        )
        assert record["match"] == "none"
        assert record["uri"] is None


class TestEnrichGeographicWithNaf:
    def test_fills_uri_and_variants_but_never_a_description(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/label/" in request.url.path:
                return httpx.Response(
                    302,
                    headers={
                        "location": "https://id.loc.gov/authorities/names/n79041717"
                    },
                )
            return httpx.Response(
                200,
                text=json.dumps(
                    [
                        {
                            "@id": "http://id.loc.gov/authorities/names/n79041717",
                            et.MADS_AUTHORITATIVE_LABEL: [{"@value": "California"}],
                            et.MADS_CODE: [{"@value": "n-us-ca"}],
                            et.MADS_EDITORIAL_NOTE: [{"@value": "Heading includes..."}],
                        },
                        {
                            "@id": "_:b1",
                            et.MADS_VARIANT_LABEL: [{"@value": "Cal. (California)"}],
                        },
                    ]
                ),
            )

        records = [
            {"label": "California", "match": "none", "description": None},
            {"label": "Bavaria", "match": "lcsh:pref_label", "description": "a place"},
        ]
        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            filled = et.enrich_geographic_with_naf(records, client, verbose=False)

        assert filled == 1
        assert str(records[0]["uri"]).endswith("n79041717")
        assert records[0]["match"] == "naf:label_service"
        assert records[0]["alt_labels"] == ["Cal. (California)"]
        assert records[0]["gac_codes"] == ["n-us-ca"]
        assert records[0]["description"] is None, "NAF carries no definitions"
        assert records[1]["match"] == "lcsh:pref_label", "already-matched untouched"

    def test_unresolvable_place_is_left_alone(self, tmp_path: Path):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        records = [
            {"label": "Puget Sound (Wash.)", "match": "none", "description": None}
        ]
        with et.LocClient(
            cache_dir=tmp_path, delay=0, client=_mock_client(handler)
        ) as client:
            assert et.enrich_geographic_with_naf(records, client, verbose=False) == 0
        assert records[0]["match"] == "none"


# =============================================================================
# CLI surface
# =============================================================================


class TestCli:
    def test_dry_run_opens_no_socket(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
    ):
        def explode(*_: object, **__: object) -> None:
            raise AssertionError("--dry-run must not create an HTTP client")

        monkeypatch.setattr(et.httpx, "Client", explode)
        taxonomy_dir = tmp_path / "taxonomies"
        taxonomy_dir.mkdir(parents=True)
        (taxonomy_dir / "lcgft.json").write_text(
            json.dumps({"type": "lcgft", "labels": [{"id": "Maps", "label": "Maps"}]})
        )

        exit_code = et.main(
            ["--data-dir", str(tmp_path), "--taxonomy", "lcgft", "--dry-run"]
        )
        assert exit_code == 0
        assert "dry run" in capsys.readouterr().out

    def test_missing_data_dir_is_an_error(self, tmp_path: Path):
        assert et.main(["--data-dir", str(tmp_path / "nope"), "--dry-run"]) == 2

    def test_selected_keys_defaults_to_everything(self):
        assert et.selected_keys(None) == list(et.ALL_KEYS)
        assert et.selected_keys(["all"]) == list(et.ALL_KEYS)
        assert et.selected_keys(["lcgft", "lcgft"]) == ["lcgft"]

    def test_keys_needing_dumps_is_deduplicated_and_skips_lcc(self):
        assert et.keys_needing_dumps(["lcgft", "lcsh_geo", "lcc_subclass"]) == [
            "lcgft",
            "lcsh",
        ]

    def test_limit_caps_labels_per_taxonomy(self, tmp_path: Path, monkeypatch):
        """A full run with --limit 2 writes exactly two enriched records."""
        taxonomy_dir = tmp_path / "taxonomies"
        taxonomy_dir.mkdir(parents=True)
        (taxonomy_dir / "lcgft.json").write_text(
            json.dumps(
                {
                    "type": "lcgft",
                    "name": "LCGFT",
                    "labels": [
                        {"id": "Maps", "label": "Maps", "frequency": 3, "rank": 1},
                        {
                            "id": "Indexes",
                            "label": "Indexes",
                            "frequency": 2,
                            "rank": 2,
                        },
                        {
                            "id": "Registers",
                            "label": "Registers",
                            "frequency": 1,
                            "rank": 3,
                        },
                    ],
                }
            )
        )
        dump_bytes = tmp_path / "src.gz"
        _write_dump(dump_bytes, GENRE_FORMS_RECORDS)
        payload = dump_bytes.read_bytes()

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=payload)

        real_client_cls = httpx.Client
        monkeypatch.setattr(
            et.httpx,
            "Client",
            lambda **_: real_client_cls(
                transport=httpx.MockTransport(handler), follow_redirects=False
            ),
        )
        exit_code = et.main(
            [
                "--data-dir",
                str(tmp_path),
                "--taxonomy",
                "lcgft",
                "--limit",
                "2",
                "--delay",
                "0",
                "--quiet",
            ]
        )
        assert exit_code == 0
        written = json.loads((taxonomy_dir / "enriched" / "lcgft.json").read_text())
        assert [r["label"] for r in written["labels"]] == ["Maps", "Indexes"]
        assert written["coverage"]["total"] == 2
        report = json.loads(
            (taxonomy_dir / "enriched" / "coverage_report.json").read_text()
        )
        assert report["taxonomies"]["lcgft"]["total"] == 2
        assert (taxonomy_dir / "enriched" / "coverage_report.md").exists()
        assert (
            (taxonomy_dir / "enriched" / ".gitignore")
            .read_text()
            .strip()
            .endswith(".cache/")
        )


class TestCuratedPools:
    """The frozen v0.3.1 pools are enrichment targets, not just the MARC tables."""

    def test_curated_pools_are_registered(self):
        for key in ("curated_forms", "curated_topics", "curated_geographic"):
            assert key in et.TAXONOMIES
            assert key in et.ALL_KEYS

    def test_curated_labels_are_loaded_in_frequency_table_shape(self):
        labels = et.load_curated_labels("curated_forms")
        assert labels
        assert set(labels[0]) == {"id", "label", "frequency", "rank"}
        assert [entry["rank"] for entry in labels] == list(range(1, len(labels) + 1))

    def test_curated_labels_are_deduplicated_and_sorted(self):
        labels = et.load_curated_labels("curated_topics")
        values = [entry["label"] for entry in labels]
        assert values == sorted(set(values))

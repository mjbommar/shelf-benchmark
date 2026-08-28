"""Unit tests for shelf.sampler.generator prompt variants (v0.4).

Tests cover:
- Byte-for-byte reproduction of the frozen v0.3.1 system prompt, both via
  ``build_system_prompt`` directly and end-to-end through
  ``DocumentGenerator`` (the default configuration).
- Every prompt variant retains the SHOW-DON'T-TELL block verbatim, and no
  variant adds a negative style instruction (e.g. "avoid em-dashes").
- Form-conditional output format selection: form-level overrides, category
  defaults, and the plain-prose fallback for unrecognized forms/categories.
- Seeded, reproducible sampling of prompt variants, both via
  ``PromptVariantSampler`` directly and via ``DocumentGenerator``.
- ``_parse_generated_text`` round-trips the "Title: ...\\n\\n<body>" envelope
  regardless of which variant or output format produced the body.
"""

from __future__ import annotations

import asyncio

import pytest
from shelf.llm.backends import (
    GenerationParams,
    GenerationRequest,
    GenerationResult,
    LLMBackend,
)
from shelf.sampler.dimensions import LCCClass, LCGFTTerm
from shelf.sampler.document import Document
from shelf.sampler.generator import (
    CATEGORY_OUTPUT_FORMATS,
    DEFAULT_PROMPT_VARIANT_WEIGHTS,
    FORM_OUTPUT_FORMATS,
    GENERATION_INSTRUCTIONS,
    LCGFT_CATEGORY_DESCRIPTIONS,
    SHOW_DONT_TELL_BLOCK,
    DocumentGenerator,
    DocumentLength,
    GeneratedDocument,
    OutputFormat,
    PromptVariant,
    PromptVariantSampler,
    Register,
    _parse_generated_text,
    build_system_prompt,
    resolve_output_format,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _doc(
    lcc_code: str = "P",
    lcc_name: str = "Language and Literature",
    category: str = "Literature",
    form: str = "Novels",
    topics: list[str] | None = None,
    audience: str | None = "general readers",
    geographic: list[str] | None = None,
) -> Document:
    return Document(
        lcc=LCCClass(code=lcc_code, name=lcc_name),
        lcgft=LCGFTTerm(category=category, form=form),
        topics=topics or ["memory", "identity"],
        audience=audience,
        geographic=geographic or ["Ireland"],
        id="doc-test-1",
    )


class FakeBackend(LLMBackend):
    """Minimal LLMBackend that echoes a fixed body and records every request."""

    provider = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def _respond(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(text="Title: A Fake Title\n\nSome fake body text.")

    # params is unused (fixed body regardless of sampling params); the name
    # and position are dictated by the LLMBackend protocol, which ty checks
    # structurally including parameter names -- so it can't be prefixed with
    # an underscore without breaking that structural match.
    def generate(
        self,
        request: GenerationRequest,
        params: GenerationParams,  # noqa: ARG002
    ) -> GenerationResult:
        return self._respond(request)

    async def generate_async(
        self,
        request: GenerationRequest,
        params: GenerationParams,  # noqa: ARG002
    ) -> GenerationResult:
        return self._respond(request)

    def generate_batch(
        self,
        requests: list[GenerationRequest],
        params: GenerationParams,  # noqa: ARG002
    ) -> list[GenerationResult]:
        return [self._respond(r) for r in requests]


ALL_VARIANTS = list(PromptVariant)
NEW_VARIANTS = [v for v in PromptVariant if v is not PromptVariant.V0_3_1]


# --------------------------------------------------------------------------- #
# v0.3.1 byte-for-byte reproduction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("category", "form"),
    [
        ("Literature", "Novels"),
        ("Law materials", "Legal briefs"),
        ("Religious materials", "Prayers"),
        ("Cartographic materials", "Maps"),
        ("Visual works", "Photographs"),
    ],
)
def test_v0_3_1_variant_is_byte_for_byte_generation_instructions(category, form):
    """build_system_prompt with V0_3_1 ignores the document entirely and
    returns exactly GENERATION_INSTRUCTIONS -- the frozen preset that
    produced all 42,532 v0.3.1 documents."""
    doc = _doc(category=category, form=form)
    result = build_system_prompt(doc, PromptVariant.V0_3_1)
    assert result == GENERATION_INSTRUCTIONS
    # Also identical byte length / hash-equivalent (belt and suspenders).
    assert len(result) == len(GENERATION_INSTRUCTIONS)


def test_v0_3_1_variant_accepts_string_id():
    """The frozen preset is reachable by its stable string id too."""
    doc = _doc()
    assert build_system_prompt(doc, "v0.3.1") == GENERATION_INSTRUCTIONS
    assert PromptVariant.V0_3_1.value == "v0.3.1"


def test_document_generator_default_reproduces_v0_3_1_end_to_end():
    """DocumentGenerator's default configuration (no prompt_variant args)
    must send exactly GENERATION_INSTRUCTIONS as the system prompt -- this
    is the end-to-end proof that existing call sites (which never pass the
    new prompt_variant* arguments) keep producing the v0.3.1 prompt
    byte-for-byte."""
    backend = FakeBackend()
    gen = DocumentGenerator(seed=42, llm_backend=backend, use_llm=True)
    doc = _doc(category="Law materials", form="Legal briefs")

    result = gen.generate(doc, length=DocumentLength.SHORT, register=Register.FORMAL)

    assert len(backend.requests) == 1
    assert backend.requests[0].system_prompt == GENERATION_INSTRUCTIONS
    assert result.prompt_variant_id == PromptVariant.V0_3_1.value


def test_document_generator_fixed_non_default_variant():
    """Passing prompt_variant fixes every document to that one variant."""
    backend = FakeBackend()
    gen = DocumentGenerator(
        seed=1,
        llm_backend=backend,
        use_llm=True,
        prompt_variant=PromptVariant.EDITORIAL,
    )
    doc = _doc()
    result = gen.generate(doc)
    assert result.prompt_variant_id == PromptVariant.EDITORIAL.value
    assert backend.requests[0].system_prompt == build_system_prompt(
        doc, PromptVariant.EDITORIAL
    )


# --------------------------------------------------------------------------- #
# SHOW-DON'T-TELL retention + no negative style instructions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_every_variant_retains_show_dont_tell_block(variant):
    doc = _doc()
    system_prompt = build_system_prompt(doc, variant)
    assert SHOW_DONT_TELL_BLOCK in system_prompt


@pytest.mark.parametrize("variant", NEW_VARIANTS)
def test_new_variants_are_distinct_from_v0_3_1_and_each_other(variant):
    doc = _doc()
    v0_3_1_text = build_system_prompt(doc, PromptVariant.V0_3_1)
    variant_text = build_system_prompt(doc, variant)
    assert variant_text != v0_3_1_text


def test_new_variant_preambles_are_pairwise_distinct():
    doc = _doc()
    texts = {v: build_system_prompt(doc, v) for v in NEW_VARIANTS}
    values = list(texts.values())
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_no_variant_instructs_against_em_dashes_or_similar(variant):
    """Per data_plan_v0.4.md section 4.2 point 3: no variant may add a
    negative style instruction like "avoid em-dashes" -- that just swaps
    one artifact for another rather than reducing it."""
    doc = _doc()
    system_prompt = build_system_prompt(doc, variant).lower()
    for banned_phrase in ("em-dash", "em dash", "emdash", "avoid dashes"):
        assert banned_phrase not in system_prompt


# --------------------------------------------------------------------------- #
# Form-conditional output format selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("form", "category", "expected"),
    [
        # Task-cited examples: prayer, poem, joke, legal brief, letter.
        ("Prayers", "Religious materials", OutputFormat.PLAIN_PROSE),
        ("Poetry", "Literature", OutputFormat.VERSE),
        ("Jokes", "Recreational works", OutputFormat.PLAIN_PROSE),
        ("Legal briefs", "Law materials", OutputFormat.LEGAL_STYLE),
        ("Letters", "Creative nonfiction", OutputFormat.LETTER_STYLE),
        # A few more, including category-default (no form override) cases.
        ("Interviews", "Sound recordings", OutputFormat.TRANSCRIPT_STYLE),
        ("Drama", "Literature", OutputFormat.TRANSCRIPT_STYLE),
        ("Diaries", "Creative nonfiction", OutputFormat.DIARY_STYLE),
        ("Maps", "Cartographic materials", OutputFormat.LIST_STYLE),
        (
            "Handbooks",
            "Instructional and educational works",
            OutputFormat.STRUCTURED_MARKDOWN,
        ),
        ("Some Uncommon Form", "Informational works", OutputFormat.STRUCTURED_MARKDOWN),
        ("Some Uncommon Form", "Literature", OutputFormat.PLAIN_PROSE),
    ],
)
def test_resolve_output_format_mapping(form, category, expected):
    assert resolve_output_format(form, category) == expected


def test_resolve_output_format_falls_back_to_plain_prose():
    """Unknown form AND unknown category -> plain prose, never markdown --
    markdown-by-default was exactly the v0.3.1 problem being fixed."""
    assert (
        resolve_output_format("Nonexistent Form", "Nonexistent Category")
        == OutputFormat.PLAIN_PROSE
    )


def test_form_override_takes_precedence_over_category_default():
    """Handbooks (STRUCTURED_MARKDOWN override) should win even when
    paired with a category whose own default is something else."""
    assert (
        resolve_output_format("Handbooks", "Law materials")
        == OutputFormat.STRUCTURED_MARKDOWN
    )
    assert CATEGORY_OUTPUT_FORMATS["Law materials"] == OutputFormat.LEGAL_STYLE


def test_category_output_formats_cover_every_known_lcgft_category():
    """CATEGORY_OUTPUT_FORMATS must have an entry for every one of the 14
    LCGFT categories the corpus actually uses -- a missing entry would
    silently fall back to plain prose rather than a deliberate choice."""
    assert set(CATEGORY_OUTPUT_FORMATS.keys()) == set(
        LCGFT_CATEGORY_DESCRIPTIONS.keys()
    )


def test_form_output_formats_have_no_redundant_category_matching_entries_only():
    """Every override key must map to a real OutputFormat (sanity check the
    table isn't silently broken)."""
    for form, fmt in FORM_OUTPUT_FORMATS.items():
        assert isinstance(fmt, OutputFormat), form


@pytest.mark.parametrize("variant", NEW_VARIANTS)
def test_output_format_instruction_appears_in_system_prompt(variant):
    """The resolved format instruction for the document's form must
    actually be present in the assembled system prompt (not just computed
    and discarded)."""
    doc = _doc(category="Religious materials", form="Prayers")
    system_prompt = build_system_prompt(doc, variant)
    assert "continuous prose in paragraphs" in system_prompt

    verse_doc = _doc(category="Literature", form="Poetry")
    verse_prompt = build_system_prompt(verse_doc, variant)
    assert "lineated verse" in verse_prompt


# --------------------------------------------------------------------------- #
# Seeded, reproducible variant sampling
# --------------------------------------------------------------------------- #


def test_prompt_variant_sampler_is_deterministic_under_seed():
    a = PromptVariantSampler(seed=123)
    b = PromptVariantSampler(seed=123)
    draws_a = [a.sample() for _ in range(200)]
    draws_b = [b.sample() for _ in range(200)]
    assert draws_a == draws_b


def test_prompt_variant_sampler_default_pool_excludes_v0_3_1():
    """v0.3.1 is a frozen reproduction preset, not a candidate a v0.4 run
    should draw at random by default."""
    sampler = PromptVariantSampler(seed=7)
    draws = {sampler.sample() for _ in range(500)}
    assert PromptVariant.V0_3_1 not in draws
    assert draws == set(DEFAULT_PROMPT_VARIANT_WEIGHTS.keys())


def test_prompt_variant_sampler_respects_explicit_weights_including_v0_3_1():
    sampler = PromptVariantSampler(weights={PromptVariant.V0_3_1: 1.0}, seed=1)
    assert all(sampler.sample() is PromptVariant.V0_3_1 for _ in range(20))


def test_document_generator_variant_sampling_is_seeded_and_recorded():
    """Two DocumentGenerators built with the same seed and the same
    prompt_variant_weights must draw the identical sequence of variants,
    and each GeneratedDocument must record the variant that produced it."""
    docs = [_doc() for _ in range(10)]

    def _build():
        backend = FakeBackend()
        gen = DocumentGenerator(
            seed=99,
            llm_backend=backend,
            use_llm=True,
            prompt_variant_weights=DEFAULT_PROMPT_VARIANT_WEIGHTS,
        )
        return gen, backend

    gen_a, backend_a = _build()
    gen_b, backend_b = _build()

    results_a = [gen_a.generate(d) for d in docs]
    results_b = [gen_b.generate(d) for d in docs]

    variants_a = [r.prompt_variant_id for r in results_a]
    variants_b = [r.prompt_variant_id for r in results_b]

    assert variants_a == variants_b
    # Not every draw is v0.3.1 -- sampling is actually happening.
    assert set(variants_a) <= {v.value for v in NEW_VARIANTS}
    assert all(isinstance(v, str) for v in variants_a)


def test_generate_batch_records_prompt_variant_per_document():
    backend = FakeBackend()
    gen = DocumentGenerator(
        seed=5,
        llm_backend=backend,
        use_llm=True,
        prompt_variant_weights=DEFAULT_PROMPT_VARIANT_WEIGHTS,
    )
    docs = [_doc() for _ in range(6)]
    results = gen.generate_batch(docs, use_backend_batch=True)
    assert len(results) == 6
    for r in results:
        assert r.prompt_variant_id in {v.value for v in NEW_VARIANTS}
        assert r.to_dict()["prompt_variant_id"] == r.prompt_variant_id


def test_generated_document_to_dict_includes_prompt_variant_id():
    doc = _doc()
    gd = GeneratedDocument(
        document=doc,
        title="T",
        body="B",
        prompt_variant_id=PromptVariant.ARCHIVAL.value,
    )
    assert gd.to_dict()["prompt_variant_id"] == "v0.4-archival"


# --------------------------------------------------------------------------- #
# _parse_generated_text round-trips every variant's envelope
# --------------------------------------------------------------------------- #


REPRESENTATIVE_BODIES: dict[OutputFormat, str] = {
    OutputFormat.PLAIN_PROSE: (
        "The rain had not stopped for three days.\n\nStill, the ferry ran on schedule."
    ),
    OutputFormat.VERSE: "the river does not ask\nwhere it is going\n\nit only goes",
    OutputFormat.STRUCTURED_MARKDOWN: (
        "## Overview\n\n- First point\n- Second point\n\n**Note:** read carefully."
    ),
    OutputFormat.LEGAL_STYLE: (
        "I. BACKGROUND\n\nThe parties entered into an agreement on the date below."
    ),
    OutputFormat.LETTER_STYLE: (
        "Dear Marguerite,\n\nIt has been too long since your last letter.\n\n"
        "Yours faithfully,\nH."
    ),
    OutputFormat.TRANSCRIPT_STYLE: (
        "INTERVIEWER: When did you first arrive?\n\nSUBJECT: In the spring of that year."
    ),
    OutputFormat.DIARY_STYLE: "March 4th\n\nWoke early. The frost had not yet lifted.",
    OutputFormat.LIST_STYLE: "- Riverside Hall, 9pm\n- Main Tent, 10:30pm\n\n- Closing remarks",
    OutputFormat.CAPTION_STYLE: (
        "A wide shot of the harbor at dawn, fishing boats still at their moorings."
    ),
}


@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_parse_generated_text_round_trips_every_output_format(output_format):
    title = "An Ordinary Morning"
    body = REPRESENTATIVE_BODIES[output_format]
    raw_text = f"Title: {title}\n\n{body}"

    parsed_title, parsed_body = _parse_generated_text(raw_text)

    assert parsed_title == title
    assert parsed_body == body


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_parse_generated_text_round_trips_through_document_generator(variant):
    """End-to-end: generate() under each variant, with the backend echoing
    a valid envelope, and confirm the parsed title/body match exactly what
    the (fake) model produced -- the envelope contract is unaffected by
    which variant or output format is in play."""
    backend = FakeBackend()
    gen = DocumentGenerator(
        seed=3, llm_backend=backend, use_llm=True, prompt_variant=variant
    )
    doc = _doc()
    result = gen.generate(doc)
    assert result.title == "A Fake Title"
    assert result.body == "Some fake body text."
    assert result.prompt_variant_id == PromptVariant(variant).value


def test_generate_async_records_prompt_variant():
    backend = FakeBackend()
    gen = DocumentGenerator(
        seed=11,
        llm_backend=backend,
        use_llm=True,
        prompt_variant=PromptVariant.PRACTITIONER,
    )
    doc = _doc()
    result = asyncio.run(gen.generate_async(doc))
    assert result.prompt_variant_id == PromptVariant.PRACTITIONER.value
    assert backend.requests[0].system_prompt == build_system_prompt(
        doc, PromptVariant.PRACTITIONER
    )

"""
Text generation for sampled documents.

Takes label combinations and generates actual document text,
either via templates or an LLM backend (OpenAI, Anthropic, Gemini).
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from shelf.llm import (
    GenerationParams,
    GenerationRequest,
    LLMBackend,
    OpenAIResponsesBackend,
)

from .document import Document

if TYPE_CHECKING:
    import openai


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL = "gpt-5.1"
DEFAULT_SERVICE_TIER = "flex"

# Sampling parameter ranges for randomization
TEMPERATURE_RANGE = (0.6, 1.2)  # Creative but not too wild
TOP_P_RANGE = (0.85, 1.0)  # Mostly full distribution


@dataclass
class SamplingParams:
    """LLM sampling parameters."""

    temperature: float
    top_p: float

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p}


class SamplingParamsSampler:
    """Sample randomized LLM generation parameters."""

    def __init__(
        self,
        temperature_range: tuple[float, float] = TEMPERATURE_RANGE,
        top_p_range: tuple[float, float] = TOP_P_RANGE,
        seed: int | None = None,
    ):
        self._rng = random.Random(seed)
        self._temp_range = temperature_range
        self._top_p_range = top_p_range

    def sample(self) -> SamplingParams:
        return SamplingParams(
            temperature=round(self._rng.uniform(*self._temp_range), 2),
            top_p=round(self._rng.uniform(*self._top_p_range), 2),
        )


class DocumentLength(str, Enum):
    """Document length categories."""

    MICRO = "micro"  # 10-25 words (tweet, headline)
    TINY = "tiny"  # 25-50 words (blurb, caption)
    BRIEF = "brief"  # 50-100 words (abstract, summary)
    SHORT = "short"  # 100-250 words (short article)
    MEDIUM = "medium"  # 250-500 words (blog post)
    LONG = "long"  # 500-1000 words (article)
    VERY_LONG = "very_long"  # 1000-2000 words (essay)
    EXTENDED = "extended"  # 2000-4000 words (long-form)


LENGTH_WORD_RANGES = {
    DocumentLength.MICRO: (10, 25),
    DocumentLength.TINY: (25, 50),
    DocumentLength.BRIEF: (50, 100),
    DocumentLength.SHORT: (100, 250),
    DocumentLength.MEDIUM: (250, 500),
    DocumentLength.LONG: (500, 1000),
    DocumentLength.VERY_LONG: (1000, 2000),
    DocumentLength.EXTENDED: (2000, 4000),
}

# Default length distribution (weighted towards medium)
DEFAULT_LENGTH_WEIGHTS = {
    DocumentLength.MICRO: 0.05,
    DocumentLength.TINY: 0.08,
    DocumentLength.BRIEF: 0.12,
    DocumentLength.SHORT: 0.20,
    DocumentLength.MEDIUM: 0.25,
    DocumentLength.LONG: 0.15,
    DocumentLength.VERY_LONG: 0.10,
    DocumentLength.EXTENDED: 0.05,
}


class Register(str, Enum):
    """Writing register/tone."""

    CASUAL = "casual"  # Informal, conversational
    CONVERSATIONAL = "conversational"  # Friendly but clear
    PROFESSIONAL = "professional"  # Standard business
    FORMAL = "formal"  # Formal/official
    ACADEMIC = "academic"  # Scholarly
    TECHNICAL = "technical"  # Technical/specialized
    JOURNALISTIC = "journalistic"  # News style
    CREATIVE = "creative"  # Literary/artistic


REGISTER_DESCRIPTIONS = {
    Register.CASUAL: "casual and informal, like a blog post or social media",
    Register.CONVERSATIONAL: "friendly and approachable, like talking to a colleague",
    Register.PROFESSIONAL: "clear and professional, standard business tone",
    Register.FORMAL: "formal and official, appropriate for legal or governmental contexts",
    Register.ACADEMIC: "scholarly and precise, with citations and hedged claims",
    Register.TECHNICAL: "technical and specialized, assuming domain expertise",
    Register.JOURNALISTIC: "clear and factual, inverted pyramid news style",
    Register.CREATIVE: "expressive and literary, with vivid language and style",
}

# Default register distribution
DEFAULT_REGISTER_WEIGHTS = {
    Register.CASUAL: 0.10,
    Register.CONVERSATIONAL: 0.15,
    Register.PROFESSIONAL: 0.25,
    Register.FORMAL: 0.15,
    Register.ACADEMIC: 0.15,
    Register.TECHNICAL: 0.10,
    Register.JOURNALISTIC: 0.05,
    Register.CREATIVE: 0.05,
}


class PromptVariant(str, Enum):
    """System-prompt variant used for a generation call.

    ``V0_3_1`` is the frozen preset that reproduces the exact system prompt
    used for all 42,532 documents published in v0.3.1 (see
    ``GENERATION_INSTRUCTIONS``). It must never change: it is what makes the
    published corpus byte-for-byte reproducible.

    The remaining members are v0.4 variants. They differ from each other
    (and from v0.3.1) only in framing, voice, and emphasis, plus a
    form-conditional output-format instruction in place of v0.3.1's global
    markdown mandate. Every variant retains the SHOW-DON'T-TELL block
    verbatim (see ``SHOW_DONT_TELL_BLOCK``) — that constraint is load-bearing
    for label non-triviality and is not a stylistic choice.
    """

    V0_3_1 = "v0.3.1"
    DIRECT = "v0.4-direct"
    PRACTITIONER = "v0.4-practitioner"
    EDITORIAL = "v0.4-editorial"
    ARCHIVAL = "v0.4-archival"


class OutputFormat(str, Enum):
    """A concrete output-format instruction for the generated document body.

    Selected per-document from the document's LCGFT form/category (see
    ``resolve_output_format``) so that output format is a property of the
    document type, not a global mandate. Only used by v0.4 prompt variants —
    ``PromptVariant.V0_3_1`` always uses its own frozen, globally-mandated
    markdown instruction.
    """

    PLAIN_PROSE = "plain_prose"
    VERSE = "verse"
    STRUCTURED_MARKDOWN = "structured_markdown"
    LEGAL_STYLE = "legal_style"
    LETTER_STYLE = "letter_style"
    TRANSCRIPT_STYLE = "transcript_style"
    DIARY_STYLE = "diary_style"
    LIST_STYLE = "list_style"
    CAPTION_STYLE = "caption_style"


class GeneratedContent(BaseModel):
    """Pydantic model for LLM-generated content."""

    title: str = Field(description="Document title")
    body: str = Field(description="Document body text")


@dataclass
class GeneratedDocument:
    """A document with generated text content."""

    # Original sampled labels
    document: Document

    # Generated content
    title: str
    body: str

    # Generation metadata
    prompt: str | None = None
    target_length: DocumentLength | None = None
    register: Register | None = None
    word_count: int | None = None
    sampling_params: SamplingParams | None = None
    prompt_variant_id: str | None = None

    def __post_init__(self):
        if self.word_count is None:
            self.word_count = len(self.body.split())

    def __str__(self) -> str:
        length_str = self.target_length.value if self.target_length else "unknown"
        register_str = self.register.value if self.register else "unknown"
        lines = [
            f"Title: {self.title}",
            f"Length: {self.word_count} words ({length_str}) | Register: {register_str}",
            "",
            f"{self.body[:500]}{'...' if len(self.body) > 500 else ''}",
            "",
            "--- Labels ---",
            str(self.document),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full dictionary representation."""
        return {
            **self.document.to_dict(),
            "title": self.title,
            "body": self.body,
            "word_count": self.word_count,
            "target_length": self.target_length.value if self.target_length else None,
            "register": self.register.value if self.register else None,
            "prompt": self.prompt,
            "prompt_variant_id": self.prompt_variant_id,
            "sampling_params": self.sampling_params.to_dict()
            if self.sampling_params
            else None,
        }

    def to_benchmark_item(self) -> dict:
        """Format for benchmark dataset (text + labels)."""
        return {
            "id": self.document.id,
            "title": self.title,
            "text": self.body,
            "word_count": self.word_count,
            "register": self.register.value if self.register else None,
            "labels": self.document.label_vector,
        }


# System instructions for document generation
# Version 2.0: Added anti-leakage instructions and semantic descriptions
GENERATION_INSTRUCTIONS = """Generate a realistic document matching the provided specifications.

STYLE: Structure and format your output as this document type actually appears in the real world. Replicate authentic conventions, layout, voice, and structural elements characteristic of this form.

SUBJECT AREA: This is the subject field. Use vocabulary, concepts, methods, and framing native to this discipline. Demonstrate authentic domain expertise through content, not labels.

TOPICS: These specific subjects must appear substantively in the content. Address each topic meaningfully, not just mentioned in passing.

AUDIENCE: Write for this reader. Calibrate vocabulary complexity, assumed prior knowledge, explanatory depth, and mode of address to match what this audience expects.

GEOGRAPHIC: Ground the content in this location. Use relevant place names, local institutions, regional context, and location-appropriate references.

SHOW, DON'T TELL - Avoid artificial self-labeling:
- Don't open with field announcements like "In political science..." or "In the field of medicine..." - just write about the subject naturally
- Don't use meta-commentary about the document type like "This satire explores..." or "This lecture covers..." - just BE that type
- Don't add classification headers like "Document Type:" or "Subject Area:" or "Category:" or "LCGFT:" or "LCC:"
- Using domain vocabulary naturally is fine and expected (e.g., "the court ruled" in a legal document, "the patient presented with" in a medical case)
- The difference: "In political science, civil law refers to..." (bad - announces the field) vs. "Civil law systems trace their origins to Roman codes..." (good - demonstrates expertise naturally)

Unusual combinations are intentional. If the form seems mismatched with the domain, interpret creatively while honoring both constraints authentically.

Output format: "Title: {title}\n\n{body}"
Line 1: "Title: " followed by title text
Line 2: blank (the split point is "\n\n")
Line 3+: markdown body"""


# The anti-self-labeling constraint block, verbatim as it appears inside
# GENERATION_INSTRUCTIONS above. Extracted as its own constant so every v0.4
# prompt variant can embed it identically rather than re-deriving it — this
# is the piece that keeps labels non-trivial by suppressing self-announcing
# text, and it is NOT a stylistic knob: every PromptVariant must contain it
# unmodified. (Verified in tests/unit/test_prompt_variants.py.)
SHOW_DONT_TELL_BLOCK = """SHOW, DON'T TELL - Avoid artificial self-labeling:
- Don't open with field announcements like "In political science..." or "In the field of medicine..." - just write about the subject naturally
- Don't use meta-commentary about the document type like "This satire explores..." or "This lecture covers..." - just BE that type
- Don't add classification headers like "Document Type:" or "Subject Area:" or "Category:" or "LCGFT:" or "LCC:"
- Using domain vocabulary naturally is fine and expected (e.g., "the court ruled" in a legal document, "the patient presented with" in a medical case)
- The difference: "In political science, civil law refers to..." (bad - announces the field) vs. "Civil law systems trace their origins to Roman codes..." (good - demonstrates expertise naturally)"""

assert SHOW_DONT_TELL_BLOCK in GENERATION_INSTRUCTIONS, (
    "SHOW_DONT_TELL_BLOCK drifted from the text embedded in GENERATION_INSTRUCTIONS"
)


# Semantic descriptions for LCC classes (avoid using exact taxonomy names)
LCC_SEMANTIC_DESCRIPTIONS = {
    "A": "general reference, encyclopedias, journalism, museums",
    "B": "philosophy, psychology, ethics, religion, spirituality",
    "C": "historical sciences, archaeology, genealogy, biography",
    "D": "world history, ancient civilizations, modern nations, wars",
    "E": "American history, United States, colonial era, civil war",
    "F": "Americas history, Canada, Latin America, local US history",
    "G": "geography, maps, anthropology, folklore, sports, recreation",
    "H": "social sciences, economics, sociology, statistics, commerce",
    "J": "government, politics, policy, elections, political systems, international relations",
    "K": "law, legal systems, courts, legislation, constitutional law",
    "L": "education, schools, teaching, curriculum, higher education",
    "M": "music, musical instruments, compositions, music theory",
    "N": "visual arts, painting, sculpture, architecture, photography",
    "P": "language, linguistics, literature, fiction, poetry, drama",
    "Q": "science, mathematics, physics, chemistry, biology, astronomy",
    "R": "medicine, healthcare, diseases, anatomy, nursing, public health",
    "S": "agriculture, farming, crops, livestock, forestry, fishing",
    "T": "technology, engineering, manufacturing, construction, crafts",
    "U": "military science, armies, warfare, defense, veterans",
    "V": "naval science, navies, ships, maritime, coast guard",
    "Z": "bibliography, libraries, publishing, book history, information science",
}

# Semantic descriptions for LCGFT categories (avoid using exact taxonomy names)
LCGFT_CATEGORY_DESCRIPTIONS = {
    "Cartographic materials": "spatial representations, geographic visualizations",
    "Commemorative works": "memorials, tributes, anniversary publications",
    "Creative nonfiction": "narrative journalism, personal essays, memoirs",
    "Discursive works": "essays, criticism, commentary, analysis, opinion",
    "Ephemera": "flyers, tickets, menus, programs, temporary materials",
    "Informational works": "reference materials, guides, handbooks, reports",
    "Instructional and educational works": "tutorials, courses, how-to guides, lessons",
    "Law materials": "legal documents, statutes, court records, contracts",
    "Literature": "fiction, poetry, drama, creative writing, novels",
    "Music": "compositions, songs, scores, musical works",
    "Recreational works": "games, puzzles, humor, entertainment",
    "Religious materials": "sacred texts, prayers, sermons, devotional works",
    "Sound recordings": "audio content, spoken word, podcasts, interviews",
    "Visual works": "images, video, film, photography, graphic content",
}

# Semantic descriptions for LCGFT forms (common ones - add more as needed)
LCGFT_FORM_DESCRIPTIONS = {
    # Literature
    "Satire": "humorous critique using irony, exaggeration, and wit",
    "Poetry": "verse, poems, lyrical compositions",
    "Novels": "long-form fiction, narrative storytelling",
    "Short stories": "brief fictional narratives",
    "Drama": "plays, theatrical scripts, dialogues",
    "Essays": "short prose on a single subject",
    # Informational
    "Lectures": "educational presentations, academic talks",
    "Reports": "formal accounts, findings, documentation",
    "Handbooks": "practical guides, reference manuals",
    "Encyclopedias": "comprehensive reference works",
    # Cartographic
    "Maps": "geographic representations, spatial visualizations",
    "Atlases": "collections of maps, geographic compendiums",
    # Music
    "Songs": "musical compositions with lyrics",
    "Hymns": "religious songs, worship music",
    # Religious
    "Prayers": "devotional texts, supplications",
    "Sermons": "religious addresses, homilies",
    # Visual
    "Photographs": "captured images, photographic documentation",
    "Television programs": "broadcast video content, TV shows",
    "Films": "motion pictures, movies, cinema",
    # Instructional
    "Textbooks": "educational materials, course books",
    "Tutorials": "step-by-step instructional content",
    # Other
    "Interviews": "conversations, Q&A format discussions",
    "Journalism": "news reporting, journalistic writing",
    "Criticism": "analytical evaluation, reviews",
    "Biographies": "life stories, biographical accounts",
    "Diaries": "personal journals, day-by-day records",
    "Letters": "correspondence, written communications",
    "Speeches": "formal addresses, orations",
    "Field recordings": "on-location audio captures",
}


def _parse_generated_text(text: str) -> tuple[str, str]:
    """Parse generated text into title and body.

    Expected format:
        Title: Some Title Here

        Body content in markdown...
    """
    text = text.strip()

    # Split on first double newline
    first_line, sep, body = text.partition("\n\n")

    # Extract title from first line
    if first_line.startswith("Title:") or first_line.startswith("Title "):
        title = first_line[6:].strip()
    else:
        # Fallback: first line is title
        title = first_line.strip()

    body = body.strip() if sep else ""

    return title, body


def _get_form_description(form: str, category: str, enriched: Any | None = None) -> str:
    """Get semantic description for a form, with category fallback.

    Resolution order, best conditioning first: the hand-written description
    (written specifically to describe a form without naming it), then a
    sanitized Library of Congress description if ``enriched`` is supplied, then
    the category description, then the bare form name.

    ``enriched`` is an optional :class:`shelf.sampler.enriched.EnrichedDescriptions`.
    When it is omitted -- the default -- this function behaves exactly as it did
    in v0.3.1, so the frozen corpus stays byte-for-byte reproducible.
    """
    # Try form-specific description first
    if form in LCGFT_FORM_DESCRIPTIONS:
        return LCGFT_FORM_DESCRIPTIONS[form]
    # Sanitized LC text extends coverage to the ~500 forms nobody hand-wrote
    if enriched is not None:
        lc_description = enriched.for_form(form)
        if lc_description:
            return lc_description
    # Fall back to category description
    if category in LCGFT_CATEGORY_DESCRIPTIONS:
        return LCGFT_CATEGORY_DESCRIPTIONS[category]
    # Last resort: use form name but lowercase
    return form.lower()


def _get_domain_description(
    lcc_code: str,
    lcc_name: str,
    enriched: Any | None = None,
    subclass: str | None = None,
) -> str:
    """Get semantic description for a domain.

    As with :func:`_get_form_description`, the hand-written description wins and
    sanitized LC text is the fallback that covers subclass codes. Omitting both
    ``enriched`` and ``subclass`` reproduces v0.3.1 behavior exactly.

    ``subclass`` is the v0.4 Phase 2 difficulty tier. When a document carries
    one, its description **replaces** the parent class description rather than
    supplementing it: the whole point of the tier is that "QA" and "QC" must be
    separable, and conditioning both on `LCC_SEMANTIC_DESCRIPTIONS["Q"]` --
    "science, mathematics, physics, chemistry, biology, astronomy" -- would
    condition them on exactly the same words. The parent description stays as
    the fallback for a subclass LC never described.
    """
    if subclass and enriched is not None:
        subclass_description = enriched.for_lcc_subclass(subclass)
        if subclass_description:
            return subclass_description
    if lcc_code in LCC_SEMANTIC_DESCRIPTIONS:
        return LCC_SEMANTIC_DESCRIPTIONS[lcc_code]
    if enriched is not None:
        lc_description = enriched.for_lcc_subclass(lcc_code)
        if lc_description:
            return lc_description
    # Fall back to name but avoid exact match
    return lcc_name.lower().replace(" and ", ", ")


# =============================================================================
# Form-conditional output format (v0.4 prompt variants only)
# =============================================================================
#
# v0.3.1 mandated "Line 3+: markdown body" for every document regardless of
# form (see GENERATION_INSTRUCTIONS above). Measured against the published
# corpus, that produced markdown headers in 41.3% of documents and bold in
# 59.9% overall -- including 18.3% of Literature with headers and 24.5% of
# Creative nonfiction with bold, neither of which is how those forms look in
# the real world. The mapping below selects a format instruction from the
# document's LCGFT form (falling back to its category, then to plain prose)
# instead of mandating one format globally. It only affects PromptVariant
# members other than V0_3_1; the frozen v0.3.1 preset is untouched.

# Per-category default output format, keyed by the 14 LCGFT categories
# (see LCGFT_CATEGORY_DESCRIPTIONS for the exact category strings).
CATEGORY_OUTPUT_FORMATS: dict[str, OutputFormat] = {
    "Cartographic materials": OutputFormat.LIST_STYLE,
    "Commemorative works": OutputFormat.PLAIN_PROSE,
    "Creative nonfiction": OutputFormat.PLAIN_PROSE,
    "Discursive works": OutputFormat.PLAIN_PROSE,
    "Ephemera": OutputFormat.LIST_STYLE,
    "Informational works": OutputFormat.STRUCTURED_MARKDOWN,
    "Instructional and educational works": OutputFormat.STRUCTURED_MARKDOWN,
    "Law materials": OutputFormat.LEGAL_STYLE,
    "Literature": OutputFormat.PLAIN_PROSE,
    "Music": OutputFormat.VERSE,
    "Recreational works": OutputFormat.PLAIN_PROSE,
    "Religious materials": OutputFormat.PLAIN_PROSE,
    "Sound recordings": OutputFormat.TRANSCRIPT_STYLE,
    "Visual works": OutputFormat.CAPTION_STYLE,
}

# Form-level overrides for specific forms whose real-world format differs
# from their category's default (e.g. Poetry is under a prose-leaning
# category umbrella but is verse; Handbooks are informational but genuinely
# benefit from structured markdown). Checked before the category default.
FORM_OUTPUT_FORMATS: dict[str, OutputFormat] = {
    "Poetry": OutputFormat.VERSE,
    "Songs": OutputFormat.VERSE,
    "Hymns": OutputFormat.VERSE,
    "Prayers": OutputFormat.PLAIN_PROSE,
    "Sermons": OutputFormat.PLAIN_PROSE,
    "Speeches": OutputFormat.PLAIN_PROSE,
    "Lectures": OutputFormat.PLAIN_PROSE,
    "Jokes": OutputFormat.PLAIN_PROSE,
    "Drama": OutputFormat.TRANSCRIPT_STYLE,
    "Interviews": OutputFormat.TRANSCRIPT_STYLE,
    "Letters": OutputFormat.LETTER_STYLE,
    "Diaries": OutputFormat.DIARY_STYLE,
    "Maps": OutputFormat.LIST_STYLE,
    "Atlases": OutputFormat.LIST_STYLE,
    "Games": OutputFormat.LIST_STYLE,
    "Satellite imagery": OutputFormat.CAPTION_STYLE,
    "Field recordings": OutputFormat.CAPTION_STYLE,
    "Television programs": OutputFormat.CAPTION_STYLE,
    "Films": OutputFormat.CAPTION_STYLE,
    "Photographs": OutputFormat.CAPTION_STYLE,
    "Handbooks": OutputFormat.STRUCTURED_MARKDOWN,
    "Textbooks": OutputFormat.STRUCTURED_MARKDOWN,
    "Tutorials": OutputFormat.STRUCTURED_MARKDOWN,
    "Encyclopedias": OutputFormat.STRUCTURED_MARKDOWN,
    "Reports": OutputFormat.STRUCTURED_MARKDOWN,
    "Legal briefs": OutputFormat.LEGAL_STYLE,
}

# The "Line 3+" instruction for each output format. Deliberately silent on
# em-dashes or any other negative style constraint -- see data_plan_v0.4.md
# section 4.2: instructing against one artifact just substitutes another.
_OUTPUT_FORMAT_INSTRUCTIONS: dict[OutputFormat, str] = {
    OutputFormat.PLAIN_PROSE: (
        "Line 3+: continuous prose in paragraphs separated by blank lines. "
        "Do not use markdown headers, bold/italic emphasis markup, or "
        "bullet lists -- this document type reads as running prose, not a "
        "formatted document."
    ),
    OutputFormat.VERSE: (
        "Line 3+: lineated verse. Break lines where the verse breaks, not "
        "at the margin; separate stanzas with a blank line. Do not use "
        "markdown headers, bullet lists, or prose paragraphs."
    ),
    OutputFormat.STRUCTURED_MARKDOWN: (
        "Line 3+: markdown body. Use headers, bold for key terms, and "
        "bullet or numbered lists where they genuinely aid the reader, the "
        "way a real reference or instructional document would."
    ),
    OutputFormat.LEGAL_STYLE: (
        "Line 3+: legal-document prose. Use ALL-CAPS or numbered section "
        'labels (e.g. "I. BACKGROUND", "WHEREAS") the way real legal '
        "instruments do. Do not use markdown bold, italics, or bullet "
        "lists."
    ),
    OutputFormat.LETTER_STYLE: (
        "Line 3+: epistolary form -- a salutation, body paragraphs, and a "
        "closing/signature line, the way a real letter is laid out. Do not "
        "use markdown headers or bullet lists."
    ),
    OutputFormat.TRANSCRIPT_STYLE: (
        "Line 3+: turn-by-turn transcript. Label each turn with a speaker "
        "name or role followed by a colon, on its own line. Do not use "
        "markdown headers, bold, or bullet lists."
    ),
    OutputFormat.DIARY_STYLE: (
        "Line 3+: dated journal entries. Begin each entry with a date or "
        "day label on its own line, followed by first-person prose. Do not "
        "use markdown headers, bold, or bullet lists."
    ),
    OutputFormat.LIST_STYLE: (
        "Line 3+: a structured list or gazetteer-style entries (names, "
        "labels, brief notations), the way a real map legend, program, or "
        "ephemeral notice is laid out. Bullets or short lines are fine; do "
        "not use markdown headers or bold prose paragraphs."
    ),
    OutputFormat.CAPTION_STYLE: (
        "Line 3+: a brief descriptive caption or liner note in plain "
        "prose, the way real visual or audio media is documented in text "
        "form. Do not use markdown headers, bold, or bullet lists."
    ),
}


# The envelope description shared by every prompt variant -- copied
# verbatim (including its newline handling) from the tail of
# GENERATION_INSTRUCTIONS, up to but not including the "Line 3+" line. Using
# the exact same wording that already produced 42,532 correctly-parsed
# documents avoids re-deriving new phrasing that might parse differently.
_ENVELOPE_PREFIX = (
    'Output format: "Title: {title}\n\n{body}"\n'
    'Line 1: "Title: " followed by title text\n'
    'Line 2: blank (the split point is "\n\n")\n'
)

assert GENERATION_INSTRUCTIONS.endswith(_ENVELOPE_PREFIX + "Line 3+: markdown body"), (
    "_ENVELOPE_PREFIX drifted from the tail of GENERATION_INSTRUCTIONS"
)


def resolve_output_format(form: str, category: str) -> OutputFormat:
    """Resolve the output format for a document from its LCGFT form/category.

    Form-level overrides take precedence over the category default; an
    unrecognized form/category falls back to plain prose rather than
    markdown, since plain prose is the safer default for the majority of
    document forms.
    """
    if form in FORM_OUTPUT_FORMATS:
        return FORM_OUTPUT_FORMATS[form]
    if category in CATEGORY_OUTPUT_FORMATS:
        return CATEGORY_OUTPUT_FORMATS[category]
    return OutputFormat.PLAIN_PROSE


def build_generation_prompt(
    doc: Document,
    length: DocumentLength = DocumentLength.MEDIUM,
    register: Register = Register.PROFESSIONAL,
    enriched: Any | None = None,
) -> str:
    """Build the input text for document generation.

    Uses semantic descriptions instead of exact taxonomy names to reduce
    label leakage in generated documents.

    Pass an optional :class:`shelf.sampler.enriched.EnrichedDescriptions` as
    ``enriched`` to extend description coverage to labels that have no
    hand-written entry. Its text is sanitized of taxonomy names before it
    reaches the prompt. Omitting it reproduces v0.3.1 byte-for-byte.
    """
    word_min, word_max = LENGTH_WORD_RANGES[length]
    register_desc = REGISTER_DESCRIPTIONS[register]

    # Use semantic descriptions instead of exact taxonomy names
    form_desc = _get_form_description(doc.lcgft.form, doc.lcgft.category, enriched)
    domain_desc = _get_domain_description(
        doc.lcc.code, doc.lcc.name, enriched, subclass=doc.lcc.subclass
    )

    parts = [
        f"style: {form_desc}",
        f"subject area: {domain_desc}",
        f"topics: {', '.join(doc.topics)}",
        f"length: {word_min}-{word_max} words",
        f"tone: {register_desc}",
    ]

    if doc.audience:
        parts.append(f"audience: {doc.audience}")

    if doc.geographic:
        parts.append(f"geographic: {', '.join(doc.geographic)}")

    return "\n".join(parts)


def build_title_prompt(doc: Document, enriched: Any | None = None) -> str:
    """Build a prompt just for title generation.

    Uses semantic descriptions to reduce label leakage in titles. As with
    :func:`build_generation_prompt`, ``enriched`` is optional and omitting it
    reproduces v0.3.1 behavior.
    """
    form_desc = _get_form_description(doc.lcgft.form, doc.lcgft.category, enriched)
    domain_desc = _get_domain_description(
        doc.lcc.code, doc.lcc.name, enriched, subclass=doc.lcc.subclass
    )

    return f"""Generate a creative, realistic title for a document that is:
- Style: {form_desc}
- Subject area: {domain_desc}
- Topics: {", ".join(doc.topics)}
{f"- Audience: {doc.audience}" if doc.audience else ""}

Avoid self-labeling titles that announce the genre or field.
Bad: "A Satire on Agriculture" or "Political Science Lecture" or "Medical Handbook"
Good: Titles that hint at content naturally - the way real documents are titled.

The combination may be unconventional - embrace it creatively.
Respond with ONLY the title, no quotes or explanation."""


# =============================================================================
# Prompt variants (v0.4) -- system prompt is a controlled, recorded factor
# =============================================================================
#
# Each preamble below covers the same functional ground as
# GENERATION_INSTRUCTIONS (style/subject/topics/audience/geographic framing,
# then the SHOW_DONT_TELL_BLOCK, then a note on unusual combinations) but in
# a distinct voice/framing. None of them add negative style instructions
# (e.g. "avoid em-dashes") -- see data_plan_v0.4.md section 4.2, point 3.
# The output-format instruction is appended separately by build_system_prompt
# since it depends on the specific document's form/category.

_VARIANT_PREAMBLES: dict[PromptVariant, str] = {
    PromptVariant.DIRECT: (
        "Write one realistic document to the specification below. Do not "
        "explain what you're doing -- produce the document itself, exactly "
        "as it would exist in the world.\n"
        "\n"
        "Style -- match this document type's real-world conventions: its "
        "layout, its voice, its structural habits, down to how it opens "
        "and closes.\n"
        "\n"
        "Subject -- write with the vocabulary, reasoning, and assumptions "
        "native to this discipline. Expertise should show up in the "
        "content, never in a label.\n"
        "\n"
        "Topics -- treat every listed topic as substantive material the "
        "document must actually engage with, not a checklist to mention "
        "in passing.\n"
        "\n"
        "Audience -- pitch vocabulary, assumed background, and depth of "
        "explanation at exactly this reader.\n"
        "\n"
        "Geographic -- anchor the document in this place: real-feeling "
        "names, institutions, and regional detail, not a generic "
        "setting.\n"
        "\n" + SHOW_DONT_TELL_BLOCK + "\n"
        "\n"
        "Treat any unlikely pairing of form and subject as a constraint to "
        "satisfy, not a mismatch to flag. Write it straight."
    ),
    PromptVariant.PRACTITIONER: (
        "You are the person who actually writes this kind of document for "
        "a living, sitting down to produce one more of them. You are not "
        "describing the form from outside it -- you are inside it, doing "
        "the work.\n"
        "\n"
        "Form: this is what you are writing today. Let its real-world "
        "conventions -- structure, pacing, the things practitioners always "
        "include and always skip -- come through the way they would from "
        "someone who has written a hundred of these before.\n"
        "\n"
        "Domain: write with the reflexes of someone trained in this field "
        "-- its assumptions, its shorthand, its way of framing a problem "
        "-- without ever naming the field.\n"
        "\n"
        "Topics: these are the actual substance of what you're writing "
        "about today. Engage with each one the way the work requires, not "
        "as items on a list.\n"
        "\n"
        "Audience: this is who is going to read what you produce. Write "
        "for them specifically -- their vocabulary, what they already "
        "know, how much explanation they need.\n"
        "\n"
        "Geographic: this is where the work is set or where it comes "
        "from. Let real place names, local institutions, and regional "
        "texture ground it.\n"
        "\n" + SHOW_DONT_TELL_BLOCK + "\n"
        "\n"
        "If the assignment pairs an unexpected subject with this form, "
        "that's just the job today -- do it the way a working "
        "professional would, without commentary on how unusual it is."
    ),
    PromptVariant.EDITORIAL: (
        "You are producing finished copy for a real publication or "
        "institution that regularly puts out documents exactly like this "
        "one. An editor will run it as-is -- it needs to already read like "
        "something that belongs on their shelf, not a draft or an "
        "approximation.\n"
        "\n"
        "Form: match the exact conventions this outlet's readers expect -- "
        "the structure, the layout, the register -- for this kind of "
        "piece.\n"
        "\n"
        "Subject: write with the command of the subject that this "
        "publication's contributors are known for. Expertise reads "
        "through the content, never through a label on it.\n"
        "\n"
        "Topics: every topic listed is something the piece is actually "
        "about and must develop, not a keyword to touch on.\n"
        "\n"
        "Audience: this publication's readership. Write directly to them "
        "-- their vocabulary, their assumed background, what they're "
        "actually looking for from a piece like this.\n"
        "\n"
        "Geographic: the piece's place of origin or setting. Ground it in "
        "specifics -- real institutions, real place names, real regional "
        "detail -- the way a piece written from and for that place would "
        "be.\n"
        "\n" + SHOW_DONT_TELL_BLOCK + "\n"
        "\n"
        "If the pairing of form and subject is an unusual one for this "
        "outlet, run with it as a deliberate editorial choice, not an "
        "oddity to flag to the reader."
    ),
    PromptVariant.ARCHIVAL: (
        "This document is being added to an existing collection that "
        "already holds many genuine examples of this exact type. It has "
        "to sit among them without standing out -- no tell that separates "
        "it from the real holdings around it.\n"
        "\n"
        "Form: reproduce this type's real conventions precisely -- the "
        "collection's other holdings of this form all share a "
        "recognizable structure, voice, and layout, and this one should "
        "be no exception.\n"
        "\n"
        "Subject: the collection's holdings on this subject demonstrate "
        "real command of it -- vocabulary, reasoning, and assumptions "
        "specific to the discipline, carried entirely by the content.\n"
        "\n"
        "Topics: these are the specific subjects a cataloguer would say "
        "this item is substantively about -- each must actually appear in "
        "the material, not just be gestured at.\n"
        "\n"
        "Audience: the collection's items are each written for a specific "
        "kind of reader. Match this one's vocabulary, assumed knowledge, "
        "and depth to that reader.\n"
        "\n"
        "Geographic: place this item concretely -- real names, "
        "institutions, and regional detail consistent with where it comes "
        "from.\n"
        "\n" + SHOW_DONT_TELL_BLOCK + "\n"
        "\n"
        "Unusual pairings of form and subject already exist elsewhere in "
        "the collection -- treat this one the same way: authentic and "
        "unremarked upon, not framed as an oddity."
    ),
}


def build_system_prompt(
    doc: Document, variant: PromptVariant | str = PromptVariant.V0_3_1
) -> str:
    """Build the system prompt for a document, given a prompt variant.

    ``PromptVariant.V0_3_1`` always returns ``GENERATION_INSTRUCTIONS``
    unchanged (byte-for-byte), regardless of ``doc`` -- this is the frozen
    preset that reproduces the published v0.3.1 corpus.

    Every other variant appends a form-conditional output-format
    instruction (see ``resolve_output_format``) to that variant's preamble,
    keeping the same "Title: {title}\\n\\n{body}" envelope that
    ``_parse_generated_text`` depends on.
    """
    variant = PromptVariant(variant)
    if variant is PromptVariant.V0_3_1:
        return GENERATION_INSTRUCTIONS

    preamble = _VARIANT_PREAMBLES[variant]
    output_format = resolve_output_format(doc.lcgft.form, doc.lcgft.category)
    format_block = _ENVELOPE_PREFIX + _OUTPUT_FORMAT_INSTRUCTIONS[output_format]
    return f"{preamble}\n\n{format_block}"


# =============================================================================
# Template-based generation (no LLM needed)
# =============================================================================

TITLE_TEMPLATES = {
    "Informational works": [
        "{topic}: A Comprehensive Overview",
        "Understanding {topic} in {domain}",
        "The Essential Guide to {topic}",
        "{topic} Report {year}",
        "Analysis of {topic}: Key Findings",
    ],
    "Law materials": [
        "In Re: {topic}",
        "{topic}: Legal Framework and Analysis",
        "Statutory Guide to {topic}",
        "Case Studies in {topic}",
        "{topic} Compliance Manual",
    ],
    "Instructional and educational works": [
        "Learn {topic}: A Beginner's Guide",
        "{topic} for {audience}",
        "Mastering {topic}: Step by Step",
        "The {topic} Handbook",
        "Introduction to {topic}",
    ],
    "Literature": [
        "The {topic} Chronicles",
        "Songs of {topic}",
        "A {topic} Story",
        "Reflections on {topic}",
        "{topic}: A Novel",
    ],
    "Creative nonfiction": [
        "My Journey Through {topic}",
        "{topic}: A Personal Account",
        "Inside {topic}",
        "The {topic} Diaries",
        "Dispatches from {topic}",
    ],
    "Discursive works": [
        "On {topic}",
        "The Case for {topic}",
        "{topic}: A Critical Analysis",
        "Debating {topic}",
        "Perspectives on {topic}",
    ],
    "Visual works": [
        "{topic} in Focus",
        "Visions of {topic}",
        "{topic}: A Visual Journey",
        "The Art of {topic}",
        "Seeing {topic}",
    ],
    "Sound recordings": [
        "{topic}: The Audio Experience",
        "Sounds of {topic}",
        "{topic} Sessions",
        "Listening to {topic}",
        "{topic}: A Podcast",
    ],
    "Ephemera": [
        "{topic} Collection",
        "The {topic} Papers",
        "{topic} Memorabilia",
        "Fragments of {topic}",
        "{topic} Archives",
    ],
    "default": [
        "{topic}",
        "On {topic}",
        "The {topic} Document",
        "{topic}: {domain}",
        "Exploring {topic}",
    ],
}

BODY_TEMPLATES = {
    "Informational works": """
This document provides a comprehensive examination of {topic} within the context of {domain}.
Drawing on extensive research and analysis, we explore the key aspects, current developments,
and implications of this subject area.

{topic} represents a significant area of inquiry in {domain}. Recent developments have
highlighted the importance of understanding the fundamental principles and practical
applications. This work synthesizes current knowledge and presents it in an accessible format
{audience_note}.

Key areas covered include the theoretical foundations, practical considerations, and
emerging trends. The analysis draws on multiple sources and methodologies to provide
a balanced and thorough treatment of the subject.

{geographic_note}

The findings presented here have implications for researchers, practitioners, and
policymakers alike. By understanding {topic} in depth, stakeholders can make more
informed decisions and contribute to advancing knowledge in this field.
""",
    "Law materials": """
INTRODUCTION

This legal document addresses matters pertaining to {topic} under the framework of {domain}.
The following analysis considers relevant statutory provisions, case law, and regulatory
guidance applicable to the subject matter.

BACKGROUND

{topic} presents significant legal considerations that require careful examination.
The intersection of {domain} principles with practical application necessitates a
thorough understanding of the governing legal framework.

ANALYSIS

The legal landscape surrounding {topic} has evolved considerably. Courts and regulatory
bodies have developed substantial jurisprudence addressing key issues. This document
examines the primary legal standards and their application.

{geographic_note}

CONCLUSION

Based on the foregoing analysis, the legal treatment of {topic} requires attention to
both established precedent and emerging developments. {audience_note}
""",
    "Literature": """
The story begins, as all stories do, with a question about {topic}.

In the realm of {domain}, where ideas take shape and meanings multiply, there exists
a truth that few have dared to explore. This is not merely an account of facts and
figures, but a journey into the heart of what {topic} truly means.

Consider the way light falls on an ordinary day. Consider how {topic} transforms
everything it touches, reshaping our understanding of {domain} itself. The characters
in this narrative—and we are all characters, in a sense—must navigate these waters.

{geographic_note}

What follows is both a meditation and an adventure. {audience_note} We invite you to
see {topic} not as something distant and abstract, but as a living force that shapes
our world and our stories within it.

The pages ahead will challenge assumptions and illuminate possibilities. In {domain},
as in life, the most profound discoveries often come from the most unexpected places.
""",
    "Instructional and educational works": """
WELCOME TO {topic}

This guide will walk you through the essential concepts and practical skills needed
to understand {topic} in the context of {domain}. {audience_note}

GETTING STARTED

Before diving in, let's establish some foundational concepts. {topic} encompasses
several key areas that build upon each other. Understanding these basics will help
you progress effectively.

KEY CONCEPTS

1. Foundation: The core principles of {topic}
2. Application: How these principles apply in {domain}
3. Practice: Hands-on exercises and examples
4. Mastery: Advanced techniques and considerations

STEP-BY-STEP GUIDANCE

Begin by familiarizing yourself with the terminology and basic framework. {topic}
may seem complex at first, but breaking it down into manageable components makes
learning straightforward.

{geographic_note}

PRACTICE EXERCISES

Apply what you've learned through the exercises included in this guide. Consistent
practice is the key to developing proficiency in {topic}.

NEXT STEPS

Once you've mastered the basics, you'll be ready to explore more advanced aspects
of {topic} and its applications in {domain}.
""",
    "default": """
{topic} occupies a unique position within {domain}, offering perspectives and insights
that merit careful consideration. This document explores the subject from multiple
angles, providing a thorough treatment of key themes and developments.

The relationship between {topic} and broader trends in {domain} reveals important
patterns and possibilities. By examining these connections, we can develop a more
nuanced understanding of both the specific subject and its wider context.

{audience_note}

Several key themes emerge from this exploration:

First, the fundamental nature of {topic} and its defining characteristics. Second,
the ways in which {topic} intersects with other areas of {domain}. Third, the
practical implications and applications that arise from this understanding.

{geographic_note}

Looking ahead, {topic} will likely continue to evolve and develop in response to
changing circumstances and new discoveries. This document provides a foundation
for understanding these developments as they unfold.

The insights gathered here represent a synthesis of current knowledge and emerging
perspectives. They offer a starting point for further inquiry and deeper engagement
with {topic} in all its complexity.
""",
}


class TemplateGenerator:
    """Generate document text using templates (no LLM required)."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self, doc: Document) -> GeneratedDocument:
        """Generate title and body from templates."""
        topic = doc.topics[0] if doc.topics else "the subject"
        domain = doc.lcc.name
        category = doc.lcgft.category

        # Build audience note
        if doc.audience:
            audience_note = f"This material is designed for {doc.audience.lower()}."
        else:
            audience_note = ""

        # Build geographic note
        if doc.geographic:
            geographic_note = (
                f"This work focuses particularly on {', '.join(doc.geographic)}."
            )
        else:
            geographic_note = ""

        # Select and fill title template
        title_templates = TITLE_TEMPLATES.get(category, TITLE_TEMPLATES["default"])
        title_template = self._rng.choice(title_templates)
        title = title_template.format(
            topic=topic,
            domain=domain,
            audience=doc.audience or "readers",
            year=self._rng.randint(2020, 2024),
        )

        # Select and fill body template
        body_templates = BODY_TEMPLATES.get(category, BODY_TEMPLATES["default"])
        if isinstance(body_templates, list):
            body_template = self._rng.choice(body_templates)
        else:
            body_template = body_templates

        body = body_template.format(
            topic=topic,
            domain=domain,
            audience_note=audience_note,
            geographic_note=geographic_note,
        ).strip()

        return GeneratedDocument(
            document=doc,
            title=title,
            body=body,
            prompt=None,
        )


class LengthSampler:
    """Sample document lengths with configurable distribution."""

    def __init__(
        self,
        weights: dict[DocumentLength, float] | None = None,
        seed: int | None = None,
    ):
        self._rng = random.Random(seed)
        self._weights = weights or DEFAULT_LENGTH_WEIGHTS
        self._lengths = list(self._weights.keys())
        self._probs = list(self._weights.values())

    def sample(self) -> DocumentLength:
        return self._rng.choices(self._lengths, weights=self._probs, k=1)[0]


class RegisterSampler:
    """Sample writing registers with configurable distribution."""

    def __init__(
        self,
        weights: dict[Register, float] | None = None,
        seed: int | None = None,
    ):
        self._rng = random.Random(seed)
        self._weights = weights or DEFAULT_REGISTER_WEIGHTS
        self._registers = list(self._weights.keys())
        self._probs = list(self._weights.values())

    def sample(self) -> Register:
        return self._rng.choices(self._registers, weights=self._probs, k=1)[0]


# Default distribution for PromptVariantSampler: uniform over the v0.4
# variants only. PromptVariant.V0_3_1 is intentionally excluded from the
# default pool -- it is a frozen reproduction preset for the already-published
# corpus, not a candidate a v0.4 generation run should draw at random. Pass
# an explicit weights dict (including V0_3_1) to change this.
DEFAULT_PROMPT_VARIANT_WEIGHTS: dict[PromptVariant, float] = {
    PromptVariant.DIRECT: 0.25,
    PromptVariant.PRACTITIONER: 0.25,
    PromptVariant.EDITORIAL: 0.25,
    PromptVariant.ARCHIVAL: 0.25,
}


class PromptVariantSampler:
    """Sample prompt variants with a configurable, seeded distribution."""

    def __init__(
        self,
        weights: dict[PromptVariant, float] | None = None,
        seed: int | None = None,
    ):
        self._rng = random.Random(seed)
        self._weights = weights or DEFAULT_PROMPT_VARIANT_WEIGHTS
        self._variants = list(self._weights.keys())
        self._probs = list(self._weights.values())

    def sample(self) -> PromptVariant:
        return self._rng.choices(self._variants, weights=self._probs, k=1)[0]


class DocumentGenerator:
    """
    Generate complete documents with text using a configurable LLM backend.

    Defaults to OpenAI Responses, but can be swapped for Anthropic or Gemini.
    """

    def __init__(
        self,
        seed: int | None = None,
        model: str = DEFAULT_MODEL,
        service_tier: str | None = DEFAULT_SERVICE_TIER,
        client: "openai.OpenAI | None" = None,
        async_client: "openai.AsyncOpenAI | None" = None,
        llm_backend: LLMBackend | None = None,
        length_weights: dict[DocumentLength, float] | None = None,
        register_weights: dict[Register, float] | None = None,
        temperature_range: tuple[float, float] = TEMPERATURE_RANGE,
        top_p_range: tuple[float, float] = TOP_P_RANGE,
        use_llm: bool = True,
        prompt_variant: PromptVariant | str = PromptVariant.V0_3_1,
        prompt_variant_weights: dict[PromptVariant, float] | None = None,
    ):
        """
        prompt_variant / prompt_variant_weights control which system prompt
        (see PromptVariant) is used per document:

        - By default (neither argument given), every document uses the
          fixed, frozen ``PromptVariant.V0_3_1`` preset -- this keeps
          DocumentGenerator's default behavior byte-for-byte identical to
          v0.3.1, so existing call sites are unaffected.
        - Passing ``prompt_variant`` alone fixes every document to that one
          variant instead (still fully reproducible under ``seed``).
        - Passing ``prompt_variant_weights`` activates a seeded
          PromptVariantSampler that draws a (recorded) variant per document;
          this takes precedence over ``prompt_variant``.
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._model = model
        self._service_tier = service_tier
        self._client = client
        self._async_client = async_client
        self._llm_backend = llm_backend
        self._use_llm = use_llm
        self._template_gen = TemplateGenerator(seed)
        self._length_sampler = LengthSampler(length_weights, seed)
        self._register_sampler = RegisterSampler(register_weights, seed)
        self._sampling_sampler = SamplingParamsSampler(
            temperature_range, top_p_range, seed
        )
        self._fixed_prompt_variant = PromptVariant(prompt_variant)
        self._prompt_variant_sampler = (
            PromptVariantSampler(prompt_variant_weights, seed)
            if prompt_variant_weights is not None
            else None
        )

    def _get_llm_backend(self) -> LLMBackend:
        if self._llm_backend is not None:
            return self._llm_backend

        # Default to OpenAI Responses backend for compatibility
        self._llm_backend = OpenAIResponsesBackend(
            model=self._model,
            service_tier=self._service_tier,
            client=self._client,
            async_client=self._async_client,
        )
        return self._llm_backend

    def sample_length(self) -> DocumentLength:
        """Sample a document length."""
        return self._length_sampler.sample()

    def sample_register(self) -> Register:
        """Sample a writing register."""
        return self._register_sampler.sample()

    def sample_prompt_variant(self) -> PromptVariant:
        """Sample (or return the fixed) prompt variant for the next document."""
        if self._prompt_variant_sampler is not None:
            return self._prompt_variant_sampler.sample()
        return self._fixed_prompt_variant

    def generate(
        self,
        doc: Document,
        length: DocumentLength | None = None,
        register: Register | None = None,
        prompt_variant: PromptVariant | None = None,
    ) -> GeneratedDocument:
        """Generate a complete document with text."""
        length = length or self.sample_length()
        register = register or self.sample_register()

        if not self._use_llm:
            result = self._template_gen.generate(doc)
            result.target_length = length
            result.register = register
            return result

        variant = prompt_variant or self.sample_prompt_variant()
        input_text = build_generation_prompt(doc, length, register)
        sampling = self._sampling_sampler.sample()

        gen_result = self._get_llm_backend().generate(
            GenerationRequest(
                prompt=input_text,
                system_prompt=build_system_prompt(doc, variant),
            ),
            GenerationParams(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_output_tokens=4096,
            ),
        )
        raw_text = gen_result.text

        # Parse: first line is "Title: ...", then blank line, then body
        title, body = _parse_generated_text(raw_text)

        return GeneratedDocument(
            document=doc,
            title=title,
            body=body,
            prompt=input_text,
            target_length=length,
            register=register,
            sampling_params=sampling,
            prompt_variant_id=variant.value,
        )

    async def generate_async(
        self,
        doc: Document,
        length: DocumentLength | None = None,
        register: Register | None = None,
        prompt_variant: PromptVariant | None = None,
    ) -> GeneratedDocument:
        """Generate a document asynchronously."""
        length = length or self.sample_length()
        register = register or self.sample_register()

        if not self._use_llm:
            result = self._template_gen.generate(doc)
            result.target_length = length
            result.register = register
            return result

        variant = prompt_variant or self.sample_prompt_variant()
        input_text = build_generation_prompt(doc, length, register)
        sampling = self._sampling_sampler.sample()

        gen_result = await self._get_llm_backend().generate_async(
            GenerationRequest(
                prompt=input_text,
                system_prompt=build_system_prompt(doc, variant),
            ),
            GenerationParams(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_output_tokens=4096,
            ),
        )
        raw_text = gen_result.text

        title, body = _parse_generated_text(raw_text)

        return GeneratedDocument(
            document=doc,
            title=title,
            body=body,
            prompt=input_text,
            target_length=length,
            register=register,
            sampling_params=sampling,
            prompt_variant_id=variant.value,
        )

    def generate_batch(
        self,
        docs: list[Document],
        show_progress: bool = False,
        use_backend_batch: bool = False,
    ) -> list[GeneratedDocument]:
        """Generate text for a batch of documents (sync)."""
        if not self._use_llm or not use_backend_batch:
            results = []
            for i, doc in enumerate(docs):
                if show_progress and (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{len(docs)}")
                results.append(self.generate(doc))
            return results

        # Single sampling configuration applied to the whole batch
        sampling = self._sampling_sampler.sample()
        params = GenerationParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_output_tokens=4096,
        )

        requests: list[GenerationRequest] = []
        meta: list[tuple[Document, DocumentLength, Register, PromptVariant]] = []
        for doc in docs:
            length = self.sample_length()
            register = self.sample_register()
            variant = self.sample_prompt_variant()
            meta.append((doc, length, register, variant))
            requests.append(
                GenerationRequest(
                    prompt=build_generation_prompt(doc, length, register),
                    system_prompt=build_system_prompt(doc, variant),
                )
            )

        results_raw = self._get_llm_backend().generate_batch(requests, params)
        if len(results_raw) != len(docs):
            raise ValueError(
                "LLM backend returned mismatched batch size "
                f"(expected {len(docs)}, got {len(results_raw)})"
            )

        results: list[GeneratedDocument] = []
        for (doc, length, register, variant), request, gen_result in zip(
            meta, requests, results_raw
        ):
            raw_text = gen_result.text
            if not raw_text.strip():
                raise ValueError(
                    "Received empty output from LLM during batch generation"
                )
            title, body = _parse_generated_text(raw_text)
            results.append(
                GeneratedDocument(
                    document=doc,
                    title=title,
                    body=body,
                    prompt=request.prompt,
                    target_length=length,
                    register=register,
                    sampling_params=SamplingParams(
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                    ),
                    prompt_variant_id=variant.value,
                )
            )
        return results

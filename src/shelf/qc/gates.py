"""QC gates G1-G5 and G7 (§12 of ``docs/data_plan_v0.4.md``).

Every generated document passes through these gates before it enters the
corpus. Each gate produces a :class:`GateResult` that is both a pass/fail
verdict *and* a recorded metric -- per §12, several gates ("record the delta,
never silently keep", "report the fraction covered, not just a boolean") are
explicitly required to publish a measurement, not just a bit.

G6 (near-duplicate) is scored separately in :mod:`shelf.qc.dedup` because it
needs corpus-wide context (a MinHash/LSH index across all documents seen so
far) rather than being a pure function of one document. G4 (self-labeling)
reuses :func:`shelf.sampler.leakage.scan_document` rather than reimplementing
label-leakage detection.

Example:
    from shelf.qc.gates import run_gates

    result = run_gates(
        doc_id="doc-1",
        raw_text="Title: Solar Panels\\n\\nSolar panels convert sunlight...",
        word_count=42,
        target_word_range=(25, 50),
        topics=["Solar energy"],
    )
    result.passed          # True/False
    result.to_columns()    # {"qc_parse": True, "qc_length_delta": 0, ...}
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from shelf.sampler.generator import _parse_generated_text
from shelf.sampler.leakage import LeakageReport, scan_document

__all__ = [
    "Gate",
    "GateResult",
    "QCResult",
    "check_language",
    "check_length",
    "check_parse",
    "check_parse_fields",
    "check_refusal",
    "check_self_label",
    "check_topic_coverage",
    "run_gates",
]


class Gate(str, Enum):
    """The seven QC gates from §12 of the data plan."""

    PARSE = "g1_parse"
    LANGUAGE = "g2_language"
    LENGTH = "g3_length"
    SELF_LABEL = "g4_self_label"
    TOPIC_COVERAGE = "g5_topic_coverage"
    NEAR_DUPLICATE = "g6_near_duplicate"
    REFUSAL = "g7_refusal"


@dataclass(frozen=True)
class GateResult:
    """The outcome of one gate applied to one document.

    Attributes:
        gate: Which gate produced this result.
        passed: Whether the document clears the gate's threshold.
        value: The recorded metric (a delta, a fraction, a bool) -- this is
            what gets stored as the ``qc_*`` column, independent of the
            pass/fail verdict. Several gates require exactly this
            distinction: length adherence must "record the delta, never
            silently keep", and topic coverage must "report the fraction
            covered, not just a boolean".
        detail: Extra diagnostic context (e.g. matched phrases, target
            range). Not part of the stored column, useful for debugging and
            for building rejection-reason breakdowns in promotion checks.
    """

    gate: Gate
    passed: bool
    value: Any = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QCResult:
    """The full set of per-document gate outcomes (G1-G5, G7).

    G6 is attached after the fact (see :meth:`with_near_duplicate`) because
    it depends on corpus-wide state that a single document's fields cannot
    provide.
    """

    doc_id: str
    parse: GateResult
    language: GateResult
    length: GateResult
    self_label: GateResult
    topic_coverage: GateResult
    refusal: GateResult
    near_duplicate: GateResult | None = None

    @property
    def gates(self) -> tuple[GateResult, ...]:
        """All gate results attached so far, in G1..G7 order."""
        ordered = (
            self.parse,
            self.language,
            self.length,
            self.self_label,
            self.topic_coverage,
            self.near_duplicate,
            self.refusal,
        )
        return tuple(result for result in ordered if result is not None)

    @property
    def passed(self) -> bool:
        """Whether the document clears every attached gate."""
        return all(result.passed for result in self.gates)

    def failed_gates(self) -> tuple[Gate, ...]:
        """Which gates this document failed, in G1..G7 order."""
        return tuple(result.gate for result in self.gates if not result.passed)

    def with_near_duplicate(self, result: GateResult) -> QCResult:
        """Return a copy of this result with G6 attached.

        Args:
            result: The G6 :class:`GateResult`, typically produced by
                ``shelf.qc.dedup.NearDuplicateIndex.add``.
        """
        return QCResult(
            doc_id=self.doc_id,
            parse=self.parse,
            language=self.language,
            length=self.length,
            self_label=self.self_label,
            topic_coverage=self.topic_coverage,
            refusal=self.refusal,
            near_duplicate=result,
        )

    def to_columns(self) -> dict[str, Any]:
        """Serialize to the ``qc_*`` columns named in §13 of the data plan."""
        return {
            "qc_parse": self.parse.passed,
            "qc_language": self.language.passed,
            "qc_length_delta": self.length.value,
            "qc_selflabel": self.self_label.passed,
            "qc_topic_coverage": self.topic_coverage.value,
            "qc_near_dup": None
            if self.near_duplicate is None
            else self.near_duplicate.value,
            "qc_refusal": self.refusal.passed,
            "qc_passed": self.passed,
        }


# =============================================================================
# G1 -- Parse
# =============================================================================


def check_parse(raw_text: str | None) -> GateResult:
    """G1: the raw generation envelope must be non-empty and split cleanly.

    This is the v0.3.1 empty-body bug: ``_parse_generated_text`` silently
    returns an empty body when the model does not emit a blank line after
    the title, and nothing downstream noticed until 84 such documents were
    caught and filtered post hoc (v0.3.1). Applying this gate *before* a
    document enters the corpus is the fix; retroactive filtering is not the
    correct place for it.

    Args:
        raw_text: The raw text returned by the generator backend, before
            parsing into title/body.
    """
    if raw_text is None or not raw_text.strip():
        return GateResult(
            Gate.PARSE,
            False,
            value=False,
            detail={"reason": "empty_text", "title": "", "body": ""},
        )
    title, body = _parse_generated_text(raw_text)
    return _parse_verdict(title, body)


def check_parse_fields(title: str | None, body: str | None) -> GateResult:
    """G1, given already-split ``title``/``body`` fields.

    Use this when the raw pre-parse envelope is unavailable -- notably, the
    published HuggingFace corpus only retains post-parse ``title``/``body``
    columns, not the raw LLM output. Semantics match :func:`check_parse`.
    """
    return _parse_verdict(title or "", body or "")


def _parse_verdict(title: str, body: str) -> GateResult:
    title_ok = bool(title.strip())
    body_ok = bool(body.strip())
    passed = title_ok and body_ok
    reason = None
    if not passed:
        reason = "empty_title" if not title_ok else "empty_body"
    return GateResult(
        Gate.PARSE,
        passed,
        value=passed,
        detail={"title": title, "body": body, "reason": reason},
    )


# =============================================================================
# G2 -- Language
# =============================================================================

# A small, high-frequency set of English function words. No language-detection
# dependency is declared for this project, so this gate is a deliberately
# simple heuristic rather than a statistical language model: function-word
# density is a strong, cheap discriminator between English and both other
# languages and non-linguistic text, and is stable even on very short
# ("micro", 10-25 word) documents.
_ENGLISH_FUNCTION_WORDS: frozenset[str] = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "at",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "his",
        "her",
        "their",
        "our",
        "your",
        "not",
        "no",
        "so",
        "than",
        "then",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "some",
        "most",
        "more",
        "less",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "into",
        "through",
        "about",
        "between",
        "among",
        "over",
        "under",
        "after",
        "before",
        "during",
        "without",
        "within",
        "against",
        "up",
        "down",
        "out",
        "off",
        "also",
        "only",
        "just",
        "there",
        "here",
        "such",
        "other",
        "same",
        "each",
        "both",
        "few",
        "many",
        "much",
        "most",
    ]
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def check_language(
    text: str,
    *,
    stopword_threshold: float = 0.12,
    ascii_ratio_threshold: float = 0.5,
) -> GateResult:
    """G2: the document must be (heuristically) English.

    Combines two signals so that both non-Latin scripts and non-English
    Latin-script text are caught:

    1. ASCII-letter ratio among alphabetic characters -- near zero for
       Chinese, Arabic, Cyrillic, etc.
    2. English function-word density among tokens -- low for Spanish,
       French, German, etc. even though they share the Latin alphabet.

    Args:
        text: Document body text.
        stopword_threshold: Minimum fraction of tokens that must be common
            English function words.
        ascii_ratio_threshold: Minimum fraction of alphabetic characters
            that must be ASCII letters.
    """
    if not text or not text.strip():
        return GateResult(
            Gate.LANGUAGE, False, value=0.0, detail={"reason": "empty_text"}
        )

    alpha_chars = [c for c in text if c.isalpha()]
    ascii_ratio = (
        sum(1 for c in alpha_chars if _ASCII_LETTER_RE.match(c)) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )

    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    stopword_ratio = (
        sum(1 for w in tokens if w in _ENGLISH_FUNCTION_WORDS) / len(tokens)
        if tokens
        else 0.0
    )

    passed = (
        ascii_ratio >= ascii_ratio_threshold and stopword_ratio >= stopword_threshold
    )
    return GateResult(
        Gate.LANGUAGE,
        passed,
        value=round(stopword_ratio, 4),
        detail={"ascii_ratio": round(ascii_ratio, 4), "token_count": len(tokens)},
    )


# =============================================================================
# G3 -- Length adherence
# =============================================================================


def check_length(
    word_count: int,
    target_word_range: tuple[int, int] | None,
    *,
    tolerance: float = 0.25,
    min_tolerance_words: int = 5,
) -> GateResult:
    """G3: how far ``word_count`` falls outside ``target_word_range``.

    Per §12/§13, the delta is *always* recorded (``value``), never collapsed
    to a boolean -- callers that only look at ``passed`` still get the exact
    signed overshoot/undershoot via ``value`` and ``qc_length_delta``.

    Documents whose target range brackets very few words (e.g. ``micro``,
    10-25 words) are handled correctly here because tolerance scales with
    the range's own width, not a fixed word count -- a document with a
    10-25 word target and 27 actual words is a 2-word overshoot, not a
    "too short" false positive from some fixed global tolerance.

    Args:
        word_count: Actual word count of the generated body.
        target_word_range: ``(min, max)`` words requested, or ``None``/
            ``(0, 0)`` if no target was set (always passes; nothing to
            compare against).
        tolerance: Fraction of the target range's width allowed as slack
            beyond the bounds before the gate fails.
        min_tolerance_words: Floor on the slack, so narrow ranges are not
            unreasonably strict.
    """
    if not target_word_range or tuple(target_word_range) == (0, 0):
        return GateResult(
            Gate.LENGTH, True, value=0, detail={"reason": "no_target_range"}
        )

    lo, hi = target_word_range
    if word_count < lo:
        delta = word_count - lo
    elif word_count > hi:
        delta = word_count - hi
    else:
        delta = 0

    span = max(hi - lo, 1)
    allowed = max(min_tolerance_words, round(tolerance * span))
    passed = abs(delta) <= allowed
    return GateResult(
        Gate.LENGTH,
        passed,
        value=delta,
        detail={"target_range": [lo, hi], "allowed_slack": allowed},
    )


# =============================================================================
# G4 -- Self-labeling leakage (delegates to shelf.sampler.leakage)
# =============================================================================


def check_self_label(text: str, labels: Iterable[str] = ()) -> GateResult:
    """G4: scan for self-labeling / taxonomy-leakage phrases.

    Thin wrapper around :func:`shelf.sampler.leakage.scan_document`, which
    already implements this gate and is exercised by its own test suite.
    Not reimplemented here.

    Args:
        text: Document body text.
        labels: Taxonomy label strings that must not appear verbatim (e.g.
            the 14 LCGFT categories). Optional -- omit to check only for
            self-labeling phrasing, not verbatim label leakage.
    """
    report: LeakageReport = scan_document(text, labels)
    return GateResult(
        Gate.SELF_LABEL, report.is_clean, value=report.is_clean, detail=report.as_dict()
    )


# =============================================================================
# G5 -- Topic coverage
# =============================================================================

# Requested topics/geographic labels are short phrases ("Climate change",
# "Labor"). A strict verbatim-phrase check passes almost always (measured:
# 87.2% of topic terms already appear verbatim in the v0.3.1 corpus) and so
# is not a meaningful gate on its own. Instead this scores, per topic, the
# fraction of the topic's own significant words that appear (with simple
# suffix tolerance for plurals/derivations, e.g. "govern" matches
# "governance"), then reports the *document-level fraction of topics
# covered* as the recorded metric -- not a single pass/fail bit.
_TOPIC_STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "of",
        "the",
        "in",
        "on",
        "for",
        "to",
        "with",
        "as",
        "by",
        "at",
        "from",
        "or",
    ]
)


def _topic_significant_words(topic: str, *, min_length: int = 3) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(topic)]
    return [w for w in words if len(w) >= min_length and w not in _TOPIC_STOPWORDS]


def _topic_word_fraction(text_lower: str, topic: str) -> float:
    words = _topic_significant_words(topic)
    if not words:
        return 1.0  # nothing meaningful to require
    hits = sum(
        1 for w in words if re.search(rf"(?<!\w){re.escape(w)}\w*(?!\w)", text_lower)
    )
    return hits / len(words)


def check_topic_coverage(
    text: str,
    topics: Sequence[str],
    *,
    per_topic_threshold: float = 0.6,
    coverage_threshold: float = 0.0,
) -> GateResult:
    """G5: what fraction of the requested topics appear *verbatim* in the text.

    **This gate is informational, not gating** (``coverage_threshold=0.0`` by
    default), because its natural reading is inverted.

    A high verbatim rate is not a quality signal -- it is the string-matching
    shortcut. Measured on the published v0.3.1 corpus, **87.2% of topic terms
    appear verbatim** and 78.5% of documents contain every one of theirs, which
    is precisely why `topic_classification` is largely solvable by string
    matching (§11.1). ``GENERATION_INSTRUCTIONS`` asks models to *show, not
    tell*, so a document that addresses "Religion" and "Aesthetics" through a
    parish organ rather than by naming them is doing the right thing.

    Gating on it inverts corpus quality: on the v0.4 run, Claude Opus 5 scored
    34.9% and GPT-5.6 Sol 75.1%, and treating that as a pass rate would have
    rejected 78% of the better generator's output.

    Raise ``coverage_threshold`` deliberately if you want it to gate; the
    fraction is always recorded in ``value`` either way.

    Args:
        text: Document body text.
        topics: Requested topic labels for this document (e.g.
            ``doc.topics``).
        per_topic_threshold: Fraction of a single topic's significant words
            that must appear for that topic to count as "covered".
        coverage_threshold: Fraction of topics that must be covered for the
            document to pass. Defaults to 0.0 -- informational only.
    """
    if not topics:
        return GateResult(
            Gate.TOPIC_COVERAGE, True, value=1.0, detail={"reason": "no_topics"}
        )

    text_lower = text.lower()
    per_topic = {topic: _topic_word_fraction(text_lower, topic) for topic in topics}
    covered = sum(1 for frac in per_topic.values() if frac >= per_topic_threshold)
    coverage = covered / len(topics)
    passed = coverage >= coverage_threshold
    return GateResult(
        Gate.TOPIC_COVERAGE,
        passed,
        value=round(coverage, 4),
        detail={"per_topic": {k: round(v, 4) for k, v in per_topic.items()}},
    )


# =============================================================================
# G7 -- Refusal / boilerplate
# =============================================================================

_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bi can(?:not|'t) (?:help|assist|comply|fulfill|provide)\b",
        r"\bi'?m sorry,? but\b",
        r"\bas an ai\b",
        r"\bas an ai language model\b",
        r"\bas a large language model\b",
        r"\bi do not have the ability to\b",
        r"\bi am not able to\b",
        r"\bi won'?t be able to\b",
        r"\bunable to (?:fulfill|complete|assist with) (?:this|that) request\b",
        r"\bi must decline\b",
        r"\bagainst (?:my|the) content policy\b",
        r"\bi'?m (?:just )?an ai\b",
    )
)

_TRUNCATION_MARKERS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\[(?:content )?truncated\]",
        r"<\|[^|]*\|>",  # stray special/control tokens
        r"\.\.\.\s*$",
    )
)

_SENTENCE_END_CHARS = ".!?\"'”’)]}:;,"

# Markdown/structural decoration that legitimately ends a document without
# ending a *sentence*: a table cell (`|`), emphasis markers (`*`, `_`), a
# code fence (`` ` ``), a blockquote/heading marker (`>`, `#`), or a blank
# fill-in-the-blank field (`_`). Stripped before judging whether the text
# "ends on a sentence", since SHELF documents legitimately end mid-table,
# mid-list, or on a signature-block placeholder for several LCGFT forms
# (Legislative materials, Calendars, Diagrams, Greeting cards, etc.).
_STRUCTURAL_TRAILING_CHARS = "*_`|>#~ \t"


def _looks_truncated(text: str) -> tuple[bool, bool]:
    """Split truncation detection into a high-precision and a soft signal.

    Returns:
        ``(hard_truncated, missing_terminal_punctuation)``.

        ``hard_truncated`` is based only on explicit artifacts (an empty
        body, an explicit "[truncated]" marker, a stray control token, or a
        trailing ellipsis) and is what actually fails the gate.

        ``missing_terminal_punctuation`` -- "a body long enough to have
        real sentences should end on one" -- sounds reasonable but is not
        reliable across SHELF's genre diversity: validated against the
        published corpus, this signal alone flags ~14% of documents, and
        manual inspection shows the overwhelming majority are legitimate
        non-prose forms (tables, diagrams, signature blocks, fill-in
        fields, list/caption styles) rather than truncated output, while
        real refusal/truncation boilerplate is caught at <0.02% by the
        patterns above. It is recorded for visibility but does not gate.
    """
    stripped = text.rstrip()
    if not stripped:
        return True, True
    hard_truncated = any(pattern.search(stripped) for pattern in _TRUNCATION_MARKERS)
    core = stripped.rstrip(_STRUCTURAL_TRAILING_CHARS)
    missing_punctuation = bool(core) and (
        len(core.split()) >= 15 and core[-1] not in _SENTENCE_END_CHARS
    )
    return hard_truncated, missing_punctuation


def check_refusal(text: str) -> GateResult:
    """G7: refusal boilerplate ("I can't help with...", "As an AI...") and
    truncation artifacts (explicit truncation markers, stray control
    tokens, a trailing ellipsis).

    A document that simply lacks terminal sentence punctuation is *not*
    failed on that basis alone -- see :func:`_looks_truncated` -- but is
    reported in ``detail['missing_terminal_punctuation']`` for visibility.
    """
    if not text or not text.strip():
        return GateResult(
            Gate.REFUSAL, False, value=True, detail={"reason": "empty_text"}
        )
    matched = [p.pattern for p in _REFUSAL_PATTERNS if p.search(text)]
    hard_truncated, missing_punctuation = _looks_truncated(text)
    flagged = bool(matched) or hard_truncated
    return GateResult(
        Gate.REFUSAL,
        not flagged,
        value=flagged,
        detail={
            "matched_patterns": matched,
            "truncated": hard_truncated,
            "missing_terminal_punctuation": missing_punctuation,
        },
    )


# =============================================================================
# Convenience: run G1-G5 and G7 together
# =============================================================================


def run_gates(
    doc_id: str,
    *,
    raw_text: str | None = None,
    title: str = "",
    body: str = "",
    word_count: int | None = None,
    target_word_range: tuple[int, int] | None = None,
    topics: Sequence[str] = (),
    taxonomy_labels: Iterable[str] = (),
    length_tolerance: float = 0.25,
    topic_coverage_threshold: float = 0.0,  # informational; see check_topic_coverage
) -> QCResult:
    """Run every per-document gate (G1-G5, G7) and collect the results.

    G6 is not included: near-duplicate detection needs corpus-wide state
    (see :mod:`shelf.qc.dedup`). Attach it afterwards with
    ``result.with_near_duplicate(...)``.

    Args:
        doc_id: Identifier for the document (for traceability only).
        raw_text: Raw pre-parse generator output. When given, G1 parses it
            and the resulting title/body feed the remaining gates. Mutually
            exclusive in effect with ``title``/``body`` (those are ignored
            if ``raw_text`` is provided).
        title: Already-parsed title, used when ``raw_text`` is unavailable
            (e.g. scoring the published corpus, which only retains
            post-parse fields).
        body: Already-parsed body, same caveat as ``title``.
        word_count: Body word count; computed from ``body`` if omitted.
        target_word_range: ``(min, max)`` words requested for G3.
        topics: Requested topic labels for G5.
        taxonomy_labels: Taxonomy label strings G4 must not find verbatim.
        length_tolerance: Passed through to :func:`check_length`.
        topic_coverage_threshold: Passed through to
            :func:`check_topic_coverage` as ``coverage_threshold``.
    """
    if raw_text is not None:
        parse = check_parse(raw_text)
    else:
        parse = check_parse_fields(title, body)

    resolved_body = str(parse.detail.get("body") or "")
    wc = word_count if word_count is not None else len(resolved_body.split())

    return QCResult(
        doc_id=doc_id,
        parse=parse,
        language=check_language(resolved_body),
        length=check_length(wc, target_word_range, tolerance=length_tolerance),
        self_label=check_self_label(resolved_body, taxonomy_labels),
        topic_coverage=check_topic_coverage(
            resolved_body, topics, coverage_threshold=topic_coverage_threshold
        ),
        refusal=check_refusal(resolved_body),
    )

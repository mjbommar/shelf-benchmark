"""Taxonomy label-leakage detection and sanitization.

SHELF's labels are only non-trivial because generated documents do not announce
their own class. `build_generation_prompt` is built around this: it feeds the
model *semantic descriptions* rather than taxonomy names, and
``GENERATION_INSTRUCTIONS`` forbids self-labeling openings and classification
headers.

That invariant is fragile in two places, and this module guards both.

**Prompt-side.** Descriptions harvested from LC hierarchies frequently name the
taxonomy they belong to. Measured on the enriched LCGFT export, **25.7% of
descriptions name an LCGFT category verbatim** -- "Maps" is described as *"a kind
of Cartographic materials, Informational works, Visual works"*, and
`lcgft_category` is itself a 14-way prediction target. Hierarchy-derived
descriptions leak at 38.6%; genuine LC scope notes leak at only 7.2%. Feeding
the raw text into a prompt would steer the generated document toward the very
words the benchmark asks a model to predict.

**Document-side.** Even with a clean prompt, a generator may self-label anyway.
This is QC gate G4 in the data plan.

Both checks share one implementation because they are the same question asked of
different text.

Example:
    from shelf.sampler.leakage import find_leaked_labels, sanitize_description

    find_leaked_labels("a kind of Cartographic materials", LCGFT_CATEGORIES)
    # ('Cartographic materials',)

    sanitize_description("a kind of Cartographic materials; includes Atlases",
                         LCGFT_CATEGORIES)
    # 'includes Atlases'
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "LeakageReport",
    "find_labels_in_labelling_context",
    "find_leaked_labels",
    "has_self_labeling",
    "sanitize_description",
    "scan_document",
]

# Phrases that announce a document's own type or field. The generation prompt
# forbids exactly these, so their presence in output means the instruction was
# ignored, not that the topic happens to be discussed.
_SELF_LABEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Taxonomy CODES are unambiguous, so they are matched anywhere in the text,
    # not just at line start. Measured on the published v0.3.1 corpus, 202
    # documents (0.47%) carry one of these, and the line-anchored version missed
    # most of them: they appear inside parentheses ("**Disciplina (LCC: Language
    # and Literature)**") and inside YAML blocks ("Tipo (LCGFT): Field
    # recordings"), never at the start of a line.
    # The optional bracket handles the real corpus form "Tipo (LCGFT): ...",
    # where the code is parenthesised and the colon falls outside.
    re.compile(r"\b(lcgft|lcsh|lcdgt|lcc)\s*[)\]]?\s*:", re.I),
    # Generic words like "category" and "genre" occur in ordinary prose, so
    # those stay anchored to a line start where they read as a header.
    re.compile(
        r"^\s*(document\s+type|subject\s+area|category|genre)\s*:",
        re.I | re.M,
    ),
    re.compile(r"\bin the field of\b", re.I),
    re.compile(
        r"\bthis (?:satire|lecture|poem|essay|report|guide|article|document)\b[^.]{0,40}\b(?:explores|covers|examines|presents)\b",
        re.I,
    ),
    re.compile(
        r"^\s*(?:in|within)\s+(?:political science|medicine|law|philosophy|economics|sociology)\s*,",
        re.I | re.M,
    ),
)

# Connective fragments left behind once a leaked label is removed.
_DANGLING = re.compile(
    r"^\s*(?:a kind of|kind of|type of|used for|includes?)\s*[,;:]?\s*|"
    r"\s*[,;]\s*(?=[,;])|^\s*[,;]\s*|\s*[,;]\s*$",
    re.I,
)


@dataclass(frozen=True)
class LeakageReport:
    """What a leakage scan found in one piece of text."""

    leaked_labels: tuple[str, ...] = ()
    self_labeling: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """Whether the text is free of both leakage kinds."""
        return not self.leaked_labels and not self.self_labeling

    def as_dict(self) -> dict[str, list[str] | bool]:
        """Serialize for storage as a QC column."""
        return {
            "leaked_labels": list(self.leaked_labels),
            "self_labeling": list(self.self_labeling),
            "is_clean": self.is_clean,
        }


def _label_pattern(label: str) -> re.Pattern[str]:
    """Word-boundary pattern for a label, case-insensitive.

    Boundaries matter: "Music" as a category must not fire on "musicology",
    and must fire on "Music." at the end of a clause.
    """
    return re.compile(rf"(?<!\w){re.escape(label)}(?!\w)", re.I)


def find_leaked_labels(
    text: str,
    labels: Iterable[str],
    *,
    min_length: int = 4,
) -> tuple[str, ...]:
    """Return the taxonomy labels that appear verbatim in ``text``.

    Args:
        text: Text to scan.
        labels: Taxonomy label strings that must not appear.
        min_length: Labels shorter than this are ignored. Single letters and
            two-character codes ("G", "KF") occur constantly in ordinary prose
            and would make every scan positive.

    Returns:
        Matching labels, longest first so that a match on "Informational works"
        is reported ahead of a match on "works".
    """
    if not text:
        return ()
    found = [
        label
        for label in labels
        if label and len(label) >= min_length and _label_pattern(label).search(text)
    ]
    return tuple(sorted(set(found), key=lambda s: (-len(s), s)))


def has_self_labeling(text: str) -> tuple[str, ...]:
    """Return self-labeling phrases found in ``text`` (QC gate G4)."""
    if not text:
        return ()
    hits: list[str] = []
    for pattern in _SELF_LABEL_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(match.group(0).strip()[:60])
    return tuple(hits)


# A label only counts as leakage in a *document* when it appears in a labelling
# context. Several LCGFT categories -- "Music", "Literature", "Ephemera" -- are
# also ordinary English words, and a document about a parish music director will
# naturally contain "music". GENERATION_INSTRUCTIONS explicitly permits domain
# vocabulary and forbids only announcing the class.
#
# Measured: applying the bare-word check to generated documents flagged 15-20%
# of them, essentially all on ordinary uses of "Music" and "Literature". The
# context requirement takes that to the real rate.
_LABELLING_CONTEXT = (
    r"(?:^|[\n(\[])\s*(?:{label})\s*[)\]]?\s*[:\u2014-]"  # "Music:" / "(Music):"
    r"|(?:category|genre|form|class|type|subject)\s*[:=]\s*{label}"  # "Category: Music"
    r"|\((?:lcgft|lcc|lcsh)\s*[:=]?\s*{label}"  # "(LCGFT: Music"
)


def find_labels_in_labelling_context(
    text: str,
    labels: Iterable[str],
) -> tuple[str, ...]:
    """Return labels that appear as an explicit label, not as vocabulary.

    This is the check to use on *generated documents*.
    :func:`find_leaked_labels` -- a bare word-boundary match -- remains the
    right check for *descriptions* headed into a prompt, where any occurrence
    is unwanted.
    """
    if not text:
        return ()
    found = []
    for label in labels:
        if not label:
            continue
        pattern = _LABELLING_CONTEXT.format(label=re.escape(label))
        if re.search(pattern, text, re.I | re.M):
            found.append(label)
    return tuple(sorted(set(found), key=lambda s: (-len(s), s)))


def scan_document(
    text: str,
    labels: Iterable[str] = (),
    *,
    require_labelling_context: bool = True,
) -> LeakageReport:
    """Scan generated text for leaked taxonomy labels and self-labeling.

    Args:
        text: The generated document.
        labels: Taxonomy labels that must not be *announced*.
        require_labelling_context: When True (the default, and correct for
            documents), a label counts only if it appears as a label rather
            than as ordinary vocabulary. Set False for prompt-bound text, where
            any occurrence matters.
    """
    leaked = (
        find_labels_in_labelling_context(text, labels)
        if require_labelling_context
        else find_leaked_labels(text, labels)
    )
    return LeakageReport(
        leaked_labels=leaked,
        self_labeling=has_self_labeling(text),
    )


def sanitize_description(
    description: str,
    labels: Sequence[str],
    *,
    min_length: int = 4,
) -> str:
    """Strip taxonomy label names out of a description.

    Intended for descriptions harvested from LC hierarchies before they are
    placed in a generation prompt. Removing the label names leaves the genuinely
    semantic remainder ("includes Aeronautical charts, Bottle-charts") which
    still conditions the generator without naming the prediction target.

    Longest labels are removed first so that stripping "works" does not damage
    "Informational works" before it is matched.

    Args:
        description: Raw description text.
        labels: Taxonomy labels to remove.
        min_length: Labels shorter than this are left alone.

    Returns:
        The sanitized description, or an empty string if nothing survives --
        an empty result is meaningful and the caller must fall back rather
        than use it.
    """
    if not description:
        return ""

    targets: list[str] = [
        label for label in set(labels) if label and len(label) >= min_length
    ]
    targets.sort(key=len, reverse=True)

    cleaned = description
    for label in targets:
        cleaned = _label_pattern(label).sub("", cleaned)

    # Collapse the punctuation debris left where labels used to be.
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _DANGLING.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,;.])", r"\1", cleaned)
        cleaned = cleaned.strip(" ,;:.-")

    return cleaned.strip()

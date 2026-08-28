"""Enriched Library of Congress descriptions for generation prompts.

`scripts/enrich_taxonomies.py` fetches real LC scope notes from id.loc.gov and
writes them to ``data/taxonomies/enriched/``. That closed a hard blocker: LCC
subclass description coverage went from 16/100 to 96/100, and LCGFT forms from
17/554 to 412/554, which is what makes §4.1's pool expansion and Phase 2's
subclass tier possible at all.

The enriched text cannot be used raw, for two reasons this module handles.

**Most of it is derived, not verbatim.** Only ~19% are genuine LC scope notes;
the rest are templated from LC broader/narrower hierarchies. Those templates
*name the taxonomy*: measured on the LCGFT export, **25.7% of descriptions
contain an LCGFT category verbatim** -- "Maps" is described as *"a kind of
Cartographic materials, Informational works, Visual works"* -- and
``lcgft_category`` is itself a 14-way prediction target. Every description is
therefore passed through :func:`shelf.sampler.leakage.sanitize_description`
before it can reach a prompt.

**The hand-written descriptions are better.** The small curated dicts in
``generator.py`` were written specifically to describe a form *semantically*
without naming it. Where one exists it wins; LC text is the fallback that
extends coverage to labels nobody hand-wrote.

Resolution order, highest quality first:

1. curated hand-written description (``LCGFT_FORM_DESCRIPTIONS`` etc.)
2. verbatim LC scope note, sanitized
3. LC hierarchy/variant-derived description, sanitized
4. ``None`` -- the caller keeps its own existing fallback

Loading is **opt-in**. The default generation path does not consult this module,
so v0.3.1 prompts remain byte-for-byte reproducible.

**What sanitizing does and does not remove.** Measured over the 461 enriched
forms: exact word-boundary occurrences of a label or category are eliminated
(**0 remaining**), but **20.2%** of descriptions still contain a morphological
relative of their own label -- "Tourist maps" is described as *"maps designed for
tourists"*. That is deliberate. ``GENERATION_INSTRUCTIONS`` already permits
domain vocabulary ("the court ruled" in a legal document); what it forbids is
*announcing* the class. Stemming these away would strip the legitimate
vocabulary that makes conditioning work at all.

**Prefer ``verbatim_only=True`` for prompts.** Derived descriptions vary from
excellent to useless -- "Legal briefs" gets real LC prose, "Abstracts" collapses
to "Derivative works" and "Maps" to a bare list of narrower terms. Verbatim-only
keeps 190 forms, 382 topics, and 18 LCC subclasses; the full set keeps 461,
1,983, and 96. Coverage versus conditioning quality is a real trade and the
caller should make it explicitly.

Example:
    from shelf.sampler.enriched import EnrichedDescriptions

    enriched = EnrichedDescriptions.load()
    enriched.for_form("Abstracts")        # sanitized LC text, or None
    enriched.for_lcc_subclass("QA")       # 'mathematics, ...'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shelf.sampler.leakage import find_leaked_labels, sanitize_description

__all__ = [
    "DEFAULT_ENRICHED_DIR",
    "EnrichedDescriptions",
    "EnrichedEntry",
]

DEFAULT_ENRICHED_DIR = Path("data/taxonomies/enriched")

# Files consulted, in the order they are merged. Curated exports come last so
# they overwrite the bulk frequency tables for labels present in both -- the
# curated set is the vocabulary v0.3.1 actually used.
_FORM_FILES = ("lcgft.json", "curated_lcgft_forms.json")
_TOPIC_FILES = ("lcsh_topical_top2000.json", "curated_lcsh_topics.json")
_GEO_FILES = ("lcsh_geo_top500.json", "curated_geographic.json")
_LCC_FILES = ("lcc_subclass_top100.json",)


@dataclass(frozen=True)
class EnrichedEntry:
    """One label's enriched description and its provenance."""

    label: str
    description: str
    source: str
    verbatim: bool
    uri: str | None = None
    sanitized: bool = False
    removed_labels: tuple[str, ...] = ()

    @property
    def is_derived(self) -> bool:
        """Whether the text was templated rather than quoted from LC."""
        return not self.verbatim


def _load_labels(path: Path) -> list[dict[str, Any]]:
    """Read a taxonomy export, tolerating a missing file."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    labels = payload.get("labels") if isinstance(payload, dict) else payload
    return labels if isinstance(labels, list) else []


def _as_bool(value: Any) -> bool:
    """Coerce the JSON export's mixed bool/string encoding."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


@dataclass
class EnrichedDescriptions:
    """Sanitized LC descriptions keyed by label, per taxonomy."""

    forms: dict[str, EnrichedEntry] = field(default_factory=dict)
    topics: dict[str, EnrichedEntry] = field(default_factory=dict)
    geographic: dict[str, EnrichedEntry] = field(default_factory=dict)
    lcc_subclasses: dict[str, EnrichedEntry] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        enriched_dir: Path | str = DEFAULT_ENRICHED_DIR,
        forbidden_labels: tuple[str, ...] = (),
        verbatim_only: bool = False,
    ) -> EnrichedDescriptions:
        """Load and sanitize every enriched export.

        Args:
            enriched_dir: Directory written by ``scripts/enrich_taxonomies.py``.
            forbidden_labels: Labels that must never appear in a description --
                typically the LCGFT categories and any other prediction target.
                Defaults to the 14 LCGFT categories. The label's own name is
                always stripped in addition to these.
            verbatim_only: Keep only genuine LC scope notes and discard
                hierarchy-derived text. Derived descriptions vary wildly in
                usefulness -- "Legal briefs" gets real LC prose while
                "Abstracts" collapses to "Derivative works" -- so a run that
                needs dependable conditioning should set this and accept the
                lower coverage.

        Returns:
            A populated instance. Missing files yield empty maps rather than an
            error, so a repo without the enrichment step still works.
        """
        from shelf.evaluate.registry import LCGFT_CATEGORIES

        forbidden = forbidden_labels or tuple(LCGFT_CATEGORIES)
        base = Path(enriched_dir)

        return cls(
            forms=cls._build(base, _FORM_FILES, forbidden, verbatim_only=verbatim_only),
            topics=cls._build(
                base, _TOPIC_FILES, forbidden, verbatim_only=verbatim_only
            ),
            geographic=cls._build(
                base, _GEO_FILES, forbidden, verbatim_only=verbatim_only
            ),
            lcc_subclasses=cls._build(
                base, _LCC_FILES, forbidden, key="id", verbatim_only=verbatim_only
            ),
        )

    @staticmethod
    def _build(
        base: Path,
        filenames: tuple[str, ...],
        forbidden: tuple[str, ...],
        key: str = "label",
        verbatim_only: bool = False,
    ) -> dict[str, EnrichedEntry]:
        """Merge files into one sanitized map."""
        merged: dict[str, EnrichedEntry] = {}
        for name in filenames:
            for row in _load_labels(base / name):
                label = str(row.get(key) or row.get("label") or "").strip()
                raw = str(row.get("description") or "").strip()
                if not label or not raw:
                    continue
                if verbatim_only and not _as_bool(row.get("description_verbatim")):
                    continue

                # Strip the label's OWN name too. `form_classification` is a
                # 133-way prediction target, so a description of "Prayers" that
                # reads "Rosaries (Prayer books), Litanies" leaks the answer
                # just as surely as naming a category does.
                targets = (*forbidden, label)
                leaked = find_leaked_labels(raw, targets)
                cleaned = sanitize_description(raw, targets) if leaked else raw
                if not cleaned:
                    # Sanitizing consumed the whole description; better to have
                    # nothing and fall back than to prompt with an empty string.
                    continue

                merged[label] = EnrichedEntry(
                    label=label,
                    description=cleaned,
                    source=str(row.get("description_source") or "unknown"),
                    verbatim=_as_bool(row.get("description_verbatim")),
                    uri=(str(row["uri"]) if row.get("uri") else None),
                    sanitized=bool(leaked),
                    removed_labels=leaked,
                )
        return merged

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def for_form(self, form: str) -> str | None:
        """Sanitized description for an LCGFT form, if one exists."""
        entry = self.forms.get(form)
        return entry.description if entry else None

    def for_topic(self, topic: str) -> str | None:
        """Sanitized description for an LCSH topic, if one exists."""
        entry = self.topics.get(topic)
        return entry.description if entry else None

    def for_geographic(self, place: str) -> str | None:
        """Sanitized description for a geographic heading, if one exists."""
        entry = self.geographic.get(place)
        return entry.description if entry else None

    def for_lcc_subclass(self, code: str) -> str | None:
        """Sanitized description for an LCC subclass code, if one exists."""
        entry = self.lcc_subclasses.get(code)
        return entry.description if entry else None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def coverage(self) -> dict[str, dict[str, int]]:
        """Per-taxonomy counts of entries, verbatim notes, and sanitizations.

        Publishing this alongside a corpus makes the quality of the conditioning
        text auditable: a run whose descriptions are 80% templated is a
        different experiment from one whose descriptions are LC prose.
        """
        report: dict[str, dict[str, int]] = {}
        for name, mapping in (
            ("forms", self.forms),
            ("topics", self.topics),
            ("geographic", self.geographic),
            ("lcc_subclasses", self.lcc_subclasses),
        ):
            entries = list(mapping.values())
            report[name] = {
                "total": len(entries),
                "verbatim_scope_notes": sum(1 for e in entries if e.verbatim),
                "derived": sum(1 for e in entries if e.is_derived),
                "sanitized": sum(1 for e in entries if e.sanitized),
            }
        return report

    def audit_leakage(self, forbidden_labels: tuple[str, ...]) -> dict[str, list[str]]:
        """Return any label whose stored description still leaks.

        This should always be empty. It exists so the invariant is checkable
        rather than assumed, since a leak here contaminates generated documents
        with the words a model is asked to predict.
        """
        offenders: dict[str, list[str]] = {}
        for name, mapping in (
            ("forms", self.forms),
            ("topics", self.topics),
            ("geographic", self.geographic),
            ("lcc_subclasses", self.lcc_subclasses),
        ):
            bad = [
                entry.label
                for entry in mapping.values()
                if find_leaked_labels(entry.description, forbidden_labels)
            ]
            if bad:
                offenders[name] = bad
        return offenders

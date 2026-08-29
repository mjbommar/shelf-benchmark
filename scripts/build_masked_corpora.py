"""Build masked and sham-masked copies of a corpus, for the leakage ablation.

The question is not whether label leakage inflates scores -- we already know
SHELF carries about three times the natural verbatim-label rate. The question
is whether it changes *which model wins*.

Answering that needs two conditions, not one:

``masked``
    Remove the label's own words from the document.

``sham``
    Remove an equal number of randomly chosen tokens from the same document.

Without the sham arm a score drop cannot be attributed to leakage, because
masking also shortens and damages the text. Sham removes the same amount of
material while carrying no label signal, so the difference between the two is
the part attributable to the label.

**The variant rule is frozen** (docs/PREREGISTRATION.md §4) and deliberately
modest. For a label L we remove, case-insensitively at word boundaries: L
itself; L with a trailing "s"; every whitespace-separated token of L that is
at least five characters; and each such token with a trailing "s". No
stemming, no lemmatisation. Calling this "morphological variants" would
overstate it.

Masking is applied identically to every corpus. Masking only the synthetic
one would confound the intervention with the corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
MIN_TOKEN_LEN = 5
MASK = ""  # deletion, not a placeholder token


def variants(label: str) -> list[str]:
    """The frozen variant list for one label. Order: longest first."""
    if not label:
        return []
    out: set[str] = set()
    lab = label.strip()

    def _add(w: str) -> None:
        if not w:
            return
        out.add(w)
        if not w.lower().endswith("s"):  # avoid "Workss"
            out.add(w + "s")

    _add(lab)
    for tok in re.split(r"[\s,/&()-]+", lab):
        tok = tok.strip()
        if len(tok) >= MIN_TOKEN_LEN:
            _add(tok)
    return sorted(out, key=len, reverse=True)


def compile_patterns(labels: list[str]) -> list[re.Pattern[str]]:
    pats = []
    seen: set[str] = set()
    for lab in labels:
        for v in variants(lab):
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            pats.append(re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE))
    return pats


def mask_text(text: str, pats: list[re.Pattern[str]]) -> tuple[str, int]:
    """Remove label words. Returns (masked_text, tokens_removed)."""
    before = len(text.split())
    out = text
    for p in pats:
        out = p.sub(MASK, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out, max(0, before - len(out.split()))


def sham_text(text: str, n_remove: int, rng: random.Random) -> str:
    """Remove n_remove randomly chosen tokens, matched to the masked count."""
    toks = text.split()
    if n_remove <= 0 or n_remove >= len(toks):
        return text
    drop = set(rng.sample(range(len(toks)), n_remove))
    return " ".join(t for i, t in enumerate(toks) if i not in drop)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-masked", required=True)
    ap.add_argument("--out-sham", required=True)
    ap.add_argument(
        "--label-fields",
        default="lcc_name,topics,lcgft_form,lcgft_category",
        help="Fields whose values are removed from the text.",
    )
    ap.add_argument(
        "--lcc-names",
        default="",
        help=(
            "JSON map of lcc_code -> lcc_name. Required for corpora that carry "
            "the code but not the name: LCSHBench has only lcc_code, so without "
            "this the masker finds no label text and removes nothing, which "
            "would silently make its masked arm identical to its unmasked one."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import polars as pl

    src = Path(args.source)
    om, osh = Path(args.out_masked), Path(args.out_sham)
    om.mkdir(parents=True, exist_ok=True)
    osh.mkdir(parents=True, exist_ok=True)
    fields = [f for f in args.label_fields.split(",") if f]
    lcc_names: dict[str, str] = {}
    if args.lcc_names:
        lcc_names = json.loads(Path(args.lcc_names).read_text())
    rng = random.Random(args.seed)

    stats = {"removed_tokens": 0, "docs": 0, "docs_changed": 0}
    for split in SPLITS:
        f = src / f"{split}.parquet"
        if not f.exists():
            continue
        df = pl.read_parquet(f)
        rows = df.to_dicts()
        masked_rows, sham_rows = [], []
        for r in rows:
            text = r.get("text") or r.get("body") or ""
            labels: list[str] = []
            # Derive the class name when the corpus carries only the code.
            if lcc_names and not r.get("lcc_name"):
                code = r.get("lcc_code")
                if code and code in lcc_names:
                    labels.append(lcc_names[code])
            for fld in fields:
                v = r.get(fld)
                if v is None:
                    continue
                labels.extend(str(x) for x in (v if isinstance(v, list) else [v]) if x)
            pats = compile_patterns(labels)
            mt, n_removed = mask_text(text, pats)
            st = sham_text(text, n_removed, rng)
            stats["docs"] += 1
            stats["removed_tokens"] += n_removed
            stats["docs_changed"] += 1 if n_removed else 0
            masked_rows.append({**r, "text": mt, "body": mt})
            sham_rows.append({**r, "text": st, "body": st})
        pl.DataFrame(masked_rows, infer_schema_length=None).write_parquet(
            om / f"{split}.parquet"
        )
        pl.DataFrame(sham_rows, infer_schema_length=None).write_parquet(
            osh / f"{split}.parquet"
        )
        logger.info(f"  {split:<11} {len(rows):>6,} rows")

    d = stats["docs"] or 1
    logger.info(
        f"\n{src.name}: {stats['docs']:,} docs, "
        f"{stats['docs_changed'] / d * 100:.1f}% changed, "
        f"{stats['removed_tokens'] / d:.1f} tokens removed per doc on average"
    )
    logger.info(f"  masked -> {om}\n  sham   -> {osh}")
    (om / "_masking_stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

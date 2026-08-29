"""Generate Croissant 1.0 metadata for the SHELF dataset.

NeurIPS Datasets and Benchmarks requires machine-readable Croissant
metadata. The copy Hugging Face generates automatically is a stub: no
``conformsTo``, no ``distribution``, and no ``recordSet``, so it does not
describe the fields a reader would consume.

This writes a complete record: one FileSet per config, one RecordSet per
config with typed fields, and the provenance a reader needs to know which
slice they are holding.

    python scripts/build_croissant.py --output data/hf_release/croissant.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = "mjbommar/SHELF"
BASE = f"https://huggingface.co/datasets/{REPO}"

# config -> (human description, caveat that a consumer must not miss)
CONFIGS: dict[str, tuple[str, str]] = {
    "all": (
        "Every synthetic SHELF document pooled into one corpus (62,899).",
        "Not generator balanced: the largest generator is 47.7% of the corpus. "
        "Use v0_4_core when generator balance matters.",
    ),
    "default": (
        "The v0.3.1 corpus (42,532 documents).",
        "Split at document level, not spec level, so near-duplicate "
        "realisations of one specification may straddle splits.",
    ),
    "v0_4_core": (
        "Generator-balanced v0.4 corpus (18,345 documents, 15 generators).",
        "Largest generator share 9.24%. Split on spec_id.",
    ),
    "v0_4_supplement": (
        "Supplementary single-generator v0.4 documents (1,043).",
        "Carries no LCC subclass label: an intended subclass tier was "
        "generated from parent-class descriptions and is not shipped.",
    ),
    "v0_4_minimal_pairs": (
        "Minimal pairs differing on one facet (687 documents).",
        "Paired by pair_id; do not split a pair across folds.",
    ),
    "v0_4_holdout": (
        "Held-out v0.4 documents (292).",
        "Reserved for contamination checks.",
    ),
    "transfer_lcshbench": (
        "English LCSHBench records: real catalogue records from Harvard, "
        "Columbia and Princeton with real LCC classes (4,924).",
        "NATURAL text, not synthetic, and catalogue metadata rather than "
        "running prose (median 596 characters). A second transfer control. "
        "Do not pool with transfer_gutenberg; report them separately.",
    ),
    "transfer_gutenberg": (
        "Project Gutenberg passages, human written and human catalogued (3,016).",
        "NATURAL text, not synthetic. This is the transfer control. Never "
        "pool it into a training corpus: SHELF-to-Gutenberg transfer is the "
        "measurement it exists to support.",
    ),
}

# Field name -> (croissant dataType, description)
FIELDS: dict[str, tuple[str, str]] = {
    "id": ("sc:Text", "Stable document identifier."),
    "title": ("sc:Text", "Document title."),
    "body": ("sc:Text", "Document text, excluding the title."),
    "text": ("sc:Text", "Body text. Excludes the title to avoid label leakage."),
    "lcc_code": (
        "sc:Text",
        "Library of Congress Classification class, one of 21 (A-Z).",
    ),
    "lcc_name": ("sc:Text", "Human-readable name of the LCC class."),
    "lcc_uri": ("sc:Text", "URI of the LCC class in the LC linked data service."),
    "lcgft_category": ("sc:Text", "Genre/Form category, one of 14."),
    "lcgft_form": ("sc:Text", "Genre/Form term, one of 133."),
    "topics": ("sc:Text", "Topic labels drawn from LCSH."),
    "geographic": ("sc:Text", "Geographic region label."),
    "audience": ("sc:Text", "Intended audience (LCDGT-derived)."),
    "register": ("sc:Text", "Writing register."),
    "model": ("sc:Text", "Generating model. Provider routing prefixes normalised."),
    "word_count": ("sc:Integer", "Word count of the body."),
    "source_config": ("sc:Text", "Config this row came from (present in `all`)."),
    "source_version": ("sc:Text", "Dataset version this row came from."),
    "spec_id": ("sc:Text", "Content hash of the document specification (v0.4)."),
    "split": ("sc:Text", "Split assignment: train, validation, or test."),
}


def record_set(config: str, description: str, caveat: str) -> dict[str, Any]:
    fields = [
        {
            "@type": "cr:Field",
            "@id": f"{config}/{name}",
            "name": name,
            "description": desc,
            "dataType": dtype,
            "source": {
                "fileSet": {"@id": f"{config}-files"},
                "extract": {"column": name},
            },
        }
        for name, (dtype, desc) in FIELDS.items()
    ]
    return {
        "@type": "cr:RecordSet",
        "@id": config,
        "name": config,
        "description": f"{description} {caveat}",
        "field": fields,
    }


def build() -> dict[str, Any]:
    distribution: list[dict[str, Any]] = [
        {
            "@type": "cr:FileObject",
            "@id": "repo",
            "name": "repo",
            "description": "The SHELF dataset repository on Hugging Face.",
            "contentUrl": BASE,
            "encodingFormat": "git+https",
            "sha256": "main",
        }
    ]
    record_sets: list[dict[str, Any]] = []
    for config, (desc, caveat) in CONFIGS.items():
        distribution.append(
            {
                "@type": "cr:FileSet",
                "@id": f"{config}-files",
                "name": f"{config}-files",
                "description": f"Parquet shards for the {config} config.",
                "containedIn": {"@id": "repo"},
                "encodingFormat": "application/x-parquet",
                "includes": f"{config}/*.parquet",
            }
        )
        record_sets.append(record_set(config, desc, caveat))

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataBiases": "cr:dataBiases",
            "dataCollection": "cr:dataCollection",
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "personalSensitiveInformation": "cr:personalSensitiveInformation",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            "containedIn": "cr:containedIn",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "rai": "http://mlcommons.org/croissant/RAI/",
            "samplingRate": "cr:samplingRate",
            "equivalentProperty": "cr:equivalentProperty",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "SHELF",
        "alternateName": ["Synthetic Harness for Evaluating LLM Fitness"],
        "description": (
            "SHELF is a factorial benchmark for bibliographic classification "
            "built on Library of Congress taxonomies (LCC, LCGFT, LCSH, "
            "LCDGT). Documents are model generated from content-addressed "
            "specifications, so generator and label are independent by "
            "construction and genre is orthogonal to subject. A natural-text "
            "Project Gutenberg slice is included as a transfer control. "
            "SHELF scores do not transfer to natural text in absolute terms: "
            "a lexical classifier scoring 0.887 in-domain scores 0.313 on "
            "Gutenberg, and the failure is symmetric."
        ),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": BASE,
        "citeAs": (
            "@misc{shelf2026, title={SHELF: A Factorial Benchmark for "
            "Bibliographic Classification}, author={Bommarito, Michael}, "
            "year={2026}, howpublished={\\url{" + BASE + "}}}"
        ),
        "version": "0.4.0",
        "datePublished": "2026-08-27",
        "keywords": [
            "bibliographic classification",
            "Library of Congress Classification",
            "text embeddings",
            "synthetic evaluation data",
            "taxonomic representation",
        ],
        "isLiveDataset": False,
        "distribution": distribution,
        "recordSet": record_sets,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/hf_release/croissant.json")
    args = ap.parse_args()

    doc = build()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    logger.info(f"Wrote {out}")
    logger.info(f"  conformsTo : {doc['conformsTo']}")
    logger.info(f"  configs    : {len(doc['recordSet'])}")
    logger.info(f"  fields each: {len(doc['recordSet'][0]['field'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

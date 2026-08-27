# Hosting, licensing, and maintenance

Written for the NeurIPS Datasets and Benchmarks checklist, which requires
a hosting plan, a licence, and a statement of who maintains the artifact.

## Hosting

The dataset lives at
[huggingface.co/datasets/mjbommar/SHELF](https://huggingface.co/datasets/mjbommar/SHELF),
public, with 13 configs. Hugging Face serves it over the standard datasets
API, so no reviewer needs credentials or a special client:

```python
from datasets import load_dataset
ds = load_dataset("mjbommar/SHELF", "all")
```

Machine-readable metadata is published at `croissant.json` in the same
repository, conforming to Croissant 1.0 and validated with `mlcroissant`.
Hugging Face also auto-generates a Croissant record at
`/api/datasets/mjbommar/SHELF/croissant`; that copy is a stub with no
`recordSet`, so the hand-built file is the authoritative one.

Code lives in a separate repository and is versioned independently. Every
result file records the git commit, the dataset checksum, the random seed,
and the sklearn and Python versions that produced it.

## Licensing

The dataset is CC-BY-4.0.

Two components carry their own provenance and are not ours to relicense:

- **The Gutenberg transfer slice** derives from Project Gutenberg texts.
  It is included as a natural-text control. Users redistributing it should
  check Project Gutenberg's terms for their jurisdiction.
- **Taxonomy labels** derive from Library of Congress vocabularies (LCC,
  LCGFT, LCSH, LCDGT), which the Library of Congress publishes as open
  data.

The document text is model generated. We make no claim that model output
is copyrightable and impose no restriction on it beyond the licence above.

## Versioning

Configs are additive. A published config is never silently rewritten: when
a slice is found to be wrong, it is republished under a name that says what
it is, and the card records the change. Two such corrections have already
happened and are documented on the card:

- A planned LCC subclass tier was generated from parent-class descriptions
  rather than subclass descriptions, so it carries no subclass label. It
  ships as `v0_4_supplement`, described for what it is, and the tier is
  not claimed.
- One model was reachable through two provider routes and appeared under
  two ids, splitting its own share. Ids are now normalised, with
  `provider_served` retaining the routing detail.

The `default` config is frozen so that published v0.3.1 baselines stay
valid. New work goes into new configs.

## Maintenance

Maintained by the author. Issues and corrections go through the Hugging
Face repository's discussion tab and the code repository's issue tracker.

The commitment is narrow and therefore keepable: **correctness fixes and
provenance corrections, not feature growth.** A benchmark that keeps
changing is not a benchmark. If a defect is found in a published config,
the fix is a new config plus a card entry, not an in-place edit.

## Known limitations, stated up front

These belong in a maintenance plan because they bound what the artifact
should be used for.

- **Scores do not transfer to natural text.** A lexical classifier
  scoring 0.8932 in-domain scores 0.3010 on Gutenberg, and the failure is
  symmetric (0.5261 natural in-domain, 0.2954 transferred the other way).
  A SHELF score is not an estimate of catalogue performance.
- **No human ceiling.** No annotation round has been run, so absolute
  scores have no upper reference. LCSHBench reports a human ceiling of
  86.9% exact for its own task; we have no equivalent.
- **The pooled `all` config is not generator balanced.** The largest
  generator is 47.7% of it. Generator-sensitive results belong on
  `v0_4_core`.
- **Source bias is unmeasured.** Neural retrievers are known to prefer
  model-written text. Since the corpus is entirely model written, a
  per-model bias term of unknown size sits underneath any ranking.

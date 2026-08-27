# Transfer between corpora

Measured 2026-08-27 with `scripts/transfer_matrix.py`. TF-IDF plus
logistic regression, 21-class LCC, macro-F1, `zero_division=0.0`. The probe
has no pretraining, so contamination in the natural corpora cannot explain
any gap here.

## Full corpora

Each corpus at its natural size: SHELF 62,899, Gutenberg 3,016,
LCSHBench 4,924 (English).

| train \ test | shelf | gutenberg | lcshbench |
|---|---|---|---|
| **shelf** | **0.8873** | 0.3133 | 0.4113 |
| **gutenberg** | 0.2836 | **0.5101** | 0.2135 |
| **lcshbench** | 0.4442 | 0.2800 | **0.5559** |

## Size-balanced control

Every corpus subsampled to 3,016 documents, the size of the smallest. This
rules out the obvious confound: SHELF is twenty times larger than
Gutenberg, so its diagonal could have been a training-set-size artifact.

| train \ test | shelf | gutenberg | lcshbench |
|---|---|---|---|
| **shelf** | **0.8052** | 0.2341 | 0.3831 |
| **gutenberg** | 0.2869 | **0.4969** | 0.2162 |
| **lcshbench** | 0.3866 | 0.2474 | **0.5282** |

## What the matrix shows

**1. The original transfer result reproduces.** The four cells reported in
the v0.4 plan land within 0.02 of the earlier measurement (0.8932 /
0.3010 / 0.5261 / 0.2954 against 0.8873 / 0.3133 / 0.5101 / 0.2836). It
was not an artifact of the older corpus.

**2. SHELF is lexically easier, and it is not a size effect.** The
in-domain advantage survives balancing: 0.8052 against 0.4969 and 0.5282
at identical sample size. Synthetic documents carry more surface signal
about their own label than natural text does. This is the saturation
finding, and balancing is what makes it a finding rather than an artifact.

**3. The new result: natural-to-natural transfer is the worst in the
matrix.** Gutenberg and LCSHBench are both human written and human
catalogued, and they transfer to each other at 0.2162 and 0.2474 -- worse
than SHELF transfers to LCSHBench (0.3831). Two genuine catalogue corpora
agree with each other less than the synthetic corpus agrees with either.

The third point changes the argument. The obvious objection to SHELF is
that synthetic data does not transfer to real text, and Gill et al.
(arXiv:2505.22830) give that objection its strongest published form. The
matrix says the premise is too narrow: **cross-corpus transfer failure is
general, not synthetic-specific.** A model fitted to Gutenberg prose does
not classify catalogue records either.

That does not rescue absolute transfer -- SHELF scores still do not
predict catalogue performance, and we should keep saying so. It does mean
the honest framing is corpus mismatch rather than synthetic deficiency,
and that the right comparison for any transfer claim is a natural-to-
natural baseline. Papers that report synthetic-to-natural degradation
without one are reporting an effect they have not isolated.

## Caveat on LCSHBench

LCSHBench rows are catalogue metadata -- title, abstract, table of
contents -- with a median of 596 characters. Gutenberg rows are running
prose. Some of the Gutenberg-to-LCSHBench gap is length and register
rather than taxonomy. The finding survives this because it is a comparison
of *pairs*, and the synthetic-to-LCSHBench pair beats the
natural-to-LCSHBench pair on the same target.

## Reproducing

```bash
python scripts/transfer_matrix.py \
    --corpus shelf=data/hf_dataset/all \
    --corpus gutenberg=data/hf_dataset/transfer_gutenberg \
    --corpus lcshbench=data/hf_dataset/transfer_lcshbench \
    --balance
```

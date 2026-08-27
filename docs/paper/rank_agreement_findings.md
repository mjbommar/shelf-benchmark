# Rank agreement

Measured 2026-08-27. 22 embedding models scored on 21-class LCC
classification against three corpora, then their rankings correlated.
Intervals are 2,000 bootstrap resamples **of models**, since the question is
whether an ordering would hold for a different model set.

| pair | n | Spearman | 95% CI | Kendall |
|---|---|---|---|---|
| SHELF vs Gutenberg | 22 | **0.878** | [0.64, 0.99] | 0.783 |
| SHELF vs LCSHBench | 22 | **0.781** | [0.42, 0.97] | 0.687 |
| Gutenberg vs LCSHBench | 22 | 0.963 | [0.86, 0.99] | 0.852 |

## What this settles

**Absolute scores do not transfer; rankings do.** These are separate
properties and SHELF now has a measurement of each:

| property | result |
|---|---|
| absolute score | fails — 0.8873 in-domain against 0.3133 on Gutenberg |
| model ranking | holds — rho 0.878 against Gutenberg, 0.781 against LCSHBench |

A benchmark can be useless for predicting what a model will score in
production and still be reliable for choosing between models. SHELF is that
benchmark, and both halves are measured rather than asserted.

For scale, the closest published precedent is Majurski and Matuszek (TMLR
2026), who report Spearman 0.91 between synthetic and human-curated
benchmarks in a different domain. SHELF's 0.878 against natural prose sits
just below it.

## The caution that survives

Natural-to-natural agreement is the highest cell (0.963). Gutenberg and
LCSHBench order models more alike than SHELF orders them like either.

The intervals overlap -- [0.64, 0.99] against [0.86, 0.99] -- so this is
**not** a demonstrated difference, and it should not be reported as one.
What can be said is that nothing in these data shows SHELF ranking models
*better* than a natural corpus would. The defensible claim is that SHELF
ranks models about as well as natural bibliographic data does, not better.

## A caveat retired

Before the full sweep, a partial run over 12 models put SHELF against
LCSHBench at 0.350 and appeared to show weak agreement. That was
underpowered; at n = 22 the same pair reaches 0.781. Partial sweeps were
flagged as preliminary at the time and should not be cited.

An expected explanation also failed. LCSHBench is catalogue metadata with a
median of 596 characters and Gutenberg is running prose, so a modality gap
looked like a natural reason for weak agreement across that pair. But those
two corpora agree with each other at 0.963 *despite* the modality gap, so
modality plainly does not prevent rank agreement and cannot excuse a low
correlation.

## Reproducing

```bash
python scripts/rank_agreement.py \
    --corpus shelf=results/v0.3.0/baselines \
    --corpus gutenberg=results/transfer_gutenberg/baselines \
    --corpus lcshbench=results/transfer_lcshbench/baselines \
    --task lcc_classification
```

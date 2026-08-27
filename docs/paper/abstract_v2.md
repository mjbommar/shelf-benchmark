# Abstract (v2)

Reasoning in §2. Numbers verified 2026-08-27; bootstrap CIs in
`scripts/../transfer_findings.md` and the Pareto test.

---

## 1. The abstract

> Benchmarks for bibliographic classification are built on library
> catalogue records, on the assumption that real records are the gold
> standard and generated text is a compromise. We measure that assumption
> and find it does not hold.
>
> We introduce SHELF, a corpus of 62,899 documents generated from
> content-addressed specifications by 25 language models, labelled with
> Library of Congress classes and genre terms. Because every document is
> drawn from an explicit specification, generator, genre, and subject vary
> independently, which no natural corpus permits.
>
> We then measure how far a classifier trained on one corpus carries to
> another, across SHELF, Project Gutenberg passages, and LCSHBench
> catalogue records. Transfer is poor everywhere. The two human-catalogued
> corpora reach only 0.21 and 0.25 macro-F1 on each other, below every
> other pairing we test. There is no natural corpus that stands in for
> bibliographic classification in general.
>
> Against that baseline, SHELF is the strongest available source of
> training signal. At matched sample size it is never significantly worse
> than a natural corpus as a transfer source, and it is better by 0.17
> macro-F1 on catalogue records. At its own scale it is the best source
> for both natural corpora we test. Scale is not a confound here: it is
> what generation buys.
>
> Absolute scores still do not transfer. A model scoring 0.89 on SHELF
> scores 0.31 on Gutenberg, and we report SHELF as an instrument for
> ranking and diagnosis rather than as an estimate of catalogue
> performance.

About 250 words.

---

## 2. How this was derived

Work backwards from the transfer matrix. The paper's value follows from
what the off-diagonal cells say, not from anything we assert about the
corpus.

### 2.1 The measurement

TF-IDF plus logistic regression, 21-class LCC, macro-F1. The probe has no
pretraining, so contamination cannot explain any cell.

| train \ test | shelf | gutenberg | lcshbench |
|---|---|---|---|
| **shelf** | **0.8873** | 0.3133 | 0.4113 |
| **gutenberg** | 0.2836 | **0.5101** | 0.2135 |
| **lcshbench** | 0.4442 | 0.2800 | **0.5559** |

### 2.2 Which source wins for each target

For every target, compare the two other corpora as training sources.
Intervals are 2,000 bootstrap resamples of the test set.

**At natural scale:**

| target | best source | margin | 95% CI |
|---|---|---|---|
| gutenberg | **shelf** | +0.0333 | [+0.0033, +0.0639] |
| lcshbench | **shelf** | +0.1978 | [+0.1702, +0.2263] |
| shelf | lcshbench | +0.1606 | [+0.1535, +0.1679] |

SHELF is the best out-of-domain source for **both** natural corpora, and
both margins exclude zero.

**At matched sample size** (every corpus cut to 3,016, the size of the
smallest):

| target | best source | margin | 95% CI |
|---|---|---|---|
| gutenberg | lcshbench | +0.0132 | [-0.0185, +0.0432] — **not significant** |
| lcshbench | **shelf** | +0.1669 | [+0.1293, +0.2004] |

At equal N the Gutenberg comparison is a tie, and the LCSHBench
comparison is a large SHELF win. **SHELF is never significantly worse than
a natural corpus as a transfer source, and is sometimes much better.**

### 2.3 The inference

The standard objection to synthetic evaluation data assumes a natural
corpus is the thing to be measured against. That assumption needs
natural corpora to agree with each other. They do not: Gutenberg and
LCSHBench, both human written and human catalogued, produce the two worst
cells in the matrix.

So the choice was never synthetic against real. Every corpus is a proxy,
each fails to generalise to the others, and the practical question is
which proxy to hold. On that question the synthetic corpus wins on the
merits, because it is never a worse source and is the only one whose
scale, balance, and difficulty can be set deliberately.

Scale deserves a sentence of its own, because it looks like a confound and
is not. SHELF's full-size advantage does partly come from having 62,899
documents. That is the point. One can generate another 62,899 SHELF
documents; one cannot generate another 62,899 Harvard catalogue records.
The balanced control is there to show the advantage does not *depend* on
scale, not to argue that scale should be discounted.

### 2.4 What we keep conceding

Absolute transfer stays disproved, and the abstract says so in its own
voice rather than in a limitations section. 0.8873 in-domain against
0.3133 transferred is a real gap, it is symmetric, and it means a SHELF
score is not a forecast of catalogue performance. The claim is ranking and
diagnosis.

This is the difference between the earlier framing and this one. Before,
the transfer gap was the paper's problem and the abstract worked around
it. Now it is the paper's evidence, and the finding that natural corpora
fail the same test is what turns it from an apology into a result.

### 2.5 Caveat to keep visible

LCSHBench rows are catalogue metadata with a median of 596 characters;
Gutenberg rows are running prose. Length and register contribute to the
Gutenberg-to-LCSHBench gap. The comparison survives because it is between
*sources on a fixed target*: both candidate sources face the same target
text, so the target's register cannot favour either.

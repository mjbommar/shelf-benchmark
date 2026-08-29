# Pre-registration: cross-formulation validation and the leakage ablation

Frozen 2026-08-29, **before any of the experiments below have been run**.
Supersedes the decision rules in `CROSS_TASK_VALIDATION_PLAN.md`, which were
qualitative. Written in response to six reviewer amendments, all accepted.

The point of freezing this is that the analyses below can contradict the
paper. If the decision rule is written after the numbers, it is not a rule.

---

## 0. What is already known, and what is being tested

Established: subject-classification rankings agree across SHELF, Gutenberg,
and LCSHBench at Spearman 0.877 and 0.777. SHELF carries about three times
the natural verbatim-label rate (19.1% against 6.1%, length-controlled).

Open, and tested here:

1. Does rank agreement hold for retrieval, clustering, and pairs, or only
   for classification?
2. Does label leakage change *which model wins*, or only the scores?

---

## 1. Scope of any claim these experiments can support

**These are four formulations of one label on overlapping documents. They are
not four independent validations of the benchmark.** Classification,
retrieval, clustering, and pairs reuse the same documents and the same
representations, so their agreement is correlated by construction.

The strongest claim available if everything holds:

> Subject-based model rankings transfer across three corpora and four
> evaluation formulations.

It will **not** license claims about genre, audience, register, geography, or
instruction following, none of which exist in both natural corpora. The
paper states this limit explicitly whatever the outcome.

---

## 2. Fixed analysis choices

**Model intersection.** Every correlation uses the identical set of models
scored on all corpora for that task. The intersection is computed once,
reported as `n`, and never adjusted after seeing results. Fine-tuned models
stay excluded (they are trained on SHELF).

**Primary metrics, frozen now:**

| task | primary | secondary |
|---|---|---|
| classification | macro-$F_1$ | accuracy, micro-$F_1$ |
| retrieval | **binary nDCG@10** | graded nDCG@10, MRR, recall@10 |
| clustering | **ARI** | v-measure, NMI |
| pairs | **AUC** | average precision, $F_1$, accuracy |

Binary nDCG is primary for retrieval because graded relevance derives from
the LCC hierarchy, and the natural corpora lack the subclass metadata that
grading leans on. Graded stays secondary and is reported.

ARI is primary for clustering because v-measure is not chance-corrected and
scored 0.0869 on shuffled labels in our own control.

**Intervals.** Every correlation carries a 2,000-sample bootstrap interval
resampling **models**. Every score difference carries one resampling the
appropriate unit (documents for classification and retrieval; see §5 for
pairs).

---

## 3. Decision rule for "rank agreement holds"

Computed per task, SHELF against each natural corpus.

- **Supports the broad claim:** Spearman point estimate $\geq 0.6$ with the
  lower bound of the 95% interval $> 0$, for **at least three of the four**
  formulations against **both** natural corpora.
- **Supports a narrowed claim:** the above holds for classification but fails
  for two or more other formulations. The title becomes *Subject
  Classification Rankings Transfer*, and the abstract states which
  formulations did not.
- **Contradicts the claim:** the lower bound crosses zero for classification
  against either natural corpus at the fixed model intersection. The paper's
  central result is withdrawn and rewritten around the transfer matrix alone.

**Contradictory results across tasks are reported, not averaged.** If
retrieval agrees and clustering does not, the paper says so and speculates in
one sentence at most.

---

## 4. Masking ablation, with controls

**Applied identically to all three corpora.** Masking only SHELF would confound
the intervention with the corpus.

**Variant list, frozen before running.** For a label $L$ the masker removes,
case-insensitively at word boundaries: $L$ itself; $L$ with a trailing
plural; each whitespace-separated token of $L$ of five characters or more;
and each such token with a trailing plural. Nothing else. Word-boundary exact
removal is what `sanitize_description` already does; the plural and
token-level rules are the extent of what "close variant" means here, and the
plan does not claim morphological analysis it has not implemented.

**Four conditions per corpus:**

| condition | what is removed |
|---|---|
| unmasked | nothing |
| masked | label terms per the frozen rule |
| **sham** | an equal number of randomly chosen tokens, matched per document to the count masking removed |
| — | (sham uses a fixed seed and is reported alongside every masked result) |

The sham condition is the control that separates *removing label signal* from
*damaging the document*. A masked-versus-unmasked change that the sham
reproduces is not evidence about leakage.

**Reported for classification and retrieval:**

1. Score change, masked and sham against unmasked, per corpus.
2. Rank correlation, masked against unmasked, per corpus.
3. Rank correlation, masked SHELF against masked natural corpora.

**Interpretation, fixed in advance:**

- Masking lowers scores but preserves ranking, and sham does not: leakage
  inflates scores uniformly. The ranking claim survives and is strengthened.
- Masking changes the ranking beyond the sham's effect: leakage was
  distorting model selection. This is a finding against the current corpus,
  and masking becomes a fix for the next generation rather than a robustness
  check.
- Masked-against-natural agreement exceeds unmasked-against-natural: leakage
  was actively moving SHELF away from natural behaviour. Same conclusion,
  stated more strongly.

---

## 5. Pair construction

One mining policy, applied identically to all three corpora, documented in
the release:

- 2,000 positive and 2,000 negative pairs per corpus, so chance is AUC 0.5.
- Positive and negative quotas balanced across the 21 LCC classes.
- **Random negatives**, not hard negatives, because hard-negative mining
  depends on an embedding model and would couple the task to one of the
  systems under test.
- No duplicate pairs, and no pair of a document with itself.
- A document may appear in several pairs; the cap is recorded.

**Document-clustered bootstrap.** 4,000 pairs built from far fewer documents
are not 4,000 independent observations. Intervals for pair metrics resample
**documents**, and every pair containing a resampled document travels with
it.

---

## 6. Clustering stability

K-means currently runs with `n_init=10` at a single fixed `random_state`, so
all models share one initialisation draw.

- Run each clustering evaluation at **five fixed seeds** (0, 1, 2, 3, 4).
- Report per-model median ARI across seeds as the primary value.
- Report rank stability across seeds: the Spearman correlation between the
  model ordering at each pair of seeds.
- **If median across-seed rank stability is below 0.9, clustering is dropped
  from the rank-agreement claim** and reported as too unstable to carry it.
  That threshold is fixed now.

---

## 7. Class composition

Measured before running, and it defuses part of the concern: LCC prevalence
is close to uniform in all three corpora — the ratio of most to least common
class is 1.1x in SHELF, 1.0x in Gutenberg, and 1.1x in LCSHBench. The corpora
are not competing on different class mixtures.

Prevalence is not difficulty, so:

- Macro aggregation stays primary throughout, which already weights classes
  equally.
- Per-class scores are **not** currently emitted: classification results
  carry accuracy, macro/micro/weighted F1 and counts only. An earlier version
  of this section promised them. They require a change to the classification
  evaluator and are listed as outstanding rather than claimed.
**True prevalence — corrected 2026-08-29 after a second review.** An earlier
version of this section reported Gutenberg at 1.01x and asserted that its real
distribution was "not recoverable from this repository". **Both were wrong.**
1.01x is the stratified evaluation slice, which is the very quantity the
review said must not be passed off as true prevalence, and the real figure was
already computed by this project's own pipeline and committed:

| source | file (tracked) | n | max/min |
|---|---|---|---|
| Project Gutenberg eligible pool | `data/transfer/gutenberg/manifest.json` | 58,077 | **306x** |
| LCSHBench English dev, as published | LCSHBench | 4,204 | 1.19x |
| Gutenberg *evaluation slice* (stratified) | `transfer_gutenberg` | 3,016 | 1.01x |

Gutenberg's real pool is dominated by P (Language and Literature): P 54.3\%; D 9.7\%; B 6.4\%; A 4.6\%.
`data/transfer/gutenberg/distribution_report.json`, also tracked, already
carried a skew notice saying so.

So the honest statement is: **the evaluated Gutenberg slice is a 21-class
stratified sample drawn from a pool with 306-fold class imbalance.** That is a
deliberate design choice, not a property of the corpus, and any claim about
ecological performance on Project Gutenberg would need the slice rebuilt at
natural prevalence. LCSHBench needs no such caveat at 1.19x.

- **Correction, added after review.** An earlier version of this section said
  "the original distribution remains the primary ecological result." That was
  empty: no result in this project uses any natural corpus's original class
  distribution. Gutenberg is uniform *because it was built uniform* --- 142 to
  144 documents in each of 21 classes is a stratified sample, not Project
  Gutenberg's true LCC prevalence, which is not reported anywhere here. The
  honest statement is that all three evaluated slices are near-balanced by
  construction, so composition cannot explain a ranking difference between
  them, and that no ecological result exists. Producing one requires
  resampling Gutenberg to its natural prevalence, which is future work.

---

## 8. Order

A (retrieval and clustering, no new code) → C (masking with sham) →
B (pairs) → D (frontier models, optional and not required for a preprint).

C precedes B because the leakage question is the more serious objection and
does not depend on pairs.

---

## 9. What gets reported regardless

Every phase's result, including the ones that weaken the paper. The paper
already carries five findings that went the wrong way. This document exists
so that a sixth cannot be quietly dropped.

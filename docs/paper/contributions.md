# SHELF paper contributions

Rewritten 2026-08-27. The previous version asserted six novelty claims,
five of which do not survive a check against the literature. What follows
records what we can defend, what we cannot, and the evidence for each.

Framing lives in [outline_v2.md](outline_v2.md). Read that first.

---

## What we cannot claim

Each of these appeared in the previous draft and has to go. Keeping them
costs credibility on the claims that do hold.

| Retired claim | Why |
|---|---|
| First "domain-complete" synthetic benchmark | Invented category, and the exact claim shape Raji et al. (arXiv:2111.15366) argue fails construct validity. Naming the phenomenon narrowly is stronger. |
| Synthetic generation ensures zero overlap with pretraining data | Overstated. It buys a date: documents generated in 2026 cannot sit in an embedder trained before them. It does not buy freedom from source bias. |
| First benchmark to address contamination through synthesis | LiveBench, LiveCodeBench, GSM1k, and PhantomWiki all precede us. PhantomWiki does it while deliberately avoiding language models. |
| Sparse outperforms neural, challenging scaling assumptions | Not novel. FinMTEB reports bag-of-words beating every dense model on financial semantic similarity; BEIR established BM25 as a strong zero-shot baseline in 2021. Report it as corroboration. |
| First benchmark with efficiency-adjusted rankings | Unverified and implausible as stated. Drop the precedence claim; keep the analysis. |
| First bibliographic benchmark | LCSHBench (arXiv:2606.04382) and SemEval-2025 Task 5 got there first, on real catalogue records with real cataloguer labels. Verified 2026-08-27 by loading LCSHBench from the hub. |

Two presentational fixes:

- **Do not write "document understanding."** In this literature it already
  means visually rich document analysis (DocVQA, UDOP, DUDE, VRDU), so it
  routes reviewers to the wrong prior work. Use "bibliographic
  classification" or "taxonomic representation quality."
- **Do not pool `default` with `v0_4_core` when reporting corpus size.**
  Pooling returns gpt-5.2 to 47.7% of the corpus against 9.2% in
  `v0_4_core`. Report slices separately, or report the pooled `all`
  config and state the imbalance.

---

## What we can claim

### 1. The synthetic-to-natural transfer gap

The strongest result in the project, and currently the only one with no
counterpart in the literature for this task.

TF-IDF with logistic regression, 21-class LCC, macro-F1:

| Train -> Test | macro-F1 |
|---|---|
| SHELF -> SHELF | 0.8932 |
| SHELF -> Gutenberg | 0.3010 |
| Gutenberg -> Gutenberg | 0.5261 |
| Gutenberg -> SHELF | 0.2954 |

Three properties carry it. The failure is symmetric, so it is not that
synthetic text is simply easier. The natural in-domain control sits well
above the transferred score, so it is not that natural text is hard. And
the classifier has no pretraining, so contamination cannot explain it.

**Evidence status:** measured. Needs restating on v0.4 and the pooled
corpus before publication.

### 2. The design is the contribution, and it is load-bearing

Every point claim above falls to prior work. That is normal for a resource
paper -- BEIR contained no novel dataset, MTEB no novel task. What matters
is whether the combination enables a measurement that a single-axis
benchmark cannot produce. Ours does: the transfer result requires at least
four design elements holding at once.

The ablation table in [outline_v2.md](outline_v2.md) §2.1 gives the full
mapping of design element to confound removed to measurement enabled.

**Evidence status:** argued, and checkable from the table.

### 3. Generator as a controlled factor

Fifteen generators realising one identical spec block, split on `spec_id`,
with generator-by-label independence measured (Cramer's V <= 0.028) and
generator attribution measured at 93.1%. Largest generator share 9.24% in
`v0_4_core`.

E5-Mistral and Cosmopedia used many models. Neither reports generator as
an axis with paired realisations of a single specification.

**Evidence status:** measured and published.

### 4. Genre orthogonal to subject by construction

The 21 x 133 cross product with measured independence. Natural corpora
correlate genre with subject, so they cannot separate "does the model read
the subject" from "does the model read the register." QUEST comes closest
and has no genre axis.

**Evidence status:** measured.

### 5. Negative results retained

Register clustering came in at chance (pooled ARI 0.0010 against a 0.0013
flat baseline) and the task was retired rather than reported. The
geographic label defect (76.4% of two-tag documents spanning different
regions) was measured, fixed, and reported with a shuffled-label control.
A prose-rewrite A/B was run and rejected on its own evidence.

Bean et al. (arXiv:2511.04703) found only 16.0% of 445 benchmarks conduct
any statistical testing. Publishing this material is differentiating.

**Evidence status:** measured.

---

## Claims needing evidence we do not yet have

### Rank agreement

Absolute transfer is disproved -- by us. Rank agreement is the claim that
remains, and it must be measured rather than asserted: does SHELF rank
models the way natural bibliographic data ranks them?

Precedents say the target is right and the numbers can be high: Majurski
and Matuszek report Spearman 0.91 against human-curated benchmarks (TMLR
2026), and YourBench reports rho = 1 for rankings from document-grounded
synthetic evaluation. Neither is bibliographic, so ours is a new
measurement rather than a replication.

We now have two natural comparison corpora: the Gutenberg slice, and
LCSHBench, which carries real LCC labels on real records.

**Evidence status:** not yet run. This is the largest open item.

### Source bias

Neural retrievers prefer model-written text (Dai et al., KDD 2024). Our
headline output is a model ranking, so a per-model bias term of unknown
size sits underneath it. Measure it on matched pairs and report it beside
each score as a second axis.

**Evidence status:** not yet run.

### A human ceiling

No annotation round has been run, so absolute scores have no scale. The
399-document kit is built and smoke-tested. Bowman and Dahl
(arXiv:2104.02145) make reliable annotation a benchmark criterion.

**Evidence status:** infrastructure ready, not run.

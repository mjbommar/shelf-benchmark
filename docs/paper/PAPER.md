# SHELF: paper draft

Consolidated 2026-08-27. Supersedes `outline_v2.md` as the working document.
Every number below is measured and traceable to a script in this repository;
nothing is carried over from earlier drafts on trust.

Companion documents: `contributions.md` (claim audit),
`transfer_findings.md`, `rank_agreement_findings.md`,
`topic_leakage_experiment.md`, `hosting_and_maintenance.md`.

---

## Title

> **Rankings Transfer, Scores Do Not: A Factorial Benchmark for
> Bibliographic Classification**

The title states the finding rather than naming the artifact. Avoid
"document understanding" everywhere: in this literature it means visually
rich document analysis (DocVQA, UDOP, DUDE) and will route reviewers to the
wrong prior work.

**Venue: ICLR 2027** — abstract 18 September 2026, paper 25 September 2026
(AoE). Fallback is the NeurIPS **Evaluations & Datasets Track**, renamed from
"Datasets and Benchmarks" in 2026, whose 2027 call is not yet published.
Croissant metadata, hosting plan, and licence are already in place, which the
NeurIPS track requires and ICLR does not penalise.

---

## Abstract

In `abstract_v2.md`. The load-bearing sentence: absolute scores do not carry
across corpora, model rankings do, and both are measured.

---

## 1. Introduction

Open on saturation, which is verifiable rather than rhetorical. MTEB's
English v2 classification suite tops out at 77 labels (Banking77) and its top
twelve models sit between 0.89 and 0.92. Then the gap: across all 1,443 MTEB
tasks there is no Library of Congress task, no Dewey, no MARC, no cataloguing
task of any kind (verified by grep over the installed package). Footnote the
trap: `LccSentimentClassification` is Leipzig Corpora sentiment, not Library
of Congress.

State the contribution as a measurement, not a resource.

---

## 2. Related work

Six subsections, closest neighbour first. Full citation list and
verification log in `outline_v2.md` §7.

1. **Automated subject cataloguing.** Frank and Paynter (JASIST,
   10.1002/asi.10360) for the classical form of our primary task; Annif for
   production practice; SemEval-2025 Task 5 / LLMs4Subjects; LCSHBench
   (arXiv:2606.04382), which we also use as a test corpus. One sentence on
   the distinction: that line assigns headings to real records at extreme
   label scale; we measure representation quality under a controlled design.
2. **Embedding benchmarks and saturation.** MTEB, BEIR, MMTEB, HUME.
3. **Domain benchmarks and the case against them.** FinMTEB, ChemTEB,
   CaseHOLD; then Raji et al., Bean et al., Bowman and Dahl, with an
   explicit statement of which criteria we meet.
4. **Synthetic evaluation data.** InPars/Promptagator/SWIM-IR, E5-Mistral,
   PhantomWiki. Then Gill et al. (arXiv:2505.22830), the sharpest objection,
   answered in §6 with a measurement rather than an argument.
5. **Contamination.** GSM1k, rephrased-sample undetectability, MTEB issue
   #1036. State the limit of our own claim.
6. **Instruction following and exclusion.** FollowIR, InstructIR,
   Promptriever, NevIR, ExcluIR, CoDeR, QUEST, CSFCube. Use **V@k** from
   CoDeR, not our coined name.

**Caution on FinMTEB.** Its rank correlations were computed over seven
models. Cite the point estimates and absolute drops; do not lean on the
significance test.

---

## 3. Corpus

Content-addressed `DocumentSpec`; the same spec issued to every generator;
splitting on `spec_id`.

**Scale.** 62,899 synthetic documents (config `all`), plus 3,016 natural
Project Gutenberg passages held out as a transfer control.

**Generators.** The balanced slice `v0_4_core` is 18,345 documents from 15
current-generation frontier models across 11 labs — Anthropic, OpenAI,
Google, Alibaba, DeepSeek, Zhipu, Moonshot, MiniMax, Meta, Mistral, xAI —
with the largest at 9.24%. Compare E5-Mistral (one model) and Cosmopedia
(one model).

**Cite the right config.** Generator diversity belongs to `v0_4_core`; scale
belongs to `all`, where pooling returns GPT-5.2 to 47.7%. Claiming both from
one number is the error a reviewer will catch.

**Splitting.** Document-level splitting straddles 598 of 600 specs;
`spec_id` grouping straddles none.

**QC.** Gates G1–G7. Report the empty-body defect and its fix.

---

## 4. Tasks

21-way LCC; 133-way form, where the headroom is; pair classification over
the S0–S6 relation ladder; instruction retrieval reported as
chance-normalised **V@k** lift, because base rates differ 5.6× across facets
(0.048 for "different LCC class" against 0.0086 for "different form").

---

## 5. Result 1 — transfer

TF-IDF plus logistic regression, macro-F1, `zero_division=0.0`. No
pretraining, so contamination cannot explain any cell.

| train \ test | shelf | gutenberg | lcshbench |
|---|---|---|---|
| **shelf** | **0.8873** | 0.3133 | 0.4113 |
| **gutenberg** | 0.2836 | **0.5101** | 0.2135 |
| **lcshbench** | 0.4442 | 0.2800 | **0.5559** |

Size-balanced control at 3,016 documents each, which rules out SHELF's
20-fold size advantage as the explanation:

| train \ test | shelf | gutenberg | lcshbench |
|---|---|---|---|
| **shelf** | **0.8052** | 0.2341 | 0.3831 |
| **gutenberg** | 0.2869 | **0.4969** | 0.2162 |
| **lcshbench** | 0.3866 | 0.2474 | **0.5282** |

Three findings:

1. **Absolute scores do not transfer.** 0.8873 in-domain against 0.3133,
   symmetric.
2. **SHELF is lexically easier, and it is not a size effect.** The in-domain
   advantage survives balancing.
3. **Natural-to-natural is the worst pairing in the matrix.** Two
   human-catalogued corpora reach 0.2135 and 0.2800 on each other. Transfer
   failure is general, not synthetic-specific — so a synthetic-to-natural
   degradation reported without a natural-to-natural baseline has not
   isolated its effect. This is the answer to Gill et al.

---

## 6. Result 2 — rank agreement

22 embedding models scored on each corpus; bootstrap over **models**, since
the question is whether an ordering holds for a different model set.

| pair | Spearman | 95% CI | Kendall |
|---|---|---|---|
| SHELF vs Gutenberg | **0.878** | [0.64, 0.99] | 0.783 |
| SHELF vs LCSHBench | **0.781** | [0.42, 0.97] | 0.687 |
| Gutenberg vs LCSHBench | 0.963 | [0.86, 0.99] | 0.852 |

**This is the paper's central claim.** Transfer and rank agreement are
different properties. A benchmark is used to choose between models, which
needs the second, not the first. Majurski and Matuszek report 0.91 between
synthetic and human-curated benchmarks in another domain; 0.878 against
natural prose sits just below.

**Stated honestly:** natural-to-natural is the highest cell at 0.963. The
intervals overlap, so the difference is not demonstrated, and the claim is
that SHELF ranks models *about as well as* natural bibliographic data —
never better.

---

## 7. Result 3 — how much label signal sits on the surface

A corpus-difficulty diagnostic that generalises, reported with a natural
baseline, which is what makes it interpretable.

Verbatim `lcc_name` in its own document, full running text, length
controlled to 200 words:

| corpus | full text | at 200 words |
|---|---|---|
| Gutenberg (natural) | 8.9% | 6.1% |
| SHELF | 23.2% | 18.7% |

Real documents contain their own descriptive terms — a zero baseline would
be strange, and it is why extractive summarisation works. SHELF carries
about 3× the natural rate, which partly explains its lexical ceiling.

**QC has measurable effect.** Length-controlled, by generation:

| config | `lcc_name` | `topics` | `lcgft_form` |
|---|---|---|---|
| v0.3.1 | 21.7% | 76.6% | 7.2% |
| v0.4 core | **13.8%** | **44.5%** | **1.5%** |

Most benchmark papers assert their filters work. This quantifies it.

**A prompt fix exists and is characterised, not adopted.** Topics are passed
verbatim while form and subject get semantic descriptions, which is the
mechanism. An instruction not to echo topic words cuts verbatim echo from
82.2% to 4.4% on one generator and 61.5% to 23.1% on another, at a cost of
8–14 points of blind-judge topic coverage. Details and the corrected
recommendation are in `topic_leakage_experiment.md`.

---

## 8. What a SHELF score licenses

- **Yes:** ranking embedding models for bibliographic classification;
  diagnosing which facet a model reads; controlled ablation of generator,
  genre, and subject.
- **No:** predicting absolute performance on catalogue text. We disproved
  that ourselves and report it as a result rather than a limitation.

---

## 9. Limitations

1. **No human ceiling.** No annotation round has run, so absolute scores
   have no upper reference. LCSHBench reports 86.9% exact for its task; we
   have no equivalent. The 399-document kit is built.
2. **Source bias unmeasured.** Neural retrievers prefer model-written text
   (Dai et al., KDD 2024). The corpus is entirely model-written, so a
   per-model bias term of unknown size sits under any ranking. The 15-model,
   11-lab balance diffuses it but does not measure it.
3. **`all` is not generator balanced.** Largest generator 47.7%.
4. **Modality.** LCSHBench is catalogue metadata (median 596 characters),
   Gutenberg is prose. Reported separately, never averaged. Note that these
   two agree at 0.963 *across* that gap, so modality does not prevent rank
   agreement.
5. **Prompt-fix evidence is thin.** Two generators, n≈90 judgements per arm.

---

## 10. Contributions

1. The transfer matrix, including the natural-to-natural baseline that the
   objection to synthetic data has been missing.
2. Rank agreement measured against two natural corpora, one of them real
   catalogue records.
3. A factorial design whose elements are each justified by a confound they
   remove (`outline_v2.md` §2.1) — the combination is load-bearing, since
   the transfer measurement needs four of them at once.
4. Generator as a reported axis: 15 frontier models, 11 labs, max share
   9.24%.
5. Verbatim-label-rate-against-natural-baseline as a reusable corpus
   diagnostic, plus a measured before/after showing QC works.
6. Negative results retained: register clustering at chance and retired; a
   geographic label defect measured and fixed; a prose-rewrite A/B rejected
   on its own evidence; a prompt fix characterised and *not* adopted.

---

## 11. Schedule

Targeting **ICLR 2027**: abstract 18 September 2026, paper 25 September
2026. Twenty-two and twenty-nine days out.

No human annotation round — decided, not deferred by oversight. Absolute
scores therefore have no upper reference, stated in Limitations. The paper
survives this because its claims are about *rankings*, and a ranking needs
no ceiling to be interpretable.

Remaining before submission is writing, not measurement: the evidence base
(transfer matrix, rank agreement at n=22, leakage diagnostics with a natural
baseline) is complete.

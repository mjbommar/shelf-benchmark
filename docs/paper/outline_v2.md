# SHELF paper outline (v2)

Supersedes `contributions.md` and `related_work_notes.md`, both of which
carry claims that will not survive review (§7 below).

Every external fact here was checked against a primary source on
2026-08-27. Checks are marked [v]. Unchecked claims inherited from the
literature review are marked [u] and must be verified before drafting.

---

## 1. The claim

One sentence, and the whole paper serves it:

> Synthetic corpora can be built with experimental controls that natural
> corpora cannot offer, and the resulting scores do not transfer to
> natural text — we measure both, on the same task, with the same
> classifier.

That is a measurement paper, not a leaderboard paper. It survives the
existence of LCSHBench. A leaderboard paper does not.

**Venue: NeurIPS Datasets and Benchmarks.** Both outline agents reached
this independently. The track guidance explicitly protects work that
"fills a critical gap for a smaller, long-tail research area" [u]. Croissant
metadata, a hosting plan, and a licence are mandatory; the dataset is
already public, so this is work we can finish.

---

## 2. What we can and cannot claim

The literature check killed several claims outright. Better to find that
now than in a review.

| Claim | Status |
|---|---|
| First bibliographic benchmark | **Dead.** LCSHBench [v] and SemEval-2025 Task 5 got there first, on real catalogue records. |
| First to address contamination by synthesis | **Dead.** LiveBench, LiveCodeBench, GSM1k, PhantomWiki. |
| "Domain-complete" coverage of human knowledge | **Drop it.** This is the exact shape of claim Raji et al. wrote their paper to reject. Name the phenomenon narrowly instead. |
| Sparse beats dense, challenging scaling assumptions | **Reframe as corroboration.** FinMTEB reports the same for finance. |
| Instruction-following retrieval is new | **Dead.** MTEB already ships FollowIR, InstructIR, mFollowIR, NevIR. |
| Violation metrics are new | **Dead.** CoDeR defines V@k. Adopt their term rather than coining `contrast_violation`. |

What survives is **not** a list of point claims. It is the design.

### 2.1 The combination is the contribution

Every point claim above falls to a prior paper. That is normal, and it is
not fatal. BEIR contained no novel dataset; it was novel as a zero-shot
heterogeneous aggregation. MTEB contained no novel task. LegalBench is 162
tasks, none of them individually new. Resource papers earn their place by
what the assembly makes possible, not by owning a primitive.

The test a reviewer applies is whether the combination is **load-bearing
or merely additive**. Additive means we did many good things in one place.
Load-bearing means the design enables a measurement that no single-axis
benchmark can produce. Ours is load-bearing, and the transfer result is
the proof: it exists *because* of the architecture, not beside it.

State it as an instrument, not a collection:

> SHELF is a factorial instrument for bibliographic classification. Each
> design element removes a confound that would otherwise make the headline
> measurement uninterpretable.

The argument is best made as a design ablation — for each element, the
measurement it protects:

| Design element | Confound it removes | Measurement it enables |
|---|---|---|
| One spec block realised by 15 generators | Generator identity confounded with content | Generator effect reportable as an axis, not noise |
| Split on `spec_id` | Near-duplicate leakage across splits | Honest generalisation (598/600 specs straddle without it) |
| Genre orthogonal to subject by construction | Genre-subject correlation in all natural corpora | Whether a model reads subject or reads register |
| Label semantics fixed across corpora | Taxonomy drift between synthetic and natural | **The transfer measurement itself** |
| Lexical baseline with no pretraining | Contamination | Transfer gap attributable to distribution, not memorisation |
| Chance-normalised violation | Base rates differ 5.6x across facets | Violation rates comparable across tasks |
| Relation ladder S0-S6 | "Similar" collapsing distinct relations | Graded difficulty rather than binary pairs |

Read down the last column: no existing benchmark has the row set, and the
transfer measurement requires at least four of the rows at once. That is
the holistic claim, made checkable.

**Why this is worth arguing rather than conceding.** Bean et al. reviewed
445 benchmarks and found only 16.0% conduct any statistical testing [u].
Doing the whole package properly is empirically rare, so "we did all of it
and here is what it bought" is a real differentiator — provided we show
what it bought.

### 2.2 The four elements, restated

1. **Generator as a controlled factor.** Fifteen generators over one
   identical spec block, split on `spec_id`, with generator-by-label
   independence measured (Cramer's V 0.016 (subject) and 0.027 (genre category)). E5-Mistral and Cosmopedia
   used many models; neither reports generator as an axis.
2. **Genre orthogonal to subject by construction.** Real corpora correlate
   the two, so they cannot separate the questions. Ours can.
3. **A natural-text control inside the same benchmark.** The Gutenberg
   slice shares the label space, which is what makes transfer measurable
   at all. Most synthetic benchmarks have no natural arm.
4. **Negative results kept.** Register clustering came in at chance and we
   retired the task rather than report the number. The geographic label
   defect was measured, fixed, and reported with a shuffled-label control.

---

## 3. Section plan

### §1 Introduction

Open on the saturation numbers, which are verifiable rather than
rhetorical. MTEB's English v2 classification suite tops out at 77 labels
(Banking77), and the top twelve models sit between 0.89 and 0.92 [u]. HUME
puts the best model above the human average [u]. Then the gap: across all
1,443 MTEB tasks there is **no** Library of Congress task, no Dewey, no
MARC, no cataloguing task of any kind [v, verified by grep over the
installed package]. Note the trap in a footnote — `LccSentimentClassification`
is Leipzig Corpora sentiment, not Library of Congress [v] — because a
reader will otherwise think we missed it.

State the contribution as a measurement, not a resource.

### §2 Related work

Six subsections. Order matters: meet the closest neighbour early.

1. **Automated subject cataloguing.** Lead here, do not bury it.
   Frank and Paynter 2004 for the classical version of our primary task.
   Annif for production practice. SemEval-2025 Task 5. LCSHBench, stated
   plainly, including its human ceiling. Then one sentence on the
   distinction: that line assigns headings to real records at extreme
   label scale; we measure representation quality under a controlled
   design at moderate scale. Different questions.
2. **Embedding benchmarks and saturation.** MTEB, BEIR, MMTEB. The four
   documented failures with numbers.
3. **Domain benchmarks and the case against them.** FinMTEB, ChemTEB,
   CaseHOLD. Then Raji et al., Bean et al., Bowman and Dahl — and an
   explicit statement of which criteria we meet and which we do not.
4. **Synthetic evaluation data.** The query-generation line, E5-Mistral,
   PhantomWiki (which avoids language models on purpose, and we must say
   why we chose otherwise). Then the objections, each answered with a
   measurement from our own corpus rather than an argument.
5. **Contamination.** GSM1k, rephrased-sample undetectability, MTEB issue
   #1036. State the limit of our own claim: documents generated in 2026
   cannot sit in an embedder trained before them. That is all synthesis
   buys. It does not buy freedom from source bias.
6. **Instruction following and exclusion.** FollowIR, InstructIR,
   Promptriever, NevIR, ExcluIR, CoDeR, QUEST, CSFCube.

**Caution on FinMTEB.** Its rank correlations were computed over seven
models [u]. At n=7 a rank correlation almost cannot reach significance,
so "not correlated" is close to guaranteed by construction. Cite the point
estimates and the absolute drops. Do not lean on the significance test — a
reviewer who reads the table will notice, and it costs us credibility we
need elsewhere.

### §3 Corpus construction

Spec blocks and content-addressed `spec_id`. The generator panel. The QC
gates G1-G7 and what each removed. Splitting on `spec_id`, with the
measured leakage it prevents: document-level splitting straddles 598 of
600 specs, grouped splitting straddles none.

Report the empty-body defect and its fix honestly. It hit short documents
hardest and two generators worst.

### §4 Tasks and metrics

The 21-way LCC task. The 133-way form task, which is where the headroom
is. Pair classification over the relation ladder S0-S6. Instruction
retrieval with anchor and contrast facets, reported as chance-normalised
lift, because raw violation rates are not comparable across tasks: chance
is 0.048 for "different LCC class" against 0.0086 for "different form."

Adopt V@k from CoDeR.

### §5 The transfer result — the centre of the paper

Give this its own section. TF-IDF and logistic regression, 21-class LCC,
macro-F1:

| Train -> Test | macro-F1 |
|---|---|
| SHELF -> SHELF | 0.8932 |
| SHELF -> Gutenberg | 0.3010 |
| Gutenberg -> Gutenberg | 0.5261 |
| Gutenberg -> SHELF | 0.2954 |

Three properties make this publishable on its own. The failure is
symmetric, so it is not simply that synthetic text is easier. The natural
in-domain control (0.5261) sits far above the transferred score, so the
gap is not natural-text difficulty. And because the classifier has no
pretraining, Gutenberg's contamination cannot explain it.

Say what the number means: a SHELF score does not predict an absolute
score on real catalogue text. We measured that ourselves and we report it
as a result, not a limitation.

### §6 What a SHELF score licenses

The rebuttal section, and the one most likely to decide the outcome.

Absolute transfer is disproved — by us. **Rank agreement is the claim that
remains, and it must be measured, not asserted.** Majurski and Matuszek
validated synthetic benchmarks against human-curated ones at Spearman 0.91
[u]; that is the template. Run the same correlation between model rankings
on SHELF, on our Gutenberg slice, and on LCSHBench, which is public, CC0,
and carries an `lc_class` field — real LCC labels on real records [v,
loaded from the hub: dev 18,993 / test 3,353, 15 languages, 22 classes].

**This experiment does not exist yet and the paper does not work without
it.** It is the single largest piece of remaining work.

Then the source-bias control. Neural retrievers prefer model-written text
by over 30% [u], and our headline output is a model ranking. Measure the
per-model preference gap on matched pairs and report it as a second axis
beside each score.

### §7 Limitations

No human annotation round. Bowman and Dahl make reliable annotation a
criterion, and without a ceiling "0.918" has no scale to be read against.
The 399-document kit is built and smoke-tested. **My recommendation: run
it before submission.** It is two cataloguers' time and it converts the
weakest objection into a strength. You descoped human labelling for the
data phase, which was right then; this is a different decision, made for
the paper.

Also state: no subclass tier ships in v0.4, and why (§8).

---

## 4. Terminology

Do not write "document understanding." In this literature it already means
visually rich document analysis — DocVQA, UDOP, DUDE. A reader will go to
the wrong prior work. Use "bibliographic classification" or "taxonomic
representation quality." Costs nothing, collides with nothing.

---

## 5. Reporting the corpus

Report slices separately. Never pool `default` with `v0_4_core` — our own
dataset card says why, since pooling returns the largest generator to
roughly half the combined corpus and destroys the balance v0.4 exists to
provide.

Current corpus, as published: 62,899 synthetic documents plus 3,016
natural Gutenberg documents held out as a transfer control, 65,915 total.
`v0_4_core` is 18,345 documents from 15 generators, largest share 9.24%.

---

## 6. Framing

The earlier version of this outline posed a choice between a measurement
paper and a resource paper. That framing was too narrow. The design
argument (§2.1) makes them one paper:

> SHELF is a factorial instrument. The transfer gap is what the instrument
> measured first, and it is the evidence that the controls work.

The resource and the finding support each other. The architecture is
justified by the measurement it enabled; the measurement is credible
because of the controls. Neither half stands alone, which is the normal
shape of a good benchmark paper.

**What still decides the outcome is rank agreement.** If SHELF ranks models
the way natural bibliographic data ranks them, the instrument is useful
beyond the one finding. Two precedents say this is the right target and
that the numbers can be high: Majurski and Matuszek report Spearman 0.91
against human-curated benchmarks [v, TMLR 2026], and YourBench reports
Spearman rho = 1 for rankings from document-grounded synthetic evaluation
[v, cited 16]. Neither is bibliographic, so ours is a new measurement in a
new domain, not a replication.

We now have two natural comparison corpora, which is better than one:
our Gutenberg slice, and LCSHBench, which is public, CC0, and carries real
LCC labels on real records [v].

Run that experiment next. It does not decide *whether* to write the paper
any more — it decides how strong §6 is.

---

## 7. Citation verification log

Checked against primary sources on 2026-08-27.

**Both citations I had flagged as doubtful are real.** A Google Scholar
search returned nothing for either, and I wrongly read that as evidence of
absence. Scholar indexing is not a existence test -- the arXiv export API
was also returning empty for known-good control IDs the same day. Verified
directly:

- **Gill, Ravichander, Marasovic, "What Has Been Lost with Synthetic
  Evaluation?"** arXiv:2505.22830. Confirmed from the arXiv abstract, and
  the load-bearing sentence is verbatim: LLM-generated dataset variants
  are "often valid according to the annotation guidelines, at a fraction
  of the cost," but "are less challenging for LLMs than their
  human-authored counterparts," which "calls for critically reassessing
  the immediate use of this increasingly prevalent approach to benchmark
  creation." Case studies are CondaQA (negation) and DROP (quantities).
  **This is the sharpest objection to SHELF and it must be cited.**
- **Frank and Paynter, "Predicting Library of Congress classifications
  from Library of Congress subject headings."** JASIST,
  DOI 10.1002/asi.10360, issued 2003-10-28. Confirmed via Crossref. This
  is the classical version of SHELF's primary task and belongs in §2 of
  related work.

Verified in the same pass: LCSHBench (loaded from the hub -- dev 18,993 /
test 3,353, 15 languages, 22 LCC classes, `lc_class` field present), MTEB
carrying no cataloguing task of any kind (grep over 1,443 installed task
definitions), Majurski and Matuszek (TMLR 2026, Spearman 0.91 / Pearson
0.74), YourBench, HUME (ICLR 2026), Dai et al. (KDD 2024).

**One caution on HUME.** The literature review summarised it as showing
models have passed the human ceiling. The paper's own abstract resists
that reading: "Rather than treating low human performance as a ceiling to
surpass..." Cite the gap, not a ceiling-crossing claim.

**One addition the review missed.** El Assadi, Muennighoff, and Lee, "The
Embedder's Dilemma: LLMs Are Better, but at What Cost?" (arXiv:2608.12875,
August 2026), 26 embedding models from 118M to 14B. Recent, from the MTEB
maintainers, directly relevant to §2 of related work.

**Still unverified [u].** The FinMTEB n=7 correlation caveat, the MTEB
saturation figures, the Bean et al. 16.0% statistic, and the Dai et al.
"over 30%" magnitude are all quoted from the literature review and have
not been checked against the primary sources. Check before drafting.

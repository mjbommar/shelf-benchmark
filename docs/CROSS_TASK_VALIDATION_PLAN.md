# Plan: broaden the validation, and test whether leakage decides rankings

Written 2026-08-29 in response to a review point that the paper's central
claim — *rankings transfer* — rests on one of four available task
formulations, and that the leakage measurement is descriptive where it could
be causal.

Both criticisms are correct. This plan closes them in cost order, cheapest
first, and each phase produces a result that stands whether or not the later
phases run.

---

## What the constraint actually is

The paper says genre labels do not exist for the natural corpora, and uses
that to justify testing one task. Two things are wrong with it.

`lcc_retrieval` and `lcc_clustering` are defined on `text` and `lcc_code`
only, and both natural corpora carry both fields. **Verified 2026-08-29: both
tasks run on Gutenberg with no code change** (TF-IDF reaches v-measure 0.2572
on clustering; retrieval ran over 627 queries against a 2,389-document
corpus). They were never run there. Nothing prevented it.

And the label claim is itself partly false. Gutenberg carries LCSH topics on **all** 3,016 documents, and
`lcgft_form` (22 distinct) and `lcgft_category` (7) on 686 of them. Only LCSHBench is subject-only.

So the true constraint is narrower: **subject is the only label present in
all three corpora at full coverage** — and subject alone supports four task
formulations.

---

## Phase A — two more formulations, no new code

Run `lcc_retrieval` and `lcc_clustering` for all 22 models on
`transfer_gutenberg` and `transfer_lcshbench`, then recompute rank agreement
per task.

- **Cost:** compute only. Corpora are 3,016 and 4,924 documents, so each
  sweep is far smaller than the pooled one. Estimate 1–2 hours.
- **Code:** none.
- **Output:** rank agreement on three formulations instead of one.
- **Risk:** LCSHBench documents are short catalogue metadata; retrieval over
  a 3,400-document corpus may be too easy or too sparse. Report it, do not
  quietly drop it.

**Decision rule.** If rank agreement holds across classification, retrieval,
and clustering, the paper's claim is materially stronger and the title is
earned. If it holds for classification only, the title must narrow to
*Ranking Transfers for Subject Classification* and the paper says so.

## Phase B — the fourth formulation

Build same-subject pair sets for both natural corpora and score all models.

- **Cost:** a small script. `mine_stratified_pairs` in
  `src/shelf/hub/hard_negatives.py` already does the mining; what is missing
  is writing the result to `<SHELF_DATA_DIR>/pairs/<name>/<split>.parquet`,
  which is where `pair.py` looks.
- **Estimate:** 1–2 hours including the sweep.
- **Note:** `same_lcc_pairs` already declares `primary_metric="auc_roc"`, so
  the F1 problem does not recur here.
- **Balance the pairs deliberately.** 2,000 positive and 2,000 negative, as
  the SHELF pair sets use, so chance is 0.5 and comparable across corpora.

## Phase C — the masking ablation

The higher-value experiment, and the one nobody has run.

1. Mask exact subject names and close morphological variants from SHELF
   documents. Reuse `sanitize_description` in `src/shelf/sampler/leakage.py`,
   which already removes a label at word boundaries; extend it to the
   document body.
2. Rerun subject classification and subject retrieval on masked SHELF.
3. Report three things, not one:
   - **score change** — does masking lower absolute scores?
   - **rank correlation, masked against unmasked** — does masking change
     *which model wins*?
   - **rank correlation, masked against natural** — does masking move SHELF
     *closer to* the natural corpora?

- **Cost:** masker plus two sweeps over 62,899 documents. Estimate 3–4 hours.
- **Why it matters more than the score change:** we already know SHELF
  carries about three times the natural verbatim-label rate. What is unknown
  is whether that inflation is *uniform* — a constant offset that leaves the
  ordering intact — or *selective*, favouring models that exploit surface
  matching. Only the second would undermine the ranking claim.

**The result to watch for.** If masked-against-natural rank agreement is
*higher* than unmasked-against-natural, then leakage was actively distorting
model selection, and masking is not a robustness check but a corpus fix that
belongs in the next generation.

## Phase D — frontier-scale extension, optional

The evaluated set stops at 336M parameters. Adding two or three current
large embedding models would test whether the ranking result survives at the
scale practitioners actually deploy.

- **Cost:** download and inference for large models; several hours, and it
  may exceed available GPU memory.
- **Value:** answers a limitation the paper currently only states.
- **Recommendation:** do it last, and only if A–C hold. If they do not, the
  paper's claims narrow and frontier coverage is beside the point.

---

## Order and rationale

A, then C, then B, then D.

A is free and covers two formulations. C is the highest-value unknown and
does not depend on B. B completes the set but is the weakest of the three,
because pair tasks were the family where three of six carried no signal at
all. D is contingent on the rest.

## What changes in the paper either way

- §5 gains a per-task rank-agreement table instead of a single row.
- §9 loses the incorrect claim about genre labels and gains an accurate one.
- A masking subsection joins §6, since that is where surface signal lives.
- If any phase contradicts the current result, it is reported. The paper
  already carries five findings that went the wrong way; a sixth is not a
  problem.

## A checklist item this exposes

`docs/EVALUATION_CHECKLIST.md` has no rule that catches this. It checks
whether a sweep is complete, whether a sample is biased, whether an interval
supports a claim — but nothing asks whether the *evidence covers the scope of
the claim it supports*. A title-level claim of "rankings transfer" resting on
one of four available formulations passes every existing check. Added as A4.

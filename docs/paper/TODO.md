# SHELF paper — work queue

Dependency-ordered. Steps 1-3 are a strict chain; step 4 runs in parallel
and needs no compute. Status: `[ ]` open, `[~]` in progress, `[x]` done.

Framing and evidence audit live in `outline_v2.md`.

---

## Critical path

### 1. Recover the baseline table
- [x] Fix the overwrite. The real site was `scripts/baselines/run_all.py`,
      not `run_baselines.py:698`: it wrote `summary.json` from only the
      models the current invocation ran. Added
      `harvest_existing_results()`, which folds every per-task result in
      the output directory into the summary before scoring, with the
      current run winning over stale copies.
- [x] Re-aggregate v0.3.0. Recovered 1 -> 360 result cells and 1 -> 25
      SHELF scores, nothing lost. Added `--aggregate-only` to rebuild
      from disk without running anything.
- [x] Regression test: 8 tests in
      `tests/unit/test_baseline_summary_merge.py`. Full suite 1,977 pass.
- [x] **Confirmed the headline does not reproduce.** The recovered table
      puts TF+SVD at 0.5008, rank 10 — matching the earlier recomputation
      (0.502), not the published 0.679. TF-IDF+SVD is 0.5163, rank 3.
      Any paper text citing 0.679 must change.

### 2. v0.4 baseline sweep
- [ ] Run 22 models x 16 tasks on v0.4. Local encoders, so compute only,
      no API spend.
- [ ] **Decision needed:** v0.4 only, or v0.3.0 and v0.4 side by side?
      Side by side costs a second sweep but turns the version bump into
      evidence that generator balancing changed scores — which supports
      the factorial-design argument in `outline_v2.md` §2.1.
- [ ] Produces the paper's headline table and the SHELF ranking for §3.

### 3. Rank agreement — closes outline §6
- [ ] Run the same 22 models on **LCC classification only** across:
      SHELF v0.4, the Gutenberg slice, and LCSHBench (`lc_class`).
      One task, not sixteen.
- [ ] Spearman between the three rankings, with confidence intervals.
- [ ] LCSHBench is public, CC0, 15 languages, dev 18,993 / test 3,353,
      22 LCC classes. Verified loadable from the hub 2026-08-27.
- [ ] Targets from precedent: Majurski and Matuszek 0.91 (TMLR 2026),
      YourBench rho = 1. Neither is bibliographic, so this is a new
      measurement rather than a replication.

---

## Parallel track (no compute)

### 4. Retire dead claims
- [x] `contributions.md` rewritten around what is defensible; six retired
      claims tabulated with reasons. `abstract.md`,
      `related_work_notes.md`, `key_findings.md`, and
      `reviewer_concerns.md` bannered as superseded.
- [x] "domain-complete" removed from `CLAUDE.md`, replaced with the
      factorial-instrument framing plus the two measured cautions.
- [x] "document understanding" flagged and replaced in the live docs.
- [ ] Adopt V@k from CoDeR rather than the coined `contrast_violation`.

### 5. Verify two citations before they enter a bibliography
- [x] Gill et al. (arXiv:2505.22830) — **real, verified from the arXiv
      abstract.** My Scholar-based doubt was wrong; Scholar indexing is
      not an existence test. The load-bearing sentence is verbatim:
      variants are "less challenging for LLMs than their human-authored
      counterparts." Must be cited — it is the sharpest objection.
- [x] Frank and Paynter — **real, verified via Crossref.** JASIST,
      DOI 10.1002/asi.10360, issued 2003-10-28. The classical version of
      SHELF's primary task. Subrahmanyam 2006 (LRTS) is a useful
      additional citation on cross-library LCC consistency.
- [x] Fix the HUME characterisation. The review summarised it as models
      passing the human ceiling; the paper's abstract resists that
      reading ("Rather than treating low human performance as a ceiling
      to surpass..."). Cite the gap, not a ceiling-crossing.
- [x] Add El Assadi, Muennighoff, Lee, "The Embedder's Dilemma"
      (arXiv:2608.12875, Aug 2026). Missed by the literature review,
      from the MTEB maintainers, 26 models.

---

## Open decisions

- [ ] **Venue and timing.** NeurIPS Datasets and Benchmarks 2026 closed in
      late spring; today is 2026-08-27, so that is the 2027 cycle and
      there is no schedule pressure. ICLR's cycle typically closes in late
      September, which would be about four weeks. Confirm real dates —
      the answer decides whether step 6 is affordable.
- [ ] **Human annotation round.** The 399-document kit is built and
      smoke-tested. Without it, absolute scores have no scale and the
      "no reliable annotation" objection (Bowman and Dahl) has no
      rebuttal. My recommendation is to run it; it is the first thing I
      would cut if the venue is four weeks out. Descoped for the data
      phase, which was right then — this is a separate decision.

---

## Submission requirements (NeurIPS D&B)

- [ ] Croissant machine-readable metadata.
- [ ] Hosting, licensing, and maintenance plan.
- [ ] Data and code reachable by all reviewers at submission.

---

## Done

- [x] Publish v0.4 to `mjbommar/SHELF` (12 configs, version 0.4.0).
- [x] Fix release metadata: 15 generators at 9.24% largest share; corpus
      total 62,899 synthetic + 3,016 natural = 65,915; subclass tier
      withdrawn and republished as `v0_4_supplement`.
- [x] Verify LCSHBench, MTEB's absence of any cataloguing task (grep over
      1,443 installed tasks), Majurski and Matuszek, YourBench, HUME,
      Dai et al.
- [x] Reframe the contribution around design rather than point claims
      (`outline_v2.md` §2.1).

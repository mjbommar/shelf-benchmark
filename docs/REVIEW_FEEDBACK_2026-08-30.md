# Review feedback: expanded encoder evaluation

Date: 2026-08-30

## Verdict

The expanded experiments materially strengthen the paper. The scientific evidence is now sufficient for the paper's carefully limited claim: SHELF preserves model rankings for the shared subject label across four correlated task formulations, but does not predict absolute performance on natural corpora.

The manuscript and repositories are not yet ready to publish. Several stale factual statements remain in the paper, and the benchmark checkout is not a clean, frozen result artifact.

## Findings to fix

### 1. Reconcile stale model counts

The current experiment contains 27 configurations and 25 distinct model systems after excluding the OGBert weight duplicate and the second context-budget configuration of GTE-ModernBERT from rank correlations.

The paper states this correctly near the beginning of `latex/sections/05_ranking.tex`, but later says that the sweep contains 22 configurations and 21 sets of weights. `latex/sections/03_corpus.tex` also still refers to a 22-model comparison. Update every model-count statement consistently.

### 2. Correct the confidence-interval table caption

The caption for the detailed formulation table in `latex/sections/05_ranking.tex` describes the previous sensitivity result. It currently says that capable-only analysis removes four OGBert variants plus RoBERTa and that one interval includes zero.

The current restricted analysis removes three distinct OGBert systems plus RoBERTa, and every reported interval remains above zero. The caption should match the table and current artifact.

### 3. Distinguish configurations from distinct encoders

The additions comprise five configurations but only four distinct sets of weights:

- Granite Embedding Small R2
- GTE-ModernBERT at 2,048 tokens
- GTE-ModernBERT at 8,192 tokens, using the same weights
- EmbeddingGemma 300M
- Qwen3 Embedding 0.6B

Therefore, phrases such as "five 2025 encoders" and "25 distinct embedding models, five of them released in 2025" are inaccurate. Prefer wording such as "five new configurations representing four 2025 encoders, with one encoder evaluated at two context budgets."

This wording appears in `paper.yaml`, `latex/sections/04b_baselines.tex`, `latex/sections/04c_retrieval.tex`, and `latex/sections/05_ranking.tex`.

### 4. Update the stale natural-to-natural correlation

The final part of `latex/sections/05_ranking.tex` still reports Gutenberg versus LCSHBench as 0.960. The updated report and table give 0.963.

### 5. Freeze a coherent publication artifact

At review time, the benchmark checkout contained 716 modified and 273 untracked files. The paper checkout was also ahead of its remote and contained a modified rebuilt PDF.

The pooled result manifest describes only the most recent incremental GTE-ModernBERT-8k invocation rather than the complete aggregate directory. Before publication, provide a final aggregate manifest or equivalent ledger that identifies all included models, tasks, result files, exclusions, code revision, corpus revision, and generation procedure.

Then reconcile the generated artifacts deliberately, rebuild the PDF, commit both repositories, push them, and verify that the remote revisions match the intended local revisions.

## Scientific assessment

The expanded model panel resolves the most important remaining sensitivity concern.

- All four subject formulations pass the frozen decision rule against both natural corpora.
- Full-set classification agreement is 0.885 against Gutenberg and 0.792 against LCSHBench.
- The capable-only classification sensitivity against LCSHBench improves from 0.539 with an interval crossing zero to 0.655 with a 95% interval of [0.22, 0.93].
- The other capable-only formulations also pass against both natural corpora.
- Clustering uses five-seed medians and remains above the preregistered median-stability floor on all corpora.
- Gutenberg clustering stability weakened from a median of 0.959 to 0.932, and one seed pair is 0.892. The manuscript discloses this appropriately instead of redefining the frozen rule.
- The GTE-ModernBERT 2k-versus-8k experiment shows a negligible average effect across measured cells, supporting the decision to retain only one context configuration in the rank correlation.

The central remaining scientific limitations are scope rather than an unaddressed confound: one shared subject label, correlated formulations, no matched human-written reference, source bias not directly measured, and a model panel that remains below frontier scale. The manuscript generally states these limitations candidly.

## Verification performed

- Regenerated the cross-formulation report with the intended exclusions: exact match to `results/transfer/cross_formulation.json`.
- Checked all manuscript table values: all 170 values matched their result artifacts.
- Ran the test suite: 1,977 tests passed.
- Ran the paper validation and PDF build: all available gates passed.
- Confirmed that the broad preregistered verdict remains `BROAD CLAIM SUPPORTED`.

## Publication recommendation

After correcting the stale prose and producing a clean, provenance-complete result release, the paper is suitable for arXiv. The evidence is stronger and more practically relevant than in the earlier 21-model analysis. The remaining corrections should be treated as required factual and release-hygiene work, not as reasons to run another large experiment.

# Working state: adding 2025 small encoders

Written 2026-08-30, mid-sweep. This file exists so the work can be picked up
without the conversation that produced it. Delete it when the work lands.

## Goal

Add five 2025 small-encoder configurations to the sweep and to the paper, end
to end, using only the single free GPU (device 1 under
`CUDA_DEVICE_ORDER=PCI_BUS_ID`; device 0 has another tenant).

| key | model | seq | batch |
|---|---|---|---|
| `granite_small_r2` | ibm-granite/granite-embedding-small-english-r2 (48M) | 2048 | 16 |
| `gte_modernbert_2k` | Alibaba-NLP/gte-modernbert-base (149M) | 2048 | 16 |
| `gte_modernbert_8k` | same weights, longer context | 8192 | 4 |
| `embeddinggemma_300m` | google/embeddinggemma-300m (308M) | 2048 | 16 |
| `qwen3_embed_600m` | Qwen/Qwen3-Embedding-0.6B (596M) | 2048 | 8 |

Each model runs six arms: `pooled` (all tasks), `transfer_gutenberg` and
`transfer_lcshbench` (classification, retrieval, clustering), and same-subject
pairs on all three. 33 cells per model when complete.

## Sweep status

Driver: `$SCRATCH/newmodels.sh`, log `/tmp/shelf_newmodels.log`, progress
`$SCRATCH/newmodels_progress.log`. One writer at a time, memory-capped,
retries once and reports `FAIL` on a non-zero exit.

Complete: `granite_small_r2`, `gte_modernbert_2k`, `embeddinggemma_300m`,
`qwen3_embed_600m`. Running: `gte_modernbert_8k` (the truncation arm).
Zero CUDA OOMs across the whole sweep.

Re-running the driver is safe: every call passes `--skip-existing`.

## Findings that change the paper

These are measured, not provisional. Sections needing rewrite once all five
models land:

1. **EmbeddingGemma beats TF-IDF on subject classification**, 0.8887 against
   0.8686. Sections 4b and 4c lean on the sparse baseline leading subject
   classification. It still leads *register* classification (0.6358), so the
   claim narrows rather than disappears.
2. **Genre-form headroom shrank.** Section 6b is built on the task topping out
   at 0.2106. EmbeddingGemma reaches 0.2605, a 24% relative jump.
3. **Not recency dominance, but close.** With four models in, 2025 encoders
   lead four of five tasks. Qwen3-0.6B takes retrieval (0.7104 against MPNet's
   0.6806) and clustering (0.5152 against the old best 0.4482); EmbeddingGemma
   takes classification and genre form. But **MPNet (2020) still leads pair
   classification at 0.8474**, no 2025 model beats it, and `gte_modernbert_2k`
   sits at #12 on that task despite being new. Three different 2025 models hold
   the three new top spots rather than one dominating, which supports section
   6b's claim that the tasks measure different capabilities.

   Ceilings that moved: clustering 0.4482 -> 0.5152 (+15%), genre form
   0.2106 -> 0.2605 (+24%), subject classification 0.8686 -> 0.8887.
4. **Model counts move everywhere.** 30 configured models now. Every "21
   distinct models" and "22 configurations" figure in sections 4b, 4c, 5, 6b,
   7 and the abstract has to be recomputed, not edited by hand.

## Analysis to redo after the sweep

All of these read the results directories, so they must be re-run, not patched:

- `scripts/rank_agreement.py` and `scripts/cross_formulation_agreement.py`
  (both full and `--exclude-models` restricted variants)
- `scripts/task_rank_divergence.py`
- `scripts/clustering_stability_v2.py` for the new models, then
  `cross_formulation_agreement.py --clustering-medians`
- `scripts/verify_paper_numbers.py` last, as the gate

## Traps that already cost time here

- **Three scripts need the weight-duplicate exclusion**, not one:
  `rank_agreement.py`, `masking_ablation.py`, `cross_formulation_agreement.py`.
  Fixing one and assuming the rest is how the duplicate survived.
- **`gte_modernbert_2k` and `gte_modernbert_8k` share one weights file.** They
  are not the ogbert case: different truncation gives different embeddings, so
  they are legitimately different systems. But a weight-hash dedup will collapse
  them. Use the 2k entry in the main model set and treat 8k as the truncation
  experiment; do not let both into one rank correlation.
- **The masking arms are a closed experiment** over the original 21-model set.
  `check_evaluation.py` will report them biased once new models exist. Gate
  them with `--exclude-models "finetune,granite_small_r2,gte_modernbert,embeddinggemma,qwen3_embed"`,
  which passes on all six arms.
- **`create_model` used to discard `max_seq_length` and prompts.** Fixed in
  `f3eff8b`. Every earlier result ran at whatever context its own config
  defaulted to; the old models all sit at or below 512, so they were
  effectively matched, but that was luck rather than design.
- **No model gets its model-card prompt**, old or new. Uniform handicap, worth
  stating in the paper.
- **Process patterns match the shell that runs them.** `pgrep -f
  'bash /tmp/.*driver\.sh'` killed this session's own shell. Enumerate `/proc`
  and exclude your own ancestry. See checklist F8.
- **Sub-command arguments need their flag.** Task names passed positionally
  make argparse exit 2 and evaluate nothing. See checklist F9.

## Outstanding from the adversarial review

Three priority fixes are done (restricted range, transfer matrix rebuild,
false sentences). Remaining lower-rated findings, not yet addressed:

- Section 6 does not say how little of the SHELF-Gutenberg gap masking
  explains (roughly a tenth).
- Table 5's caption describes one model intersection where there are three.
- The abstract's first sentence says all 62,899 documents come from
  specifications; 42,532 do not, as the next sentence says.
- The clustering stability gate was computed including the weight duplicate.
- Two claims lack a locatable artifact: the 0.5008 rebuild figure, and the
  jitter sensitivity covering only four of the reported correlations.
- The prompt-study intervals are not document-clustered, and its judge is a
  single uncalibrated model, which the paper does not say.

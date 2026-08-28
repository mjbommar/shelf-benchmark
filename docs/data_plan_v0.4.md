# SHELF v0.4 — Corpus Modernization & Enhancement Plan

**Scope**: the evaluation *data* only — corpus composition, label granularity, controls, QC,
and versioning. Harness bugs (SHELF-score renormalization, pair-metric floor, truncation and
prompt handling in the sentence-transformer adapter) and paper framing are out of scope here
and not yet written up.

**Date**: 2026-08-26
**Baseline**: v0.3.1 (`mjbommar/SHELF`, 42,532 docs)
**Budget target**: $20–50 per generator model

---

## 1. Evidence base

Everything below was measured against the published v0.3.1 dataset on 2026-08-26, not assumed.

| # | Measurement | Value |
|---|---|---|
| 1 | Corpus generator share | **94.1% GPT-5.x** (gpt-5.2 30,028 = 70.6%; gpt-5.1 10,006 = 23.5%) |
| 2 | Minority generators | 5 of 9 models contribute ~100 docs each (0.24% apiece) |
| 3 | Generator attribution from text | **93.1% accuracy** (TF-IDF+LR, 4 generators balanced at 999, chance 25%); gpt-5.1 vs gpt-5.2 alone ≈ 89% |
| 4 | Generator × label confounding | **None** — Cramér's V ≤ 0.023 for lcc_code / lcgft_category / register / target_length, all p > 0.13 |
| 5 | LCC task headroom | TF-IDF+LR = **0.892 macro-F1**; per-length: 22-word docs **0.754**, 1,899-word docs 0.918 |
| 6 | Corpus integrity | 0 exact-duplicate texts, 0 duplicate spec tuples, 0 cross-split spec collisions |
| 7 | Short-document mass | 12.9% of docs under 50 words (micro 1,988 + tiny 3,511) |
| 8 | Length distribution | mean 637 words, median 322, p90 1,854, p99 3,539, max 6,203 |
| 9 | Taxonomy utilization | topics **112 of 2,000** (5.6%); geographic **44 of 500** (8.8%); forms **133 of 554** (24%) |
| 10 | Topic label quality | The 112 topics in use are top-level abstractions (Art, Music, History, Culture, Religion…), each ~3,400 docs at near-uniform frequency |
| 11 | Unused label fields | `topics` (multi-label, mean 2.49/doc, uniform 1–4), `audience` (24 values, 70.2% coverage), `lcgft_form` (133 values, retrieval-only) |
| 12 | Opening diversity | 25,592 distinct opening trigrams; top-100 = 13.0% of corpus; max single opener 0.59% |
| 13 | LLM slop markers | "delve" 0.2%, "tapestry" 0.5%, "In conclusion" 0.2%, "Moreover" 0.4% of docs |
| 14 | Style fingerprints | em-dash in **71.6%** of docs; markdown headers 41.3%, bullets 34.1%, bold 59.9%, tables 2.5% |
| 15 | Markdown by genre | Models do adapt (headers 63.5% instructional vs 18.3% literature), but literature/creative-nonfiction markdown rates remain unrealistically high |

Three of these deserve emphasis.

**Finding 3 + 4 together.** Each generator leaves a strong lexical fingerprint, but that
fingerprint is *orthogonal to the labels* in v0.3.1 — the sampling was clean. So there is no
active confound to fix. What there *is* is a missing capability: at 94% one family, the corpus
cannot answer "does an embedder generalize across generators?" Balancing turns a latent risk
into a first-class evaluation axis.

**Finding 5.** A pure bag-of-words model classifies a **22-word** document into its LC class
77% of the time, and only 17 F1 points separate that from a 1,900-word document. The primary
task is close to lexically saturated: it rewards domain-vocabulary matching, not document
understanding. This is the deepest data-level problem, and no amount of newer generator models
fixes it. Difficulty has to be built in deliberately.

**Findings 12–14 together.** Prompt quality is *better* than expected — openings are genuinely
diverse and slop markers are rare, so blanket prompt randomization would solve a problem that
does not exist. The real artifacts are narrower and structural: a globally mandated markdown
output format, and an em-dash rate of 71.6% that almost certainly feeds the 93% generator
attribution in Finding 3. Fix those two specifically (§4), don't randomize indiscriminately.

---

## 2. Problems, ranked

| ID | Problem | Addressed by |
|---|---|---|
| P1 | Generator monoculture (94.1% GPT-5.x) | Phase 1 |
| P2 | Primary task lexically saturated | Phase 2, Phase 3 |
| P3 | Label granularity too coarse (21 classes) | Phase 2 |
| P4 | Pair items randomly constructed → no hard negatives | Phase 3 |
| P5 | No held-out slice for the circularity objection | Phase 4 |
| P6 | No human-validated subset or human ceiling | Phase 5 |
| P7 | Micro/tiny stratum (12.9%) may be unlearnable noise for some tasks | Phase 1 (length rebalance) |
| P8 | Taxonomy under-utilization (5.6% of topics, 8.8% of geography, 24% of forms) | Phase 0 (§4.1) |
| P9 | Three well-formed label fields sit unused as tasks | §11 |
| P10 | Single hidden system prompt behind all 42,532 docs; mandated markdown; em-dash tell | Phase 0 (§4.2) |
| P11 | English-only corpus while most modern embedders are multilingual | Stated limitation; v0.5 decision (§11.7) |
| P12 | Single corpus / single spec draw — no replicate, so corpus idiosyncrasy is unestimable | Phase 1 spec blocks |
| P13 | No natural, human-authored transfer slice; all holdouts are synthetic | Phase 6 |
| P14 | Composite score treats every item as equally precise; no difficulty or relation strata | §11.4–11.5, §13 |

**Already sound, do not touch**: no duplicates, no split leakage, no generator-label
confounding, title-leakage already excluded from the `text` field.

---

## 3. Design principles

1. **Freeze v0.3.1.** It stays published and evaluable so existing baselines remain comparable.
   v0.4 is additive plus new slices, never a silent replacement.
2. **Generator is a controlled factor, not incidental.** Equal N per generator, and the *same
   spec block* given to every generator (see §5).
3. **Difficulty by construction.** Finer labels and lexically-controlled items, not just more
   documents.
4. **Every document carries provenance and QC flags.** Pass rates per generator are themselves
   a reportable result.
5. **Budget-bounded and resumable.** Hard per-model spend caps, spec-hash-keyed queue, no
   re-spend on restart.
6. **Replicate, don't just balance.** Independent spec blocks and an external transfer slice, so
   corpus-level and generation-process-level idiosyncrasy are estimable rather than assumed away.
7. **No composite without strata.** Every headline number is accompanied by per-stratum results
   (difficulty, relation, length, generator family) and an uncertainty-aware aggregate.
8. **Challenge slices are diagnostics, not training data.** The adversarial and transfer slices
   must never be optimized against.

---

## 4. Phase 0 — Generation infrastructure (no token spend)

Prerequisites before any money is spent.

- **Add OpenRouter and xAI backends** to `src/shelf/llm/backends.py`, which today has only
  OpenAI/Anthropic/Gemini. Both are OpenAI-chat-compatible, so a `base_url`-parameterized
  subclass covers them. OpenRouter is what unlocks the open-weight families.
- **Batch API support** — 50% discount across the board. Anthropic Message Batches, OpenAI
  Batch, Gemini batch, and OpenRouter's `:batch` model ids. All Phase 1 costs below assume batch
  pricing where available.
  > **Correction (verified 2026-08-26)**: the `:batch` suffix is *not* just a pricing variant.
  > A sync call to `google/gemini-3.7-flash:batch` returns
  > `{"error":{"message":"This model is only available through the Batch API. Use the
  > /api/beta/batches endpoint instead.","code":404}}`. Batch requires the real submit/poll/collect
  > flow against `POST /api/beta/batches`, not a model-id change. Implemented in
  > `OpenRouterBackend`.
- **Live pricing.** Replace the hardcoded `MODEL_PRICING` table in
  `scripts/generate_documents.py` (last updated December 2025, now 8 months stale) with a
  fetch from `https://openrouter.ai/api/v1/models`, cached per run and recorded in the manifest.
- **Cost ledger with a hard cap.** Per-request token accounting to
  `data/artifacts/cost_ledger.jsonl`; abort the run for a model when its cap is hit. Non-negotiable
  given per-model budgets.
- **Resumable spec queue** keyed by spec hash, so a crash or a rate-limit storm doesn't re-spend.
- **Reasoning-token control.** Set minimal/no reasoning effort where supported and cap
  `max_output_tokens`. Thinking tokens bill as output and are the main cost-overrun risk; the
  estimates below assume ~2,400 output tokens for thinking models, ~900 for non-thinking.
- **Record the serving provider.** OpenRouter may route one model id to different backends with
  different quantization. Capture the top-level `provider` field from each response into the
  artifact. **Confirmed empirically**: two calls to `meta-llama/llama-4-maverick` minutes apart
  were served by *DeepInfra* and then *DigitalOcean*. Routing variance is real, not theoretical;
  pin it with `ProviderRouting.pin(...)` where reproducibility matters.
- **Reasoning control is per-model, not per-provider** (verified). `grok-4.3` accepts
  `reasoning_effort:"none"`; `grok-4.6` rejects that value; non-reasoning models reject the
  parameter entirely. Backends must detect a reasoning-parameter rejection and retry once without
  it, or a single unsupported model kills a 1,500-document run.

### 4.1 Coverage expansion (sampler change, no extra token cost)

The cheapest available difficulty increase. Every expansion below is a change to
`src/shelf/sampler/dimensions.py` and the taxonomy file it loads — the generation cost per
document is unchanged.

| Dimension | v0.3.1 | v0.4 target | Source file (already on disk) |
|---|---|---|---|
| topics | 112 | **500–1,000** | `lcsh_topical_top500.json` / `top1000.json` |
| geographic | 44 | **200+** | `lcsh_geo_top500.json` |
| lcgft_form | 133 | **300+** | `lcgft.json` (554 available) |
| lcc_subclass | — | 60–100 | `lcc_subclass_top100.json` (Phase 2) |

Topic expansion matters most. The current 112 are top-level LCSH abstractions at near-uniform
frequency — "Art" as a topic label is close to contentless, which is why topic-derived tasks
are weak. Moving to top-500 or top-1,000 gives specific headings with real discriminative
content.

> **The LCGFT frequency table is not purely LCGFT.** Enrichment could not match 103 of the 554
> "forms" to any LCGFT record because they are not LCGFT terms: form subdivisions (`Juvenile
> works`, `Sources`, `Tables`), French RVM headings (`Statistiques`, `Rapports techniques`), MeSH
> publication types (`Practice Guideline`), and outright junk (`Census, 1990`, `United States`).
> Three entries in the LCC subclass table — `IN`, `PAR`, `NOT` — are extraction artifacts, not LC
> codes. Clean these before expanding the form pool, or the expansion imports noise as labels.

**Do not add** `sudoc_agency_top50` or `corp_names_top100`. Those are provenance metadata, not
content dimensions, and would add label noise rather than coverage.

#### 4.1.1 The real blocker is descriptions, not pool size *(measured 2026-08-26)*

Expanding the pools is genuinely free. Generating *good documents* from the expanded pools is
not, and this correction matters more than the expansion itself.

All 14 taxonomy files under `data/taxonomies/` — 4,829 labels — have **zero** populated
`description`, `uri`, `alt_labels`, `broader`, or `narrower` fields. They are pure MARC frequency
rank tables, not the id.loc.gov linked-data extraction the field names imply. The semantic
descriptions the generation prompt actually uses are small hardcoded dicts in
`sampler/generator.py`:

| Source | Entries | Coverage of the expansion target |
|---|---|---|
| `LCC_SEMANTIC_DESCRIPTIONS` | 21 | **16 of 100** LCC subclasses |
| `LCGFT_FORM_DESCRIPTIONS` | 29 | **17 of 554** ranked forms |
| `LCGFT_CATEGORY_DESCRIPTIONS` | 14 | 14 categories (complete) |
| `lcgft_hierarchy.json` | 439 terms / 23 categories | **132 of 554** ranked forms (23.8%) |

Consequences:

- **Phase 2 is blocked** until ~84 LCC subclass descriptions exist. `_get_domain_description`
  falls back to the bare code name, which is precisely the self-labeling the prompt forbids.
- **Form expansion degrades quietly.** `_get_form_description` falls back to `form.lower()`, so
  a prompt for one of the ~283 undescribed forms reads `style: abstracts` instead of a scope
  note — weaker conditioning and a self-labeling risk.
- The hierarchy file uses **23** categories while the corpus uses **14**, so it cannot be used
  directly to assign `lcgft_category` to newly added forms.

**Fix (implemented, with a caveat)**: `scripts/enrich_taxonomies.py` now pulls real LC data —
bulk SKOS/RDF for LCSH and LCGFT, the id.loc.gov classification API for LCC. Coverage went from
16/100 to **96/100** LCC subclasses, 17/554 to **412/554** LCGFT forms, and **1,931/2,000** LCSH
topics.

> **But most of it is derived, not verbatim.** Only ~19% of the new descriptions are genuine LC
> scope notes; the rest are templated from LC broader/narrower hierarchies. And those templates
> **name the taxonomy**: measured on the LCGFT export, **25.7% of descriptions contain an LCGFT
> category verbatim** — "Maps" is described as *"a kind of Cartographic materials, Informational
> works, Visual works"*, and `lcgft_category` is a 14-way prediction target. Hierarchy-derived
> descriptions leak at **38.6%**; real scope notes leak at only 7.2%.
>
> Feeding these into a prompt raw would reintroduce exactly the label leakage
> `build_generation_prompt` exists to prevent. `shelf.sampler.leakage` now guards it:
> `find_leaked_labels` detects it, `sanitize_description` strips it. Applied to the LCGFT export
> it takes leakage from **114/451 (25.3%) to 0**, with 8.0% of descriptions reduced to empty
> (the caller must fall back for those). The same module implements QC gate **G4** for generated
> documents, and correctly does *not* flag legitimate domain vocabulary.

**Wired in** via `shelf.sampler.enriched.EnrichedDescriptions`, which loads every enriched
export, sanitizes each description against the LCGFT categories **and the label's own name**, and
exposes it to `build_generation_prompt` / `build_title_prompt` through an optional `enriched=`
argument. Omitting it reproduces v0.3.1 byte-for-byte, which the prompt-variant tests verify.

Resolution order is hand-written description → sanitized LC scope note → category fallback → bare
label, so the curated text (written specifically to describe a form without naming it) always
wins and LC text only extends coverage.

Effect on a form with no hand-written entry:

```
without enrichment:  style: reference materials, guides, handbooks, reports
with enrichment:     style: documents representing the arguments of one or more parties and
                            other documents submitted to, or generated by, a court in a
                            particular case
```

Post-sanitization audit over the real export: **zero** exact label or category leaks. A residual
**20.2%** of descriptions contain a morphological relative of their own label ("Tourist maps" →
"maps designed for tourists"). That is left alone deliberately — `GENERATION_INSTRUCTIONS`
already permits domain vocabulary and forbids only *announcing* the class, so stemming it away
would strip the conditioning signal itself.

Use `verbatim_only=True` where conditioning quality matters more than coverage: derived text
ranges from excellent ("Legal briefs") to useless ("Abstracts" → "Derivative works"). Verbatim
keeps 190 forms / 382 topics / 18 LCC subclasses; the full set keeps 461 / 1,983 / 96. `scripts/loc_taxonomies.py` already
documents that LCSH, LCGFT, and LCDGT are available as bulk SKOS/RDF from id.loc.gov; only LCC
is not (it needs the LC Classification Outline, which is scrapeable). Authoring these from the
authoritative source is a prerequisite for §4.1 and Phase 2, not an optional polish step.

**Non-prose material.** Only 2.5% of documents contain tables. LCGFT has inherently tabular
forms (statistics, indexes, registers, catalogs) that current embedders handle poorly. Sampling
these deliberately opens genuinely untested territory at no extra cost.

### 4.2 Prompt variants (make the prompt a controlled factor)

Findings 12–13 show openings are already diverse and slop is rare, so the fix is targeted, not
a blanket randomization.

1. **Drop the globally mandated markdown instruction.** The system prompt currently ends with
   `Line 3+: markdown body`, and it shows: 41.3% of docs carry headers, 59.9% use bold. Models
   partially adapt (63.5% headers for instructional works vs 18.3% for literature), but 18.3%
   of *Literature* having markdown headers and 24.5% of *creative nonfiction* using bold is not
   how those documents look. Make output format **form-conditional** instead of global.
2. **Author 3–5 system-prompt variants** and sample among them, storing `prompt_variant_id` on
   every document. This is the principled version of "randomize more": prompt effect becomes
   measurable rather than a hidden constant baked into all 42,532 existing documents, and a
   variant that produces degenerate output can be identified and excluded after the fact.
3. **Do not add "avoid em-dashes"** to any variant. The 71.6% em-dash rate is a real fingerprint,
   but instructing against it substitutes one artifact for another. Let variant diversity and
   generator diversity reduce it, and measure the result.
4. **Keep the SHOW-DON'T-TELL block in every variant.** It is load-bearing — it is what keeps
   labels non-trivial by suppressing self-announcing text. Verify with QC gate G4 rather than
   loosening it.

Prompt variants are introduced **in v0.4 slices only**. v0.3.1's single-prompt corpus stays
intact so existing baselines remain comparable.

#### 4.2.1 Validated on real generations *(2026-08-26)*

**Implemented and measured.** 5 prompt variants (`v0.3.1` frozen + `direct`, `practitioner`,
`editorial`, `archival`), all retaining the SHOW-DON'T-TELL block verbatim, plus a
form-conditional `OutputFormat` (9 formats) replacing the global markdown mandate.

A/B on **300 real generations** (`meta-llama/llama-4-maverick`, 60 docs per arm, LONG length,
restricted to forms whose target format is *not* markdown):

| Arm | n | headers | bullets | bold | **any markdown** | mean words |
|---|---|---|---|---|---|---|
| `v0.3.1` | 60 | 23.3% | 10.0% | 11.7% | **26.7%** | 627 |
| `v0.4-direct` | 60 | 0.0% | 0.0% | 0.0% | **0.0%** | 533 |
| `v0.4-practitioner` | 60 | 0.0% | 1.7% | 0.0% | **1.7%** | 534 |
| `v0.4-editorial` | 60 | 0.0% | 1.7% | 0.0% | **1.7%** | 530 |
| `v0.4-archival` | 60 | 0.0% | 1.7% | 0.0% | **1.7%** | 541 |

Pooled: v0.3.1 16/60 vs. new variants 3/240. **Fisher exact p < 0.00001, odds ratio 28.7.**

Spurious markdown on documents that should not have it drops from 26.7% to ~1.3%. Note the side
effect: the new variants produce ~15% shorter documents at the same length target, so length
weights may need recalibrating for v0.4.

An earlier n=10-per-arm version of this test was **not** conclusive — a single document carried
the entire difference. It is recorded here only because the powered rerun replaced it.

---

## 5. Phase 1 — Generator-balanced core *(main spend)*

**15 generators × 1,500 documents = 22,500 documents.**

The key design choice: **draw the specs once and give the identical spec block to every
generator.** Generator then becomes the only varying factor, which buys, at no extra cost:

- paired cross-generator comparison (15 realizations of every spec)
- a natural "same content, different generator" pair task
- direct measurement of generator effect on every downstream metric
- a train-on-family-A / test-on-family-B generalization split

> **Critical consequence**: 15 realizations of one spec are near-duplicates by construction.
> **Split by `spec_id`, never by document id.** All realizations of a spec must land in the
> same split, or you manufacture exactly the train/test leakage v0.3.1 currently avoids.

**Use three independent spec blocks of 500, not one block of 1,500.** A single shared block gives
maximum paired control but leaves block-level idiosyncrasy unestimable — one spec draw can reward
its own sampling accident the same way one corpus can reward its own writing style. Three blocks
drawn under different sampler seeds preserve all 1,500 paired comparisons per generator *and* add
a between-block variance estimate, at identical cost. Record `spec_block_id` and report
between-block variance alongside every headline number.

This is the corpus-level replicate that a single shared corpus otherwise cannot provide.

**Implemented**: `shelf.sampler.specs` provides `DocumentSpec` (content-addressed, frozen),
`SpecBlock` (immutable, checksummed), `draw_spec_blocks()`, JSONL persistence with
tamper-and-truncation detection, and `assign_blocks_to_splits()` which splits **within** each
block so split is never confounded with draw. Verified on real draws: three blocks of 200 share
zero spec ids, are deterministic under seed, and an edited or truncated block file is rejected on
load rather than silently changing the split.

### Roster and cost (batch pricing where available, 1,500 docs each)

Costs assume 550 input tokens and 2,400 output tokens for reasoning models / 900 for
non-reasoning, priced against the live OpenRouter catalogue on 2026-08-26 and **cross-checked
against each provider's own pricing page**. Where the two disagree the *higher* figure is used.

| Generator | Role | Cost |
|---|---|---|
| `anthropic/claude-opus-5:batch` | Anthropic flagship | $47.06 |
| `openai/gpt-5.6-sol:batch` | OpenAI flagship | $37.65 † |
| `x-ai/grok-4.6` | xAI flagship | $23.25 |
| `google/gemini-3.1-pro-preview:batch` | Google flagship | $22.43 |
| `anthropic/claude-sonnet-5:batch` | Anthropic mid | $18.82 |
| `moonshotai/kimi-k2.6` | Moonshot (OW) | $15.18 |
| `mistralai/mistral-medium-3-5` | Mistral (OW) | $11.36 |
| `qwen/qwen3.5-397b-a17b` | Qwen (OW) | $8.75 |
| `z-ai/glm-5` | Zhipu (OW) | $7.41 |
| `google/gemini-3.7-flash:batch` | Google small | $7.06 † |
| `deepseek/deepseek-v4-pro` | DeepSeek (OW) | $6.98 |
| `minimax/minimax-m3` | MiniMax (OW) | $4.57 |
| `anthropic/claude-haiku-4.5:batch` | Anthropic small | $3.79 |
| `openai/gpt-5.6-luna:batch` | OpenAI small | $2.24 |
| `meta-llama/llama-4-maverick` | Meta (OW) | $1.25 |
| **Total** | **15 generators, 22,500 docs** | **$217.80** |

**Every model stays under the $50 ceiling**, with Opus 5 highest at $47.06.

#### Reconciled against the real pricing tool *(2026-08-27)*

`scripts/generate_documents.py --dry-run` prices the roster against the live catalogue. Running
the full 15-model, 22,500-document Phase 1 at `--reasoning-effort none`:

| | Total | Max single model |
|---|---|---|
| Plan estimate above (2,400 output tokens, reasoning **on**) | $217.80 | $47.06 (Opus 5) |
| **Tool projection** (~900 output tokens, reasoning **off**) | **$89.33** | $17.21 (Opus 5) |
| **Tool worst case** (what `BudgetGuard` actually reserves) | **$170.61** | $32.88 (Opus 5) |

The entire gap is the output-token assumption. Reasoning tokens bill as output, so disabling
reasoning roughly halves the cost — and every model lands inside the $50 cap with room to spare.

> **This is a quality trade, not free money.** Frontier models with thinking disabled may write
> worse documents, and nothing here measures that. The prompt-variant A/B (§4.2.1) used a
> non-reasoning model and says nothing about it either. Before banking the $89 figure, run a
> small reasoning-on vs reasoning-off comparison on one or two flagship models and check document
> quality against the QC gates. Budget with the worst-case column, not the projected one.

† **Unresolved 2× pricing uncertainty on two roster models.** OpenRouter and the provider's own
page disagree by exactly a factor of two:

| Model | OpenRouter | Provider page |
|---|---|---|
| `openai/gpt-5.6-sol` | $2.00 / $10.00 | $4.00 / $20.00 (platform.openai.com) |
| `google/gemini-3.7-flash` | $0.375 / $1.875 | $0.75 / $3.75 (ai.google.dev, through 2026-12-31) |

Both were confirmed by reading each source directly; what was *not* confirmed is which one is
actually billed. The discrepancy is inconsistent within each family — `gpt-5.6-luna` and
`gpt-5.6-terra` match OpenAI exactly, and `gemini-3.6-flash` matches Google exactly — so it is
not a systematic OpenRouter markdown. Budget with the conservative figure above, pass
`VERIFIED_NATIVE_PRICES` as `overrides=`, and let the ledger record actuals. Both models remain
comfortably inside budget either way.

Verify both prices again immediately before spending: Google's rate rises to $1.50 / $7.50 on
2027-01-01, and OpenAI describes Sol's pricing as promotional "at least through 2026-11-21".

### Length rebalance (fixes P7)

Move from the current weighted-toward-medium distribution to near-uniform across
brief/short/medium/long/very_long. **Drop the `micro` bucket** (10–25 words) from the core
slice — a 15-word snippet is not a meaningful classification instance — or retain it only as an
explicitly labeled diagnostic stratum excluded from headline metrics. Keep `tiny` and
`extended` as the tails.

---

## 6. Phase 2 — Difficulty tier: LCC subclass

`data/taxonomies/lcc_subclass_top100.json` is already extracted and currently **unused**. It
gives 100 real LC subclasses (QA vs QC vs QH, KF vs KJ, etc.), which demands within-domain
discrimination that domain vocabulary alone cannot supply — the direct antidote to P2/P3.

- **Sample uniformly over ~60–100 subclasses**, not by LC frequency. The raw frequencies are
  wildly skewed (KF alone is 122,484 of the reference corpus); uniform sampling is required, as
  it already is for top-level LCC.
- Target **6,000 documents** across 3 strong, cheap generators
  (`deepseek/deepseek-v4-pro`, `qwen/qwen3.5-397b-a17b`, `z-ai/glm-5`) ≈ **$38**.
- Expected outcome: TF-IDF degrades sharply where it currently sits at 0.892. If it does not,
  that is itself a finding worth reporting.
- Needs subclass descriptions for prompting — **now resolved**: 80/80 of the usable pool have a
  sanitized description via `EnrichedDescriptions.for_lcc_subclass`.

### 6.1 Plumbing built — and a pilot says the premise is shaky *(2026-08-27)*

`LCCSubclassSampler` yields **80 usable subclasses** (the `IN`/`PAR`/`NOT` extraction artifacts
and 16 bare main-class letters excluded, `JX` dropped for having no resolvable description),
sampled **uniformly**: 80 distinct codes at 1.59% max share, where frequency weighting would put
**59.9% on `KF` alone**. Subclass is carried through `DocumentSpec` (hashed into `spec_id` only
when set, so existing blocks still verify), into the prompt via the enriched description, into
the dataset schema, and registered as `lcc_subclass_classification`.

**Then a 240-document pilot tested whether the tier does what it is for.** 20 subclasses
concentrated in only 5 parents (T, H, Q, V, U — four each), which is the hardest within-domain
case. TF-IDF + logistic regression:

| Task | Classes | macro-F1 |
|---|---|---|
| parent class | 5 | 0.9452 |
| **subclass** | 20 | **0.7585** |
| within parent Q (QA/QB/QD/QH) | 4 | **0.9286** |
| within parent V | 4 | 0.8651 |
| within parent T | 4 | 0.7952 |
| within parent H | 4 | 0.7833 |
| within parent U (UA/UB/UF/UG) | 4 | 0.6333 |

**The mechanism is measurable**: **36.1% of each subclass's description vocabulary appears
verbatim in its own documents** (range 15.8%–58.3%). The caption-hierarchy descriptions are
keyword lists — "geomagnetism" is in QC's, "cytology" in QH's — and generators copy them
straight through. The shortcut is re-lexicalized at finer grain rather than defeated.

> **State the caveat plainly: this is not a clean comparison to LCC-21's 0.892.** That figure came
> from 25,518 training documents; this pilot had ~8 per class. More data almost certainly pushes
> the subclass number **up**, not down — so 0.7585 is a *floor* on how lexical the tier is, and
> the honest reading is that Phase 2 at full scale would land closer to LCC-21 than the plan
> assumed. Within-parent Q at 0.9286 on eight examples per class is the clearest warning.

#### 6.2 The prose rewrite was tried and it failed *(2026-08-27)*

The obvious fix was to rewrite the caption hierarchies into prose with no enumerable term list.
`scripts/rewrite_subclass_descriptions.py` did that for all 96 subclasses, cutting mean
source-vocabulary overlap from the caption baseline to **10.6%**, and the output reads well —
QA/QB/QD/QH become clearly distinct sentences about what each field studies.

A controlled A/B settles it. Same **20 pinned subclasses**, same seeds, same document specs,
same generator (`claude-haiku-4.5`), 240 documents per arm, both arms 240/240:

| Arm | description copy rate | TF-IDF macro-F1 | MiniLM macro-F1 |
|---|---|---|---|
| caption hierarchy | 29.3% | **0.6393** | 0.8152 |
| prose rewrite | **35.5%** | **0.7240** | 0.8469 |

**The rewrite made both metrics worse**: copy rate up 6.2pp, TF-IDF up 0.0848. The likely reason
is the opposite of the intuition behind it — a comma-separated term list is awkward to absorb
into prose, whereas a fluent sentence hands the generator ready-made phrases. **Option 2 is
rejected on evidence.** Do not ship the prose descriptions; keep the caption hierarchies.

**And the tier is better than the earlier pilot suggested.** With captions, TF-IDF reaches 0.6393
while MiniLM reaches 0.8152 — a **0.18 gap**, meaning the subclass task carries real non-lexical
signal. Against LCC-21's TF-IDF 0.892, a lexical baseline loses ~0.25 here. The first pilot's
0.7585 came from a *different generator* (`llama-4-maverick`), where TF-IDF and MiniLM landed
within a point of each other.

> **The generator changes the answer.** llama-4-maverick produced documents a bag-of-words model
> classified as well as a neural one; claude-haiku-4.5 produced documents where the neural model
> leads by 0.18 on identical labels and specs. Any claim about how lexical a SHELF task is has to
> name the generator that produced the documents — which is itself an argument for the
> generator-balanced design in Phase 1.

**Decision: proceed with Phase 2 using the original caption descriptions.** Report the tier as
finer-grained and genuinely harder for lexical baselines, with the copy rate and this A/B
published alongside. Do not describe it as eliminating lexical saturation.

Caveat carried: 240 documents, ~8 training examples per class. The A/B is internally valid
(identical subclasses and seeds, only descriptions differ), but the absolute numbers will move at
Phase 2 scale.

---

## 7. Phase 3 — Lexically-controlled hard items

Attacks the vocabulary shortcut head-on and gives the pair tasks real dynamic range.

**Generated minimal pairs** (~2,000 pairs / 4,000 docs, ≈ **$45**): hold topics, audience,
register, and length constant; vary *exactly one* factor —
- same subject, different form (jokes vs. lecture about the same topic)
- same form, different subject
- same topics, different LCC class

**Mined hard negatives** (free, no generation): mine the existing corpus for
maximum-lexical-overlap / different-label pairs and minimum-overlap / same-label pairs. This
costs nothing and should be built first — it may supply enough signal that fewer generated
pairs are needed.

Both feed new pair-classification and retrieval item sets that a lexical baseline cannot solve
by vocabulary matching alone.

---

## 8. Phase 4 — Held-out contamination probe

**~1,000 documents**, from the newest available generators, **held out of train entirely** and
published as a separate config. ≈ **$15**.

Purpose: answer the circularity objection quantitatively rather than by argument. If embedder
performance on the holdout slice matches performance on the core slice, generator identity
demonstrably isn't driving results. Finding 3 (93% attribution accuracy) makes this test
necessary rather than optional.

---

## 9. Phase 5 — Human validation

**300–500 documents**, dual-coded for LCC class and LCGFT category by someone with cataloging
familiarity. Costs time, not tokens.

Delivers inter-annotator agreement and a **human ceiling** — without which "model X reaches
0.89" has no interpretable upper bound. This is the one component that cannot be generated.

**Status: infrastructure complete, labeling deferred** *(2026-08-27)*. The annotation work is
explicitly out of scope for this phase. Everything around it is built and verified, so the round
can start whenever two cataloguers are available:

- `scripts/build_annotation_kit.py` → `data/annotation/v0.4/`: **399 documents**, all 21 LCC
  classes at exactly 19 each, 13 generators represented, gold labels withheld into a separate
  file, two independent coding sheets, and instructions generated from the live registry rather
  than hand-copied.
- `scripts/score_annotations.py`: Cohen's κ between coders, each coder against the generated
  label, and the derived human ceiling. Smoke-tested against simulated coders at known accuracy
  (0.75 / 0.70 → κ=0.490, ceiling 0.722).

Two design choices worth keeping when the round runs: documents under 60 words are excluded,
because a 15-word document is not classifiable by anyone and including them would measure the
corpus's short tail rather than coder agreement; and coders are asked to force a choice and flag
`uncertain` rather than leave a blank, since a forced choice plus a flag carries more information
than a gap.

Until the round runs, **model scores have no human ceiling to be reported against**, and that
limitation should be stated wherever v0.4 results are published.

---

## 10. Phase 6 — Natural-data transfer slice

**The largest gap in this plan before now.** Phase 4's holdout is still synthetic — newer
generators, same generation process. It cannot answer whether SHELF performance transfers to
documents no LLM wrote.

Build a **natural, human-authored, human-catalogued transfer slice** with real LC labels. The
tooling largely exists: `scripts/analyze_marc_frequencies.py` already parses MARC 050 (LCC),
650 (LCSH), and 655 (LCGFT) via `pymarc`.

**Candidate source — Project Gutenberg.** Public-domain full text with human-assigned LC classes
in the RDF metadata (~75k works). No licensing friction, no scraping, labels assigned by
cataloguers rather than by us.

Known and unavoidable skews, which must be **reported, not hidden**:

- heavily weighted to P (Language and Literature) and pre-1929 works
- book-length documents, unlike SHELF's 25–4,000 word range
- English-dominant

Mitigation: stratify and subsample toward the flattest achievable LCC distribution, chunk long
works into passages with document-level labels, and publish the realized distribution alongside
results. This is a **transfer slice, not a balanced corpus** — that is an acceptable role for it.

> **Critical caveat**: Project Gutenberg is in the pretraining data of essentially every model
> evaluated. Results on this slice are **not** contamination-free and must never be presented as
> such. That is precisely what makes the comparison informative: SHELF is the clean-synthetic
> condition, Gutenberg is the contaminated-natural condition, and the *gap between them* is the
> measurement. Report them as two conditions, never pooled.

Supplementary sources worth evaluating: US government documents (SuDoc-classified, public
domain, full text via govinfo) would add non-literary subject coverage where Gutenberg is
weakest.

**Cost: $0 in tokens.** This is engineering and cataloguing work.

**Framing consequence.** With this slice in place, the defensible claim becomes: *SHELF is the
controlled capability test; the transfer suite is the external check.* SHELF alone does not
establish general-purpose embedding quality, and should not be presented as though it does.

### 10.1 Built, and the transfer gap is large *(measured 2026-08-27)*

**3,016 passages from 1,008 Project Gutenberg works, 48 works in each of all 21 LCC classes.**
The plan predicted an unavoidably P-heavy sample; flattening by *works* rather than passages beat
that — normalized entropy **1.0000**, total variation distance from SHELF's test split **0.0100**.

The flat headline hides real skew and the report says so: the sampling fraction runs **0.15% for
P (48 of 31,524 works) to 46.60% for V (48 of 103)** — a 306× range. V and K are near-exhausted
draws with far less within-class variety than equal N implies.

**The transfer result** (LCC classification, 21 classes, TF-IDF + logistic regression, macro-F1):

| Train → Test | macro-F1 | Accuracy |
|---|---|---|
| SHELF → SHELF | **0.8932** | 0.8927 |
| **SHELF → Gutenberg** | **0.3010** | 0.2948 |
| Gutenberg → Gutenberg | 0.5261 | 0.5381 |
| Gutenberg → SHELF | 0.2954 | 0.3355 |

**A model that scores 0.89 on SHELF scores 0.30 on human-written, human-catalogued text — a 66%
drop.** Three things make this more than "natural data is harder":

1. Gutenberg in-domain reaches **0.5261**, so a model fitted on natural data does **75% better**
   on natural data than one fitted on SHELF. The ceiling is not the constraint; the domain shift
   is.
2. Transfer fails **symmetrically** (0.301 out, 0.295 back). The two corpora carry genuinely
   different lexical→label mappings rather than one simply being noisier.
3. Because this baseline is TF-IDF, which has no pretraining, **Gutenberg's contamination is
   irrelevant to this particular measurement**. The gap is real domain shift, not memorization.

Per-class transfer is highly uneven: **R 0.58, E 0.52, M 0.52** (stable technical vocabularies)
against **A 0.01, C 0.05, J 0.06, Z 0.10, G 0.10** — the diffuse or historically-shifted classes.

**This is the strongest available argument for the plan's positioning.** Strong SHELF performance
substantially overstates ability on real bibliographic material, and any claim about
general-purpose document understanding has to be made against the transfer slice, not against
SHELF alone.

Caveats to carry with the number: it is a lexical baseline and a neural embedder may behave
differently; Gutenberg skews pre-1930 and book-length, so era and genre shift are confounded with
natural-vs-synthetic; and Gutenberg's LCC assignments carry no agency provenance in the RDF, so
whether they are LC copy cataloguing or volunteer assignment is unverified.

---

## 11. Task additions

The selection principle: LCC-21 is saturated (TF-IDF 0.892), so add tasks with **headroom**, and
prefer ones the data already supports over ones needing new generation.

### 11.1 No new documents required

| Task | Why | Depends on |
|---|---|---|
| **Form classification (133-way)** | Label exists, task doesn't. Far more headroom than 21-way LCC, and form is the dimension models demonstrably struggle with | — |
| **Multi-label topic classification** | `topics` is well-formed (mean 2.49/doc, uniform 1–4) and completely unused; SHELF currently has no multi-label task at all | §4.1 topic expansion to be meaningful |

**Both implemented and measured against a TF-IDF lexical baseline** *(2026-08-26)*:

| Task | Classes | TF-IDF macro-F1 | Headroom vs LCC |
|---|---|---|---|
| `lcc_classification` (control) | 21 | 0.883–0.892 | — |
| **`form_classification`** | 133 | **0.33–0.38** | **~0.50 absolute** |
| `topic_classification` (multi-label) | 112 | 0.760 macro-F1 | small on F1 |

**`form_classification` is the win.** Every one of the 133 forms is present in all three splits
(rarest test class n=22) and only one collapses to F1=0, so the gap is genuine difficulty, not
sparsity. Worst classes are semantically overlapping genres — `News articles` 0.00, `Fiction`
0.027, `Personal narratives` 0.037.

**`topic_classification` is mostly string matching, and the measurement says so.** Independently
verified: **87.2% of topic terms appear verbatim in the document body**, and **78.5% of documents
contain every one of their topics verbatim**. Compare forms at **8.7%** and LCC names at 24.1% —
that contrast is exactly why form is hard and topic is not. Report this task on **subset accuracy
(0.317)**, where the difficulty actually lives, not on F1.

> Multi-label F1 also swings violently on threshold policy: the same features give macro-F1 0.760
> with `class_weight="balanced"` and 0.379 without. Any multi-label number is meaningless without
> stating the threshold policy, which is why threshold-free ranking metrics (LRAP, mAP, coverage
> error) are reported alongside.

**Both now run end-to-end through `run_all.py`** *(closed 2026-08-27)*. Three fixes were needed:
`_create_evaluator` dispatch for `TaskType.MULTILABEL`, a second dispatch in the direct-model
evaluation path, and `multilabel` added to 22 models' `supports` lists.

Two aggregate-honesty bugs surfaced while closing it, both now fixed:

- `compute_shelf_score` emitted **0.0000** for a model with no weighted task type, which reads as
  "scored zero" rather than "not scored". Such models are now omitted from the aggregate with a
  logged note. This is the same failure mode as the renormalization bug that flattered
  partial-coverage models.
- `print_summary_table` iterated a **hardcoded** list of four task types, so `multilabel` results
  ran, were saved, and never appeared in any summary. The display now appends every task type
  actually present, so a new task cannot silently vanish.

Measured end-to-end with the config's `tfidf` model (TF-IDF + **SVD-256**):

| Task | TF-IDF+SVD-256 | Raw TF-IDF (no SVD) |
|---|---|---|
| `lcc_classification` | 0.8832 | 0.883–0.892 |
| `form_classification` | **0.2443** | 0.33–0.38 |
| `topic_classification` | 0.5429 | 0.760 |

SVD compression to 256 dimensions costs little on the 21-way task but a great deal on the
133-way and 112-label ones — a useful reminder that "TF-IDF" results depend heavily on whether
the pipeline reduces dimensionality. The multi-label head also over-predicts badly:
`label_cardinality_pred` 5.47 against a true 2.48.
| **Graded-relevance retrieval** | Not a new task — a **fix** to the existing three. The LCC hierarchy gives graded judgments free: same subclass = 3, same class = 2, same category = 1, else 0. Current binary relevance over ~5% of a 34k corpus is why NDCG is so hard to interpret | Phase 2 for subclass |

**Implemented** *(2026-08-27)*. The existing NDCG **could not** express partial credit — it took
`relevant_ids: set[str]` and hardcoded `gain = 1.0`. `ndcg_at_k` now accepts a set (bit-identical
to before) or a gain mapping, with `compute_graded_retrieval_metrics` alongside. Gains come from
`strata.classify_relation`.

| Task | BM25 | TF-IDF+SVD | MiniLM-L6 | Qwen3-Emb-0.6B |
|---|---|---|---|---|
| lcc_retrieval | 0.481 → **0.589** | 0.621 → **0.717** | 0.592 → **0.688** | 0.691 → **0.791** |
| form_retrieval | 0.070 → **0.142** | 0.101 → **0.217** | 0.071 → **0.155** | 0.093 → **0.189** |
| category_retrieval | 0.283 → **0.214** | 0.444 → **0.332** | 0.319 → **0.239** | 0.377 → **0.285** |

LCC and form rise (near misses now score); category falls, because its ideal ranking now demands
form-level precision inside the category. Model ordering is preserved on all three, and the spread
on form_retrieval widens from 0.070–0.101 to 0.142–0.217 — the discrimination the fix was for.
Graded metrics are currently **secondary**; promoting them to primary is a one-line registry
change plus two test assertions.

### 11.2 Enabled by the v0.4 corpus

| Task | Why | Depends on |
|---|---|---|
| **Hierarchical LCC (class → subclass)** | Where the real difficulty lives. Report hierarchical F1 and LCA distance, not flat accuracy | Phase 2 |
| **Cross-generator robustness** | Every task re-scored under a train-on-family-A / test-on-family-B split. Directly answers whether embedders track content or generator style (Finding 3) | Phase 1 |

### 11.3 Highest novelty: instruction-following retrieval

Prioritize this one. The factorial design lets SHELF pose queries **no natural corpus can
support cleanly** — *"find documents on the same subject but in a different genre"* — because in
real corpora genre and subject are correlated and the request is ill-posed. It also targets
precisely the instruct-embedders (Qwen3-Embedding, E5-instruct, Instructor) that the current
task suite cannot distinguish.

Needs new item construction, but no new documents.

#### 11.3.1 Built, and the honest verdict is split *(measured 2026-08-27)*

Four instruction tasks are registered, each an **anchor** group (facets that must match) and a
**contrast** group (facets that must differ), with relevance defined by the instruction rather
than by any single label field. Queries are prefixed `Instruct: {instruction}\nQuery: {text}`.

Beyond the IR metrics they report `contrast_violation@k` — the fraction of top-k that is exactly
what the instruction forbade — and, crucially, `contrast_violation_lift@k` against the rate a
random ranking would give. Without the lift a violation rate is uninterpretable: chance is 0.048
for "different LCC class" but 0.0086 for "different form".

**ndcg@10, 1,500 queries:**

| Task | BM25 | TF-IDF+SVD | MiniLM-L6 | Qwen3-Emb-0.6B |
|---|---|---|---|---|
| same_subject_diff_form | 0.4353 | **0.5712** | 0.4900 | 0.6266 |
| same_form_diff_subject | 0.0386 | 0.0470 | 0.0381 | **0.0601** |
| same_topic_diff_subject | **0.2228** | 0.1945 | 0.1861 | 0.2097 |
| same_audience_diff_register | 0.0461 | 0.0626 | 0.0551 | **0.0742** |

**Two of the four do not measure instruction-following, and the lexical baseline proves it.**
On `same_subject_diff_form` TF-IDF beats MiniLM (0.571 vs 0.490) — its contrast clause excludes
only 1 of 133 forms, so it is `lcc_retrieval` wearing an instruction. On
`same_topic_diff_subject` BM25 is the best model outright, consistent with the measured 87%
verbatim-topic rate. Keep both as controls, not as results.

**`same_form_diff_subject` works, and nothing solves it.** Every model sits at the floor
(0.038–0.060), and `contrast_violation@10` runs 0.44–0.62 against a 0.048 chance rate — a **9–13×
lift**. Independently reproduced: TF-IDF at violation@10 **0.606**, chance 0.0476, **lift 12.7×**.
Models fill their top-10 with precisely the documents the instruction excluded. Headroom is real
(relevant sets ≈244 of 34,025).

> **The "distinguishes instruct-embedders" claim in §11.3 is not demonstrated.**
> Qwen3-Embedding-0.6B, a genuine instruct-embedder given its documented prompt format, is best
> on 3 of 4 tasks *and* has the **highest** violation lift (12.9 and 13.7). It has better facet
> representations; it is not obeying the contrast clause either. What is established is narrower
> and still worth publishing: the task is well-posed, not lexically solvable, and currently
> unsolved by anything tested.

**Excluded from the SHELF aggregate.** These are `TaskType.RETRIEVAL`, so leaving them in
`tasks.retrieval` would silently turn the retrieval average from 3 tasks into 7 and move every
model's score — one task alone swung a test aggregate from 0.577 to 0.727. `shelf_score.exclude_tasks`
now keeps them reported but out of the aggregate until the two broken ones are settled.

**A bug this surfaced.** `CachedEmbedder` raised on any cache miss, and instruction tasks prefix
their queries, so those texts are never in a cache built from raw corpus text. Every dense model
failed these tasks while sparse models — which bypass the cache — passed, which is exactly the
shape of failure that looks like a finding. The cache now takes an optional `fallback` embedder.

**Reranking** is worth adding alongside it if you want to open a model class: cross-encoders and
late-interaction models (LFM2.5-ColBERT, mLateOn) are entirely untestable on SHELF today.

### 11.4 Relation strata — graded, not binary

Generalizes the graded-relevance item in §11.1. Every document pair sits at a defined position on
a bibliographic relation ladder, and pair/retrieval items should be sampled **balanced across
strata** rather than as "same label / different label":

| Stratum | Definition |
|---|---|
| S0 identical facets | same LCC subclass, same form, same topics |
| S1 same subclass | same LCC subclass, differing form or topics |
| S2 same class | same LCC class, different subclass |
| S3 sibling class | different class, same LCGFT category |
| S4 same form only | same form, unrelated subject |
| S5 same topic only | shared topics, different class and form |
| S6 unrelated | no shared facet |

Two requirements borrowed directly from the OpenGloss protocol:

1. **Do not collapse every non-match into a negative.** Preserve the ordinal structure as graded
   targets. S2 and S6 are both "not the same subclass" and are not remotely equivalent.
2. **Balance lexical overlap and label frequency across strata.** Otherwise a model wins S4/S5 by
   vocabulary overlap alone — which, given Finding 5, is exactly the failure mode to expect.

**Measured on the real v0.3.1 corpus** (`shelf.evaluate.strata`, 4,000 pairs each, seed 42):

| Stratum | Random pairs (current negatives) | Same-LCC pairs (current positives) |
|---|---|---|
| S0 identical facets | 0.00% | 0.00% |
| S1 same subclass | 0.00% | 0.00% |
| S2 same class | 5.25% | **100.00%** |
| S3 same form only | 0.88% | 0.00% |
| S4 same category only | 5.80% | 0.00% |
| S5 same topic only | 6.98% | 0.00% |
| S6 unrelated | **81.10%** | 0.00% |

This quantifies the problem precisely: the existing `same_lcc_pairs` task is
**100% S2 positives against 81% S6 negatives** — "same subject vs. totally unrelated". The hard
strata (S3/S4/S5) make up only 13.7% of random draws, which is why the task saturates. The root
cause is in `shelf.hub.dataset.generate_pairs`, which draws negatives with
`random.sample(all_labels, 2)`. Those pairs already exist in the corpus, so stratum-balanced
resampling costs **$0** and needs no new generation. S0/S1 are structurally unreachable in v0.3.1
(no subclass metadata, and zero duplicate spec tuples), and become reachable only with Phase 2.

**Graded labels are already present and already discarded.** The published `topic_overlap_pairs`
config carries 4-valued labels — the number of shared topics — with real mass in every level
(test split: `{0: 1600, 1: 1200, 2: 800, 3: 219}`). But `compute_pair_metrics` in
`metrics/pair.py` binarizes them to `0 vs non-0` before scoring. A graded relevance signal that
the corpus already contains is being thrown away at evaluation time. Preserving it is not new
data collection; it is not deleting what is there.

**Implemented**: `shelf.evaluate.strata` (relation ladder + difficulty bands) and
`shelf.hub.hard_negatives.mine_stratified_pairs` (index-driven quota mining). Measured on 12,000
real documents, all five reachable strata fill to quota in 0.1s with zero mislabelled pairs;
S3 costs 1.05 attempts per pair via the form index versus ~113 under rejection sampling. Requests
for unreachable strata return an explicit shortfall in a `MiningReport` rather than hanging or
silently short-counting.

### 11.5 Difficulty strata

Report easy/medium/hard separately for every task. Without this, a model can post a strong
aggregate purely by exploiting trivial lexical cues, and the composite hides it.

**Operational definition** (reproducible, no human labelling): score every item by the margin of
a lexical baseline (BM25 or TF-IDF+LR). Items the lexical baseline resolves with high confidence
are **easy**; items it gets wrong or resolves near its decision boundary are **hard**. Store the
resulting `difficulty` on the item and publish the distribution.

This makes the vocabulary shortcut measurable rather than merely suspected, and turns Finding 5
from a limitation into a reported axis.

**Measured on the real `lcc_classification` test split** (TF-IDF+LR, n=8,507, tercile bands, with
the negative-margin floor forcing all baseline errors into `hard`):

| Band | n | Share | TF-IDF accuracy | Mean words |
|---|---|---|---|---|
| hard | 2,836 | 33.3% | 0.675 | 469 |
| medium | 2,835 | 33.3% | 1.000 | 588 |
| easy | 2,836 | 33.3% | 1.000 | 833 |

Difficulty tracks length steeply — **55.1% of `micro` documents are hard vs. 20.6% of
`extended`** — confirming that length and difficulty are entangled and must be reported jointly
rather than either one alone. Note that `medium` and `easy` are both perfectly solved by the
baseline: the split between them is a *confidence* distinction, not a correctness one, and should
be described that way.

### 11.6 Context-window conditions

Evaluate at **explicit truncation budgets — 512 / 2,048 / 8,192 tokens** — rather than letting
each model silently truncate at its own `max_seq_length`. Given that 46% of SHELF documents
exceed 512 tokens, context-length advantage is currently folded invisibly into every score.
Making it a reported condition converts a confound into a finding.

### 11.7 Do not expand

- **Clustering.** There are already 12 clustering tasks, and register and geographic clustering
  return V-measure ~0.004–0.009 for *every* model — no discriminative variance whatsoever.
  Adding more is negative value. Either fix them (cluster *within* an LCC class to control the
  dominant factor) or retire them.

  > **Geographic clustering has a ground-truth defect** *(measured 2026-08-26)*. 33.4% of the
  > corpus carries two geographic tags, and **76.4% of those tag pairs map to different regions**
  > — `['Paris','Brazil']`, `['Beijing','Florida']`, `['Tokyo','Africa']`. The task label comes
  > from `get_region_from_list()`, which takes **the first tag**. So **38.5% of all
  > geographically-labelled test documents carry an arbitrary region label**: the document really
  > is about two regions and the ground truth silently picks one.
  >
  > Restricting evaluation to unambiguous single-region documents (n=3,391 of 5,559) moves
  > MiniLM's V-measure from **0.0048 to 0.0103 (+117%)** and ARI from 0.0007 to 0.0036.
  >
  > So label noise is real and roughly *doubles* the measurable signal — but the absolute score
  > stays near zero, so it is **not** the whole explanation. Both problems are live: fix the
  > labels (free), and replace flat k-means with subject-conditional clustering. Until then, the
  > claim "embeddings don't capture geographic content" is not supported by this task.
  >
  > Related generation defect: the prompt instructs "Ground the content in this location", so a
  > spec of `geographic: France, Texas` produces incoherent grounding. A real generated sample
  > opened *"As we stand today in the city of Tokyo, North Carolina…"*. Multi-region specs should
  > either be dropped or restricted to same-region pairs at sampling time (QC gate territory).

#### 11.7.1 Resolved: fix geography, retire register *(measured 2026-08-27)*

`GeographicLabelPolicy` (`FIRST` / `UNAMBIGUOUS_ONLY` / `ALL_REGIONS`) and
`SubjectConditionalClusteringEvaluator` are implemented. All numbers below are MiniLM on the real
test split, unambiguous labels only, **with a shuffled-label control** — because conditional
clustering makes the per-class problem smaller and easier, so a raw score is not interpretable.

| Metric | Real | Shuffled (chance) | **Lift over chance** |
|---|---|---|---|
| flat ARI | 0.0036 | −0.0004 | **+0.0041** |
| conditional ARI, pooled | 0.0067 | −0.0001 | **+0.0068** |
| conditional ARI, macro | 0.1120 | −0.0015 | **+0.1135** |
| conditional V-measure, pooled | 0.1455 | **0.0534** | +0.0921 |
| conditional V-measure, macro | 0.2584 | **0.0869** | +0.1715 |

**Two corrections this control forces.**

1. **Report ARI, not V-measure, for conditional clustering.** Shuffled labels score V-measure
   **0.0869 macro / 0.0534 pooled** — a third of the apparent signal is structural. ARI is
   chance-corrected and shuffles to ≈0. Any registered conditional task must use
   `ari_pooled` as its primary metric; `v_measure` there is not comparable to `v_measure` on a
   flat task.
2. **The honest effect is modest, not dramatic.** Conditioning raises the chance-corrected ARI
   lift from **+0.0041 to +0.0068** — real, and about 1.7×, but the macro figure (+0.1135)
   overstates it roughly 17× because macro averages over classes and small classes get easy
   problems. Report pooled.

**Verdicts:**

- **Geographic clustering — fix and keep.** Label policy plus subject-conditioning both help and
  the lift is positive in all 21 LCC classes. Report `UNAMBIGUOUS_ONLY` + conditional + pooled
  ARI, alongside the flat number for continuity.
- **Register clustering — retire.** Holding subject constant does not move it: pooled ARI 0.0010
  against a flat baseline of 0.0013, with per-class ARI ranging −0.004 to 0.054, i.e. noise. This
  is a genuine negative result about the data, not a fixable defect. Retire
  `register_clustering` and its `_hdbscan` / `_agglomerative` siblings, or report the null
  explicitly and exclude it from any aggregate.

**Both acted on** *(2026-08-27)*. `geographic_clustering_conditional` is registered, dispatched
to `SubjectConditionalClusteringEvaluator` with `UNAMBIGUOUS_ONLY` applied automatically, and
wired into the baseline config. Its primary metric is `ari_pooled`. Running it end-to-end
reproduces the manual measurement exactly — `ari_pooled` 0.0067, `ari_macro` 0.1120,
`v_measure_macro` 0.2584, 3,391 unambiguous documents, all 21 classes clustered, none skipped —
so the headline is now the chance-corrected figure rather than the inflated one.

The three `register_clustering*` tasks are kept registered and reported but added to
`shelf_score.exclude_tasks`: the null result stays visible and cannot dilute the aggregate.

Two integration bugs surfaced only by running it through the real harness rather than in
isolation: the task was registered but absent from the config's task list (so it silently ran
nothing), and `evaluate_embedder()` did not accept the `save_samples` argument the runner passes
to every clustering evaluator.
- **Audience classification.** 24 values but only 70.2% coverage and likely weak signal. Low
  priority relative to everything above.
- **Multilingual.** The real structural gap — most modern embedders being benchmarked are
  multilingual while SHELF is English-only. But it multiplies the corpus and needs
  native-speaker validation. Name it as a stated limitation in v0.4; make it a v0.5 decision
  rather than half-doing it.

---

## 12. QC gates

Applied to every generated document *before* it enters the corpus. Results stored as columns,
pass rates published per generator.

| Gate | Check |
|---|---|
| G1 Parse | Non-empty; splits cleanly into `Title:` + body (the v0.3.1 empty-body bug) |
| G2 Language | English (retain existing filter) |
| G3 Length adherence | \|actual − target\| within tolerance; record the delta, never silently keep |
| G4 Self-labeling leakage | Scan for "Document Type:", "In the field of…", genre announcements. The system prompt forbids these — verify rather than trust |
| G5 Topic coverage | Requested topics actually appear (lemma or embedding match) |
| G6 Near-duplicate | MinHash/SimHash against the full corpus at ~0.9. **Mandatory** now that one spec goes to 15 generators |
| G7 Refusal/boilerplate | "I can't help with…", "As an AI…", truncation artifacts |

G6 is the gate that protects the Phase 1 design. G4 protects the central "show, don't tell"
premise that makes the labels non-trivial.

### 12.2 What the gates found in the published v0.3.1 corpus *(2026-08-27)*

Running the suite over all 42,532 documents produced two defects in the *shipped* corpus, not
just in the tooling.

**Explicit taxonomy self-labelling: 833 documents (1.96%).** `GENERATION_INSTRUCTIONS` forbids
"classification headers like `Document Type:` or `LCGFT:` or `LCC:`". They are there anyway:

```
1. **Folk Music (LCGFT: Folk music)**
**Disciplina (LCC: Language and Literature)**
Tipo (LCGFT): Field recordings (Sound recordings)
```

These documents announce the answer to `lcgft_category_classification` and `lcc_classification`
in their own text. G4's original pattern **missed almost all of them** because it was anchored to
a line start, and real leakage never appears there — it is parenthesised or inside a YAML block.
G4 now matches taxonomy codes anywhere, tolerating the `(LCGFT):` bracket form, while keeping
generic words like "category" and "genre" line-anchored so ordinary prose does not trip it.

**Non-English documents: ~639 (1.50%) on a conservative estimate.** Spanish, German, Italian,
French and Portuguese, in a corpus documented as English-only and already filtered once for
language. Real examples: *"TERMO DE COOPERAÇÃO TÉCNICA Nº 017/2025"*, *"République Française —
Ville de Paris"*, *"REPUBBLICA ITALIANA — ESTRATTO DI ACCORDO"*.

> **A correction worth recording.** The QC agent reported G2's ~6% flag rate as essentially all
> false positives from keyword-dense structured English, and declined a fix on the grounds that
> it had only one true catch. Inspecting a random sample of the flagged documents, **three of six
> were genuinely non-English**. G2 has real precision problems, but it is also finding a real
> defect, and the two must not be conflated. A proper language-ID dependency (`langdetect`,
> `fasttext lid.176`) is the right fix; the function-word heuristic cannot separate these cases.

Both defects argue for the same thing: gates must run **before** publication, and their pass
rates must be published per generator. v0.3.1 shipped with ~2% self-labelled and ~1.5%
non-English documents that nothing caught.

### 12.3 Two gates were miscalibrated, and running them on new data exposed it

Applied to the v0.4 generation in flight, the gates initially retained **22.3% of Opus 5** and
54.7% of GPT-5.6 Sol. Neither number was real.

**G4 fired on ordinary vocabulary.** Several LCGFT categories -- `Music`, `Literature`,
`Ephemera` -- are also everyday English words. A document about a parish music director contains
"music"; a court filing contains "literature". A bare word-boundary match flagged **15-20%** of
generations on exactly that. `GENERATION_INSTRUCTIONS` permits domain vocabulary and forbids only
*announcing* the class, so document scanning now requires a **labelling context** (`Category:
Music`, `(LCGFT: Music)`, `Music:` at line start) while descriptions bound for a prompt keep the
strict bare-word check. Re-measured:

| Corpus | Flagged |
|---|---|
| v0.3.1 published | **2.48%** |
| v0.4, Claude Opus 5 | **0.18%** |
| v0.4, GPT-5.6 Sol | **0.47%** |

The new generation is an order of magnitude cleaner than the shipped corpus on this axis.

**G5's semantics were inverted.** It measures whether requested topics appear *verbatim*. But
high verbatim coverage is the string-matching shortcut, not quality — 87.2% of v0.3.1 topic terms
appear verbatim, which is exactly why `topic_classification` is largely solvable by string
matching. Gating on it rewards telling over showing: Opus 5 scored **34.9%** and Sol **75.1%**,
so the gate was rejecting 78% of the *better* generator's output. G5 is now informational by
default (`coverage_threshold=0.0`), with the fraction always recorded.

After both fixes, retention is **86.6% (Opus 5) / 93.4% (Sol)**, and the residual is almost
entirely G2's known language-detection precision problem.

> **The lesson generalizes.** Both gates passed review against v0.3.1 and looked reasonable there.
> They broke on the first genuinely new corpus, in opposite directions — one too strict, one
> measuring the wrong sign. Gates must be calibrated against the data they will actually gate,
> and a pass rate that varies by 40 points between two frontier generators should be treated as
> evidence about the gate before it is treated as evidence about the generators.

### 12.1 Promotion checks (run before any slice is published)

Adapted from the OpenGloss v1.3 promotion contract. A slice is not published until the manifest
report establishes:

- exact rows generated, retained, and rejected per generator, with per-gate rejection counts
- unique/missing `spec_id`s and realized coverage of every sampled dimension
- realized distributions by generator, difficulty, relation stratum, length bucket, and
  `prompt_variant_id` — sampling weights **and** realized counts, which are not the same thing
- split sizes plus every removed cross-split spec collision
- near-duplicate rate within and across generators
- **SHA-256 hashes for every split and for the immutable spec blocks**, published with the release
- overlap audit between SHELF and the natural transfer slice, with decontaminated results reported
  wherever overlap is material

**Any later revision is a new named data condition.** It does not silently replace these pins in
an existing version — the same rule that keeps v0.3.1 comparable.

---

## 13. Schema additions

Additive only; every v0.3.1 field is preserved.

```
spec_id                  # shared across generators — the split key
slice                    # core | subclass | minimal_pair | holdout
lcc_subclass             # Phase 2
lcc_subclass_name
generator_family         # openai | anthropic | google | xai | deepseek | ...
generator_release_date
generator_provider       # OpenRouter serving backend
pair_id, pair_role       # Phase 3 minimal pairs
prompt_variant_id        # §4.2 — which system-prompt variant produced this doc
output_format            # form-conditional format actually requested (§4.2)
spec_block_id            # §5 — which independent spec draw this came from
difficulty               # §11.5 — easy | medium | hard, from lexical-baseline margin
relation_stratum         # §11.4 — S0..S6, on pair/retrieval items
source_type              # synthetic | natural (Phase 6)
qc_parse, qc_language, qc_length_delta, qc_selflabel,
qc_topic_coverage, qc_near_dup, qc_refusal
```

Existing fields whose *value space* widens under §4.1 (`topics`, `geographic`, `lcgft_form`)
keep their names and types — only the sampler's label pool changes.

## 14. Splitting

**Implemented and demonstrated** *(2026-08-27)*. `SplitConfig(group_by="spec_id")` splits over
groups and expands back to documents, with a post-condition that fails loudly if any group
straddles. Measured on a Phase-1-shaped corpus (3 blocks × 200 specs × 8 generators = 4,800 docs):

| Protocol | Specs straddling splits |
|---|---|
| document-level (v0.3.1 default) | **598 of 600** |
| `group_by="spec_id"` | **0** |

Group counts land exactly on the 60/20/20 ratio (360/120/120 groups), all three spec blocks
appear in every split so split is not confounded with draw, and the result is deterministic under
seed. `group_by=None` remains the default and reproduces v0.3.1 splitting byte-for-byte.

- Split on **`spec_id`**, not document id — non-negotiable given Phase 1.
- Stratify by `(lcc_code, lcgft_category)` as today, plus `slice`.
- Holdout slice never enters train.
- Retain seed 42 and the existing `SplitConfig` machinery.

## 15. Versioning

| Version | Contents |
|---|---|
| v0.3.1 | Frozen, still published, baselines remain valid |
| v0.4.0 | v0.3.1 + core + subclass + minimal pairs |
| configs | `default`, `subclass`, `minimal_pairs`, `holdout`, + existing pair configs |

## 16. Budget

| Phase | Docs | Cost |
|---|---|---|
| 1 — Generator-balanced core | 22,500 | $218 |
| 2 — LCC subclass tier | 6,000 | $38 |
| 3 — Minimal pairs | 4,000 | $45 |
| 4 — Holdout probe | 1,000 | $15 |
| 6 — Natural transfer slice | ~5,000 | $0 (engineering only) |
| **Subtotal** | **~38,500** | **$316** |
| QC regeneration buffer (~20%) | — | $63 |
| **Total** | | **≈ $379** |

Maximum spend on any single model: **$47.06** (Opus 5, Phase 1), under conservative pricing.

**Zero-cost work**: coverage expansion (§4.1), prompt variants (§4.2), the mined hard negatives
in Phase 3, the natural transfer slice (Phase 6), and every task addition in §11 add **$0** in
generation cost — they are sampler,
config, and item-construction changes. Do these first; they may reduce how much generation the
later phases actually need.

## 16.1 What the first real Phase 1 run actually did *(2026-08-27)*

The run exposed two failure modes that no amount of dry-running would have found, and one
reporting mistake of mine.

**`--reasoning-effort none` destroyed three full model runs.** OpenRouter returns:

```
{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400}}
```

`x-ai/grok-4.6`, `google/gemini-3.1-pro-preview` and `google/gemini-3.7-flash` each returned
**1,500 errors and zero documents**. The backends already had a retry-without-reasoning path, but
its heuristic was built from the *opposite* message — a model rejecting the parameter — and
matched none of the markers here. Fixed by detecting the mandatory case as well; verified live,
all three now succeed.

> **This changes the cost model for those models.** With reasoning forced on, grok-4.6 spent
> **1,180 reasoning tokens on a 25-word document** (1,215 output tokens total). The
> reasoning-off projection does not apply to them, and the §5 worst-case column is the number to
> budget against.

**Transient burst failures are real and must be re-run.** `anthropic/claude-opus-5` produced 561
documents, then failed 939 times inside a two-minute window (03:39:25–03:41:22) with successes on
both sides, and works fine on retry. Rate limiting, not a systematic fault. `minimax/minimax-m3`
lost 274 the same way. The resumable ledger makes this recoverable — a re-run skips completed
specs and retries errors — but a run cannot be declared finished on document *counts* alone.

**A reporting mistake worth recording.** I twice reported "twelve models complete at 1,500 each,
no budget aborts" by counting **ledger rows** rather than rows with `status == "ok"`. Actual
documents at that point were **12,540, not 18,201**. Error rows carry zero tokens and zero cost,
so they are invisible in a spend total and look identical to success in a row count. Any progress
report must filter on status, and the per-model `extra` field was empty on failures, which made
diagnosis slower than it should have been.

## 16.2 Five ways "turn reasoning off to save money" silently killed a run

Every one of these returned a clean 400 or TypeError, produced zero documents, and looked like a
valid configuration in a dry run. Together they cost roughly 6,500 documents across two sessions
before each was found.

| Provider | Verbatim message | Blast radius |
|---|---|---|
| OpenRouter | `Reasoning is mandatory for this endpoint and cannot be disabled.` | 3 models × 1,500 = **0 documents** |
| Gemini | `Budget 0 is invalid. This model only works in thinking mode.` | gemini-3.1-pro-preview, **0 of 1,500** |
| Anthropic (SDK) | `Messages.create() got an unexpected keyword argument 'temperature'` | every Anthropic call |
| Anthropic (model) | `"thinking.type.disabled" is not supported by this model` | claude-fable-5, **0 of 500** |
| OpenAI | `Unsupported parameter: 'temperature'` — then `'top_p'`, one at a time | every gpt-5.6 call |

Each backend now recognises its provider's phrasing and retries with the provider default. The
OpenAI path retries **iteratively**, because that provider surfaces one rejected parameter per
attempt. The Anthropic backend **probes `messages.create`'s signature** rather than pinning an SDK
version, since the parameter was removed outright between releases.

**The generalisable rules:**

1. A cost optimisation that touches provider capabilities needs **one live call per model**
   before a batch run. Dry-running prices, not capabilities.
2. Persist failure reasons to disk. Four of these five were diagnosed only by reproducing the
   call, because the reason existed nowhere afterwards. The fifth was identified in seconds once
   `<ledger>.errors.jsonl` existed — that file paid for itself immediately.
3. Never judge a run by document count. Error rows carry zero tokens and zero cost, so a failed
   model is invisible in a spend total and indistinguishable from success in a row count.

---

## 17. Risks

| Risk | Mitigation |
|---|---|
| Shared-spec design creates near-duplicates | Spec-level splitting (§14) + G6 near-dup gate |
| Reasoning-token overrun on thinking models | Minimal reasoning effort, `max_output_tokens` cap, ledger with hard abort |
| Preview models change or disappear (`gemini-3.1-pro-preview`) | Pin and record exact model string + resolution date in the manifest |
| OpenRouter routes to varying backends/quantizations | Record `provider` per request; pin routing where it matters |
| Generator refuses certain LCC×form combinations | G7 gate; track refusal rate per generator as a reported statistic |
| Transfer slice (Gutenberg) is in every model's pretraining data | Report as the *contaminated-natural* condition explicitly; never pool with SHELF, never claim contamination-free (§10) |
| Transfer slice skew (P-class, pre-1929, book-length) mistaken for balanced coverage | Stratify, subsample, and publish the realized distribution; label it a transfer slice, not a corpus |
| Challenge/adversarial slices get optimized against and stop diagnosing | Design principle 8; keep them out of any training or tuning loop |
| 2x pricing uncertainty on `gpt-5.6-sol` and `gemini-3.7-flash` | Budget with the conservative provider figure (§5); `BudgetGuard` hard-caps per model regardless; ledger records actuals |
| Promotional pricing expires mid-project (Google 2027-01-01, OpenAI ~2026-11-21) | Re-fetch pricing immediately before each spend; `PricingTable` records `fetched_at` in the run manifest |
| A model rejects the reasoning parameter and kills a long run | Backends detect reasoning-parameter rejection and retry once without it |
| A model *requires* reasoning and rejects `--reasoning-effort none` | Same retry path, extended to the "reasoning is mandatory" message (§16.1). Cost for those models follows the worst-case column, not the reasoning-off projection |
| A provider rejects a sampling or reasoning parameter in its own dialect | **Five distinct wordings observed across four providers** — see §16.2. Every backend now detects its provider's phrasing and retries with the default. Make one live call per model before any batch run; a dry run cannot catch these |
| Transient provider bursts silently truncate a model's output | Ledger records status per request; re-run resumes and retries errors. Never judge completeness by row count — filter on `status == "ok"` |

---

## 18. Open questions

1. **Corpus size target.** 33,500 new docs roughly doubles SHELF. Is ~76k total the right size,
   or should Phase 1 N drop to keep the corpus nearer 60k?
2. **Micro bucket.** Drop entirely, or retain as an excluded diagnostic stratum?
3. **Subclass task**: replacement for the 21-class task, or an additional tier alongside it?
4. **Fable 5 / Kimi K3**: include at reduced N (~1,000, ~$47 and ~$38), or leave out?
5. **Human validation**: who codes it, and is 300–500 documents the right size?
6. **Topic expansion depth**: top-500 or top-1,000 LCSH? Deeper means more specific labels but
   thinner per-label counts — at 1,000 topics and ~76k docs a topic averages ~190 documents.
   Note this is now gated on scope-note availability (§4.1.1), not just on counts.
7. **Form classification label space**: all 300+ expanded forms, or the top-133 already in use so
   v0.3.1 documents remain usable for the task?
8. **Instruction-following retrieval** (§11.3): build for v0.4, or defer until the core corpus
   lands and item construction can be designed against the real data?
9. **Broken clustering tasks** (§11.7): fix as conditional/within-class clustering, or retire
   register and geographic clustering outright?
10. **Spec blocks**: three blocks of 500, or more blocks at smaller N? More blocks give a better
    variance estimate but fewer paired comparisons within each.
11. **Transfer slice source**: Gutenberg alone, or Gutenberg + SuDoc government documents to cover
    the non-literary subjects where Gutenberg is weakest?
12. **Difficulty definition**: lexical-baseline margin (proposed, reproducible), or an ensemble
    margin across several baselines? The former is simpler and harder to game.

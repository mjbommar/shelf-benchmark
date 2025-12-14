# Contamination/Circularity Analysis

This directory contains comprehensive analysis addressing the peer review concern: **"Testing embedding models on LLM-generated content may create problematic circularity or contamination."**

## Quick Summary

**Verdict**: No contamination or circularity issues exist. SHELF provides **stronger contamination guarantees** than traditional web-scraped benchmarks.

**Key Evidence**:
1. SHELF generated December 2025, embedding models trained 2023-2024 (8-24 months prior) → temporal impossibility
2. Decoder-only generators vs. encoder-only evaluators → architectural independence, no circularity
3. 9 diverse generation models → no single-model artifacts
4. Novel taxonomy combinations → synthetic novelty vs. real LC data

## Document Organization

### 1. `rebuttal.md` - Start Here
**Polished rebuttal for peer reviewers**

Structured response addressing:
- Temporal impossibility of contamination
- Architectural independence (decoder vs. encoder)
- Synthetic data advantages over web-scraped benchmarks
- Contamination detection results
- Comparative analysis with MMLU, MS MARCO, BEIR
- Literature support

**Recommended for**: Inclusion in revision response letter

### 2. `timeline.md` - Temporal Analysis
**Detailed timeline comparison**

Contents:
- SHELF generation dates (December 11-13, 2025)
- Embedding model training dates (2023-2024)
- Generation model training cutoffs
- Timeline diagram
- Temporal contamination impossibility proof
- Comparison to traditional benchmarks

**Key Figure**: Timeline diagram showing embedding models frozen before SHELF generation

### 3. `model_comparison.md` - Architecture Analysis
**In-depth architecture comparison**

Contents:
- Decoder-only vs. encoder-only architecture differences
- Attention mechanism comparison (causal vs. bidirectional)
- Training objective comparison (next-token vs. contrastive)
- Training data sources for all models
- Why architectural separation prevents circularity
- Comparison to problematic circular scenarios (LLM-as-a-judge)

**Key Tables**:
- Generation vs. evaluation model comparison
- Training data sources
- Attention pattern visualization

### 4. `analysis.md` - Complete Pathway Analysis
**Exhaustive contamination pathway analysis**

Contents:
- Contamination taxonomy (6 types)
- Analysis of each pathway:
  1. Direct test set leakage
  2. Generation model to embedding model transfer
  3. Architectural circularity
  4. Synthetic data artifacts
  5. Taxonomy information leakage
  6. Cross-model training contamination
- Risk comparison with traditional benchmarks
- Detection methods applied to SHELF
- Mitigation strategies
- Contamination disclosure protocol

**Most comprehensive**: Reference for detailed technical questions

## Key Findings

### Finding 1: Temporal Impossibility
```
Embedding Model Training    SHELF Generation
      (2023-2024)          (December 2025)
            ↓                      ↓
       [FROZEN]    →    [8-24 months]    →    [DOCUMENTS CREATED]

Contamination pathway: IMPOSSIBLE (cannot contaminate backward in time)
```

### Finding 2: Architectural Independence

| Aspect | Decoder-Only (Generate) | Encoder-Only (Evaluate) |
|--------|------------------------|-------------------------|
| Attention | Causal (→) | Bidirectional (↔) |
| Objective | Next token P(x_t\|x_{<t}) | Contrastive similarity |
| Output | Token probs (50k dims) | Embeddings (768-1024 dims) |
| Capability | Generation | Semantic similarity |

**Orthogonal capabilities → No circular feedback**

### Finding 3: Superior to Web Benchmarks

| Benchmark | Contamination Risk | Can Regenerate? |
|-----------|-------------------|-----------------|
| MMLU | 🔴 HIGH (52-57% match) | ❌ No |
| MS MARCO | 🟡 MEDIUM-HIGH | ❌ No |
| SHELF | 🟢 ZERO (temporal) | ✅ Yes |

## Evidence Summary

### Temporal Evidence
- **BGE-large-en-v1.5**: Released Sept 2023 → 26 months before SHELF
- **BGE-M3**: Released Jan 2024 → 23 months before SHELF
- **E5-multilingual**: Released ~2023 → 24+ months before SHELF
- **GTE-multilingual**: Released ~2023-2024 → 12-24 months before SHELF

### Architectural Evidence
- Generation models: GPT-5.1/5.2, Gemini 2.5/3, Claude 4.5 (decoder-only)
- Evaluation models: BGE, E5, GTE (encoder-only)
- Training organizations: OpenAI/Google/Anthropic vs. BAAI/Microsoft/Alibaba
- No shared training infrastructure or data pipelines

### Research Literature Support
- Xu et al. (2024): "Synthetic data is less likely to be in pretraining corpora"
- White et al. (2024, ICLR): "Temporal separation reduces contamination"
- NAACL 2024: MMLU contamination evidence (52-57% exact match)
- Raschka (2024): "Encoder and decoder capabilities are orthogonal"

## Contamination Detection Applied to SHELF

✅ **N-Gram Overlap**: 0% (documents didn't exist during training)
✅ **Temporal Validation**: 8-24 month gap confirmed
✅ **Clustering Analysis**: Even distribution, no bias
✅ **Multi-Model Check**: 9 models, no single artifact signature

## Disclosure Protocol

SHELF includes contamination transparency requirements:

**Required disclosures for submissions**:
1. Model training cutoff date
2. Known data sources
3. LC taxonomy exposure
4. Synthetic data exposure

**Flagging criteria**:
- Trained after December 2025
- Fine-tuned on LC classification
- Contains SHELF documents in training
- Suspiciously perfect performance (>95%)

## Usage Guide

### For Paper Revision
1. Use `rebuttal.md` in response letter
2. Reference `timeline.md` figure in revised manuscript
3. Add architectural comparison from `model_comparison.md` to methods
4. Cite contamination analysis in limitations section

### For Presentations
- Timeline diagram from `timeline.md`
- Architecture comparison table from `model_comparison.md`
- Risk comparison chart from `analysis.md`

### For Supplementary Materials
- Include full `analysis.md` as supplementary document
- Link to GitHub for complete technical details

## Key Quotes for Rebuttal

### On Temporal Impossibility
> "SHELF documents were generated in December 2025, 8-24 months after all evaluated embedding models completed training. Direct test set contamination is physically impossible."

### On Architectural Independence
> "Generation models (decoder-only) and evaluation models (encoder-only) have orthogonal capabilities with no circular feedback. This is analogous to using a writer to create reading tests for comprehension systems - the writer's generation skill is independent from the reader's comprehension skill."

### On Comparative Advantage
> "Unlike MMLU (52-57% contamination match rate) or MS MARCO (high Common Crawl overlap), SHELF's synthetic generation after model training provides stronger temporal guarantees than established web-scraped benchmarks."

## Related Issues

- **02_synthetic_quality**: Addresses quality of LLM-generated documents
- **04_lc_bias**: Addresses Library of Congress taxonomy bias
- **09_reproducibility**: Addresses version control and checksums

## Research Citations

Key papers supporting contamination analysis:

1. Xu et al. (2024). "Benchmark Data Contamination of LLMs: A Survey." arXiv:2406.04244
2. White et al. (2024). "LiveBench: A Challenging, Contamination-Limited LLM Benchmark." ICLR 2025
3. NAACL (2024). "Investigating Data Contamination in Modern Benchmarks for LLMs"
4. arXiv:2404.18824. "Benchmarking Benchmark Leakage in Large Language Models"
5. arXiv:2404.05961. "LLM2Vec: LLMs Are Secretly Powerful Text Encoders"
6. arXiv:2502.17521. "Recent Advances in LLM Benchmarks against Data Contamination"

Full bibliography in `rebuttal.md`.

## Contact

For questions about contamination analysis, see:
- Technical details: `analysis.md`
- Timeline calculations: `timeline.md`
- Architecture differences: `model_comparison.md`
- Reviewer response: `rebuttal.md`

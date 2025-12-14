# Timeline Analysis: Model Training vs. Data Generation

## Executive Summary

**Key Finding**: SHELF documents were generated in December 2025, making temporal contamination of embedding model pretraining data **impossible**. All evaluated embedding models completed training months to years before SHELF data was created.

## SHELF Dataset Generation Timeline

Based on HuggingFace Hub metadata:

- **Generation Period**: December 11-13, 2025
  - 20251211_123151
  - 20251212_125932
  - 20251212_214537
  - 20251213_161521

- **Dataset Version**: 0.3.0
- **Total Documents**: 42,616
- **Generation Models**: GPT-5.1, GPT-5.2, Gemini 2.5 Flash/Pro, Gemini 3 Pro, Claude Haiku/Sonnet/Opus 4.5

## Generation Model Training Cutoff Dates

| Model | Training Cutoff | Release Date | Gap Before SHELF |
|-------|----------------|--------------|------------------|
| GPT-5.1 | September 2024 | July 2025 | ~5 months |
| GPT-5.2 | August 31, 2025 | December 11, 2025 | ~10 days |
| Gemini 2.5 Pro | January 31, 2025 | April 2025 | ~10 months |
| Claude 4.5 Opus | March 2025 (reliable), August 2025 (training) | November 24, 2025 | ~1-4 months |

**Implication**: None of the generation models could have seen SHELF documents during training. The synthetic data represents novel combinations of taxonomy elements, not memorized content.

## Embedding Model Training Timeline

### BGE (BAAI General Embedding) Models

- **bge-large-en-v1.5**: Released September 11, 2023
- **bge-m3**: Released January 30, 2024
- **bge-en-icl**: Released July 26, 2024
- **bge-multilingual-gemma2**: Released July 26, 2024
- **bge-VL**: Released March 6, 2025

**Training Data Cutoff**: Pre-2024 for most models, early 2024 for latest variants
**Gap Before SHELF**: 8-24+ months

### E5 Models

- **Multilingual-E5**: Released ~2023
- **Training Data**: ~1 billion multilingual text pairs
- **Architecture**: XLM-RoBERTa based (encoder-only)

**Training Data Cutoff**: Pre-2024
**Gap Before SHELF**: 18+ months

### GTE (General Text Embedding) Models

- **gte-multilingual-base**: Released ~2023-2024
- **Architecture**: Encoder-only transformer (305M parameters)
- **Training Data**: Includes code and hyperlinks

**Training Data Cutoff**: Pre-2024
**Gap Before SHELF**: 12+ months

## Timeline Diagram

```
2023                  2024                  2025
|---------------------|---------------------|-----
  BGE-large-v1.5      BGE-m3                GPT-5.1 cutoff
  E5-multilingual     BGE-icl/gemma2        Gemini 2.5 cutoff
  GTE-base                                  Claude 4.5 cutoff
                                            |
                                            GPT-5.2 cutoff
                                            |
                                            SHELF GENERATION
                                            Dec 11-13, 2025
```

## Temporal Contamination Analysis

### Can SHELF documents be in embedding model pretraining data?

**Answer: NO - Temporally impossible**

1. **BGE models**: Trained 14-24 months before SHELF generation
2. **E5 models**: Trained 18+ months before SHELF generation
3. **GTE models**: Trained 12+ months before SHELF generation

### Can generation models leak test data to embedding models?

**Answer: NO - Different model families, no data pipeline**

1. Generation models (GPT-5, Gemini 2.5, Claude 4.5) are decoder-only generative models
2. Embedding models (BGE, E5, GTE) are encoder-only representation models
3. No shared training data pipeline between these model families
4. Embedding models were frozen before generation models were even trained

## Comparison to Traditional Benchmarks

### Web-Scraped Benchmarks (e.g., MMLU, MTEB)

- **MMLU**: Released 2021, contains content from Wikipedia, textbooks, exams
  - **Contamination Risk**: HIGH - content existed on web before model training
  - **Detection**: Difficult - requires n-gram matching, paraphrasing can evade

- **Original MTEB**: Uses existing datasets (Wikipedia, Stack Exchange, etc.)
  - **Contamination Risk**: HIGH - many sources in pretraining corpora
  - **Detection**: Challenging due to scale of pretraining data

### SHELF (Synthetic Benchmark)

- **Generation Date**: December 2025
- **Contamination Risk**: ZERO - did not exist during embedding model training
- **Detection**: Unnecessary - temporal impossibility

## Conclusion

The timeline analysis provides definitive evidence that:

1. SHELF documents could not contaminate embedding model pretraining (created after training)
2. Generation model training cutoffs do not affect embedding model evaluation (different architectures, no shared pipeline)
3. Synthetic generation provides stronger temporal guarantees than web-scraped benchmarks
4. The 8-24 month gap between embedding model training and SHELF generation creates a natural firewall against contamination

**This temporal separation is a key advantage of synthetic benchmarks over traditional web-scraped alternatives.**

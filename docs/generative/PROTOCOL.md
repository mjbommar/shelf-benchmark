# Generative probe: sample and evaluation protocol

Frozen 2026-09-01, before any test-split number was produced. The validation
sweep that chose each configuration is in [DIARY.md](DIARY.md); nothing in that
sweep touched the test split.

## Why this exists

Section 11 of the paper states that absolute scores have no human agreement
rate or expert ceiling. The best genre-form score of 0.2605 therefore cannot be
read as "the task is hard" or as "encoders are the wrong instrument". This
probe supplies a machine reference point for the subject task.

It is a reference arm. It is **not** a 28th configuration in the model panel
and takes no part in the pre-registered rank-agreement analysis, whose
population is embedding models.

## Sample

**Every document in the test split: 12,504.** No sampling, no stratification.
This is the same population, the same split, and the same documents the
encoders were scored on, so the numbers sit in the same frame as
`results/pooled/baselines/*_lcc_classification.json`.

Earlier 200-document runs were hyperparameter selection on the *validation*
split and are not reported as results.

## Task and metric

21-way LCC subject classification. Metrics from
`shelf.evaluate.metrics.classification`, the module that produced the encoder
numbers, so both arms share one implementation: macro-F1 with
`average="macro"` and `zero_division=0.0`, alongside micro-F1, weighted-F1 and
accuracy.

## The asymmetry that cannot be removed

The encoder task is a **supervised linear probe**: encode the 37,795-document
train split, fit LogisticRegression, predict test. The decoders are
**zero-shot** and see no labelled example.

A decoder scoring below an encoder is therefore not evidence of worse
representations. It is a fitted classifier against no classifier. Every table
carrying both must say so.

## Decoders, and how they differ

| model | params | interface | reasoning | own-family share of test |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.87B | label likelihood | no | 2.3% |
| Qwen3.5-2B | 2.27B | label likelihood | no | 2.3% |
| gemma-4-E2B-it | 5.12B | label likelihood | no | 8.9% |
| gpt-5.6-luna | undisclosed | constrained decoding | yes | **68.4%** |

Differences to footnote rather than hide:

1. **Prediction mechanism.** Local models rank all 21 labels by continuation
   likelihood. `gpt-5.6-luna` does not expose logprobs, and GPT-5.x returns
   empty logprobs when `json_schema` is enabled, so it is constrained by a
   JSON-schema `enum` instead. Both guarantee a prediction inside the label
   set; they reach it differently.
2. **Reasoning budget.** `gpt-5.6-luna` spends reasoning tokens before
   answering. The local models are read at a single position with none.
3. **Size.** Undisclosed for the hosted model, and Gemma's "E2B" names
   *effective* parameters against 5.12B of actual weights. The `<4B` intent of
   this probe holds for the Qwen models and not for the rest.
4. **Self-preference.** GPT and OpenAI models wrote 68.4% of the test split.
   `gpt-5.6-luna` is largely judging text its own family wrote. Accuracy is
   broken out by generator family so the size of that effect can be read
   rather than assumed.
5. **Token budgets are per-tokenizer.** A 512-token budget is 512 of that
   model's own tokens, so the underlying text differs slightly between models.

## Configurations, frozen from validation

| model | prompt | token budget |
|---|---|---|
| Qwen3.5-0.8B | letter_forced | 2048 |
| Qwen3.5-2B | cataloguer | full document |
| gemma-4-E2B-it | cataloguer | 512 |
| gpt-5.6-luna | cataloguer | 512 |

Best prompt is model-specific, which is why each was swept rather than sharing
one prompt.

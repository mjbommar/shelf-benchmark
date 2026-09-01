# Generative classification probe — working diary

Purpose: measure how small recent decoder models classify SHELF documents,
as a reference point for the encoder scores. Section 11 states the paper has
no ceiling of any kind, so the best genre-form score of 0.2605 cannot be read
as "task is hard" or "encoders are wrong tool". This probe supplies a machine
reference.

Scope, fixed before any run:
  - classification tasks only
  - decoder results are a separate reference arm, NOT a 28th configuration in
    the model panel, and NOT part of the pre-registered rank-agreement claim
  - prompt and hyperparameter sweep on a dev slice, then one frozen setting
    scored on test

## Log

### 2026-09-01 — model selection

Checked the Hub rather than relying on recall; my first query sorted by
`createdAt` and silently missed the current release, so every figure below
comes from the model API and is dated.

Qwen3.8 is current (27B weights landed 2026-08-05) but has **no small
variant**. The full official lineup:

| model | params | bf16 |
|---|---|---|
| Qwen3.8-27B | 27.78B | ~56 GB |
| Qwen3.8-Flash-Next | 180.0B | — |
| Qwen3.8-2.4T-A95B | 2.4T MoE | — |

The smallest is 27.8B, seven times the 4B ceiling set for this probe, so the
most recent Qwen *under* that ceiling is the Qwen3.5 line (2026-02-28).

Gemma 4 (2026-03-02) does have small variants. Note the E-names are
*effective* parameters, not weight counts:

| model | total params | bf16 | gated |
|---|---|---|---|
| gemma-4-E2B-it | 5.12B | 10.2 GB | no |
| gemma-4-E4B-it | 8.00B | 16.0 GB | no |

Selected, smallest-recent of each family, both fitting one 16 GB card:

  - `Qwen/Qwen3.5-0.8B`   0.87B, 1.7 GB
  - `Qwen/Qwen3.5-2B`     2.27B, 4.5 GB
  - `google/gemma-4-E2B-it` 5.12B, 10.2 GB

Comparison honesty: the best encoder in the paper is EmbeddingGemma-300M.
Even the 0.8B Qwen is ~3x that, and gemma-4-E2B is ~17x. This is a
deployable-small comparison, not a parameter-matched one, and must be
described that way.

### 2026-09-01 — environment isolation

`Qwen3.5` needs a `transformers` that knows `model_type: qwen3_5`. The project
has `5.0.0.dev0` from a git pin; PyPI is at 5.16.1.

Upgrading in place was rejected. The encoder results are frozen and 170 paper
numbers are verified against them; `transformers` governs tokenisation and
pooling for every one of those models, and MTEB's own guidance is that the
loading path changes scores. A probe must not be able to move the numbers it
is a reference for.

The probe therefore gets its own virtualenv at `.venv-gen`, which shares no
packages with the evaluation environment. Encoder results are reproduced from
`.venv` exactly as before.
### 2026-09-01 — first run degenerate, two real bugs

Qwen3.5-0.8B, letter_only, 40 docs: macro-F1 0.0048, **1 of 21 classes ever
predicted**. That is a broken harness, not a weak model. Diagnosis on three
documents showed the model ranks `A` first every time, by a wide margin:

    gold=N  top5=[('A',-4.5), ('D',-8.12), ('Q',-8.5), ...]
    gold=K  top5=[('A',-5.09), ('H',-7.03), ('D',-8.28), ...]
    gold=K  top5=[('A',-5.3),  ('E',-7.87), ('D',-7.99), ...]

`A` wins by ~3.5 nats irrespective of content: that is the label prior, not a
decision about the document.

Two causes, both fixed:

1. **No chat template.** The tokenizer has one and the checkpoint is
   instruct-tuned; scoring a raw completion is the wrong interface.
2. **Uncalibrated label priors.** Fixed by contextual calibration: score the
   labels against a content-free prompt and subtract, so what is compared is
   the evidence a document adds rather than the model's prior taste for `A`.
   Both raw and calibrated numbers are kept, because the size of the
   correction is itself worth reporting.

Also: every class letter is a single token, so all 21 can be scored from one
forward pass instead of 21. 111s per 40 documents becomes a few seconds, which
is what makes a real sweep affordable.
### 2026-09-01 — chat-mode results were invalid; surface-form bug

Read the model card instead of guessing, which found the real problem.

Qwen3.5 has a thinking mode, but its template already closes it
(`assistant\n<think>\n\n</think>\n\n`), so the next token is the answer and the
scoring position was right. The surface form was not.

The prompt ends in `\n\n`, so the model emits `A`, not ` A`. Scored at that
position on a real document:

    'A'   logprob  -3.982
    ' A'  logprob -19.263
    label mass: no-space 0.0349, with-space 0.0000

Every chat-mode cell in the first sweep compared 21 tokens holding no mass, so
those numbers were noise. Discarded to /tmp, not reported. The non-chat cells
ended in `:`, where a leading space is correct, which is why only the chat rows
looked odd.

Fix: marginalise over surface form, summing the probability of each label with
and without the leading space rather than assuming one.

Second finding from the same trace: the top continuations are `Based`, `The`,
`To`, and only 3.5% of mass sits on any bare class letter. A prompt that does
not demand a bare letter measures format compliance as much as knowledge, so an
explicit single-letter instruction becomes a sweep axis.
### 2026-09-01 — word budgets replaced by token budgets

The first sweep truncated at 100/250/500 *words*, which is indefensible twice
over. Words are not the unit the model reads (1.47 tokens per word here), and
the largest budget was not large: 500 words is 456 median tokens and leaves
32% of documents cut. Macro-F1 was still climbing across that axis
(0.3008 -> 0.3847 -> 0.4278), so the sweep stopped exactly where the trend
was still going up, which measures the budget rather than the model.

Corpus token lengths, 400 validation documents, Qwen3.5 tokenizer:

    median 456   mean 947   p90 2673   max 6341
    over 512 tokens: 43.8%      over 2048: 13.2%

The axis is now a token budget with three values that each mean something:

  - **512**  the cap 12 of 20 dense encoders in the paper run under
  - **2048** the cap the larger encoder configurations run under
  - **full** the whole document, which is what a long-context decoder can
    actually read and is the only setting that gives the decoder its best case

512 and 2048 make the comparison token-fair against the encoders; the full
condition stops the probe from understating the decoder by a limit that only
the encoders have. `truncation_audit.py` already measures the encoder side of
this, so the two are directly comparable.

Note for the writeup: at 512 tokens the decoder and the encoders see the same
text, so that column is the fair comparison. The full-document column is the
decoder's advantage and must be labelled as such, not quietly compared against
an encoder that never saw the tail of the document.
### 2026-09-01 — download setup

`hf_transfer` was not installed and `systemd-run` does not inherit the
environment, so downloads ran single-stream. Authentication still worked
because `huggingface_hub` reads `~/.cache/huggingface/token` directly, but the
token was not in the process environment either.

Installed `hf_transfer`, and later launches pass `HF_TOKEN` and
`HF_HUB_ENABLE_HF_TRANSFER=1` explicitly. The gemma-4-E2B download was already
at 6.6 of 10.2 GB when this was found and was left to finish rather than
restarted.
### 2026-09-01 — OOM on gemma, and a real inefficiency behind it

gemma-4-E2B died at `logits / final_logit_softcapping`, allocating 2.36 GiB on
a card that had 1.81 GiB free. The cause was mine, not the model's.

Only the final position's logits are ever read, but the forward pass
materialised logits for every position. Against Gemma's 262,144-token
vocabulary a 6,341-token document is 3.32 GB in bf16, and calling `.float()`
on it doubles that. Qwen survived only because 0.8B leaves more headroom.

Fixed with `logits_to_keep=1`, which computes the one row that is used. This
was costing memory on every model, not just the one that crashed.

Card 0 also holds another tenant, so the probe now runs on card 1 only.
### 2026-09-01 — sweep complete, 72 cells

Three models x 4 prompts x 2 chat x 3 token budgets, 200 validation documents,
seed 42. Best configuration per model, macro-F1 on 21-way subject:

| model | best configuration | F1 | label mass |
|---|---|---|---|
| Qwen3.5-0.8B | letter_forced, chat, 2048 tok | 0.3837 | 0.973 |
| Qwen3.5-2B | cataloguer, chat, full doc | 0.4972 | 0.704 |
| gemma-4-E2B-it | cataloguer, chat, 512 tok | 0.4804 | 0.439 |

Four findings, none of which were assumptions going in.

**The chat template is the single largest factor**, worth 2.3x to 3.0x on mean
raw F1 for every model. Scoring a raw completion against an instruct
checkpoint understates it by more than any prompt-wording choice.

**Calibration is a repair, not an improvement.** It rescues degenerate cells
(letter_only non-chat: 0.0058 -> 0.2396) but *hurts* every model on average
once the prompt is working (-0.0019, -0.0487, -0.0186), and helps in only 1 of
12 cells for Qwen3.5-2B. Reported both ways; the headline uses raw with a
working prompt, and the calibrated column stays because the size of the
correction shows how much of a weak score is prior rather than ignorance.

**Best prompt is model-specific.** `letter_forced` wins for the 0.8B model and
`cataloguer` for both larger ones. A single fixed prompt would have
misordered them, which is the argument for sweeping rather than picking one.

**More context is not uniformly better.** Qwen3.5-2B improves to the full
document (0.4748 -> 0.4894 -> 0.4972) but gemma-4-E2B is *best at 512 tokens*
and loses 0.045 with more (0.4804 -> 0.4330). So the token-fair 512 column and
the full-document column genuinely differ, and reporting only one would favour
a different model.

Next: freeze one configuration per model and score the held-out test split
with bootstrap intervals. Nothing above is a test number; all 72 cells are
validation.
### 2026-09-01 — the 200-document runs were not a valid comparison

Caught on review: the probe was scoring 200 uniformly sampled *validation*
documents while the encoders were scored on all 12,504 *test* documents. Three
mismatches, and the third is the one that matters.

| | encoder | probe as first run |
|---|---|---|
| n | 12,504 | 200 |
| split | test | validation |
| supervision | LogisticRegression fitted on 37,795 labelled documents | none |

`classification.py:388` shows the encoder task is a supervised linear probe:
encode the train split, fit a classifier, predict test. Putting 0.8887 beside a
zero-shot 0.61 would have compared a fitted classifier against no classifier,
in the encoder's favour, and I had not said so.

The 200-document runs stay as what they were, hyperparameter selection on
validation. Every reported number now comes from the full test split, through
`shelf.evaluate.metrics.classification`, the same module behind the encoder
numbers. The zero-shot versus supervised asymmetry cannot be removed and is
recorded in PROTOCOL.md and in every result record.

Also fixed here: `grep` in the launcher pipeline block-buffers, so a running
job looks dead. Use unbuffered output or write straight to a file.
### 2026-09-01 — the self-preference confound, measured

GPT and OpenAI models wrote 68.4% of the test split, so `gpt-5.6-luna` is
largely judging text its own family produced. Measured rather than assumed.

Luna's own-family accuracy is 0.6153 against 0.5882 elsewhere, a gap of
+0.0271. Small, and the ordering does not favour it: the families Luna scores
best are `qwen` (0.6289) and `x` (0.6214), with `gpt` sixth at 0.6150.

The control is a judge with almost no stake. Qwen wrote 2.3% of the corpus. If
the ordering came from family loyalty the two judges would disagree; if it came
from the documents they would agree.

    Spearman across the 12 families: 0.734      Pearson: 0.800

They agree. Documents from `claude`, `z`, `deepseek` and `minimax` are harder
for both judges, and the easy end is shared too. The ordering is a property of
the documents, not of who is grading.

Sharper still: Qwen-0.8B scores *highest* on GPT-written documents (0.4072),
above its own family's (0.3918, fifth). A model preferring its own text would
not do that.

Conclusion for the writeup: the generator-family effect is real but is document
difficulty, not self-preference, and the hosted model's score does not need a
contamination discount on this evidence. The breakdown stays in the results so
a reader can check the claim rather than take it.
### 2026-09-01 — test results, all four decoders

Full test split, 12,504 documents, frozen configurations, zero-shot.
`check_protocol.py` passes: correct n, correct split, no configuration drift,
21 of 21 classes used by every model, no dropped documents.

| model | macro-F1 | accuracy |
|---|---|---|
| Qwen3.5-0.8B | 0.3822 | 0.3912 |
| gemma-4-E2B-it | 0.4557 | 0.4712 |
| Qwen3.5-2B | 0.5075 | 0.5263 |
| gpt-5.6-luna | 0.5860 | 0.6068 |
| *best encoder, supervised probe* | *0.8887* | |

Validation chose well: predicted test within 0.0014 and 0.0102 for the two
Qwen models, so the 200-document sweep was adequate for selection even though
it was never adequate as a result.

Determinism checked by re-running Qwen3.5-0.8B twice on 500 documents:
0.3751676217 both times, bit-identical.

**Self-preference is not present.** Own-family advantage by judge: +0.0006
(Qwen-0.8B), **-0.0533** (Qwen-2B, worse on its own family), +0.0065 (Gemma),
+0.0271 (Luna). All six pairwise Spearman correlations on per-family accuracy
are positive, 0.622 to 0.902, so every judge agrees which generators wrote the
harder documents. The family effect is document difficulty. The hosted model's
score needs no contamination discount on this evidence.

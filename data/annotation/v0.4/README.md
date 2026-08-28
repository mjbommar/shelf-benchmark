# SHELF human validation kit

399 documents, stratified across LCC class and LCGFT category, sampled with
seed 42 from `data/artifacts/v0.4/phase1.jsonl`.

## What this is for

Two things that nothing else in the benchmark can supply:

1. **A human ceiling.** Model scores are uninterpretable without one.
2. **Inter-annotator agreement.** If two trained coders agree only 70% of the
   time on LCC class, then a model scoring 0.89 against the generated labels is
   measuring something other than what a cataloguer would call correct, and the
   benchmark's ceiling is the label noise rather than the task.

## Who should do this

Someone with cataloguing or library-science familiarity. Two coders working
independently. This is deliberately not crowdsourceable: the point is expert
agreement, and a non-expert's disagreement rate would measure the wrong thing.

## Procedure

1. Give each coder `sample.jsonl`, `instructions.md` and their own copy of
   `coding_sheet.csv`.
2. Coders work independently and do not discuss documents.
3. Return both completed sheets.
4. Score:

       uv run python scripts/score_annotations.py \
           --coder-a coder_a.csv --coder-b coder_b.csv --gold gold.jsonl

That reports Cohen's kappa between coders, each coder against the generated
labels, and the resulting human ceiling per task.

## What is deliberately withheld

`sample.jsonl` carries no `lcc_code`, `lcgft_category`, `lcgft_form`, `topics`
or `title`. Gold labels live in `gold.jsonl`, which coders must not see.

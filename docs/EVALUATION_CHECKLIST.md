# Evaluation integrity checklist

Every item here exists because it was violated in this project and produced a
wrong or indefensible number. Run `python scripts/check_evaluation.py` before
any results go into a paper, a dataset card, or a message to the user.

---

## A. One corpus per claim

- [ ] **A1. Every headline number comes from the same corpus.** Do not
      compute the transfer matrix on one corpus and rank agreement on
      another. *Violated: rank agreement used v0.3.0 while the paper
      described the pooled corpus.*
- [ ] **A2. The corpus is named beside every number.** "SHELF scores 0.88"
      is not a claim. "`v0_4_core` scores 0.88" is.
- [ ] **A3. Do not source two properties from two different slices and
      present them as one artifact.** Generator diversity belongs to
      `v0_4_core` (15 models, max 9.24%); scale belongs to `all` (62,899,
      max 47.7%). *Violated: claimed both from one number.*

## B. Complete before reported

- [ ] **B1. A sweep is complete or it is labelled partial in the same
      sentence as its numbers.** Not in a footnote, not in a limitation.
- [ ] **B2. Check the sample is not biased, not just small.** A partial
      sweep finishes cheap models first, so a 9-model table is a
      *small-model* table. State which models are missing. *Violated: the
      headroom table had no model above 109M.*
- [ ] **B3. Count results, never files.** A result file may contain an
      error stub. *Violated: counted 8 subclass "results" that were all
      errors.*

## C. Verify before believing

- [ ] **C1. Never report progress from file counts alone.** Confirm the
      process is alive AND its log has advanced. *Violated: reported "all
      four jobs progressing" for 70 minutes after an OOM killed them.*
- [ ] **C2. `pgrep`/`pkill` match your own shell.** Filter by RSS or use
      explicit PIDs. *Violated twice: killed my own waiter, then read a
      self-match as a live process.*
- [ ] **C3. Speed is not success.** A job that finishes suspiciously fast
      has usually failed. Check output is non-empty. *Violated: "120/120
      complete" was 120 empty rows from a 404.*
- [ ] **C4. Verify a model id against the provider before a run.**
      *Violated: `claude-haiku-4.5` is an OpenRouter alias; the native API
      404s.*

- [ ] **C5. A wrapper must not report success on a failed command.** Check
      the exit code; `echo DONE` after a pipeline reports the exit of the
      last stage, not the program. *Violated: `--task-types multilabel` is
      not a valid choice, argparse exited non-zero, the wrapper printed
      "MULTI WORKER DONE", and a whole task type was silently skipped for
      20 models. It surfaced only from a coverage count.*
- [ ] **C6. Enumerate task types from the config, never from memory.** The
      sweep ran four of the five defined types because the fifth was not in
      the list typed by hand.

## D. Statistics

- [ ] **D1. Report an interval with every correlation and every
      difference.** A point estimate alone is not a finding.
- [ ] **D2. If the interval spans zero, the claim is "not resolved", never
      "no effect".** *Violated: called a -7pp coverage change "not harmed"
      at n=90.*
- [ ] **D3. Resample the unit the claim is about.** Rank agreement over
      models resamples models, not documents.
- [ ] **D4. A single-model result is a pilot.** Add a second model before
      recommending anything. *Violated: recommended a prompt change on one
      generator; a second reversed it.*
- [ ] **D5. Preliminary numbers from partial runs are never cited later.*
      *Violated: a 0.350 correlation at n=12 became 0.781 at n=22.*

## E. Metrics

- [ ] **E1. A metric must not share vocabulary with the treatment.**
      *Violated: scored topic fidelity by cosine to the topic string while
      the treatment suppressed that string.*
- [ ] **E2. Every corpus statistic needs a natural baseline.** "87% verbatim
      labels" means nothing until a human corpus is measured the same way
      (it was 8.9%).
- [ ] **E3. Control the obvious confound before claiming an effect.**
      Length, corpus size, class balance. *The size-balanced transfer
      control is what makes that result publishable.*
- [ ] **E4. Chance-normalise any rate whose base rate varies across
      conditions.**

## F. Resources

- [ ] **F1. Check system RAM, not just GPU, before running jobs in
      parallel.** *Violated: OOM'd the machine at 61GB.*
- [ ] **F2. Run every long job under a memory cap**
      (`systemd-run --user --scope -p MemoryMax=`).
- [ ] **F3. Know the cost of a task before scaling it.** An N-by-N
      similarity over 62,899 documents is 15.8GB per matrix.
- [ ] **F4. Verify which GPU a job actually landed on**, with
      `nvidia-smi --query-compute-apps=pid,gpu_uuid` mapped through
      `--query-gpu=index,uuid`. Two traps stack here. `systemd-run --scope`
      does not inherit `CUDA_VISIBLE_DEVICES`, so pass `--setenv`. And CUDA
      orders GPUs fastest-first while `nvidia-smi` orders by PCI bus, so
      `CUDA_VISIBLE_DEVICES=1` can select the card `nvidia-smi` calls 0.
      Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. *Violated: two workers piled onto
      a GPU already running another user's job while the second sat idle,
      and the wasted capacity was read as normal load.*

## G. Reproducibility

- [ ] **G1. Never overwrite an aggregate from a partial run.** *Violated:
      one fine-tune run replaced a 25-model summary and a published
      headline could not be reproduced.*
- [ ] **G2. Re-derive a headline from stored per-task files before citing
      it.** *The 0.679 headline was actually 0.502.*
- [ ] **G3. Record the corpus, code version, seed, and library versions
      with every result.**

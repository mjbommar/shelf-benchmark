"""Score decoders on the full test split, the same population the encoders used.

The validation sweep chose one configuration per model. This scores that frozen
configuration on the whole 12,504-document test split, so the population,
split, and metric match what `results/pooled/baselines/*_lcc_classification.json`
records for the encoders.

One asymmetry cannot be removed and must be read with the numbers. The encoder
task is a supervised linear probe: encode the 37,795-document train split, fit
LogisticRegression, predict test. The decoders here are zero-shot and see no
labelled example. A decoder trailing an encoder is therefore not evidence that
its representations are worse; it is a comparison between a fitted classifier
and no classifier at all.

Metrics come from shelf.evaluate.metrics.classification, the module that
produced the encoder numbers, rather than a second implementation that could
drift from it.

Usage:
    .venv-gen/bin/python scripts/generative/run_test.py \\
        --model Qwen/Qwen3.5-2B --prompt cataloguer --tokens 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from classify import LCC, render, truncate_tokens  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, required=True, help="0 = whole document")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = every document")
    ap.add_argument("--output", default="results/generative/test_scores.jsonl")
    args = ap.parse_args()

    import polars as pl
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pl.read_parquet(f"data/hf_dataset/all/{args.split}.parquet")
    if args.limit:
        df = df.head(args.limit)
    docs, gold = df["text"].to_list(), df["lcc_code"].to_list()
    fam = [
        m.replace("anthropic/", "").split("-")[0].split("/")[0]
        for m in df["model"].to_list()
    ]
    logger.info(f"  {args.model}: {len(docs):,} documents from {args.split}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    codes = list(LCC)
    variants = []
    for c in codes:
        ids = [
            t[0]
            for t in (tok(f, add_special_tokens=False).input_ids for f in (c, f" {c}"))
            if len(t) == 1
        ]
        if not ids:
            logger.error(f"class {c} has no single-token form")
            return 2
        variants.append(ids)

    preds = []
    t0 = time.time()
    for i, d in enumerate(docs):
        s = render(args.prompt, truncate_tokens(tok, d, args.tokens))
        s = tok.apply_chat_template(
            [{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(s, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            lg = model(ids, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(lg, dim=-1)
        sc = torch.stack([torch.logsumexp(lp[v], dim=0) for v in variants])
        preds.append(codes[int(sc.argmax())])
        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            logger.info(
                f"    {i + 1:,}/{len(docs):,}  {el / (i + 1):.3f}s/doc  "
                f"eta {(len(docs) - i - 1) * el / (i + 1) / 60:.0f}m"
            )

    by_family = {}
    for f in sorted(set(fam)):
        idx = [i for i, x in enumerate(fam) if x == f]
        if len(idx) >= 50:
            by_family[f] = {
                "n": len(idx),
                "accuracy": sum(gold[i] == preds[i] for i in idx) / len(idx),
            }

    out = {
        "model": args.model,
        "prompt_style": args.prompt,
        "token_budget": args.tokens,
        "split": args.split,
        "num_samples": len(preds),
        "protocol": "zero-shot; encoders on this task use a supervised linear probe",
        "macro_f1": f1_score(gold, preds, average="macro", zero_division=0.0),
        "micro_f1": f1_score(gold, preds, average="micro", zero_division=0.0),
        "weighted_f1": f1_score(gold, preds, average="weighted", zero_division=0.0),
        "accuracy": sum(a == b for a, b in zip(gold, preds, strict=True)) / len(preds),
        "classes_used": len(set(preds)),
        "accuracy_by_generator_family": by_family,
        "seconds": round(time.time() - t0, 1),
    }
    logger.info(
        f"  DONE {args.model}: macro_f1={out['macro_f1']:.4f} "
        f"acc={out['accuracy']:.4f} ({out['classes_used']}/21 classes)"
    )
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

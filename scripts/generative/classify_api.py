"""Classify SHELF documents with a hosted model, constrained to the label set.

The local probe scores each label as a continuation, so a prediction is always
one of the 21 classes and a weak model is measured as weak rather than as
malformed. `gpt-5.6-luna` does not expose logprobs, so that method is not
available and free generation would reintroduce the parsing error the local
harness was built to avoid.

Structured outputs restore the property by a different route: a JSON schema
with an `enum` of the 21 class letters cannot return anything else. The two
arms are therefore comparable in what they constrain, and different in how:
likelihood ranking over all labels for local models, constrained decoding for
the hosted one. That difference is real and is recorded with the results
rather than smoothed over.

Two further asymmetries, stated because they bear on interpretation:

* This is a reasoning model and spends tokens thinking before answering; the
  local models were scored at a single position with no reasoning budget.
* GPT and OpenAI models wrote 68.4% of the pooled corpus and 68.4% of the test
  split. A GPT judge is being asked about text its own family largely wrote,
  so a favourable result here is not clean evidence of superior ability. The
  per-generator-family breakdown is emitted so the size of that effect can be
  read rather than assumed.

Usage:
    .venv-gen/bin/python scripts/generative/classify_api.py \\
        --model gpt-5.6-luna --prompt cataloguer --tokens 512 --n 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from classify import LCC, render, truncate_tokens  # noqa: E402

CODES = list(LCC)
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "lcc",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"lcc_class": {"type": "string", "enum": CODES}},
            "required": ["lcc_class"],
            "additionalProperties": False,
        },
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--prompt", default="cataloguer")
    ap.add_argument("--tokens", type=int, default=512, help="0 = whole document")
    ap.add_argument("--n", type=int, default=200, help="0 = every document")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument(
        "--balanced",
        action="store_true",
        help="equal quota per generator family, for the self-preference check",
    )
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    import polars as pl
    from openai import OpenAI
    from sklearn.metrics import f1_score
    from transformers import AutoTokenizer

    df = pl.read_parquet(f"data/hf_dataset/all/{args.split}.parquet")
    if args.balanced:
        # A uniform sample cannot test self-preference: GPT wrote 68% of the
        # corpus, so at n=200 every other family lands under 20 documents.
        # Draw an equal quota per generator family instead. This sample is for
        # the family contrast only -- its macro-F1 is not comparable to the
        # uniform runs, because the label mix differs.
        df = df.with_columns(
            pl.col("model")
            .str.replace(r"^anthropic/", "")
            .str.split("-")
            .list.get(0)
            .str.split("/")
            .list.get(0)
            .alias("_fam")
        )
        keep = [
            f for f, c in df["_fam"].value_counts().iter_rows() if c >= args.per_family
        ]
        df = (
            df.filter(pl.col("_fam").is_in(keep))
            .group_by("_fam")
            .head(args.per_family)
            .drop("_fam")
        )
    else:
        if args.n:
            df = df.sample(n=min(args.n, len(df)), seed=args.seed)
    docs = df["text"].to_list()
    gold = df["lcc_code"].to_list()
    fam = [
        m.replace("anthropic/", "").split("-")[0].split("/")[0]
        for m in df["model"].to_list()
    ]

    # Truncate on a real tokenizer so the budget means the same thing as it
    # does for the local models, rather than being a character guess.
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    usage = defaultdict(int)

    def one(doc: str) -> str | None:
        prompt = render(args.prompt, truncate_tokens(tok, doc, args.tokens))
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=SCHEMA,
                    max_completion_tokens=args.max_completion,
                )
                u = r.usage
                usage["prompt"] += u.prompt_tokens
                usage["completion"] += u.completion_tokens
                d = getattr(u, "completion_tokens_details", None)
                usage["reasoning"] += getattr(d, "reasoning_tokens", 0) or 0
                return json.loads(r.choices[0].message.content)["lcc_class"]
            except Exception as exc:  # transient API failure
                if attempt == 3:
                    logger.warning(f"  giving up on a document: {type(exc).__name__}")
                    return None
                time.sleep(2**attempt)
        return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        preds = list(ex.map(one, docs))

    # A dropped document is excluded from scoring and counted, never guessed:
    # imputing a label would quietly change the metric.
    keep = [i for i, p in enumerate(preds) if p is not None]
    g = [gold[i] for i in keep]
    p = [preds[i] for i in keep]

    by_family: dict[str, dict] = {}
    for f in sorted(set(fam)):
        idx = [i for i in keep if fam[i] == f]
        if len(idx) >= 50:
            by_family[f] = {
                "n": len(idx),
                "accuracy": sum(gold[i] == preds[i] for i in idx) / len(idx),
            }

    out = {
        "model": args.model,
        "prompt_style": args.prompt,
        "token_budget": args.tokens,
        "n_requested": len(docs),
        "n_scored": len(p),
        "n_failed": len(docs) - len(p),
        "split": args.split,
        "seed": args.seed,
        "method": "constrained decoding via json_schema enum (no logprobs available)",
        "macro_f1": f1_score(g, p, average="macro", zero_division=0.0),
        "accuracy": sum(a == b for a, b in zip(g, p, strict=True)) / len(p),
        "classes_used": len(set(p)),
        "accuracy_by_generator_family": by_family,
        "tokens": dict(usage),
        "seconds": round(time.time() - t0, 1),
    }
    logger.info(
        f"  {args.model} {args.prompt} tok={args.tokens or 'full'}: "
        f"macroF1={out['macro_f1']:.4f} acc={out['accuracy']:.4f} "
        f"({out['classes_used']}/21 classes, {out['n_failed']} failed, "
        f"{usage['prompt']:,}p/{usage['completion']:,}c tokens)"
    )
    if args.output:
        q = Path(args.output)
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

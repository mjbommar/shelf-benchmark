"""Sweep prompt and decoding settings for one decoder, loading it once.

Running classify.py per configuration reloads the weights every time, which
dominates the cost. This loads a model once and walks the grid, appending one
JSON line per cell so a crash keeps whatever finished.

The grid is prompt style x chat template x document budget. Calibration is not
a grid axis: both raw and calibrated scores come from the same forward passes,
so every cell reports both.

Usage:
    .venv-gen/bin/python scripts/generative/sweep.py --model Qwen/Qwen3.5-0.8B \\
        --n 200 --out results/generative/sweep_qwen08.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from classify import LCC, NULL_DOC, render, truncate_tokens  # noqa: E402

PROMPTS = ("letter_only", "letter_forced", "cataloguer", "terse")
# 512 and 2048 are the caps the paper's dense encoders run under;
# 0 is the whole document, which is what a long-context decoder sees.
BUDGETS = (512, 2048, 0)
CHAT = (True, False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import polars as pl
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = pl.read_parquet(f"data/hf_dataset/all/{args.split}.parquet")
    df = df.sample(n=min(args.n, len(df)), seed=args.seed)
    docs, gold = df["text"].to_list(), df["lcc_code"].to_list()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    codes = list(LCC)
    # Surface form is not fixed: after a colon the model emits " A", after a
    # newline it emits "A", and scoring the wrong one compares tokens holding
    # no probability mass. Collect both ids per label and marginalise.
    variants: list[list[int]] = []
    for c in codes:
        ids = []
        for form in (c, f" {c}"):
            t = tok(form, add_special_tokens=False).input_ids
            if len(t) == 1:
                ids.append(t[0])
        if not ids:
            logger.error(f"class {c} has no single-token form under this tokenizer")
            return 2
        variants.append(ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("a")

    def lp_for(style: str, chat: bool, text: str):
        s = render(style, text)
        if chat:
            s = tok.apply_chat_template(
                [{"role": "user", "content": s}],
                tokenize=False,
                add_generation_prompt=True,
            )
        ids = tok(s, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            # Only the final position is ever read. Without logits_to_keep the
            # model materialises logits for every position: a 6341-token
            # document against Gemma's 262k vocabulary is 3.3 GB in bf16, and
            # .float() doubles it. That is what OOM'd a 16 GB card.
            lg = model(ids, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(lg, dim=-1)
        # log-sum-exp over the surface forms of each label
        return torch.stack([torch.logsumexp(lp[v], dim=0) for v in variants])

    def stats(p):
        return {
            "macro_f1": f1_score(gold, p, average="macro", zero_division=0.0),
            "accuracy": sum(a == b for a, b in zip(gold, p, strict=True)) / len(gold),
            "classes_used": len(set(p)),
        }

    logger.info(
        f"{'prompt':<13}{'chat':>5}{'tokens':>8}{'raw F1':>9}{'cal F1':>9}{'used':>6}{'mass':>8}{'s':>7}"
    )
    logger.info("-" * 56)
    for style in PROMPTS:
        for chat in CHAT:
            for w in BUDGETS:
                t0 = time.time()
                bias = lp_for(style, chat, NULL_DOC)
                raw, cal = [], []
                for d in docs:
                    lp = lp_for(style, chat, truncate_tokens(tok, d, w))
                    raw.append(codes[int(lp.argmax())])
                    cal.append(codes[int((lp - bias).argmax())])
                r, c = stats(raw), stats(cal)
                mass = float(torch.exp(lp_for(style, chat, docs[0][:2000])).sum())
                rec = {
                    "model": args.model,
                    "prompt_style": style,
                    "chat_template": chat,
                    "token_budget": w,
                    "n": len(docs),
                    "split": args.split,
                    "seed": args.seed,
                    "raw": r,
                    "calibrated": c,
                    "label_mass_example": round(mass, 4),
                    "seconds": round(time.time() - t0, 1),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                logger.info(
                    f"{style:<13}{int(chat):>5}{(w or 9999):>8}{r['macro_f1']:>9.4f}"
                    f"{c['macro_f1']:>9.4f}{c['classes_used']:>6}{mass:>8.3f}{rec['seconds']:>7.0f}"
                )
    fh.close()
    logger.info(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

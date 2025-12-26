#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from transformers import AutoTokenizer

EOS_LINE = "<|endoftext|>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default='/mnt/mnt/zjdx/tyhcyq/valid_corpus_3/corpus.txt', help="Input txt (contains <|endoftext|> as separator line).")
    ap.add_argument("--output", default='/mnt/mnt/zjdx/tyhcyq/valid_corpus_3/corpus.bin', help="Output bin (uint16 flat token ids).")
    ap.add_argument("--tokenizer", default="gpt2", help="HF tokenizer name (default: gpt2).")
    ap.add_argument("--max_doc_mb", type=int, default=512,
                    help="Safety cap: abort if a single doc exceeds this size (MB).")
    ap.add_argument("--log_every", type=int, default=1000, help="Print progress every N docs.")
    args = ap.parse_args()


    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = int(tok.eos_token_id)  # GPT-2: 50256

    max_doc_bytes = args.max_doc_mb * 1024 * 1024

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    doc_buf = []
    doc_bytes = 0
    n_docs = 0
    n_tokens = 0

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, open(args.output, "wb") as fout:
        for line in fin:
            # treat a line equal to "<|endoftext|>" as doc boundary
            if line.strip() == EOS_LINE:
                text = "".join(doc_buf)
                doc_buf.clear()
                doc_bytes = 0

                # allow empty docs, but still write an EOS to keep segmentation consistent
                ids = tok.encode(text, add_special_tokens=False)
                ids.append(eos_id)

                # GPT-2 ids fit into uint16
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(fout)

                n_docs += 1
                n_tokens += arr.size

                if args.log_every > 0 and (n_docs % args.log_every == 0):
                    print(f"[tokenize] docs={n_docs} tokens={n_tokens}")

                continue

            doc_buf.append(line)
            doc_bytes += len(line.encode("utf-8", errors="replace"))

            if doc_bytes > max_doc_bytes:
                raise RuntimeError(
                    f"Single doc exceeded {args.max_doc_mb} MB before hitting <|endoftext|>. "
                    f"Likely missing separators or one doc is huge. "
                    f"Increase --max_doc_mb or fix input."
                )

        # handle tail: if file doesn't end with <|endoftext|>, flush remaining as last doc
        if doc_buf:
            text = "".join(doc_buf)
            ids = tok.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(fout)
            n_docs += 1
            n_tokens += arr.size

    print(f"Done. docs={n_docs} tokens={n_tokens} output={args.output}")

if __name__ == "__main__":
    main()

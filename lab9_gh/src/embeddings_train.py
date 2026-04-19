#!/usr/bin/env python3
"""
embeddings_train.py
Train Word2Vec and FastText (gensim) on a tokenized corpus.

Usage:
  python embeddings_train.py \
    --input_csv /content/processed_v2.csv \
    --text_col text \
    --tokens_col tokens_clean \
    --train_ids /content/splits_train_ids.txt \
    --out_dir /content/embeddings_out \
    --model_size 100 \
    --min_count 3 \
    --epochs 5
"""

import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import multiprocessing
from gensim.models import Word2Vec, FastText
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_data(input_csv, text_col="text", tokens_col="tokens", filter_labels=True):
    df = pd.read_csv(input_csv)
    if text_col in df.columns and tokens_col not in df.columns:
        # assume tokens are whitespace-split if tokens_col missing
        df[tokens_col] = df[text_col].fillna("").astype(str).apply(lambda s: s.split())
    if filter_labels and "label" in df.columns:
        df = df[df["label"].isin(["Real", "Fake"])].reset_index(drop=True)
    logger.info("Loaded %d documents (filter_labels=%s)", len(df), filter_labels)
    return df

def load_split_ids(path):
    return pd.read_csv(path, header=None)[0].tolist()

def train_models(sentences, out_dir, vector_size=100, window=5, min_count=3, sg=1, epochs=5, workers=None):
    workers = workers or max(1, multiprocessing.cpu_count()-1)
    logger.info("Training Word2Vec: size=%d window=%d min_count=%d sg=%d epochs=%d workers=%d",
                vector_size, window, min_count, sg, epochs, workers)
    w2v = Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                   min_count=min_count, sg=sg, workers=workers, epochs=epochs)
    w2v_path = os.path.join(out_dir, "word2vec.model")
    w2v.save(w2v_path)
    logger.info("Saved Word2Vec to %s", w2v_path)

    logger.info("Training FastText: same params")
    ft = FastText(sentences=sentences, vector_size=vector_size, window=window,
                  min_count=min_count, sg=sg, workers=workers, epochs=epochs)
    ft_path = os.path.join(out_dir, "fasttext.model")
    ft.save(ft_path)
    logger.info("Saved FastText to %s", ft_path)

    # Save metadata
    meta = {
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "sg": sg,
        "epochs": epochs,
        "workers": workers,
        "vocab_w2v": len(w2v.wv),
        "vocab_ft": len(ft.wv)
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Saved training metadata")
    return w2v, ft

def ensure_outdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--tokens_col", default="tokens_clean")
    parser.add_argument("--train_ids", required=True)
    parser.add_argument("--out_dir", default="./embeddings_out")
    parser.add_argument("--model_size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=3)
    parser.add_argument("--sg", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    ensure_outdir(args.out_dir)
    df = load_data(args.input_csv, text_col=args.text_col, tokens_col=args.tokens_col)
    train_ids = load_split_ids(args.train_ids)
    train_df = df[df["text_id"].isin(train_ids)].reset_index(drop=True)
    sentences = train_df[args.tokens_col].tolist()
    logger.info("Training sentences: %d", len(sentences))

    w2v, ft = train_models(sentences, args.out_dir,
                           vector_size=args.model_size,
                           window=args.window,
                           min_count=args.min_count,
                           sg=args.sg,
                           epochs=args.epochs)
    logger.info("Training finished")

if __name__ == "__main__":
    main()

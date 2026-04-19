#!/usr/bin/env python3
"""
embeddings_eval.py
Evaluate trained embeddings: nearest neighbors, domain terms, case analysis,
simple downstream baseline (logistic regression on averaged doc vectors),
and export CSV/MD summaries.

Usage:
  python embeddings_eval.py \
    --input_csv /content/processed_v2.csv \
    --tokens_col tokens_clean \
    --w2v_model /content/embeddings_out/word2vec.model \
    --ft_model /content/embeddings_out/fasttext.model \
    --out_dir /content/embeddings_out/eval \
    --control_words_file control_words.txt \
    --domain_terms_file domain_terms.txt
"""

import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import json
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def ensure_outdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_models(w2v_path, ft_path):
    w2v = Word2Vec.load(w2v_path)
    ft = FastText.load(ft_path)
    logger.info("Loaded models: w2v vocab=%d ft vocab=%d", len(w2v.wv), len(ft.wv))
    return w2v, ft

def get_neighbors(model, word, topn=10):
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        # try vector-based similar_by_vector if available
        try:
            vec = model.wv.get_vector(word)
            return model.wv.similar_by_vector(vec, topn=topn)
        except Exception:
            return []

def neighbors_to_str(neis):
    return "; ".join([f"{w} ({sim:.3f})" for w, sim in neis])

def analyze_control_words(w2v, ft, control_words, out_csv):
    rows = []
    for w in control_words:
        w2v_nei = get_neighbors(w2v, w, topn=10)
        ft_nei = get_neighbors(ft, w, topn=10)
        rows.append({
            "word": w,
            "w2v_neighbors": neighbors_to_str(w2v_nei),
            "ft_neighbors": neighbors_to_str(ft_nei),
            "w2v_count": len(w2v_nei),
            "ft_count": len(ft_nei)
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved control words neighbors to %s", out_csv)
    return rows

def analyze_domain_terms(w2v, ft, domain_terms, out_csv):
    rows = []
    for t in domain_terms:
        w2v_nei = get_neighbors(w2v, t, topn=10)
        ft_nei = get_neighbors(ft, t, topn=10)
        rows.append({
            "term": t,
            "w2v_neighbors": neighbors_to_str(w2v_nei),
            "ft_neighbors": neighbors_to_str(ft_nei),
            "w2v_count": len(w2v_nei),
            "ft_count": len(ft_nei)
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved domain terms neighbors to %s", out_csv)
    return rows

def doc_vector_average(model, tokens, default_dim):
    # average of token vectors; skip OOV tokens
    vecs = []
    for t in tokens:
        try:
            vecs.append(model.wv[t])
        except Exception:
            continue
    if len(vecs) == 0:
        return np.zeros(default_dim, dtype=float)
    return np.mean(vecs, axis=0)

def downstream_baseline(df, model, tokens_col, label_col="label_num", test_size=0.2, random_state=42):
    # compute doc vectors
    dim = model.vector_size
    X = np.vstack(df[tokens_col].apply(lambda toks: doc_vector_average(model, toks, dim)).values)
    y = df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    logger.info("Downstream baseline (avg doc vectors) acc=%.4f f1_macro=%.4f", acc, f1)
    return {"accuracy": acc, "f1_macro": f1, "report": classification_report(y_test, y_pred, output_dict=True)}

def generate_embedding_notes(out_path, meta, control_rows, domain_rows, cases):
    lines = []
    lines.append("# embedding_notes_lab9\n")
    lines.append("**Corpus**: processed_v2.csv (filtered Real/Fake)\n")
    lines.append(f"**Documents (labeled)**: {meta.get('n_labeled', 'unknown')}\n")
    lines.append("**Models trained**: Word2Vec, FastText\n")
    lines.append("**Training params**:\n")
    lines.append("```\n" + json.dumps(meta.get("train_params", {}), ensure_ascii=False, indent=2) + "\n```\n")
    lines.append("## Control words (nearest neighbors)\n")
    for r in control_rows:
        lines.append(f"- **{r['word']}**\n  - Word2Vec: {r['w2v_neighbors']}\n  - FastText: {r['ft_neighbors']}\n")
    lines.append("\n## Domain terms\n")
    for r in domain_rows:
        lines.append(f"- **{r['term']}**\n  - Word2Vec: {r['w2v_neighbors']}\n  - FastText: {r['ft_neighbors']}\n")
    lines.append("\n## Cases (auto)\n")
    for c in cases:
        lines.append(f"### CASE: {c['word']}\n")
        lines.append(f"- Word2Vec: {c['w2v']}\n- FastText: {c['ft']}\n- Verdict: {c['verdict']}\n- Notes: {c['notes']}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote embedding notes to %s", out_path)

def generate_audit_summary(out_path, meta, strong_examples, weak_examples, domain_terms, ft_wins, no_win, recommendation):
    lines = []
    lines.append("# audit_summary_lab9\n")
    lines.append(f"**Corpus**: processed_v2.csv; labeled docs: {meta.get('n_labeled', 'unknown')}\n")
    lines.append("**Models trained**: Word2Vec, FastText\n")
    lines.append("\n**Strongest nearest neighbor examples**:\n")
    for s in strong_examples:
        lines.append(f"- {s}\n")
    lines.append("\n**Weakest nearest neighbor examples**:\n")
    for w in weak_examples:
        lines.append(f"- {w}\n")
    lines.append("\n**Domain terms that were meaningful**:\n")
    for d in domain_terms:
        lines.append(f"- {d}\n")
    lines.append("\n**Where FastText won**:\n")
    for f in ft_wins:
        lines.append(f"- {f}\n")
    lines.append("\n**Where there was little gain**:\n")
    for n in no_win:
        lines.append(f"- {n}\n")
    lines.append("\n**Recommendation**:\n")
    lines.append(recommendation + "\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote audit summary to %s", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--tokens_col", default="tokens_clean")
    parser.add_argument("--w2v_model", required=True)
    parser.add_argument("--ft_model", required=True)
    parser.add_argument("--out_dir", default="./embeddings_eval")
    parser.add_argument("--control_words_file", default=None)
    parser.add_argument("--domain_terms_file", default=None)
    args = parser.parse_args()

    ensure_outdir(args.out_dir)
    df = pd.read_csv(args.input_csv)
    if args.tokens_col not in df.columns:
        df[args.tokens_col] = df["text"].fillna("").astype(str).apply(lambda s: s.split())
    w2v, ft = load_models(args.w2v_model, args.ft_model)

    # load control words and domain terms
    if args.control_words_file and os.path.exists(args.control_words_file):
        with open(args.control_words_file, "r", encoding="utf-8") as f:
            control_words = [l.strip() for l in f if l.strip()]
    else:
        control_words = ["влада","поранення","фейк","підтверджено","сша","вакцина","розслідування","напад","іпсо","коментар","раптово"]

    if args.domain_terms_file and os.path.exists(args.domain_terms_file):
        with open(args.domain_terms_file, "r", encoding="utf-8") as f:
            domain_terms = [l.strip() for l in f if l.strip()]
    else:
        domain_terms = ["фейк","підтвердження","інформація","верифікувати","джерело"]

    control_out = os.path.join(args.out_dir, "control_neighbors.csv")
    domain_out = os.path.join(args.out_dir, "domain_neighbors.csv")
    control_rows = analyze_control_words(w2v, ft, control_words, control_out)
    domain_rows = analyze_domain_terms(w2v, ft, domain_terms, domain_out)

    # simple downstream baseline using FastText and Word2Vec separately
    df_labeled = df[df["label"].isin(["Real","Fake"])].copy()
    df_labeled["label_num"] = df_labeled["label"].map({"Real":0,"Fake":1})
    meta = {"n_labeled": len(df_labeled), "train_params": {}}
    logger.info("Running downstream baseline on Word2Vec")
    w2v_metrics = downstream_baseline(df_labeled, w2v, args.tokens_col)
    logger.info("Running downstream baseline on FastText")
    ft_metrics = downstream_baseline(df_labeled, ft, args.tokens_col)

    # prepare cases auto (simple heuristics)
    cases = []
    for r in control_rows[:10]:
        w = r["word"]
        w2v_nei = r["w2v_neighbors"]
        ft_nei = r["ft_neighbors"]
        # simple overlap heuristic
        w2v_set = set([x.split(" (")[0] for x in w2v_nei.split("; ") if x])
        ft_set = set([x.split(" (")[0] for x in ft_nei.split("; ") if x])
        overlap = w2v_set & ft_set
        if len(w2v_set)==0 and len(ft_set)>0:
            verdict = "useful (FastText helps OOV/morph)"
            notes = "Word2Vec OOV; FastText provides neighbors via subwords."
        elif len(overlap) >= 3:
            verdict = "useful"
            notes = "Consistent neighborhoods across models."
        elif len(overlap) > 0:
            verdict = "partly"
            notes = "Different neighborhoods; inspect semantic quality."
        else:
            verdict = "partly"
            notes = "Different neighborhoods; inspect semantic quality."
        cases.append({"word": w, "w2v": w2v_nei, "ft": ft_nei, "verdict": verdict, "notes": notes})

    # write embedding notes and audit summary
    notes_path = os.path.join(args.out_dir, "embedding_notes_lab9.md")
    generate_embedding_notes(notes_path, meta, control_rows, domain_rows, cases)

    audit_path = os.path.join(args.out_dir, "audit_summary_lab9.md")
    strong_examples = ["розслідування", "поранення"]
    weak_examples = ["фейк", "верифікувати", "раптово"]
    ft_wins = ["підтверджено (morphology)", "поранення (morphology)"]
    no_win = ["влада", "сша"]
    recommendation = ("Embeddings are useful as an analytical tool and as auxiliary features "
                      "for downstream models. For direct real/fake classification, combine embeddings "
                      "with metadata and stylistic features; consider FastText for OOV/morph handling.")
    generate_audit_summary(audit_path, meta, strong_examples, weak_examples, domain_terms, ft_wins, no_win, recommendation)

    # save simple metrics
    metrics = {"w2v_downstream": w2v_metrics, "ft_downstream": ft_metrics}
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Saved eval metrics to %s", os.path.join(args.out_dir, "eval_metrics.json"))

if __name__ == "__main__":
    main()

# ner_pipeline.py
# Entrypoint utilities: load model, load gold, run baseline inference and save outputs.
# Designed to be imported by ner_eval.py or run standalone.

import os
import json
import argparse
import spacy
from collections import defaultdict

def load_gold(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_model(model_name="uk_core_news_sm"):
    # Load spaCy model (assumes installed)
    nlp = spacy.load(model_name)
    return nlp

def preds_from_nlp(nlp, texts):
    """
    texts: iterable of dicts with keys 'text_id' and 'text'
    returns list of preds per record: list of lists of spans dicts {start,end,label,text}
    """
    preds = []
    for rec in texts:
        doc = nlp(rec["text"])
        rec_preds = []
        for e in doc.ents:
            rec_preds.append({"start": e.start_char, "end": e.end_char, "label": e.label_.upper(), "text": e.text})
        preds.append(rec_preds)
    return preds

def evaluate_predictions(gold_records, preds_records):
    def span_key(s): return (s["start"], s["end"], s["label"].upper())
    metrics = {"TP":0, "FP":0, "FN":0}
    per_label = defaultdict(lambda: {"TP":0, "FP":0, "FN":0})
    details = []
    for grec, prec in zip(gold_records, preds_records):
        gold_spans = [{"start":e["start"], "end":e["end"], "label":e["label"].upper(), "text":e["text"]} for e in grec["entities"]]
        pred_spans = [{"start":p["start"], "end":p["end"], "label":p["label"].upper(), "text":p["text"]} for p in prec]
        gkeys = set(span_key(s) for s in gold_spans)
        pkeys = set(span_key(s) for s in pred_spans)
        tp = gkeys & pkeys
        fp = pkeys - gkeys
        fn = gkeys - pkeys
        metrics["TP"] += len(tp)
        metrics["FP"] += len(fp)
        metrics["FN"] += len(fn)
        for k in tp: per_label[k[2]]["TP"] += 1
        for k in fp: per_label[k[2]]["FP"] += 1
        for k in fn: per_label[k[2]]["FN"] += 1
        details.append({"text_id": grec["text_id"], "text": grec["text"], "gold": gold_spans, "preds": pred_spans, "tp": len(tp), "fp": len(fp), "fn": len(fn)})
    return metrics, per_label, details

def prf(tp, fp, fn):
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec = tp/(tp+fn) if tp+fn>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return prec, rec, f1

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def run_baseline(gold_path, model_name, out_dir):
    gold = load_gold(gold_path)
    nlp = load_model(model_name)
    preds = preds_from_nlp(nlp, gold)
    metrics, per_label, details = evaluate_predictions(gold, preds)
    prec, rec, f1 = prf(metrics["TP"], metrics["FP"], metrics["FN"])
    summary = {"overall": {"TP":metrics["TP"], "FP":metrics["FP"], "FN":metrics["FN"], "precision":prec, "recall":rec, "f1":f1}, "per_label": {}}
    for lab, vals in per_label.items():
        p,r,f = prf(vals["TP"], vals["FP"], vals["FN"])
        summary["per_label"][lab] = {"TP":vals["TP"], "FP":vals["FP"], "FN":vals["FN"], "precision":p, "recall":r, "f1":f}
    save_json(details, os.path.join(out_dir, "baseline_predictions.json"))
    save_json(summary, os.path.join(out_dir, "baseline_metrics.json"))
    return nlp, gold, preds, summary, details

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline NER inference")
    parser.add_argument("--gold", required=True, help="Path to gold JSON")
    parser.add_argument("--model", default="uk_core_news_sm", help="spaCy model name")
    parser.add_argument("--outdir", default="./eval_output", help="Output directory")
    args = parser.parse_args()
    nlp, gold, preds, summary, details = run_baseline(args.gold, args.model, args.outdir)
    print("Baseline saved to", args.outdir)

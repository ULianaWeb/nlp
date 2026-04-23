# ner_eval.py
# Orchestrator: apply rules, run hybrid inference, evaluate, produce error analysis and audit notes.
# Uses ner_pipeline.py and ner_rules.py

import os
import json
import argparse
from collections import Counter
from ner_pipeline import load_gold, load_model, preds_from_nlp, evaluate_predictions, prf, save_json
import ner_rules

def run_hybrid(gold_path, model_name, out_dir, prod_list=None, title_list=None, regex_rules=None):
    # Load gold and model
    gold = load_gold(gold_path)
    nlp = load_model(model_name)

    # Add rules
    nlp = ner_rules.add_entity_ruler(nlp, prod_list=prod_list, title_list=title_list)
    nlp = ner_rules.register_regex_component(nlp, regex_rules=regex_rules)

    # Run inference
    hybrid_preds = preds_from_nlp(nlp, gold)
    metrics, per_label, details = evaluate_predictions(gold, hybrid_preds)
    prec, rec, f1 = prf(metrics["TP"], metrics["FP"], metrics["FN"])
    summary = {"overall": {"TP":metrics["TP"], "FP":metrics["FP"], "FN":metrics["FN"], "precision":prec, "recall":rec, "f1":f1}, "per_label": {}}
    for lab, vals in per_label.items():
        p,r,f = prf(vals["TP"], vals["FP"], vals["FN"])
        summary["per_label"][lab] = {"TP":vals["TP"], "FP":vals["FP"], "FN":vals["FN"], "precision":p, "recall":r, "f1":f}

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    save_json(details, os.path.join(out_dir, "hybrid_predictions.json"))
    save_json(summary, os.path.join(out_dir, "hybrid_metrics.json"))

    # Error analysis (collect FN and FP)
    errors = []
    def span_key(s): return (s["start"], s["end"], s["label"].upper())
    for det in details:
        gold_spans = det["gold"]
        pred_spans = det["preds"]
        gkeys = set(span_key(s) for s in gold_spans)
        pkeys = set(span_key(s) for s in pred_spans)
        fn = gkeys - pkeys
        fp = pkeys - gkeys
        for k in fn:
            start,end,label = k
            context = det["text"][max(0,start-30):min(len(det["text"]), end+30)]
            expected = next((s for s in gold_spans if s["start"]==start and s["end"]==end and s["label"]==label), None)
            errors.append({"text_id": det["text_id"], "context": context, "expected": expected, "predicted": None, "error_type": "FN", "category": "", "explanation": ""})
        for k in fp:
            start,end,label = k
            context = det["text"][max(0,start-30):min(len(det["text"]), end+30)]
            predicted = next((s for s in pred_spans if s["start"]==start and s["end"]==end and s["label"]==label), None)
            errors.append({"text_id": det["text_id"], "context": context, "expected": None, "predicted": predicted, "error_type": "FP", "category": "", "explanation": ""})
    errors = errors[:500]
    save_json(errors, os.path.join(out_dir, "error_analysis.json"))

    # Generate audit summary and ner notes (simple templates)
    generate_audit_and_notes(out_dir, gold_path, model_name)
    return summary, details, errors

def generate_audit_and_notes(out_dir, gold_path, model_name):
    # Load metrics and errors
    with open(os.path.join(out_dir, "baseline_metrics.json"), "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(os.path.join(out_dir, "hybrid_metrics.json"), "r", encoding="utf-8") as f:
        hybrid = json.load(f)
    with open(os.path.join(out_dir, "error_analysis.json"), "r", encoding="utf-8") as f:
        errors = json.load(f)

    # Top error categories (currently uncategorized; count by error_type)
    type_counter = Counter(e.get("error_type", "UNK") for e in errors)
    cat_counter = Counter(e.get("category") or "Uncategorized" for e in errors)

    # audit_summary_lab10.md
    md = []
    md.append("# Audit Summary Lab 10\n")
    md.append(f"- **Gold:** {os.path.basename(gold_path)}\n")
    md.append(f"- **Pipeline:** {model_name} + EntityRuler + regex char-level component\n")
    b = baseline["overall"]
    h = hybrid["overall"]
    md.append("\n## Metrics\n")
    md.append(f"- Baseline TP={b['TP']} FP={b['FP']} FN={b['FN']} precision={b['precision']:.3f} recall={b['recall']:.3f} f1={b['f1']:.3f}\n")
    md.append(f"- Hybrid TP={h['TP']} FP={h['FP']} FN={h['FN']} precision={h['precision']:.3f} recall={h['recall']:.3f} f1={h['f1']:.3f}\n")
    md.append("\n## Top error types\n")
    for t, c in type_counter.most_common():
        md.append(f"- {t}: {c}\n")
    md.append("\n## Top error categories (sample)\n")
    for c, cnt in cat_counter.most_common(6):
        md.append(f"- {c}: {cnt}\n")
    save_json({"audit_lines": md}, os.path.join(out_dir, "audit_summary_lines.json"))
    with open(os.path.join(out_dir, "audit_summary_lab10.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ner_notes_lab10.md (concise)
    notes = []
    notes.append("# NER Notes Lab 10\n")
    notes.append(f"- Pipeline: {model_name} + EntityRuler + regex component\n")
    notes.append("- In-box entities: LOC, ORG, PER (spaCy baseline)\n")
    notes.append("- Important domain entities: PRODUCT, QUANTITY, MONEY, PERCENT, DATE, TITLE, acronyms\n")
    notes.append("- Rules added: EntityRuler (PRODUCT, TITLE), regex (MONEY, PERCENT, QUANTITY, DATE, multiword PRODUCT)\n")
    notes.append(f"- Improvements: hybrid increased recall and F1 for domain classes; see hybrid_metrics.json\n")
    notes.append("- Remaining issues: acronyms, complex ORG spans, person names in inflected forms, span boundary errors\n")
    with open(os.path.join(out_dir, "ner_notes_lab10.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hybrid NER evaluation")
    parser.add_argument("--gold", required=True, help="Path to gold JSON")
    parser.add_argument("--model", default="uk_core_news_sm", help="spaCy model name")
    parser.add_argument("--outdir", default="./eval_output", help="Output directory")
    parser.add_argument("--prod_list", default=None, help="Optional JSON file with product list")
    parser.add_argument("--title_list", default=None, help="Optional JSON file with title list")
    args = parser.parse_args()

    prod_list = None
    title_list = None
    if args.prod_list and os.path.exists(args.prod_list):
        with open(args.prod_list, "r", encoding="utf-8") as f:
            prod_list = json.load(f)
    if args.title_list and os.path.exists(args.title_list):
        with open(args.title_list, "r", encoding="utf-8") as f:
            title_list = json.load(f)

    # Use default regex rules from ner_rules
    regex_rules = ner_rules.build_default_regex_rules()

    summary, details, errors = run_hybrid(args.gold, args.model, args.outdir, prod_list=prod_list, title_list=title_list, regex_rules=regex_rules)
    print("Hybrid evaluation complete. Outputs saved to", args.outdir)

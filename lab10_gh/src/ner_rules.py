# ner_rules.py
# Utility module: add EntityRuler patterns and register a regex-based char-level NER component.

import re
from spacy.tokens import Span
from spacy.language import Language

# Default single-token product and title lists (can be overridden by caller)
DEFAULT_PRODUCTS = ["су-34", "су34", "рсзв", "mh17", "mh-17", "mh 17", "сапфір"]
DEFAULT_TITLES = [
    "генпрокурор","генсек","президент","експрезидент","очільник","міністр",
    "постійний представник","голова","командувач","директор","постійний","представник"
]

def add_entity_ruler(nlp, prod_list=None, title_list=None, ruler_name="entity_ruler"):
    """
    Add or replace an EntityRuler with simple dictionary patterns for PRODUCT and TITLE.
    Returns the modified nlp pipeline.
    """
    from spacy.pipeline import EntityRuler

    prod_list = prod_list or DEFAULT_PRODUCTS
    title_list = title_list or DEFAULT_TITLES

    # Remove existing ruler if present
    if ruler_name in nlp.pipe_names:
        nlp.remove_pipe(ruler_name)

    ruler = nlp.add_pipe("entity_ruler", before="ner", name=ruler_name)
    patterns = []

    # PRODUCT single-token patterns
    for p in prod_list:
        patterns.append({"label": "PRODUCT", "pattern": [{"LOWER": p.lower()}]})

    # A short multi-token product example (keeps it generic)
    patterns.append({"label": "PRODUCT", "pattern": [{"LOWER": "рятувальне"}, {"LOWER": "судно"}]})

    # TITLE patterns (single and multi-token)
    for t in title_list:
        toks = t.split()
        patterns.append({"label": "TITLE", "pattern": [{"LOWER": tok} for tok in toks]})

    ruler.add_patterns(patterns)
    return nlp


def build_default_regex_rules():
    """
    Returns a list of (label, compiled_regex) tuples for char-level matching.
    These are the same robust patterns used in the notebook: MONEY, PERCENT, QUANTITY, DATE, PRODUCT (multiword).
    """
    money_patterns = [
        r"\b\d{1,3}(?:[ \,]\d{3})*(?:-\d{1,3}(?:[ \,]\d{3})*)?\s*(?:грн\.?|гривень|гривні|uah)\b",
        r"\b\d+(?:-\d+)?\s*(?:грн\.?|гривень|гривні|uah)\b",
        r"\b\d{1,3}(?:[ \,]\d{3})*(?:-\d{1,3}(?:[ \,]\d{3})*)?\s*(?:\$|usd|доларів|долар|долари)\b",
        r"\b\d+(?:-\d+)?\s*(?:€|eur|євро)\b"
    ]
    percent_patterns = [
        r"\b\d+(?:-\d+)?\s*%\b",
        r"\b\d+(?:-\d+)?%\b"
    ]
    quantity_patterns = [
        r"\b\d{1,3}(?:[ \,]\d{3})*\+?\s*(?:км|км\.|кілометрів|кілометр|кілометри)\b",
        r"\b\d{1,3}(?:[ \,]\d{3})*\s*(?:м|м\.|метрів|метр)\b",
        r"\d{1,3}(?:[ \u00A0]\d{3})+"
    ]
    months = ["січня","лютого","березня","квітня","травня","червня","липня","серпня","вересня","жовтня","листопада","грудня"]
    date_patterns = [
        r"\b\d{1,2}\s+(?:%s)\b" % ("|".join(months)),
        r"\b(19|20)\d{2}\b",
        r"\b'\d{2}\b"
    ]
    day_pattern = r"\b(?:понеділок|вівторок|середа|четвер|п'ятниця|пятниця|субота|неділя)\b"
    multi_prod_patterns = [
        r"рятувальне\s+судно\s*(?:\"|«)?\s*сапфір(?:\"|»)?",
        r"\bрятувальне\s+судно\b"
    ]

    rules = []
    for rx in money_patterns:
        rules.append(("MONEY", re.compile(rx, flags=re.IGNORECASE)))
    for rx in percent_patterns:
        rules.append(("PERCENT", re.compile(rx, flags=re.IGNORECASE)))
    for rx in quantity_patterns:
        rules.append(("QUANTITY", re.compile(rx, flags=re.IGNORECASE)))
    for rx in date_patterns:
        rules.append(("DATE", re.compile(rx, flags=re.IGNORECASE)))
    rules.append(("DATE", re.compile(day_pattern, flags=re.IGNORECASE)))
    for rx in multi_prod_patterns:
        rules.append(("PRODUCT", re.compile(rx, flags=re.IGNORECASE)))

    return rules


def register_regex_component(nlp, regex_rules=None, component_name="regex_ner", after_pipe="entity_ruler"):
    """
    Register a char-level regex component in the spaCy pipeline.
    regex_rules: list of (label, compiled_regex). If None, uses defaults from build_default_regex_rules().
    The component will add found spans to doc.ents (avoids exact duplicates).
    """
    regex_rules = regex_rules or build_default_regex_rules()

    @Language.component(component_name)
    def regex_ner_component(doc):
        new_ents = list(doc.ents)
        text = doc.text
        for label, pattern in regex_rules:
            for m in pattern.finditer(text):
                s_char, e_char = m.start(), m.end()
                # avoid exact duplicates
                duplicate = False
                for ent in new_ents:
                    if ent.start_char == s_char and ent.end_char == e_char and ent.label_ == label:
                        duplicate = True
                        break
                if duplicate:
                    continue
                # try char_span with expand alignment
                span = doc.char_span(s_char, e_char, label=label, alignment_mode="expand")
                if span is None:
                    # fallback: approximate token boundaries
                    token_start = None
                    token_end = None
                    for i, token in enumerate(doc):
                        if token.idx <= s_char < token.idx + len(token):
                            token_start = i
                        if token.idx < e_char <= token.idx + len(token):
                            token_end = i + 1
                    if token_start is None:
                        token_start = 0
                    if token_end is None:
                        token_end = len(doc)
                    span = Span(doc, token_start, token_end, label=label)
                new_ents.append(span)
        doc.ents = tuple(new_ents)
        return doc

    # Remove existing component if present
    if component_name in nlp.pipe_names:
        nlp.remove_pipe(component_name)

    # Insert after entity_ruler if present, otherwise before ner
    if after_pipe in nlp.pipe_names:
        nlp.add_pipe(component_name, after=after_pipe)
    else:
        # try to add before ner
        if "ner" in nlp.pipe_names:
            nlp.add_pipe(component_name, before="ner")
        else:
            nlp.add_pipe(component_name)
    return nlp

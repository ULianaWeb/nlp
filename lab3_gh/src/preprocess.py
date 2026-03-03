# src/preprocess.py

import re
from typing import List, Dict

# -----------------------
# 1. CLEAN
# -----------------------

def clean_text(text: str) -> str:
    text = str(text)

    # normalize apostrophe
    text = text.replace("’", "'")

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------
# 2. NORMALIZE
# -----------------------

def normalize_text(text: str) -> str:
    # normalize dashes
    text = re.sub(r"[–—−]", "-", text)

    # normalize quotes
    text = re.sub(r"[«»“”]", '"', text)

    return text


# -----------------------
# 3. MASK PII + COUNTERS
# -----------------------

def mask_pii(text: str) -> Dict:

    stats = {
        "url_count": 0,
        "email_count": 0,
        "phone_count": 0,
        "id_count": 0
    }

    # URL
    urls = re.findall(r"http\S+|www\S+", text)
    stats["url_count"] = len(urls)
    text = re.sub(r"http\S+|www\S+", "<URL>", text)

    # EMAIL
    emails = re.findall(r"\S+@\S+", text)
    stats["email_count"] = len(emails)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)

    # PHONE
    phones = re.findall(r"\+?\d[\d\s\-]{7,}\d", text)
    stats["phone_count"] = len(phones)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", "<PHONE>", text)

    # ID (довгі цифрові послідовності 8+)
    ids = re.findall(r"\b\d{8,}\b", text)
    stats["id_count"] = len(ids)
    text = re.sub(r"\b\d{8,}\b", "<ID>", text)

    return {"text": text, "stats": stats}


# -----------------------
# 4. SENTENCE SPLIT
# -----------------------

UA_ABBREVIATIONS = ["м.", "вул.", "р.", "т.д.", "т.п."]

def sentence_split(text: str) -> List[str]:

    protected = text
    for abbr in UA_ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # не ламати числа 3.14 / 1.2.3
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    sentences = re.split(r"(?<=[.!?])\s+", protected)

    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

    return sentences


# -----------------------
# 5. MAIN PIPELINE
# -----------------------

def preprocess(text: str) -> Dict:

    cleaned = clean_text(text)
    normalized = normalize_text(cleaned)
    masked = mask_pii(normalized)

    sentences = sentence_split(masked["text"])

    return {
        "clean_text": masked["text"],
        "sentences": sentences,
        "stats": masked["stats"]
    }
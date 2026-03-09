# IE RULES
import re
import json
from pathlib import Path

city_pattern = r"\b(" + "|".join(cities) + r")\b"

date_numeric = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b")
date_text = re.compile(
    r"\b(\d{1,2})\s+(січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\s+(\d{4})"
)

amount_pattern = re.compile(
    r"(\d+)\s?(млн|тис)?\s?(грн|₴|доларів|євро|\$|€)"
)


def normalize_date_numeric(text):
    day, month, year = re.split("[./]", text)
    return f"{year}-{int(month):02d}-{int(day):02d}"


def normalize_date_text(day, month, year):
    month_num = months.get(month)
    if month_num:
        return f"{year}-{month_num}-{int(day):02d}"
    return None


def extract_dates(text):

    results = []

    for m in date_numeric.finditer(text):

        span = m.group()
        norm = normalize_date_numeric(span)

        results.append({
            "field_type": "DATE",
            "value": norm,
            "span_text": span,
            "start_char": m.start(),
            "end_char": m.end(),
            "method": "regex_date_numeric"
        })

    for m in date_text.finditer(text):

        day, month, year = m.groups()

        norm = normalize_date_text(day, month, year)

        results.append({
            "field_type": "DATE",
            "value": norm,
            "span_text": m.group(),
            "start_char": m.start(),
            "end_char": m.end(),
            "method": "regex_date_text"
        })

    return results


def extract_amounts(text):

    results = []

    for m in amount_pattern.finditer(text):

        number = int(m.group(1))
        multiplier = m.group(2)
        currency = currencies.get(m.group(3))

        if multiplier == "млн":
            number *= 1_000_000

        if multiplier == "тис":
            number *= 1_000

        results.append({
            "field_type": "AMOUNT",
            "value": number,
            "currency": currency,
            "span_text": m.group(),
            "start_char": m.start(),
            "end_char": m.end(),
            "method": "regex_amount"
        })

    return results


def extract_locations(text):

    results = []

    for m in re.finditer(city_pattern, text):

        results.append({
            "field_type": "LOCATION",
            "value": m.group(),
            "span_text": m.group(),
            "start_char": m.start(),
            "end_char": m.end(),
            "method": "city_dictionary"
        })

    return results


def extract_all(text):

    results = []

    results.extend(extract_dates(text))
    results.extend(extract_amounts(text))
    results.extend(extract_locations(text))

    return results


def evaluate_on_file(file_path):
    with open(file_path, encoding="utf8") as f:
        gold = json.load(f)

    predictions = []
    for entry in gold:
        text = entry["text"]
        extracted = extract_all(text)
        predictions.append({"entities": extracted})

    return evaluate_precision(predictions, gold)
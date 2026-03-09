# Information Extraction Policy – Lab 4

## Project
Rule-based Information Extraction for Ukrainian News

## Extracted Fields

The system extracts three types of structured information from Ukrainian news texts:

1. DATE
2. AMOUNT
3. LOCATION

Extraction is implemented using regular expressions and dictionaries.

---

# 1. DATE

## Description

The DATE field represents calendar dates mentioned in the text.

The system supports several common Ukrainian date formats.

## Supported Formats

### Numeric formats

Examples:

01.02.2024  
12/03/2023
 
### Textual formats

Examples:

12 березня 2024  
5 травня 2023 року

### Partial formats

Examples:

у травні 2023  
у вересні 2022 року

These may not always be normalized.

---

## Normalization

Normalized format:

YYYY-MM-DD

Example:

12 березня 2024 → 2024-03-12
01.02.2024 → 2024-02-01

If normalization fails:

normalized_value = null


---

# 2. AMOUNT

## Description

The AMOUNT field represents monetary values mentioned in the text.

The system extracts:

- numeric value
- currency

---

## Supported currencies

UAH  
USD  
EUR

Possible representations:

грн  
₴  
доларів  
$  
євро  
€

---

## Supported formats

Examples:

1000 грн  
5000 ₴  
$200  
200 доларів  
5 млн грн  
200 тис грн

## Normalization

Output structure:

{
"value": 5000000,
"currency": "UAH"
}

Example:

5 млн грн → value=5000000 currency=UAH


---

# 3. LOCATION

## Description

LOCATION represents Ukrainian city names mentioned in news articles.

Extraction is dictionary-based.

The dictionary contains major Ukrainian cities.

---

## Examples

Київ  
Львів  
Харків  
Одеса  
Дніпро

The system searches for exact word matches using word boundaries.

# Output Format

Each extracted entity returns:

{
"field_type": "DATE",
"value": "2024-03-12",
"span_text": "12 березня 2024",
"start_char": 0,
"end_char": 14,
"method": "regex_date_ua"
}


---

# Evaluation

Evaluation is performed using a manually annotated gold dataset.

Metrics:

Precision = correct_extractions / total_extractions


---

# Error Analysis

Common error sources:

- ambiguous numbers
- incomplete dates
- city names used as part of organization names
- currency mentions without numeric value
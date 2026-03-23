# NLP Lab 6 — Baseline Text Classification

## Overview

This project implements baseline models for binary classification of Ukrainian news articles into **Real** and **Fake** categories.

The main goal is to establish a reliable baseline using classical machine learning methods and analyze model behavior.

---

## Dataset

* Source: Ukrainian news dataset (processed in previous labs)
* Input: `processed_text`
* Target: `label` (Real / Fake)

Data splits are reused from Lab 5 to ensure consistency and prevent data leakage.

---

## Models

Two baseline models were implemented:

### Baseline 1

* TF-IDF (unigrams)
* Logistic Regression

### Baseline 2

* TF-IDF (unigrams + bigrams)
* Logistic Regression with `class_weight="balanced"`

---

## Evaluation

Metrics used:

* Accuracy
* Macro F1-score
* Per-class Precision / Recall / F1

### Test Results
Test B1 per-class evaluation:
     Class  Precision    Recall  F1-score  Support
0  Class 0   0.868000  0.947598  0.906054      458
1  Class 1   0.916084  0.798780  0.853420      328

Test B2 per-class evaluation:
     Class  Precision    Recall  F1-score  Support
0  Class 0   0.902128  0.925764  0.913793      458
1  Class 1   0.892405  0.859756  0.875776      328

---

## Key Findings

* The model performs better on **Real** news than on **Fake**.

* Fake news detection is more challenging due to:

  * linguistic similarity with real news
  * lack of strong discriminative features

* Feature analysis shows reliance on **source-related tokens**, indicating dataset bias.

---

## Error Analysis

Common error types:

* ambiguous or neutral texts
* overlap between real and fake topics
* short or low-information articles

---

## Limitations

* Model may overfit to specific sources
* No deep linguistic processing (e.g., lemmatization)
* Limited generalization

---

## How to Run

```bash
pip install -r requirements.txt
python classification_baseline.py
```

---

## Project Structure

```
project_root
│
├── notebooks/
│   └── lab6_tfidf_logistic_baseline.ipynb
│
├── src/
│   └── classification_baseline.py
│
├── docs/
│   └── audit_summary_lab6.md
│
├── data/
│   └── sample/
│       └── splits_*.txt
│
├── processed_v2.csv
├── requirements.txt
└── README.md
```

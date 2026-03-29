# NLP Lab 7 — Linear SVM and Feature Engineering

## Task

(A) Binary classification 

Improve baseline text classification model (Real vs Fake news) by exploring:

* Linear SVM
* Character n-grams
* Class imbalance handling
* Threshold tuning

---

## Models

The following models were implemented:

1. Logistic Regression (baseline from Lab 6)
2. Linear SVM (word n-grams)
3. Linear SVM (character n-grams)
4. Linear SVM (class_weight="balanced")

---

## Features

* Word n-grams (1–2)
* Character n-grams (3–5)


## Results

| Model        | Accuracy | Macro-F1 |
| ------------ | -------- | -------- |
| LogReg       |  0.8982  |  0.8947  |
| SVM word     |  0.8944  |  0.8902  |
| SVM char     |  0.9300  |  0.9276  |
| SVM balanced |  0.8893  |  0.8856  |

---

## Key Observations

* SVM generally outperforms Logistic Regression
* Character n-grams improve robustness to noisy text
* No class imbalance
* Threshold tuning allows control over precision/recall trade-off
* Char(3,5) SVM works best

---

## Error Analysis

Common error types:

* overlapping topics between real and fake news
* short or low-information texts
* ambiguous phrasing

---

## Leakage & Bias

No direct data leakage was detected.

---

## Conclusion

Linear SVM combined with TF-IDF provides a strong baseline for Ukrainian fake news classification.

Further improvements may include:

* combining word + char features
* using linguistic features
* applying transformer-based models


# Lab 03 — Lemma/POS Baseline

## Direction
A) Text Classification (real/fake news)

## Lemmatization Tool
Stanza (Ukrainian model)

## Baselines
1. TF-IDF + Logistic Regression on processed_v2
2. TF-IDF + Logistic Regression on lemma_text
3. POS filter (NOUN + ADJ)

## Results
Raw: 0.765 accuracy
Lemma: 0.78 accuracy
Lemma + POS: 0.74 accuracy

## Decision
Lemmas improve classification.
POS not useful as feature filter.

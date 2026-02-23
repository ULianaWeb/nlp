# Lab 02 – Cleaning & Normalization

Task: A – Text Classification (real/fake)
Text field: title + body

Run in Colab:
1. Upload raw.csv
2. Run notebook
3. processed_v2.csv is generated

5 hardest edge cases:
- Apostrophe variants
- UA abbreviations (м., вул.)
- URLs
- Emails
- Decimals (3.14)

Improvements:
- Deterministic pipeline
- Idempotent
- Masked PII

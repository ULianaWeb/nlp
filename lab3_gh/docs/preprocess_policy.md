Binary classification (fake vs real) – Ukrainian news.


We clean:
- Unicode normalization (NFKC)
- Extra whitespace
- Apostrophe variants

We normalize:
- Lowercasing
- Consistent whitespace

We mask:
- URL → <URL>
- Email → <EMAIL>
- Phone → <PHONE>

We DO NOT modify:
- Numbers
- Dates
- Codes
- Sentence structure

Sentence Split
Custom rule-based splitter.
Avoid splitting after abbreviations:
м., вул., р., т.д., ст., №

Examples (before → after)
1. "м. Київ." → not split
2. "ivan@gmail.com" → <EMAIL>
3. "https://site.ua" → <URL>
4. +380673504080 → <PHONE>
5. "пʼять" → "п'ять"
6. "та   решта" → "та решта"
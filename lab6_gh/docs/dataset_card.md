Назва проєкту: Disinformation UA: Fake News Classification



Задача A: Binary text classification (fake vs real)



https://www.kaggle.com/datasets/sophiamatskovych/fake-news-ua



Обсяг: 12 956 текстів



Мови: ua, en, pl, bg, ru



Ризики: повторювані новини з різних джерел, потенційна політична упередженість.



Очищення: Тексти сформовані шляхом об’єднання заголовка та основного тексту.
Середня довжина тексту становить приблизно 120 слів.
Розподіл класів демонструє збалансованість (58% real та 42% fake).
Було уніфіковано апострофи, прибрано зайві пробіли, видалено посилання, замінено символи/іноземні літери, замасковано чутливу інформацію і т. д.
Було виявлено 1.51% точних дублів та видалено їх.



План наступного кроку: застосування TF-IDF та побудова базової моделі класифікації.



Lemma/POS: Лематизація дещо покращила macro-F1 і accuracy, порівняно з raw текстами.

POS-based filtering (NOUN + ADJ only) знизило шум але іноді викинуло корисні функціональні слова.

Фінальний вибір: використовувати lemma для класифікації.



Information Extraction – Lab 4

Extracted Fields:

1\. DATE

2\. AMOUNT

3\. LOCATION

Extraction is implemented using regular expressions and dictionaries.



Ризики: значення поза словником, описки у текстах, невраховані форми запису.



Split + leakage checks - Lab 5

Risks:

* possible paraphrased duplicates
* news topics may overlap
* class imbalance could affect training



Model Baseline (Lab 6)



Baseline model:

\- TF-IDF (unigram features)

\- Logistic Regression



Test B1 per-class evaluation:

&#x20;    Class  Precision    Recall  F1-score  Support

0  Class 0   0.868000  0.947598  0.906054      458

1  Class 1   0.916084  0.798780  0.853420      328



Test B2 per-class evaluation:

&#x20;    Class  Precision    Recall  F1-score  Support

0  Class 0   0.902128  0.925764  0.913793      458

1  Class 1   0.892405  0.859756  0.875776      328



Observations



\- The model achieves higher recall for 'Real' news, indicating it is better at identifying true articles.

\- Performance on 'Fake' news is lower, especially recall, meaning some fake articles are misclassified as real.

\- This may indicate:

&#x20; - overlap in language between fake and real news

&#x20; - insufficient discriminative features for fake content



Feature Analysis



Top features indicate reliance on source-related tokens (e.g., media names), suggesting potential dataset bias.



Limitations



\- Model may rely on source names rather than content

\- Limited generalization to unseen news sources

\- No linguistic normalization (lemmatization not applied in this experiment)


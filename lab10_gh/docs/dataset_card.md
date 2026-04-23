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



Model Baseline - Lab 6



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





Modeling - Lab 7



Baseline and advanced models were trained on this dataset:



Models Used



\- Logistic Regression (TF-IDF, word n-grams)

\- Linear SVM (word n-grams)

\- Linear SVM (character n-grams)

\- Linear SVM with class balancing



\---

Findings:

\- SVM models outperform / match baseline

\- char n-grams capture subword patterns

\- class balancing doesn't improve results

\- custom\_threshold = -0.4, aiming for the least false negatives



Most popular errors:

\- overlap

\- short text



LSA/LDA - Lab 8

* теми доволі шумні і overlapping
* і LSA, і LDA дають майже однаковий результат
* LDA має більше беззмістовних тем
* LSA наче корисніша, але загалом теми не надто різні, зокрема через занадто однорідний корпус. Майже всі тексти на військову тематику, тож поділ на теми для такої задачі бінарної класифікації на цьому датасеті не надто інформативний.



Lab 9



Корпус достатньо великий для embeddings. Для базових Word2Vec / FastText (вектори 100–200) корпус достатній.

Є багато доменних термінів. Присутні терміни, пов’язані з фактчеком, джерелами та тематикою війни/здоров’я (наприклад: \*фейк, підтверджено, верифікувати, джерело, вакцина\*). Деякі доменні терміни добре представлені, інші — рідкісні.

Було виявлено трансліт і латинські вставки (`bellingcat`, `comirпaty`), токени з прикріпленою пунктуацією (`влада.`, `влада,`) та варіанти написання. Після додаткової нормалізації якість сусідів покращується.

FastText краще обробляє OOV і морфологічні варіанти; Word2Vec іноді дає більш інформативні семантичні сусіди для частих брендів і технічних термінів. Варто використовувати FastText для OOV/morph і Word2Vec як додаткове джерело семантичних фіч

Embeddings дають корисний сигнал у вашому корпусі як аналітичний інструмент. Embeddings корисні для вивчення словника, виявлення доменних груп і морфологічних варіантів. Для прямої класифікації real/fake їхня користь обмежена: вони можуть доповнювати інші ознаки, але самі по собі не гарантують суттєвого покращення без додаткових фіч і моделей.


Lab 10

Які типи сутностей реально важливі у корпусі

LOC, ORG, PER (базові)

PRODUCT, QUANTITY/DISTANCE, MONEY, PERCENT, DATE, TITLE (доменні, критичні для коректної аннотації і downstream‑задач)

Чи стандартний NER pipeline достатній

Ні. Стандартний spaCy NER дає хороше покриття для LOC/PER у простих випадках, але пропускає більшість доменних сутностей (PRODUCT, MONEY, QUANTITY, PERCENT, DATE, TITLE). Для нашого корпусу потрібен hybrid підхід: словники + char‑level regex.

Які доменні сутності виявилися проблемними

PRODUCT у багатотокенних варіантах (наприклад, «рятувальне судно "сапфір"»)

QUANTITY з тисячними роздільниками («100 000») і з позначеннями одиниць («100+ км»)

MONEY / PERCENT у форматах з дефісами та пробілами («20‑25 грн.», «50‑60%»)

TITLE та абревіатури ORG у відмінках

Чи hybrid rules дали відчутний виграш

Так. Після додавання EntityRuler + regex‑component: TP зросли, FN значно зменшилися, F1 піднявся з 0.623 → 0.846. Домени, які baseline пропускав, тепер покриті.

Які типи помилок залишаються

Абревіатури та акроніми (FN, label mismatch)

Прізвища у відмінках (FN)

Partial spans і span merge errors (FP)

Потрібні додаткові правила або тренування для стабільної класифікації складних ORG

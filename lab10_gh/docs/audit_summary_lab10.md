# Audit Summary Lab 10

## Короткий summary

* **Pipeline використано:** spaCy `uk\_core\_news\_sm` + EntityRuler + regex‑based component (char‑level).
* **Важливі сутності в задачі:** **LOC**, **ORG**, **PER**, **PRODUCT**, **QUANTITY/DISTANCE**, **MONEY**, **PERCENT**, **DATE**, **TITLE**.

## Результати (ключові метрики)

### Baseline

* **TP:** 33, **FP:** 7, **FN:** 33
* **Precision:** 0.825, **Recall:** 0.500, **F1:** 0.623

### Hybrid (pipeline + rules)

* **TP:** 52, **FP:** 5, **FN:** 14
* **Precision:** 0.912, **Recall:** 0.788, **F1:** 0.846

## Що baseline знаходив добре

* **LOC**: географічні назви в більшості випадків (висока точність і recall для LOC).
* **PER**: деякі імена/прізвища модель розпізнавала коректно.

## Які доменні / регулярні сутності baseline пропускав

* **PRODUCT, QUANTITY, PERCENT, DATE, MONEY, TITLE** — не знаходились baseline'ом; потребували правил/regex.

## Які rules були додані

* **EntityRuler**: словникові патерни для PRODUCT (су-34, mh17, рятувальне судно, сапфір) та TITLE (генпрокурор, міністр, президент тощо).
* **Regex component (char‑level)**: правила для **MONEY** (грн, $, €; діапазони, тисячні роздільники), **PERCENT** (50-60%), **QUANTITY** (100+ км, 100 000), **DATE** (день+місяць, роки), multiword PRODUCT (рятувальне судно "сапфір").

## Що вони реально покращили

* **TP** збільшився з 33 до 52 (+19).
* **FN** зменшився з 33 до 14 (−19).
* **Precision** змінилась з 0.825 до 0.912.
* Правила повністю покрили доменні категорії: PRODUCT, QUANTITY, PERCENT, DATE, MONEY, TITLE (TP для цих класів у hybrid = 1.0 recall).

## Які категорії помилок були наймасовішими

* **Organization acronym missing / Abbreviation not recognized** (НАТО, ЄС, ДНР, ППО) — багато FN та деякі label mismatches.
* **Person surname missing** (Пелосі, Кадирова, Лаврова) — FN для персональних імен.
* **Complex organization name / Partial span** — модель іноді розбиває складні ORG на частини (FP часткові спани).

Top error categories (count):

* Uncategorized: 19

## Що б робили далі

* **Розширити словники** для ORG та PER (акроніми та часті прізвища) — закрити залишкові FN.
* **Додати post‑processing фільтри**: видаляти короткі вкладені енти, виправляти злиття span (наприклад 'лаврова єс').
* **Звузити деякі regex** (за потреби) щоб зменшити FP по ORG/LOC label mismatches.
* **Зібрати більше прикладів** для ORG/PER у різних відмінках і додати до тренувального набору або до правил.


The system extracts three types of structured information from Ukrainian news texts:



1\. DATE

2\. AMOUNT

3\. LOCATION



Extraction is implemented using regular expressions and dictionaries.



\# 1. DATE



The DATE field represents calendar dates mentioned in the text.



Examples:



12 березня 2024 → 2024-03-12

{

"field\_type": "DATE",

"value": "2024-03-12",

"span\_text": "12 березня 2024",

"start\_char": 0,

"end\_char": 14,

"method": "regex\_date\_ua"

}





\# 2. AMOUNT



The AMOUNT field represents monetary values mentioned in the text.



Example:



5 млн грн → value=5000000 currency=UAH





\# 3. LOCATION



LOCATION represents Ukrainian city names mentioned in news articles.



\## Example



{

&nbsp;  "field\_type": "LOCATION",

&nbsp;  "value": "польща",

&nbsp;  "span\_text": "польщі",

&nbsp;  "start\_char": 33,

&nbsp;  "end\_char": 38,

&nbsp;  "method": "city\_dictionary"

}



Precision: 100%



Edge cases:



{"text": "The client from the USA paid 7500 USD on 15-02-2024.", "gold\_entities": \[\["USA", "COUNTRY"], \["7500 USD", "MONEY"], \["15-02-2024", "DATE"]]}

{"text": "A deposit of £1,250.50 was made on 28 Feb 2024 to Germany.", "gold\_entities": \[\["£1,250.50", "MONEY"], \["28 Feb 2024", "DATE"], \["Germany", "COUNTRY"]]}

{"text": "No payment was recorded on 2023/12/31 in Italy.", "gold\_entities": \[\["2023/12/31", "DATE"], \["Italy", "COUNTRY"]]}

{"text": "Transfer to France: 0 USD on 01.01.2024.", "gold\_entities": \[\["0 USD", "MONEY"], \["01.01.2024", "DATE"], \["France", "COUNTRY"]]}

{"text": "The total of $999,999.99 was paid in Germany on December 31st, 2023.", "gold\_entities": \[\["$999,999.99", "MONEY"], \["Germany", "COUNTRY"], \["December 31st, 2023", "DATE"]]}


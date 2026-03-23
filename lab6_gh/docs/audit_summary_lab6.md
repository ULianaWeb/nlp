# Audit Summary Lab 6



Task: Fake vs Real classification



Split: from Lab 5



Baseline 1:
Accuracy: 0.885
Macro-F1: 0.880



Baseline 2:
Accuracy: 0.898
Macro-F1: 0.895



Improvement:
Accuracy diff: 0.013
F1 diff: 0.015



Per-class metrics (Test B1):

|Class|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|Class 0|0.868|0.947598|0.906054|458|
|Class 1|0.916084|0.79878|0.85342|328|



Per-class metrics (Test B2):

|Class|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|Class 0|0.902128|0.925764|0.913793|458|
|Class 1|0.892405|0.859756|0.875776|328|



Errors analyzed: 20



Next steps:

* better preprocessing
* try lemma features
* handle source bias


# Audit Summary Lab 7

Binary classification (real or fake Ukrainian news)



Models compared:

* Logistic Regression (baseline from lab 6)
* Linear SVM (word n-grams)
* Linear SVM (char n-grams)
* Linear SVM (balanced)



Results:



LogReg:
Accuracy: 0.898
F1: 0.895



SVM word:
Accuracy: 0.894
F1: 0.890



SVM balanced:
Accuracy: 0.889
F1: 0.886



SVM char:
Accuracy: 0.930
F1: 0.928



Findings:

* SVM models outperform / match baseline
* char n-grams capture subword patterns
* class balancing doesn't improve results
* custom\_threshold = -0.4, aiming for the least false negatives



Most popular errors:

* overlap
* short text
* noisy text


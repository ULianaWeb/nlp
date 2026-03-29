from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_logreg_baseline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2),
            max_features=7000
        )),
        ("clf", LogisticRegression(max_iter=500))
    ])


def build_svm_word():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2),
            max_features=7000
        )),
        ("clf", LinearSVC())
    ])


def build_svm_char():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3,5),
            max_features=10000
        )),
        ("clf", LinearSVC())
    ])


def build_svm_balanced():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1,2),
            max_features=7000
        )),
        ("clf", LinearSVC(class_weight="balanced"))
    ])
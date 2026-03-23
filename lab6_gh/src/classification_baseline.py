import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_data(path="processed_v2.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"processed_text": "text"})
    df["text_id"] = range(len(df))
    df["label_num"] = df["label"].map({"Real": 0, "Fake": 1})
    return df


def load_splits(df):
    train_ids = pd.read_csv("splits_train_ids.txt", header=None)[0].tolist()
    val_ids = pd.read_csv("splits_val_ids.txt", header=None)[0].tolist()
    test_ids = pd.read_csv("splits_test_ids.txt", header=None)[0].tolist()

    train_df = df[df["text_id"].isin(train_ids)]
    val_df = df[df["text_id"].isin(val_ids)]
    test_df = df[df["text_id"].isin(test_ids)]

    return train_df, val_df, test_df


def build_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            max_features=5000
        )),
        ("clf", LogisticRegression(max_iter=300))
    ])
    return model


def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    return acc, f1


def main():
    df = load_data()
    train_df, val_df, test_df = load_splits(df)

    model = build_model()
    model.fit(train_df["text"], train_df["label_num"])

    acc, f1 = evaluate(model, test_df["text"], test_df["label_num"])

    print("Test Accuracy:", acc)
    print("Test Macro-F1:", f1)


if __name__ == "__main__":
    main()
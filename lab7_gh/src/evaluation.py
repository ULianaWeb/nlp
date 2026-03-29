from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, X, y):
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    return acc, f1, preds
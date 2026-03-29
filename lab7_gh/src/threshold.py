from sklearn.metrics import precision_recall_curve


def get_pr_curve(model, X, y):
    scores = model.decision_function(X)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    return precision, recall, thresholds


def apply_threshold(scores, threshold):
    return (scores >= threshold).astype(int)
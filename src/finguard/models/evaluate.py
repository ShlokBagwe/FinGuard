import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_proba, threshold=0.5):
    """Compute PR-AUC, ROC-AUC, and class-1 precision, recall and F1 at threshold.

    Takes true binary labels, predicted probabilities and threshold; returns a metrics dict.
    """
    pr_auc = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba))

    y_pred = (y_proba >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_best_threshold(y_true, y_proba):
    """Find the probability threshold that maximizes F1 on precision_recall_curve.

    Takes true labels and predicted probabilities, returning the best threshold float.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one extra precision/recall point with no threshold.
    f1_scores = (
        2
        * (precision[:-1] * recall[:-1])
        / (precision[:-1] + recall[:-1] + 1e-10)
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])

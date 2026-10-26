"""分类指标"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_NAMES = ("H", "M", "L")


def classification(y_true, y_pred, loco=False):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if not len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Empty or mismatched predictions")
    if not set(y_true).issubset({0, 1, 2}) or not set(y_pred).issubset({0, 1, 2}):
        raise ValueError("Invalid labels")
    if loco and 1 in y_true:
        raise ValueError("LOCO5 pooled test must contain H/L only")
    labels = [0, 2] if loco else [0, 1, 2]
    k = float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None
    return {
        "n": len(y_true),
        "macro_labels": [CLASS_NAMES[x] for x in labels],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "kappa": k if k is None or np.isfinite(k) else None,
        "predicted_M": int((y_pred == 1).sum()),
        "confusion_HML": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }

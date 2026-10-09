"""Binary metrics with handwritten = 1 as the positive class."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)


def binary_metrics(y, pred, score=None, prob: bool = True) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"balanced_accuracy": balanced_accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
         "formal_recall(specificity)": tn / max(1, tn + fp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if score is not None and len(set(y)) == 2:
        s = np.asarray(score, float)
        m["roc_auc"] = roc_auc_score(y, s)
        m["pr_auc"] = average_precision_score(y, s)
        if prob:
            p = np.clip(s, 1e-12, 1 - 1e-12)
            m["log_loss"] = log_loss(y, p, labels=[0, 1])
            m["brier"] = brier_score_loss(y, p)
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m.items()}

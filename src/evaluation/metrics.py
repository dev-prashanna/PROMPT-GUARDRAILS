from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "safe", 1: "prompt_injection", 2: "jailbreak"}


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute Macro F1, Accuracy, and per-class metrics for the HF Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
    }

    for i, name in LABEL_NAMES.items():
        if i < len(precision):
            metrics[f"{name}_precision"] = float(precision[i])
            metrics[f"{name}_recall"] = float(recall[i])
            metrics[f"{name}_f1"] = float(f1[i])
            metrics[f"{name}_support"] = int(support[i])

    cm = confusion_matrix(labels, predictions, labels=list(LABEL_NAMES.keys()))
    for i, row_label in LABEL_NAMES.items():
        for j, col_label in LABEL_NAMES.items():
            metrics[f"cm_{row_label}_vs_{col_label}"] = int(cm[i][j])

    report = classification_report(
        labels, predictions,
        target_names=list(LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )
    metrics["macro_precision"] = report["macro avg"]["precision"]
    metrics["macro_recall"] = report["macro avg"]["recall"]

    return metrics


def get_full_report(eval_pred) -> dict:
    """Return a full classification report dict."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return classification_report(
        labels, predictions,
        target_names=list(LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )

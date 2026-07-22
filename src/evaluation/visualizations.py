from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Safe", "Prompt Injection", "Jailbreak"]
STYLE_CONFIG = {
    "figure.figsize": (10, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}


class EvaluationVisualizer:
    """Generate publication-quality visualizations for research paper."""

    def __init__(self, output_dir: str = "paper/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(STYLE_CONFIG)

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True
    ) -> str:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=".2%" if normalize else "d",
            cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=ax, vmin=0, vmax=1 if normalize else None,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix" if normalize else "Confusion Matrix (Counts)")

        path = self.output_dir / ("confusion_matrix_normalized.png" if normalize else "confusion_matrix_counts.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix to %s", path)
        return str(path)

    def plot_loss_curves(
        self, train_losses: list[float], eval_losses: list[float]
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, "b-o", label="Training Loss", markersize=4)
        ax.plot(epochs, eval_losses, "r-o", label="Validation Loss", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / "loss_curves.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved loss curves to %s", path)
        return str(path)

    def plot_precision_recall_curves(
        self, y_true: np.ndarray, y_probs: np.ndarray
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("husl", len(LABEL_NAMES))

        for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_binary, y_probs[:, i])
            ap = average_precision_score(y_binary, y_probs[:, i])
            ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves per Class")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        path = self.output_dir / "precision_recall_curves.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved PR curves to %s", path)
        return str(path)

    def plot_class_distribution(self, labels: np.ndarray) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        unique, counts = np.unique(labels, return_counts=True)
        names = [LABEL_NAMES[int(i)] for i in unique]
        bars = ax.bar(names, counts, color=sns.color_palette("Set2", len(names)))
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Dataset Class Distribution")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(count), ha="center", va="bottom", fontweight="bold")

        path = self.output_dir / "class_distribution.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved class distribution to %s", path)
        return str(path)

    def plot_metric_comparison(self, metrics: dict[str, dict[str, float]]) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(metrics.keys())
        metric_names = set()
        for m in metrics.values():
            metric_names.update(m.keys())
        metric_names = sorted(metric_names)

        x = np.arange(len(metric_names))
        width = 0.8 / len(models)
        colors = sns.color_palette("husl", len(models))

        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            values = [model_metrics.get(m, 0) for m in metric_names]
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name, color=colors[i])

        ax.set_ylabel("Score")
        ax.set_title("Model Comparison Across Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

        path = self.output_dir / "model_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved model comparison to %s", path)
        return str(path)

    def generate_all_report_figures(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray,
        train_losses: list[float], eval_losses: list[float]
    ) -> dict[str, str]:
        paths = {}
        paths["confusion_matrix"] = self.plot_confusion_matrix(y_true, y_pred)
        paths["confusion_matrix_counts"] = self.plot_confusion_matrix(
            y_true, y_pred, normalize=False
        )
        paths["loss_curves"] = self.plot_loss_curves(train_losses, eval_losses)
        paths["pr_curves"] = self.plot_precision_recall_curves(y_true, y_probs)
        paths["class_distribution"] = self.plot_class_distribution(y_true)

        summary_path = self.output_dir / "figure_index.json"
        with open(summary_path, "w") as f:
            json.dump(paths, f, indent=2)
        logger.info("Figure index saved to %s", summary_path)
        return paths

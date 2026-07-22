from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

LABEL_NAMES = ["safe", "prompt_injection", "jailbreak"]


class EvaluationAnalyzer:
    """Deep analysis of model performance for research documentation."""

    def __init__(self, output_dir: str = "paper/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def full_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
    ) -> dict[str, Any]:
        report = classification_report(
            y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        error_analysis = self._error_analysis(y_true, y_pred, y_probs)

        results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "error_analysis": error_analysis,
        }

        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Full analysis saved to %s", results_path)
        return results

    def _error_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
    ) -> dict[str, Any]:
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]

        analysis = {
            "total_errors": int(errors.sum()),
            "error_rate": float(errors.mean()),
            "errors_per_class": {},
            "confidence_distribution": {
                "correct_mean": 0.0,
                "incorrect_mean": 0.0,
            },
        }

        correct_mask = ~errors
        if correct_mask.any():
            correct_conf = np.max(y_probs[correct_mask], axis=1)
            analysis["confidence_distribution"]["correct_mean"] = float(correct_conf.mean())

        if errors.any():
            incorrect_conf = np.max(y_probs[errors], axis=1)
            analysis["confidence_distribution"]["incorrect_mean"] = float(
                incorrect_conf.mean()
            )

        for i, name in enumerate(LABEL_NAMES):
            mask = y_true == i
            if mask.any():
                class_errors = errors[mask]
                analysis["errors_per_class"][name] = {
                    "total": int(mask.sum()),
                    "errors": int(class_errors.sum()),
                    "error_rate": float(class_errors.mean()),
                }

        return analysis

    def generate_latex_table(self, report: dict) -> str:
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Classification Results}",
            r"\label{tab:results}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Class & Precision & Recall & F1-Score & Support \\",
            r"\midrule",
        ]
        for name in LABEL_NAMES:
            if name in report:
                m = report[name]
                lines.append(
                    f"{name.replace('_', ' ').title()} & {m['precision']:.3f} & "
                    f"{m['recall']:.3f} & {m['f1-score']:.3f} & {int(m['support'])} \\\\"
                )
        if "macro avg" in report:
            m = report["macro avg"]
            lines.append(r"\midrule")
            lines.append(
                f"Macro Avg & {m['precision']:.3f} & {m['recall']:.3f} & "
                f"{m['f1-score']:.3f} & {int(m['support'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        latex = "\n".join(lines)
        path = self.output_dir / "results_table.tex"
        with open(path, "w") as f:
            f.write(latex)
        return latex

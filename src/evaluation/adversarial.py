from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdversarialEvaluator:
    """Evaluate model robustness against adversarial attacks."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def evaluate(
        self,
        model_predict_fn: Any,
        adversarial_df: pd.DataFrame,
    ) -> dict[str, Any]:
        results = {
            "total_samples": len(adversarial_df),
            "by_category": {},
            "overall": {},
        }

        categories = (
            adversarial_df["attack_category"].unique()
            if "attack_category" in adversarial_df.columns
            else ["unknown"]
        )

        all_correct = []
        all_latencies = []

        for category in categories:
            if "attack_category" in adversarial_df.columns:
                cat_df = adversarial_df[adversarial_df["attack_category"] == category]
            else:
                cat_df = adversarial_df

            correct = []
            latencies = []
            predictions = []

            for _, row in cat_df.iterrows():
                start = time.perf_counter()
                pred = model_predict_fn(row["text"])
                latency = (time.perf_counter() - start) * 1000

                is_correct = pred.get("label_id") == row["label"]
                correct.append(is_correct)
                latencies.append(latency)
                predictions.append(pred)

            correct_arr = np.array(correct)
            latencies_arr = np.array(latencies)

            results["by_category"][category] = {
                "total": len(cat_df),
                "correct": int(correct_arr.sum()),
                "accuracy": float(correct_arr.mean()),
                "avg_latency_ms": float(latencies_arr.mean()),
                "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
            }

            all_correct.extend(correct)
            all_latencies.extend(latencies)

        all_correct_arr = np.array(all_correct)
        all_latencies_arr = np.array(all_latencies)

        results["overall"] = {
            "total": len(all_correct),
            "accuracy": float(all_correct_arr.mean()),
            "avg_latency_ms": float(all_latencies_arr.mean()),
            "p95_latency_ms": float(np.percentile(all_latencies_arr, 95)),
            "p99_latency_ms": float(np.percentile(all_latencies_arr, 99)),
        }

        results["vulnerability_analysis"] = self._analyze_vulnerabilities(
            results["by_category"]
        )

        return results

    def _analyze_vulnerabilities(
        self, by_category: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        sorted_cats = sorted(
            by_category.items(), key=lambda x: x[1]["accuracy"]
        )

        return {
            "most_vulnerable": sorted_cats[0][0] if sorted_cats else None,
            "most_vulnerable_accuracy": sorted_cats[0][1]["accuracy"] if sorted_cats else None,
            "most_robust": sorted_cats[-1][0] if sorted_cats else None,
            "most_robust_accuracy": sorted_cats[-1][1]["accuracy"] if sorted_cats else None,
            "accuracy_range": (
                sorted_cats[-1][1]["accuracy"] - sorted_cats[0][1]["accuracy"]
                if sorted_cats
                else 0
            ),
            "categories_below_threshold": [
                cat for cat, data in by_category.items()
                if data["accuracy"] < 0.85
            ],
        }

    def generate_report(
        self, results: dict[str, Any], output_path: str | None = None
    ) -> str:
        lines = [
            "# Adversarial Robustness Report",
            "",
            f"## Overall Results",
            f"- Total samples: {results['overall']['total']}",
            f"- Overall accuracy: {results['overall']['accuracy']:.4f}",
            f"- Average latency: {results['overall']['avg_latency_ms']:.2f}ms",
            f"- P95 latency: {results['overall']['p95_latency_ms']:.2f}ms",
            "",
            "## Results by Attack Category",
            "",
        ]

        for category, data in sorted(results["by_category"].items()):
            status = "PASS" if data["accuracy"] >= 0.85 else "FAIL"
            lines.append(f"### {category.replace('_', ' ').title()} [{status}]")
            lines.append(f"- Samples: {data['total']}")
            lines.append(f"- Accuracy: {data['accuracy']:.4f}")
            lines.append(f"- Avg latency: {data['avg_latency_ms']:.2f}ms")
            lines.append("")

        vuln = results["vulnerability_analysis"]
        lines.append("## Vulnerability Analysis")
        lines.append(f"- Most vulnerable category: {vuln['most_vulnerable']}")
        lines.append(f"- Most robust category: {vuln['most_robust']}")
        lines.append(f"- Accuracy range: {vuln['accuracy_range']:.4f}")

        if vuln["categories_below_threshold"]:
            lines.append(f"- Categories below 85% threshold: {', '.join(vuln['categories_below_threshold'])}")

        report = "\n".join(lines)

        if output_path:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info("Adversarial report saved to %s", output_path)

        return report

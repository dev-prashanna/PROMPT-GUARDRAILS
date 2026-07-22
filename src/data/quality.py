from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class QualityChecker:
    """Data quality checks and anomaly detection."""

    def __init__(self, config: dict[str, Any]):
        data_cfg = config.get("data", {})
        preproc_cfg = data_cfg.get("preprocessing", {})
        self.min_token_length = preproc_cfg.get("min_token_length", 5)
        self.max_token_length = preproc_cfg.get("max_token_length", 500)
        self.min_text_length = 3

    def check(self, df: pd.DataFrame) -> dict[str, Any]:
        report = {
            "total_samples": len(df),
            "checks": {},
            "passed": True,
        }

        report["checks"]["completeness"] = self._check_completeness(df)
        report["checks"]["uniqueness"] = self._check_uniqueness(df)
        report["checks"]["text_quality"] = self._check_text_quality(df)
        report["checks"]["label_distribution"] = self._check_label_distribution(df)
        report["checks"]["anomalies"] = self._check_anomalies(df)

        failed_checks = [
            name for name, check in report["checks"].items() if not check.get("passed", True)
        ]
        report["passed"] = len(failed_checks) == 0
        report["failed_checks"] = failed_checks

        if not report["passed"]:
            logger.warning("Quality checks failed: %s", failed_checks)
        else:
            logger.info("All quality checks passed for %d samples", len(df))

        return report

    def _check_completeness(self, df: pd.DataFrame) -> dict[str, Any]:
        null_text = df["text"].isnull().sum() if "text" in df.columns else 0
        null_label = df["label"].isnull().sum() if "label" in df.columns else 0
        empty_text = (df["text"].astype(str).str.strip() == "").sum() if "text" in df.columns else 0

        total_issues = int(null_text + null_label + empty_text)
        return {
            "passed": total_issues == 0,
            "null_text": int(null_text),
            "null_label": int(null_label),
            "empty_text": int(empty_text),
            "completeness_rate": 1 - (total_issues / (len(df) * 2)) if len(df) > 0 else 0,
        }

    def _check_uniqueness(self, df: pd.DataFrame) -> dict[str, Any]:
        if "text" not in df.columns:
            return {"passed": True, "duplicate_count": 0}

        duplicates = df["text"].duplicated().sum()
        return {
            "passed": duplicates == 0,
            "duplicate_count": int(duplicates),
            "uniqueness_rate": 1 - (duplicates / len(df)) if len(df) > 0 else 1.0,
        }

    def _check_text_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        if "text" not in df.columns:
            return {"passed": True}

        lengths = df["text"].astype(str).str.len()
        too_short = (lengths < self.min_text_length).sum()
        too_long = (lengths > 5000).sum()

        return {
            "passed": too_short == 0 and too_long == 0,
            "too_short": int(too_short),
            "too_long": int(too_long),
            "mean_length": float(lengths.mean()),
            "std_length": float(lengths.std()),
        }

    def _check_label_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        if "label" not in df.columns:
            return {"passed": True}

        counts = df["label"].value_counts()
        max_count = counts.max()
        min_count = counts.min()
        skew_ratio = max_count / min_count if min_count > 0 else float("inf")

        return {
            "passed": skew_ratio <= 10.0,
            "distribution": {str(k): int(v) for k, v in counts.items()},
            "skew_ratio": float(skew_ratio),
            "num_classes": len(counts),
        }

    def _check_anomalies(self, df: pd.DataFrame) -> dict[str, Any]:
        if "text" not in df.columns:
            return {"passed": True, "anomaly_count": 0}

        texts = df["text"].astype(str)
        special_char_ratio = texts.apply(lambda t: sum(1 for c in t if not c.isalnum() and not c.isspace()) / max(len(t), 1))
        high_special = (special_char_ratio > 0.5).sum()

        mean_ratio = special_char_ratio.mean()
        std_ratio = special_char_ratio.std()
        statistical_outliers = (special_char_ratio > mean_ratio + 3 * std_ratio).sum()

        return {
            "passed": high_special == 0,
            "high_special_char_count": int(high_special),
            "statistical_outliers": int(statistical_outliers),
            "mean_special_ratio": float(mean_ratio),
        }

    def generate_report(self, df: pd.DataFrame, output_path: str | None = None) -> dict[str, Any]:
        report = self.check(df)

        report["summary"] = {
            "total_samples": len(df),
            "quality_score": self._compute_quality_score(report),
            "recommendations": self._generate_recommendations(report),
        }

        if output_path:
            import json
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Quality report saved to %s", output_path)

        return report

    def _compute_quality_score(self, report: dict[str, Any]) -> float:
        scores = []
        for check_name, check in report.get("checks", {}).items():
            if "completeness_rate" in check:
                scores.append(check["completeness_rate"])
            elif "uniqueness_rate" in check:
                scores.append(check["uniqueness_rate"])
            elif "skew_ratio" in check:
                scores.append(min(1.0, 10.0 / check["skew_ratio"]))
            elif check.get("passed", True):
                scores.append(1.0)
            else:
                scores.append(0.5)

        return float(np.mean(scores)) if scores else 0.0

    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        recommendations = []
        for name, check in report.get("checks", {}).items():
            if not check.get("passed", True):
                if name == "completeness":
                    recommendations.append("Remove or impute missing values")
                elif name == "uniqueness":
                    recommendations.append(f"Remove {check.get('duplicate_count', 0)} duplicates")
                elif name == "label_distribution":
                    recommendations.append(f"Address class imbalance (skew={check.get('skew_ratio', 0):.1f}x)")
                elif name == "text_quality":
                    recommendations.append("Filter out anomalous text samples")
                elif name == "anomalies":
                    recommendations.append("Review and remove anomalous samples")
        return recommendations

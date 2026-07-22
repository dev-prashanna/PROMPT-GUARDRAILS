from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical tests, confidence intervals, and effect sizes for model evaluation."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        eval_cfg = config.get("evaluation", {}).get("statistical", {})
        self.confidence_level = eval_cfg.get("confidence_level", 0.95)
        self.bootstrap_iterations = eval_cfg.get("bootstrap_iterations", 1000)
        self.alpha = 1 - self.confidence_level

    def bootstrap_ci(
        self,
        values: np.ndarray,
        metric_fn: Any = None,
        confidence_level: float | None = None,
        n_iterations: int | None = None,
    ) -> dict[str, float]:
        confidence_level = confidence_level or self.confidence_level
        n_iterations = n_iterations or self.bootstrap_iterations

        if metric_fn is None:
            metric_fn = np.mean

        bootstrapped = []
        n = len(values)
        for _ in range(n_iterations):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrapped.append(metric_fn(sample))

        bootstrapped = np.array(bootstrapped)
        lower = np.percentile(bootstrapped, (self.alpha / 2) * 100)
        upper = np.percentile(bootstrapped, (1 - self.alpha / 2) * 100)

        return {
            "mean": float(metric_fn(values)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "ci_level": confidence_level,
            "std": float(np.std(bootstrapped)),
        }

    def paired_t_test(
        self, scores_1: np.ndarray, scores_2: np.ndarray
    ) -> dict[str, Any]:
        if len(scores_1) != len(scores_2):
            raise ValueError("Score arrays must have equal length for paired t-test")

        t_stat, p_value = scipy_stats.ttest_rel(scores_1, scores_2)
        diff = scores_1 - scores_2
        effect_size = self._cohens_d(diff)

        return {
            "test": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "effect_size": effect_size,
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
        }

    def wilcoxon_test(
        self, scores_1: np.ndarray, scores_2: np.ndarray
    ) -> dict[str, Any]:
        if len(scores_1) != len(scores_2):
            raise ValueError("Score arrays must have equal length for Wilcoxon test")

        try:
            stat, p_value = scipy_stats.wilcoxon(scores_1, scores_2)
        except ValueError:
            stat, p_value = 0.0, 1.0

        return {
            "test": "wilcoxon_signed_rank",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }

    def mcnemar_test(
        self,
        correct_1: np.ndarray,
        correct_2: np.ndarray,
    ) -> dict[str, Any]:
        b01 = np.sum((~correct_1) & correct_2)
        b10 = np.sum(correct_1 & (~correct_2))

        if b01 + b10 == 0:
            return {
                "test": "mcnemar",
                "chi2": 0.0,
                "p_value": 1.0,
                "significant": False,
                "b01": int(b01),
                "b10": int(b10),
            }

        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)

        return {
            "test": "mcnemar",
            "chi2": float(chi2),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "b01": int(b01),
            "b10": int(b10),
            "effect_size": float(b01 / (b01 + b10)) if (b01 + b10) > 0 else 0.0,
        }

    def multiple_comparison_correction(
        self, p_values: list[float], method: str = "bonferroni"
    ) -> list[dict[str, Any]]:
        n = len(p_values)
        if method == "bonferroni":
            adjusted = [min(p * n, 1.0) for p in p_values]
        elif method == "holm":
            sorted_indices = np.argsort(p_values)
            adjusted_arr = np.zeros(n)
            for rank, idx in enumerate(sorted_indices):
                adjusted_arr[idx] = min(p_values[idx] * (n - rank), 1.0)
            for i in range(1, len(adjusted_arr)):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                adjusted_arr[idx] = max(adjusted_arr[idx], adjusted_arr[prev_idx])
            adjusted = adjusted_arr.tolist()
        else:
            adjusted = p_values

        return [
            {
                "original_p": p,
                "adjusted_p": a,
                "significant": a < self.alpha,
            }
            for p, a in zip(p_values, adjusted)
        ]

    def _cohens_d(self, differences: np.ndarray) -> dict[str, float]:
        mean_d = np.mean(differences)
        std_d = np.std(differences, ddof=1)
        d = mean_d / std_d if std_d > 0 else 0.0

        if abs(d) < 0.2:
            magnitude = "negligible"
        elif abs(d) < 0.5:
            magnitude = "small"
        elif abs(d) < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        return {
            "d": float(d),
            "magnitude": magnitude,
        }

    def compare_models(
        self,
        model_a_scores: np.ndarray,
        model_b_scores: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> dict[str, Any]:
        ci_a = self.bootstrap_ci(model_a_scores)
        ci_b = self.bootstrap_ci(model_b_scores)

        t_test = self.paired_t_test(model_a_scores, model_b_scores)
        wilcoxon = self.wilcoxon_test(model_a_scores, model_b_scores)

        correct_a = model_a_scores > 0.5
        correct_b = model_b_scores > 0.5
        mcnemar = self.mcnemar_test(correct_a, correct_b)

        return {
            "model_a": {
                "name": model_a_name,
                "mean": ci_a["mean"],
                "ci": [ci_a["ci_lower"], ci_a["ci_upper"]],
            },
            "model_b": {
                "name": model_b_name,
                "mean": ci_b["mean"],
                "ci": [ci_b["ci_lower"], ci_b["ci_upper"]],
            },
            "tests": {
                "paired_t_test": t_test,
                "wilcoxon": wilcoxon,
                "mcnemar": mcnemar,
            },
            "winner": model_a_name if ci_a["mean"] > ci_b["mean"] else model_b_name,
            "significant_difference": t_test["significant"],
        }

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AblationStudy:
    """Framework for running and analyzing ablation studies."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.output_dir = Path("paper/tables/ablations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def define_ablations(self) -> dict[str, dict[str, Any]]:
        return {
            "data_source": {
                "description": "Remove one data source at a time",
                "variants": [
                    {"name": "all_sources", "remove": []},
                    {"name": "no_necent", "remove": ["necent/llm-jailbreak-prompt-injection-dataset"]},
                    {"name": "no_protectai", "remove": ["protectai/restrictedresponses"]},
                    {"name": "no_hh_rlhf", "remove": ["Anthropic/hh-rlhf"]},
                ],
            },
            "augmentation": {
                "description": "Disable each augmentation technique",
                "variants": [
                    {"name": "all_augmentations", "disabled": []},
                    {"name": "no_synonym", "disabled": ["synonym_replacement"]},
                    {"name": "no_random_insertion", "disabled": ["random_insertion"]},
                    {"name": "no_back_translation", "disabled": ["back_translation"]},
                    {"name": "no_augmentation", "disabled": ["all"]},
                ],
            },
            "architecture": {
                "description": "Compare different encoder architectures",
                "variants": [
                    {"name": "deberta_v3_base", "model": "microsoft/deberta-v3-base"},
                    {"name": "bert_base", "model": "bert-base-uncased"},
                    {"name": "roberta_base", "model": "roberta-base"},
                    {"name": "distilbert_base", "model": "distilbert-base-uncased"},
                ],
            },
            "loss_function": {
                "description": "Compare loss functions for class imbalance",
                "variants": [
                    {"name": "weighted_ce", "loss": "weighted_cross_entropy"},
                    {"name": "focal_loss", "loss": "focal_loss", "gamma": 2.0},
                    {"name": "label_smoothing", "loss": "label_smoothing", "smoothing": 0.1},
                ],
            },
            "peft_rank": {
                "description": "LoRA rank ablation",
                "variants": [
                    {"name": "rank_4", "rank": 4},
                    {"name": "rank_8", "rank": 8},
                    {"name": "rank_16", "rank": 16},
                    {"name": "rank_32", "rank": 32},
                    {"name": "rank_64", "rank": 64},
                ],
            },
            "max_length": {
                "description": "Max sequence length ablation",
                "variants": [
                    {"name": "length_128", "max_length": 128},
                    {"name": "length_256", "max_length": 256},
                    {"name": "length_384", "max_length": 384},
                    {"name": "length_512", "max_length": 512},
                ],
            },
        }

    def get_ablation_config(
        self, ablation_type: str, variant_name: str
    ) -> dict[str, Any] | None:
        ablations = self.define_ablations()
        if ablation_type not in ablations:
            return None

        for variant in ablations[ablation_type]["variants"]:
            if variant["name"] == variant_name:
                return variant
        return None

    def analyze_results(
        self,
        ablation_type: str,
        results: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        baseline_name = list(results.keys())[0]
        baseline_metrics = results[baseline_name]

        analysis = {
            "ablation_type": ablation_type,
            "description": self.define_ablations().get(ablation_type, {}).get("description", ""),
            "baseline": baseline_name,
            "variants": {},
        }

        for variant_name, metrics in results.items():
            delta = {}
            for metric_name, value in metrics.items():
                if metric_name in baseline_metrics:
                    delta[metric_name] = value - baseline_metrics[metric_name]

            analysis["variants"][variant_name] = {
                "metrics": metrics,
                "delta_from_baseline": delta,
                "is_better": all(
                    delta.get(m, 0) >= 0
                    for m in ["macro_f1", "weighted_f1", "accuracy"]
                ),
            }

        analysis["ranking"] = self._rank_variants(results)
        analysis["best_variant"] = analysis["ranking"][0] if analysis["ranking"] else None

        return analysis

    def _rank_variants(
        self, results: dict[str, dict[str, float]], primary_metric: str = "macro_f1"
    ) -> list[str]:
        sorted_variants = sorted(
            results.keys(),
            key=lambda k: results[k].get(primary_metric, 0),
            reverse=True,
        )
        return sorted_variants

    def generate_latex(self, analysis: dict[str, Any], output_path: str | None = None) -> str:
        ablation_type = analysis["ablation_type"]
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Ablation Study: " + ablation_type.replace("_", " ").title() + "}",
            r"\label{tab:ablation-" + ablation_type + "}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Variant & Macro F1 & Weighted F1 & Accuracy & $\Delta$ F1 \\",
            r"\midrule",
        ]

        baseline = analysis.get("baseline", "")
        for variant_name, data in analysis.get("variants", {}).items():
            metrics = data["metrics"]
            delta = data.get("delta_from_baseline", {})
            delta_f1 = delta.get("macro_f1", 0)
            delta_str = f"{delta_f1:+.4f}" if variant_name != baseline else "---"

            label = variant_name.replace("_", " ").title()
            if variant_name == baseline:
                label += " (baseline)"

            lines.append(
                f"{label} & {metrics.get('macro_f1', 0):.4f} & "
                f"{metrics.get('weighted_f1', 0):.4f} & "
                f"{metrics.get('accuracy', 0):.4f} & {delta_str} \\\\"
            )

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        latex = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(latex)
            logger.info("Ablation LaTeX table saved to %s", output_path)

        return latex

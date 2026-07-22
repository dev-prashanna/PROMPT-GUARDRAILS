from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Cross-model comparison and latency profiling."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.results: dict[str, dict[str, Any]] = {}

    def profile_latency(
        self,
        model_path: str,
        texts: list[str],
        device: str = "cuda",
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> dict[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        latencies = []
        for text in texts[:num_warmup]:
            inputs = tokenizer(
                text, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                model(**inputs)

        for _ in range(num_runs):
            text = texts[np.random.randint(len(texts))]
            inputs = tokenizer(
                text, max_length=512, padding="max_length",
                truncation=True, return_tensors="pt",
            ).to(device)
            start = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies_arr = np.array(latencies)
        return {
            "model_path": model_path,
            "num_runs": num_runs,
            "mean_ms": float(latencies_arr.mean()),
            "std_ms": float(latencies_arr.std()),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
            "throughput_qps": 1000.0 / float(latencies_arr.mean()) if latencies_arr.mean() > 0 else 0,
        }

    def compare_architectures(
        self,
        model_configs: list[dict[str, str]],
        eval_fn: Any,
        texts: list[str] | None = None,
        device: str = "cuda",
    ) -> dict[str, Any]:
        comparison = {"models": {}, "ranking": []}

        for config in model_configs:
            name = config["name"]
            path = config.get("path", name)

            logger.info("Benchmarking model: %s", name)

            eval_result = eval_fn(path)

            latency_result = None
            if texts:
                try:
                    latency_result = self.profile_latency(path, texts, device)
                except Exception as e:
                    logger.warning("Latency profiling failed for %s: %s", name, e)

            comparison["models"][name] = {
                "architecture": config.get("type", "unknown"),
                "params": config.get("params", "unknown"),
                "evaluation": eval_result,
                "latency": latency_result,
            }

        ranked = sorted(
            comparison["models"].items(),
            key=lambda x: x[1]["evaluation"].get("macro_f1", 0),
            reverse=True,
        )
        comparison["ranking"] = [name for name, _ in ranked]

        comparison["efficiency_analysis"] = self._compute_efficiency(
            comparison["models"]
        )

        return comparison

    def _compute_efficiency(
        self, models: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        efficiency = {}
        for name, data in models.items():
            f1 = data["evaluation"].get("macro_f1", 0)
            latency = (
                data["latency"]["mean_ms"]
                if data.get("latency")
                else None
            )
            params = data.get("params", "unknown")

            if isinstance(params, str) and params.endswith("M"):
                param_count = float(params[:-1])
            else:
                param_count = 0

            efficiency[name] = {
                "f1_per_param_million": f1 / param_count if param_count > 0 else 0,
                "f1_per_latency_ms": f1 / latency if latency and latency > 0 else 0,
                "latency_efficient": latency is not None and latency < 50,
            }

        return efficiency

    def generate_comparison_table(
        self, comparison: dict[str, Any], output_path: str | None = None
    ) -> str:
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Model Architecture Comparison}",
            r"\label{tab:architecture-comparison}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Model & Params & Macro F1 & Latency (ms) & Throughput (QPS) \\",
            r"\midrule",
        ]

        for name in comparison.get("ranking", []):
            data = comparison["models"][name]
            f1 = data["evaluation"].get("macro_f1", 0)
            latency = data["latency"]["mean_ms"] if data.get("latency") else "N/A"
            throughput = (
                f"{data['latency']['throughput_qps']:.0f}"
                if data.get("latency")
                else "N/A"
            )
            params = data.get("params", "N/A")

            if isinstance(latency, float):
                latency_str = f"{latency:.2f}"
            else:
                latency_str = str(latency)

            lines.append(
                f"{name} & {params} & {f1:.4f} & {latency_str} & {throughput} \\\\"
            )

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        latex = "\n".join(lines)

        if output_path:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(latex)
            logger.info("Comparison table saved to %s", output_path)

        return latex

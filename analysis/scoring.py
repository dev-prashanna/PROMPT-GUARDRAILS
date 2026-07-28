import json
import numpy as np
from pathlib import Path
from typing import Dict, List


DEFAULT_WEIGHTS = {
    "accuracy": 0.25,
    "robustness": 0.25,
    "false_positive_rate": 0.15,
    "latency": 0.15,
    "overhead": 0.10,
    "integration": 0.10,
}


def compute_cgs(metrics: dict, weights: dict = None, max_latency: float = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    latencies = metrics.get("latency_p50_ms", 1.0)
    if max_latency is None:
        max_latency = max(latencies, 1.0) * 1.5

    components = {
        "accuracy": metrics.get("f1", 0.0),
        "robustness": metrics.get("robustness_score", 0.8),
        "false_positive_rate": 1.0 - min(metrics.get("fpr", 0.0), 1.0),
        "latency": 1.0 - min(latencies / max_latency, 1.0),
        "overhead": 1.0 - min(metrics.get("memory_peak_mb", 0) / 8192, 1.0),
        "integration": metrics.get("integration_score", 0.5),
    }

    cgs = sum(weights[k] * components[k] for k in weights if k in components)
    return round(cgs, 4)


def sensitivity_analysis(
    metrics: dict, base_cgs: float, delta: float = 0.05
) -> Dict[str, float]:
    sensitivities = {}
    for key in DEFAULT_WEIGHTS:
        perturbed_up = dict(DEFAULT_WEIGHTS)
        perturbed_up[key] = min(perturbed_up[key] + delta, 1.0)
        remaining = 1.0 - perturbed_up[key]
        other_sum = sum(v for k, v in DEFAULT_WEIGHTS.items() if k != key)
        if other_sum > 0:
            for k2 in perturbed_up:
                if k2 != key:
                    perturbed_up[k2] = DEFAULT_WEIGHTS[k2] / other_sum * remaining

        cgs_up = compute_cgs(metrics, perturbed_up)
        sensitivities[key] = round((cgs_up - base_cgs) / delta, 4)

    return sensitivities


def rank_solutions(all_metrics: Dict[str, dict], weights: dict = None) -> List[dict]:
    rankings = []
    latencies = [m.get("latency_p50_ms", 1.0) for m in all_metrics.values()]
    max_lat = max(latencies) * 1.5 if latencies else 1.0

    for name, m in all_metrics.items():
        cgs = compute_cgs(m, weights, max_lat)
        rankings.append({"name": name, "cgs": cgs, "metrics": m})

    rankings.sort(key=lambda x: x["cgs"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    return rankings

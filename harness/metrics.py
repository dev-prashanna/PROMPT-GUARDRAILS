import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ClassificationMetrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    latencies: List[float] = None

    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def fnr(self) -> float:
        denom = self.fn + self.tp
        return self.fn / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def latency_p50(self) -> float:
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0.0

    @property
    def latency_p95(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0

    @property
    def latency_p99(self) -> float:
        return float(np.percentile(self.latencies, 99)) if self.latencies else 0.0

    @property
    def latency_mean(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "accuracy": round(self.accuracy, 4),
            "latency_mean_ms": round(self.latency_mean, 2),
            "latency_p50_ms": round(self.latency_p50, 2),
            "latency_p95_ms": round(self.latency_p95, 2),
            "latency_p99_ms": round(self.latency_p99, 2),
        }


def compute_metrics(results_path: str) -> ClassificationMetrics:
    metrics = ClassificationMetrics()
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                continue
            true = r["true_label"]
            pred = r["predicted_label"]
            if true == 1 and pred == 1:
                metrics.tp += 1
            elif true == 0 and pred == 0:
                metrics.tn += 1
            elif true == 0 and pred == 1:
                metrics.fp += 1
            elif true == 1 and pred == 0:
                metrics.fn += 1
            metrics.latencies.append(r["latency_ms"])
    return metrics


def bootstrap_ci(
    values: List[float], n_bootstrap: int = 10000, ci: float = 0.95
) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    stats = []
    arr = np.array(values)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(float(np.mean(sample)))
    lower = (1 - ci) / 2
    upper = 1 - lower
    return float(np.percentile(stats, lower * 100)), float(np.percentile(stats, upper * 100))


def cohens_h(p1: float, p2: float) -> float:
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def mcnemar_test(b: int, c: int) -> float:
    from scipy.stats import chi2
    if b + c == 0:
        return 1.0
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return 1 - chi2.cdf(statistic, df=1)


def compute_all_metrics(output_dir: str) -> Dict[str, dict]:
    all_metrics = {}
    results_dir = Path(output_dir)
    for f in sorted(results_dir.glob("*.jsonl")):
        name = f.stem.rsplit("_run_", 1)[0]
        metrics = compute_metrics(str(f))
        all_metrics[f.stem] = metrics.to_dict()
    return all_metrics

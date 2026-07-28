import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_metrics(processed_dir: str) -> Dict[str, dict]:
    metrics = {}
    for f in Path(processed_dir).glob("*.json"):
        with open(f) as fh:
            metrics[f.stem] = json.load(fh)
    return metrics


def plot_comparison_radar(metrics: Dict[str, dict], output_path: str):
    categories = ["f1", "recall", "1-fpr", "accuracy"]
    labels = ["F1-Score", "Recall", "1-FPR", "Accuracy"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for name, m in metrics.items():
        values = [m.get(cat, 0) for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("Guardrail Classification Performance", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latency_comparison(metrics: Dict[str, dict], output_path: str):
    names = list(metrics.keys())
    p50 = [metrics[n].get("latency_p50_ms", 0) for n in names]
    p95 = [metrics[n].get("latency_p95_ms", 0) for n in names]
    p99 = [metrics[n].get("latency_p99_ms", 0) for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, p50, width, label="p50", color="#2196F3")
    ax.bar(x, p95, width, label="p95", color="#FF9800")
    ax.bar(x + width, p99, width, label="p99", color="#F44336")

    ax.set_xlabel("Guardrail Solution")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fpr_fnr_tradeoff(metrics: Dict[str, dict], output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, m in metrics.items():
        ax.scatter(m.get("fpr", 0), m.get("fnr", 0), s=100, label=name, zorder=5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Ideal (0,0)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("FPR vs FNR Trade-off")
    ax.legend()
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_composite_scores(cgs_scores: Dict[str, float], output_path: str):
    names = list(cgs_scores.keys())
    scores = list(cgs_scores.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, scores, color="#4CAF50")
    ax.set_xlabel("Composite Guardrail Score (CGS)")
    ax.set_title("Overall Guardrail Rankings")
    ax.set_xlim(0, 1)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_figures(metrics: Dict[str, dict], cgs_scores: Dict[str, float], output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_comparison_radar(metrics, str(out / "radar_performance.png"))
    plot_latency_comparison(metrics, str(out / "latency_comparison.png"))
    plot_fpr_fnr_tradeoff(metrics, str(out / "fpr_fnr_tradeoff.png"))
    plot_composite_scores(cgs_scores, str(out / "composite_scores.png"))

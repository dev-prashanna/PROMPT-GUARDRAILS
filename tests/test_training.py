import numpy as np
from src.evaluation.metrics import compute_metrics


class TestMetrics:
    def test_compute_metrics_perfect(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        logits = np.zeros((6, 3))
        logits[0, 0] = 10
        logits[1, 1] = 10
        logits[2, 2] = 10
        logits[3, 0] = 10
        logits[4, 1] = 10
        logits[5, 2] = 10

        metrics = compute_metrics((logits, labels))
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_compute_metrics_with_errors(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        logits = np.zeros((6, 3))
        logits[0, 0] = 10
        logits[1, 1] = 10
        logits[2, 1] = 10
        logits[3, 0] = 10
        logits[4, 2] = 10
        logits[5, 2] = 10

        metrics = compute_metrics((logits, labels))
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert "macro_precision" in metrics
        assert "macro_recall" in metrics

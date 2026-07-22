from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DecisionLogger:
    """Log per-sample predictions for audit and analysis."""

    def __init__(self, output_dir: str = "experiments/decisions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._flush_threshold = 1000

    def log_prediction(
        self,
        experiment_id: str,
        sample_id: str,
        text: str,
        true_label: int,
        predicted_label: int,
        confidence: float,
        probabilities: dict[str, float],
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "experiment_id": experiment_id,
            "sample_id": sample_id,
            "text": text[:500],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "latency_ms": latency_ms,
            "is_correct": true_label == predicted_label,
            "timestamp": time.time(),
        }
        if metadata:
            record["metadata"] = metadata

        self._buffer.append(record)
        if len(self._buffer) >= self._flush_threshold:
            self.flush()

    def log_batch(
        self,
        experiment_id: str,
        texts: list[str],
        true_labels: list[int],
        predicted_labels: list[int],
        confidences: list[float],
        probabilities: list[dict[str, float]],
        latencies_ms: list[float],
    ) -> None:
        for i in range(len(texts)):
            self.log_prediction(
                experiment_id=experiment_id,
                sample_id=f"{experiment_id}_{len(self._buffer) + i}",
                text=texts[i],
                true_label=true_labels[i],
                predicted_label=predicted_labels[i],
                confidence=confidences[i],
                probabilities=probabilities[i],
                latency_ms=latencies_ms[i],
            )

    def flush(self) -> None:
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        output_path = self.output_dir / f"decisions_{int(time.time())}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Flushed %d predictions to %s", len(self._buffer), output_path)
        self._buffer.clear()

    def load_decisions(self, experiment_id: str | None = None) -> pd.DataFrame:
        frames = []
        for f in self.output_dir.glob("decisions_*.parquet"):
            df = pd.read_parquet(f)
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if experiment_id:
            combined = combined[combined["experiment_id"] == experiment_id]
        return combined

    def compute_error_analysis(
        self, experiment_id: str
    ) -> dict[str, Any]:
        df = self.load_decisions(experiment_id)
        if df.empty:
            return {"error": "No decisions found"}

        errors = df[~df["is_correct"]]
        correct = df[df["is_correct"]]

        label_names = {0: "safe", 1: "prompt_injection", 2: "jailbreak"}
        error_by_class = {}
        for label_id, label_name in label_names.items():
            class_errors = errors[errors["true_label"] == label_id]
            class_total = df[df["true_label"] == label_id]
            error_by_class[label_name] = {
                "total": len(class_total),
                "errors": len(class_errors),
                "error_rate": len(class_errors) / len(class_total) if len(class_total) > 0 else 0,
                "avg_confidence_errors": float(class_errors["confidence"].mean()) if len(class_errors) > 0 else 0,
                "avg_confidence_correct": float(correct[correct["true_label"] == label_id]["confidence"].mean()) if len(correct[correct["true_label"] == label_id]) > 0 else 0,
            }

        return {
            "total_samples": len(df),
            "total_correct": len(correct),
            "total_errors": len(errors),
            "accuracy": len(correct) / len(df),
            "avg_confidence_correct": float(correct["confidence"].mean()) if len(correct) > 0 else 0,
            "avg_confidence_errors": float(errors["confidence"].mean()) if len(errors) > 0 else 0,
            "avg_latency_ms": float(df["latency_ms"].mean()),
            "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
            "p99_latency_ms": float(df["latency_ms"].quantile(0.99)),
            "error_by_class": error_by_class,
        }

    def get_low_confidence_errors(
        self, experiment_id: str, threshold: float = 0.6
    ) -> pd.DataFrame:
        df = self.load_decisions(experiment_id)
        if df.empty:
            return pd.DataFrame()
        return df[(~df["is_correct"]) & (df["confidence"] < threshold)]

    def get_high_confidence_errors(
        self, experiment_id: str, threshold: float = 0.9
    ) -> pd.DataFrame:
        df = self.load_decisions(experiment_id)
        if df.empty:
            return pd.DataFrame()
        return df[(~df["is_correct"]) & (df["confidence"] >= threshold)]

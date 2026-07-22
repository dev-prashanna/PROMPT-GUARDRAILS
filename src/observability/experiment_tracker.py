from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiments, metrics, and artifacts with local SQLite backend."""

    def __init__(self, db_path: str = "experiments/tracker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                config_hash TEXT NOT NULL,
                config_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'running',
                parent_experiment_id TEXT
            );
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                step INTEGER NOT NULL,
                epoch INTEGER,
                split TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                text TEXT NOT NULL,
                true_label INTEGER NOT NULL,
                predicted_label INTEGER NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT,
                latency_ms REAL,
                is_correct INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                total_samples INTEGER,
                label_distribution TEXT,
                quality_scores TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                provenance TEXT
            );
        """)
        self._conn.commit()

    def create_experiment(
        self, name: str, config: dict[str, Any], description: str = ""
    ) -> str:
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO experiments (id, name, description, config_hash, config_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (experiment_id, name, description, config_hash, json.dumps(config, default=str)),
        )
        self._conn.commit()
        logger.info("Created experiment: %s (hash=%s)", experiment_id, config_hash)
        return experiment_id

    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: int,
        epoch: int | None = None,
        split: str | None = None,
    ) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO metrics (experiment_id, metric_name, value, step, epoch, split) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, metric_name, value, step, epoch, split),
        )
        self._conn.commit()

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
    ) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO predictions "
            "(experiment_id, sample_id, text, true_label, predicted_label, "
            "confidence, probabilities, latency_ms, is_correct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                sample_id,
                text,
                true_label,
                predicted_label,
                confidence,
                json.dumps(probabilities),
                latency_ms,
                1 if true_label == predicted_label else 0,
            ),
        )
        self._conn.commit()

    def log_metrics_batch(
        self,
        experiment_id: str,
        metrics: dict[str, float],
        step: int,
        epoch: int | None = None,
        split: str | None = None,
    ) -> None:
        for name, value in metrics.items():
            self.log_metric(experiment_id, name, value, step, epoch, split)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    def get_metrics(
        self,
        experiment_id: str,
        metric_name: str | None = None,
        epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM metrics WHERE experiment_id = ?"
        params: list[Any] = [experiment_id]
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        if epoch is not None:
            query += " AND epoch = ?"
            params.append(epoch)
        query += " ORDER BY step"

        cursor = self._conn.cursor()
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_predictions(
        self, experiment_id: str, correct_only: bool | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM predictions WHERE experiment_id = ?"
        params: list[Any] = [experiment_id]
        if correct_only is not None:
            query += " AND is_correct = ?"
            params.append(1 if correct_only else 0)

        cursor = self._conn.cursor()
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE experiments SET status = ? WHERE id = ?", (status, experiment_id)
        )
        self._conn.commit()

    def list_experiments(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM experiments"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"

        cursor = self._conn.cursor()
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()

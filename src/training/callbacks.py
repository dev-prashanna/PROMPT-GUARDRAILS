from __future__ import annotations

from typing import Any

from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
)


class MetricsLoggingCallback(TrainerCallback):
    """Log detailed metrics at each evaluation step."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"[Eval Step {state.global_step}] {key}: {value:.4f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            if loss is not None:
                print(f"[Train Step {state.global_step}] loss: {loss:.4f}")


class FirewalCallbacks:
    """Configure all callbacks for the training pipeline."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.early_stop_cfg = config.get("training", {}).get("early_stopping", {})

    def get_callbacks(self) -> list[TrainerCallback]:
        callbacks = [MetricsLoggingCallback()]

        if self.early_stop_cfg.get("enabled", True):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stop_cfg.get("patience", 3),
                    early_stopping_threshold=self.early_stop_cfg.get("min_delta", 0.001),
                )
            )

        return callbacks

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FirewalCallbacks:
        return cls(config)

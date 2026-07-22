from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class FirewallPipeline:
    """End-to-end firewall pipeline for LLM security screening."""

    def __init__(self, model_path: str, config: dict[str, Any]):
        self.config = config
        self.model_path = model_path
        self.device = config.get("deployment", {}).get("serving", {}).get(
            "model_loading", {}
        ).get("device", "cuda")
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        logger.info("Loading firewall pipeline from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Firewall pipeline loaded on %s", self.device)

    def predict(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        inputs = self._tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        start = time.perf_counter()
        with __import__("torch").no_grad():
            outputs = self._model(**inputs)
        latency = (time.perf_counter() - start) * 1000

        probs = __import__("torch").softmax(outputs.logits, dim=-1)
        probs_np = probs.cpu().numpy()[0]
        pred_id = int(probs_np.argmax())

        label_map = {0: "safe", 1: "prompt_injection", 2: "jailbreak"}
        return {
            "text": text,
            "label": label_map.get(pred_id, "unknown"),
            "label_id": pred_id,
            "confidence": float(probs_np[pred_id]),
            "probabilities": {
                label_map[i]: float(p) for i, p in enumerate(probs_np)
            },
            "latency_ms": latency,
            "safe": pred_id == 0 and float(probs_np[0]) >= threshold,
        }

    def predict_batch(self, texts: list[str], threshold: float = 0.5) -> list[dict]:
        return [self.predict(text, threshold) for text in texts]

    def export_for_serving(self, output_dir: str) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(out)
        self._tokenizer.save_pretrained(out)
        logger.info("Model exported to %s", out)
        return str(out)

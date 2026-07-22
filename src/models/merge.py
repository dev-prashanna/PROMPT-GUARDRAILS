from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelMerger:
    """Merge LoRA adapter weights into the base DeBERTa-v3 model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.base_model_name = config.get("model", {}).get(
            "base_model", "microsoft/deberta-v3-base"
        )
        self.output_dir = Path(
            config.get("deployment", {}).get("merged_model_path", "deployment/merged_model")
        )

    def merge(self, peft_model_path: str, output_dir: str | None = None) -> str:
        output = Path(output_dir) if output_dir else self.output_dir
        output.mkdir(parents=True, exist_ok=True)

        logger.info("Loading base model: %s", self.base_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=self.config.get("model", {}).get("num_labels", 3),
            problem_type="single_label_classification",
        )

        logger.info("Loading LoRA adapter from: %s", peft_model_path)
        peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

        logger.info("Merging adapter weights into base model")
        merged_model = peft_model.merge_and_unload()

        logger.info("Saving merged model to: %s", output)
        merged_model.save_pretrained(output)

        tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
        tokenizer.save_pretrained(output)

        logger.info("Model merge complete. Saved to %s", output)
        return str(output)

    def load_merged(self, model_path: str | None = None):
        path = Path(model_path) if model_path else self.output_dir
        if not path.exists():
            raise FileNotFoundError(f"Merged model not found at {path}")

        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return model, tokenizer

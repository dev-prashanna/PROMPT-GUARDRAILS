from __future__ import annotations

import logging
from typing import Any

from peft import LoraConfig, TaskType, get_peft_model
from src.models.deberta_classifier import DeBERTaFirewallClassifier

logger = logging.getLogger(__name__)


class LoRAConfigurator:
    """Configure and apply LoRA PEFT to DeBERTa-v3 classifier."""

    def __init__(self, config: dict[str, Any]):
        peft_cfg = config.get("model", {}).get("peft", {})
        self.enabled = peft_cfg.get("enabled", True)
        self.target_modules = peft_cfg.get("target_modules", ["query_proj", "value_proj"])
        self.rank = peft_cfg.get("rank", 16)
        self.alpha = peft_cfg.get("alpha", 32)
        self.dropout = peft_cfg.get("dropout", 0.1)
        self.bias = peft_cfg.get("bias", "none")
        self.task_type_str = peft_cfg.get("task_type", "SEQ_CLS")
        self.merge_after = peft_cfg.get("merge_after_training", True)

    def apply(self, classifier: DeBERTaFirewallClassifier) -> Any:
        if not self.enabled:
            logger.info("LoRA disabled, returning base model")
            return classifier

        task_type = TaskType.SEQ_CLS
        lora_config = LoraConfig(
            task_type=task_type,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
        )

        peft_model = get_peft_model(classifier, lora_config)
        peft_model.print_trainable_parameters()
        logger.info(
            "Applied LoRA with rank=%d, alpha=%d, targets=%s",
            self.rank, self.alpha, self.target_modules,
        )
        return peft_model

    def get_lora_config(self) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
        )

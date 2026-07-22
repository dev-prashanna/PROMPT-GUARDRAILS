from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class DeBERTaFirewallClassifier(nn.Module):
    """DeBERTa-v3 based classifier for prompt injection and jailbreak detection."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        model_cfg = config.get("model", {})
        head_cfg = model_cfg.get("classification_head", {})
        self.base_model_name = model_cfg.get("base_model", "microsoft/deberta-v3-base")
        self.num_labels = model_cfg.get("num_labels", 3)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=self.num_labels,
            problem_type=model_cfg.get("problem_type", "single_label_classification"),
            ignore_mismatched_sizes=True,
        )
        self.config = self.model.config

        if head_cfg.get("dropout", 0) > 0:
            self._set_dropout(head_cfg["dropout"])

        trainable, total = self._count_parameters()
        logger.info(
            "Initialized DeBERTaFirewallClassifier: %d/%d trainable parameters",
            trainable, total,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def _set_dropout(self, rate: float) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def _count_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    @property
    def trainable_parameters(self) -> int:
        return self._count_parameters()[0]

    @property
    def total_parameters(self) -> int:
        return self._count_parameters()[1]

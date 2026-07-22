from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class FirewallDataset(Dataset):
    """PyTorch Dataset for prompt injection / jailbreak classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> FirewallDataset:
        return cls(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )

    def compute_class_weights(self) -> torch.Tensor:
        labels = np.array(self.labels)
        classes, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = total / (len(classes) * counts)
        weight_map = dict(zip(classes.astype(int), weights))
        return torch.tensor(
            [weight_map.get(i, 1.0) for i in range(len(classes))],
            dtype=torch.float32,
        )

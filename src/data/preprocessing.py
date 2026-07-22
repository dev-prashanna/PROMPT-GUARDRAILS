from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Clean, normalize, and tokenize text for DeBERTa-v3 input."""

    def __init__(self, config: dict[str, Any]):
        data_cfg = config.get("data", {})
        preproc_cfg = data_cfg.get("preprocessing", {})
        self.max_length = preproc_cfg.get("max_length", 512)
        self.lowercase = preproc_cfg.get("lowercase", False)
        self.strip_whitespace = preproc_cfg.get("strip_whitespace", True)
        self.deduplication = preproc_cfg.get("deduplication", True)
        self.min_token_length = preproc_cfg.get("min_token_length", 5)
        self.max_token_length = preproc_cfg.get("max_token_length", 500)

        tokenizer_name = preproc_cfg.get("tokenizer", "microsoft/deberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["text"] = df["text"].apply(self._clean_text)
        df = df[df["text"].str.strip().str.len() > 0]

        token_lengths = df["text"].apply(lambda t: len(self.tokenizer.encode(t)))
        mask = token_lengths.between(self.min_token_length, self.max_token_length)
        removed = (~mask).sum()
        if removed > 0:
            logger.info("Removed %d samples outside token length bounds", removed)
        df = df[mask].copy()

        if self.deduplication:
            before = len(df)
            df = df.drop_duplicates(subset=["text"], keep="first")
            df = df.reset_index(drop=True)
            logger.info("Deduplication: %d -> %d samples", before, len(df))

        logger.info("Preprocessed dataset: %d samples", len(df))
        return df

    def tokenize(
        self, texts: list[str], padding: str = "max_length"
    ) -> dict[str, Any]:
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    def tokenize_batch(self, df: pd.DataFrame) -> dict[str, Any]:
        return self.tokenize(df["text"].tolist())

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"http\S+|www\.\S+", "[URL]", text)
        text = re.sub(r"\s+", " ", text)
        if self.strip_whitespace:
            text = text.strip()
        if self.lowercase:
            text = text.lower()
        return text

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

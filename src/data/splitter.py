from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """Stratified train/val/test splitting with deduplication and leakage prevention."""

    def __init__(self, config: dict[str, Any]):
        split_cfg = config.get("data", {}).get("splits", {})
        self.train_ratio = split_cfg.get("train_ratio", 0.8)
        self.val_ratio = split_cfg.get("val_ratio", 0.1)
        self.test_ratio = split_cfg.get("test_ratio", 0.1)
        self.seed = split_cfg.get("random_seed", 42)
        self.stratify_by = split_cfg.get("stratify_by", "label")

    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = self._deduplicate(df)
        self._validate_no_leakage(df)

        test_size = self.test_ratio
        val_relative = self.val_ratio / (1 - test_size)

        stratify_col = df[self.stratify_by] if self.stratify_by in df.columns else None

        train_val, test = train_test_split(
            df, test_size=test_size, stratify=stratify_col, random_state=self.seed
        )

        stratify_tv = train_val[self.stratify_by] if self.stratify_by in train_val.columns else None
        train, val = train_test_split(
            train_val, test_size=val_relative, stratify=stratify_tv, random_state=self.seed
        )

        self._validate_split(train, val, test)

        logger.info(
            "Split complete: train=%d, val=%d, test=%d",
            len(train), len(val), len(test),
        )

        return {"train": train, "val": val, "test": test}

    def split_by_source(
        self, df: pd.DataFrame, holdout_source: str
    ) -> dict[str, pd.DataFrame]:
        if "source" not in df.columns:
            raise ValueError("DataFrame must have 'source' column for source-based splitting")

        train_pool = df[df["source"] != holdout_source].copy()
        test = df[df["source"] == holdout_source].copy()

        stratify_col = train_pool[self.stratify_by] if self.stratify_by in train_pool.columns else None
        train, val = train_test_split(
            train_pool,
            test_size=self.val_ratio / (1 - self.test_ratio),
            stratify=stratify_col,
            random_state=self.seed,
        )

        logger.info(
            "Source split (holdout=%s): train=%d, val=%d, test=%d",
            holdout_source, len(train), len(val), len(test),
        )

        return {"train": train, "val": val, "test": test}

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["text"], keep="first")
        df = df.reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            logger.info("Removed %d exact duplicates", removed)
        return df

    def _validate_no_leakage(self, df: pd.DataFrame) -> None:
        texts = df["text"].tolist()
        if len(texts) != len(set(texts)):
            logger.warning("Potential data leakage: duplicate texts found across dataset")

    def _validate_split(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        train_texts = set(train["text"])
        val_texts = set(val["text"])
        test_texts = set(test["text"])

        overlap_tv = train_texts & val_texts
        overlap_tt = train_texts & test_texts
        overlap_vt = val_texts & test_texts

        if overlap_tv:
            logger.warning("LEAKAGE: %d texts shared between train and val", len(overlap_tv))
        if overlap_tt:
            logger.warning("LEAKAGE: %d texts shared between train and test", len(overlap_tt))
        if overlap_vt:
            logger.warning("LEAKAGE: %d texts shared between val and test", len(overlap_vt))

        if not (overlap_tv or overlap_tt or overlap_vt):
            logger.info("No data leakage detected between splits")

    def get_split_statistics(self, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        stats = {}
        for name, df in splits.items():
            label_dist = df["label"].value_counts().to_dict() if "label" in df.columns else {}
            stats[name] = {
                "size": len(df),
                "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
                "avg_text_length": float(df["text"].str.len().mean()) if "text" in df.columns else 0,
            }
        return stats

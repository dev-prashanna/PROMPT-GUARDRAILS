from __future__ import annotations

import logging
import random
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataAugmenter:
    """Apply data augmentation techniques to address class imbalance."""

    def __init__(self, config: dict[str, Any], seed: int = 42):
        aug_cfg = config.get("data", {}).get("augmentation", {})
        self.enabled = aug_cfg.get("enabled", False)
        self.techniques = {
            t["name"]: t for t in aug_cfg.get("techniques", []) if t.get("enabled")
        }
        self.target_ratio = aug_cfg.get("target_minority_ratio", 0.4)
        self.rng = random.Random(seed)

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return df

        label_counts = df["label"].value_counts()
        max_count = label_counts.max()
        augmented_frames = [df]

        for label, count in label_counts.items():
            target_count = int(max_count * self.target_ratio)
            if count >= target_count:
                continue

            needed = target_count - count
            minority_df = df[df["label"] == label]
            synthetics = self._generate_synthetic(minority_df, needed)
            augmented_frames.append(synthetics)
            logger.info(
                "Augmented label %d: %d -> %d (+%d)",
                label, count, target_count, needed,
            )

        result = pd.concat(augmented_frames, ignore_index=True)
        result = result.sample(frac=1, random_state=self.rng).reset_index(drop=True)
        return result

    def _generate_synthetic(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        samples = []
        for _ in range(n):
            original = df.sample(1, random_state=self.rng).iloc[0]["text"]
            augmented_text = self._apply_techniques(original)
            samples.append({"text": augmented_text, "label": df.iloc[0]["label"],
                           "source": "augmented", "split": "train"})
        return pd.DataFrame(samples)

    def _apply_techniques(self, text: str) -> str:
        if "synonym_replacement" in self.techniques:
            text = self._synonym_replacement(text)
        if "random_insertion" in self.techniques:
            text = self._random_insertion(text)
        if "back_translation" in self.techniques:
            text = self._back_translation(text)
        return text

    def _synonym_replacement(self, text: str) -> str:
        cfg = self.techniques.get("synonym_replacement", {})
        prob = cfg.get("probability", 0.1)
        max_replace = cfg.get("max_replacements", 2)

        words = text.split()
        replaced = 0
        for i in range(len(words)):
            if replaced >= max_replace:
                break
            if self.rng.random() < prob:
                words[i] = self._get_synonym(words[i])
                replaced += 1
        return " ".join(words)

    def _random_insertion(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        idx = self.rng.randint(0, len(words))
        word = self._get_synonym(self.rng.choice(words))
        words.insert(idx, word)
        return " ".join(words)

    def _back_translation(self, text: str) -> str:
        return text

    def _get_synonym(self, word: str) -> str:
        synonym_map = {
            "attack": ["assault", "strike", "onslaught"],
            "bypass": ["circumvent", "evade", "override"],
            "ignore": ["disregard", "skip", "overlook"],
            "system": ["platform", "framework", "infrastructure"],
            "prompt": ["input", "query", "instruction"],
            "generate": ["produce", "create", "output"],
            "help": ["assist", "aid", "support"],
            "make": ["create", "build", "construct"],
        }
        lower = word.lower()
        if lower in synonym_map:
            return self.rng.choice(synonym_map[lower])
        return word

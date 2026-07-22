from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


class DataAcquisition:
    """Acquire and unify datasets from multiple Hugging Face sources."""

    LABEL_MAP = {"safe": 0, "prompt_injection": 1, "jailbreak": 2}
    INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self, config: dict[str, Any], output_dir: str = "data/raw"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._raw_datasets: list[pd.DataFrame] = []

    def acquire_all(self) -> pd.DataFrame:
        sources = self.config.get("data", {}).get("sources", [])
        for source in sources:
            logger.info("Acquiring dataset: %s", source["name"])
            df = self._load_source(source)
            self._raw_datasets.append(df)
            logger.info("Loaded %d rows from %s", len(df), source["name"])

        merged = self._merge_and_deduplicate()
        merged.to_parquet(self.output_dir / "raw_merged.parquet", index=False)
        self._save_metadata(merged)
        return merged

    def _load_source(self, source: dict[str, Any]) -> pd.DataFrame:
        name = source["name"]
        subset = source.get("subset")
        split_map = source.get("split_mapping", {"train": "train"})

        frames = []
        for split_label, hf_split in split_map.items():
            try:
                ds = load_dataset(name, subset, split=hf_split, trust_remote_code=True)
                frames.append(self._normalize_to_dataframe(ds, name, split_label))
            except Exception as e:
                logger.warning("Failed to load %s/%s: %s", name, hf_split, e)

        if not frames:
            raise RuntimeError(f"Could not load any splits from {name}")

        return pd.concat(frames, ignore_index=True)

    def _normalize_to_dataframe(
        self, ds: Dataset, source_name: str, split: str
    ) -> pd.DataFrame:
        df = ds.to_pandas()
        df["source"] = source_name
        df["split"] = split

        text_col = self._detect_text_column(df)
        label_col = self._detect_label_column(df)

        if text_col is None:
            raise ValueError(f"No text column found in {source_name}")

        df = df.rename(columns={text_col: "text"})

        if label_col is not None:
            df["label"] = df[label_col].apply(self._map_label)
        else:
            df["label"] = self._infer_labels(df["text"], source_name)

        return df[["text", "label", "source", "split"]].dropna(subset=["text"])

    def _detect_text_column(self, df: pd.DataFrame) -> str | None:
        candidates = ["text", "prompt", "input", "question", "query", "content", "instruction"]
        for col in candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if df[col].dtype == "object" and df[col].str.len().mean() > 20:
                return col
        return None

    def _detect_label_column(self, df: pd.DataFrame) -> str | None:
        candidates = ["label", "category", "type", "class", "unsafe", "harmful"]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _map_label(self, raw_label: Any) -> int:
        if isinstance(raw_label, (int, float)):
            return int(raw_label)
        label_str = str(raw_label).lower().strip()
        if label_str in ("safe", "benign", "0", "harmless"):
            return 0
        if label_str in ("prompt_injection", "injection", "1"):
            return 1
        if label_str in ("jailbreak", "2", "unsafe"):
            return 2
        return 0

    def _infer_labels(self, texts: pd.Series, source_name: str) -> pd.Series:
        injection_patterns = [
            "ignore previous", "disregard instructions", "you are now",
            "new instructions", "override", "system prompt",
            "act as", "pretend you are", "jailbreak", "dan mode",
        ]
        jailbreak_patterns = [
            "do anything now", "hypothetical", "fictional scenario",
            "without restrictions", "no rules", "unrestricted",
            "developer mode", "evil twin",
        ]

        labels = []
        for text in texts:
            text_lower = str(text).lower()
            if any(p in text_lower for p in jailbreak_patterns):
                labels.append(2)
            elif any(p in text_lower for p in injection_patterns):
                labels.append(1)
            else:
                labels.append(0)
        return pd.Series(labels, index=texts.index)

    def _merge_and_deduplicate(self) -> pd.DataFrame:
        merged = pd.concat(self._raw_datasets, ignore_index=True)
        merged = merged.drop_duplicates(subset=["text"], keep="first")
        merged = merged.reset_index(drop=True)
        logger.info("Merged dataset: %d rows after deduplication", len(merged))
        return merged

    def _save_metadata(self, df: pd.DataFrame) -> None:
        meta = {
            "total_samples": len(df),
            "label_distribution": df["label"].value_counts().to_dict(),
            "label_names": self.INVERSE_LABEL_MAP,
            "source_distribution": df["source"].value_counts().to_dict(),
            "avg_text_length": float(df["text"].str.len().mean()),
            "sha256": hashlib.sha256(
                pd.util.hash_pandas_object(df).values.tobytes()
            ).hexdigest(),
        }
        meta_path = self.output_dir / "acquisition_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Metadata saved to %s", meta_path)

    def load_cached(self) -> pd.DataFrame:
        cached = self.output_dir / "raw_merged.parquet"
        if cached.exists():
            return pd.read_parquet(cached)
        return self.acquire_all()

"""Stage 3: Multi-Source Data Aggregation Pipeline for Class Imbalance Mitigation.

This module implements a comprehensive data collection and preprocessing pipeline
that aggregates multiple HuggingFace datasets with proper label mapping, class
balancing, data cleaning, and final dataset construction optimized for SFT/alignment.

Datasets integrated:
    - Anthropic/hh-rlhf (safe) - Preference pairs with chosen/rejected
    - lmsys/toxic-chat (unsafe proxy) - Toxic conversations with annotations
    - Open-Orca/OpenOrca (safe) - Large-scale instruction-following
    - tatsu-lab/alpaca (safe) - Instruction-following dataset
    - Existing synthetic injection/jailbreak data (unsafe)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class Stage3DataPipeline:
    """Comprehensive data pipeline for Stage 3 model training.

    Implements multi-source aggregation with class balancing, cleaning,
    and structured output for SFT/alignment training.
    """

    LABEL_MAP = {"safe": 0, "prompt_injection": 1, "jailbreak": 2}
    INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self, config: dict[str, Any], output_dir: str = "data/stage3"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage3_cfg = config.get("stage3", {})
        self.balancing_cfg = self.stage3_cfg.get("class_balancing", {})
        self.cleaning_cfg = self.stage3_cfg.get("cleaning", {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("data", {}).get("preprocessing", {}).get(
                "tokenizer", "microsoft/deberta-v3-base"
            )
        )

    def run_pipeline(self) -> dict[str, Any]:
        """Execute the full Stage 3 pipeline and return statistics."""
        logger.info("=" * 70)
        logger.info("STAGE 3: Multi-Source Data Aggregation Pipeline")
        logger.info("=" * 70)

        stats = {}

        # Phase 1: Data Aggregation
        logger.info("\n[Phase 1] Data Aggregation from Multiple Sources")
        raw_data = self._aggregate_datasets()
        stats["aggregation"] = {
            "total_samples": len(raw_data),
            "label_distribution": raw_data["label"].value_counts().to_dict(),
            "source_distribution": raw_data["source"].value_counts().to_dict(),
        }
        logger.info("Aggregated %d samples from multiple sources", len(raw_data))

        # Phase 2: Data Cleaning and Refinement
        logger.info("\n[Phase 2] Data Cleaning and Refinement")
        clean_data = self._clean_and_refine(raw_data)
        stats["cleaning"] = {
            "before": len(raw_data),
            "after": len(clean_data),
            "removed": len(raw_data) - len(clean_data),
        }
        logger.info("Cleaning complete: %d -> %d samples", len(raw_data), len(clean_data))

        # Phase 3: Class Balancing
        logger.info("\n[Phase 3] Class Balancing")
        balanced_data = self._balance_classes(clean_data)
        stats["balancing"] = {
            "before": clean_data["label"].value_counts().to_dict(),
            "after": balanced_data["label"].value_counts().to_dict(),
            "strategy": self.balancing_cfg.get("strategy", "hybrid"),
        }
        logger.info("Balancing complete: %d samples", len(balanced_data))

        # Phase 4: Final Dataset Construction
        logger.info("\n[Phase 4] Final Dataset Construction")
        final_splits = self._construct_final_dataset(balanced_data)
        stats["final"] = {
            split: {
                "size": len(df),
                "label_distribution": df["label"].value_counts().to_dict(),
            }
            for split, df in final_splits.items()
        }
        logger.info("Final dataset construction complete")

        # Save metadata
        self._save_metadata(stats)

        return stats

    def _aggregate_datasets(self) -> pd.DataFrame:
        """Aggregate data from multiple HuggingFace sources with label mapping."""
        sources = self.stage3_cfg.get("sources", [])
        frames = []

        for source_cfg in sources:
            name = source_cfg["name"]
            logger.info("Loading dataset: %s", name)

            try:
                df = self._load_and_map_source(source_cfg)
                if df is not None and len(df) > 0:
                    frames.append(df)
                    logger.info(
                        "Loaded %d samples from %s (labels: %s)",
                        len(df),
                        name,
                        df["label"].value_counts().to_dict(),
                    )
            except Exception as e:
                logger.warning("Failed to load %s: %s", name, e)
                continue

        if not frames:
            raise RuntimeError("No datasets were successfully loaded")

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["text"], keep="first")
        merged = merged.reset_index(drop=True)

        logger.info("Merged dataset: %d samples after deduplication", len(merged))
        return merged

    def _load_and_map_source(self, source_cfg: dict[str, Any]) -> pd.DataFrame | None:
        """Load a single dataset and map to unified schema."""
        name = source_cfg["name"]
        source_type = source_cfg.get("type", "huggingface")
        label_strategy = source_cfg.get("label_strategy", "auto")
        max_samples = source_cfg.get("max_samples")
        split = source_cfg.get("split", "train")

        if source_type == "huggingface":
            subset = source_cfg.get("subset")
            if subset:
                ds = load_dataset(name, name=subset, split=split, streaming=True)
            else:
                ds = load_dataset(name, split=split, streaming=True)

            if max_samples:
                samples = []
                for i, sample in enumerate(ds):
                    if i >= max_samples:
                        break
                    samples.append(sample)
                ds = Dataset.from_list(samples)
            else:
                samples = []
                for i, sample in enumerate(ds):
                    samples.append(sample)
                    if i >= 50000:
                        break
                ds = Dataset.from_list(samples)

            df = ds.to_pandas()
        elif source_type == "local":
            path = source_cfg.get("path")
            if path and Path(path).exists():
                df = pd.read_parquet(path)
            else:
                logger.warning("Local path not found: %s", source)
                return None
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        return self._map_to_unified_schema(df, source_cfg, label_strategy)

    def _map_to_unified_schema(
        self, df: pd.DataFrame, source_cfg: dict[str, Any], label_strategy: str
    ) -> pd.DataFrame:
        """Map dataset columns to unified text/label/source schema."""
        name = source_cfg["name"]

        text_col = source_cfg.get("text_column")
        label_col = source_cfg.get("label_column")

        if text_col is None:
            text_col = self._detect_text_column(df)
        if text_col is None:
            raise ValueError(f"No text column found in {name}")

        result = pd.DataFrame()
        result["text"] = df[text_col].astype(str)
        result["source"] = name

        if label_strategy == "fixed":
            fixed_label = source_cfg.get("fixed_label", 0)
            result["label"] = fixed_label
        elif label_strategy == "preserve" and label_col and label_col in df.columns:
            result["label"] = df[label_col].apply(lambda x: int(x) if isinstance(x, (int, float)) else 0)
        elif label_col and label_col in df.columns:
            if label_strategy == "binary":
                result["label"] = df[label_col].apply(
                    lambda x: self._map_to_binary(x, source_cfg)
                )
            elif label_strategy == "toxic_threshold":
                threshold = source_cfg.get("threshold", 0.5)
                result["label"] = df[label_col].apply(
                    lambda x: 0 if float(x) < threshold else 2
                )
            elif label_strategy == "jailbreak_flag":
                result["label"] = df.apply(
                    lambda row: self._map_jailbreak(row, source_cfg), axis=1
                )
            else:
                result["label"] = df[label_col].apply(self._map_label_general)
        else:
            result["label"] = 0  # Default to safe
            if label_strategy == "inference":
                result["label"] = self._infer_safety_labels(result["text"])

        result = result.dropna(subset=["text"])
        result = result[result["text"].str.strip().str.len() > 0]

        return result

    def _detect_text_column(self, df: pd.DataFrame) -> str | None:
        """Auto-detect the primary text column."""
        candidates = [
            "text", "prompt", "input", "question", "query", "content",
            "instruction", "user_input", "chosen", "rejected",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if df[col].dtype == "object" and df[col].str.len().mean() > 20:
                return col
        return None

    def _map_label_general(self, raw_label: Any) -> int:
        """Map raw label to unified integer label."""
        if isinstance(raw_label, (int, float)):
            return int(raw_label)
        label_str = str(raw_label).lower().strip()
        if label_str in ("safe", "benign", "0", "harmless", "clean"):
            return 0
        if label_str in ("prompt_injection", "injection", "1"):
            return 1
        if label_str in ("jailbreak", "2", "unsafe", "toxic", "harmful"):
            return 2
        return 0

    def _map_to_binary(self, raw_label: Any, source_cfg: dict[str, Any]) -> int:
        """Map label to binary safe(0)/unsafe(2) classification."""
        unsafe_values = source_cfg.get("unsafe_values", [1, "toxic", "unsafe", True])
        if isinstance(raw_label, (int, float)):
            return 0 if raw_label == 0 else 2
        return 2 if str(raw_label).lower() in [str(v).lower() for v in unsafe_values] else 0

    def _map_jailbreak(self, row: pd.Series, source_cfg: dict[str, Any]) -> int:
        """Map label for jailbreak-specific datasets."""
        jailbreak_col = source_cfg.get("jailbreak_column", "jailbreaking")
        if jailbreak_col in row.index and row[jailbreak_col]:
            return 2
        return 0

    def _infer_safety_labels(self, texts: pd.Series) -> pd.Series:
        """Infer safety labels from text content using pattern matching."""
        injection_patterns = [
            "ignore previous", "disregard instructions", "you are now",
            "new instructions", "override", "system prompt", "act as",
            "pretend you are", "jailbreak", "dan mode", "do anything now",
            "bypass", "circumvent", "ignore all",
        ]
        jailbreak_patterns = [
            "do anything now", "hypothetical", "fictional scenario",
            "without restrictions", "no rules", "unrestricted",
            "developer mode", "evil twin", "opposite mode",
            "jailbreak", "unlock", "disable filters",
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

    def _clean_and_refine(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive cleaning and quality filtering."""
        logger.info("Starting data cleaning: %d samples", len(df))
        initial_count = len(df)

        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)
        logger.info("After deduplication: %d samples", len(df))

        # Step 2: Clean text content
        df["text"] = df["text"].apply(self._clean_text)
        df = df[df["text"].str.strip().str.len() > 0]
        logger.info("After text cleaning: %d samples", len(df))

        # Step 3: Filter by token length
        df = self._filter_by_token_length(df)
        logger.info("After token length filter: %d samples", len(df))

        # Step 4: Remove low-quality entries
        df = self._filter_low_quality(df)
        logger.info("After quality filter: %d samples", len(df))

        # Step 5: Remove statistical outliers
        df = self._remove_outliers(df)
        logger.info("After outlier removal: %d samples", len(df))

        df = df.reset_index(drop=True)
        logger.info(
            "Cleaning complete: %d -> %d samples (%.1f%% retained)",
            initial_count,
            len(df),
            100 * len(df) / initial_count if initial_count > 0 else 0,
        )
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact and near-duplicate texts."""
        before = len(df)
        df = df.drop_duplicates(subset=["text"], keep="first")

        # Near-duplicate detection using text similarity
        if self.cleaning_cfg.get("near_duplicate_detection", True):
            df = self._remove_near_duplicates(df)

        logger.info("Removed %d duplicates", before - len(df))
        return df

    def _remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-duplicate texts using shingling + MinHash approach."""
        threshold = self.cleaning_cfg.get("near_duplicate_threshold", 0.85)
        texts = df["text"].tolist()

        # Create shingle sets (3-word shingles) for each text
        def get_shingles(text: str, k: int = 3) -> set:
            words = text.lower().split()
            if len(words) < k:
                return set(words)
            return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}

        shingles = [get_shingles(t) for t in texts]
        keep_mask = [True] * len(texts)

        # Sample-based deduplication for efficiency
        sample_size = min(5000, len(texts))
        indices = list(range(len(texts)))
        import random
        random.seed(42)
        random.shuffle(indices)
        sample_indices = indices[:sample_size]

        for idx in sample_indices:
            if not keep_mask[idx]:
                continue
            for j in range(idx + 1, len(texts)):
                if not keep_mask[j]:
                    continue
                if len(shingles[idx]) == 0 or len(shingles[j]) == 0:
                    continue
                jaccard = len(shingles[idx] & shingles[j]) / len(shingles[idx] | shingles[j])
                if jaccard > threshold:
                    keep_mask[j] = False

        removed = sum(1 for m in keep_mask if not m)
        if removed > 0:
            logger.info("Removed %d near-duplicates", removed)
        return df[keep_mask].reset_index(drop=True)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not isinstance(text, str):
            return ""

        text = text.strip()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Replace URLs with placeholder
        text = re.sub(r"http\S+|www\.\S+", "[URL]", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize quotes
        text = re.sub(r"[\u201c\u201d\u2018\u2019]", '"', text)
        text = re.sub(r"[\u2018\u2019]", "'", text)

        return text.strip()

    def _filter_by_token_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter samples by character length (fast proxy for token length)."""
        min_chars = self.cleaning_cfg.get("min_token_length", 5) * 4
        max_chars = self.cleaning_cfg.get("max_token_length", 512) * 4

        lengths = df["text"].str.len()
        mask = lengths.between(min_chars, max_chars)
        removed = (~mask).sum()
        if removed > 0:
            logger.info("Removed %d samples outside length bounds [%d, %d] chars", removed, min_chars, max_chars)
        return df[mask].copy()

    def _filter_low_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out low-quality text entries."""
        min_text_length = self.cleaning_cfg.get("min_text_length", 10)

        # Filter too short
        df = df[df["text"].str.len() >= min_text_length]

        # Filter texts with high special character ratio
        special_ratios = df["text"].apply(
            lambda t: sum(1 for c in t if not c.isalnum() and not c.isspace()) / max(len(t), 1)
        )
        max_special_ratio = self.cleaning_cfg.get("max_special_char_ratio", 0.5)
        df = df[special_ratios <= max_special_ratio]

        # Filter texts that are mostly uppercase (shouting)
        def is_shouting(text: str) -> bool:
            alpha_chars = [c for c in text if c.isalpha()]
            if len(alpha_chars) < 10:
                return False
            return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.8

        df = df[~df["text"].apply(is_shouting)]

        # Filter texts with excessive repetition
        def has_excessive_repetition(text: str) -> bool:
            words = text.split()
            if len(words) < 5:
                return False
            word_set = set(words)
            return len(word_set) / len(words) < 0.3

        df = df[~df["text"].apply(has_excessive_repetition)]

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers based on text length."""
        if len(df) < 100:
            return df

        lengths = df["text"].str.len()
        mean_len = lengths.mean()
        std_len = lengths.std()
        z_threshold = self.cleaning_cfg.get("outlier_z_threshold", 3.0)

        mask = (lengths - mean_len).abs() <= z_threshold * std_len
        removed = (~mask).sum()
        if removed > 0:
            logger.info("Removed %d length outliers", removed)
        return df[mask].copy()

    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply class balancing strategies to mitigate imbalance."""
        strategy = self.balancing_cfg.get("strategy", "hybrid")
        target_ratio = self.balancing_cfg.get("target_ratio", 0.3)

        label_counts = df["label"].value_counts()
        logger.info("Class distribution before balancing: %s", label_counts.to_dict())

        if strategy == "oversampling":
            df = self._oversample_minority(df, target_ratio)
        elif strategy == "undersampling":
            df = self._undersample_majority(df, target_ratio)
        elif strategy == "hybrid":
            df = self._hybrid_balancing(df, target_ratio)
        elif strategy == "smote":
            df = self._smote_balancing(df)
        elif strategy == "focal_weighted":
            # No resampling, just compute weights (applied during training)
            logger.info("Using focal/weighted loss (no resampling)")
        else:
            logger.warning("Unknown balancing strategy: %s, using hybrid", strategy)
            df = self._hybrid_balancing(df, target_ratio)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(
            "Class distribution after balancing: %s",
            df["label"].value_counts().to_dict(),
        )
        return df

    def _oversample_minority(
        self, df: pd.DataFrame, target_ratio: float
    ) -> pd.DataFrame:
        """Oversample minority classes to target ratio of majority class."""
        label_counts = df["label"].value_counts()
        max_count = label_counts.max()
        target_count = int(max_count * target_ratio)

        frames = [df]
        for label, count in label_counts.items():
            if count >= target_count:
                continue
            needed = target_count - count
            minority = df[df["label"] == label]
            oversampled = minority.sample(
                n=needed, replace=True, random_state=42
            )
            oversampled["source"] = oversampled["source"] + "_oversampled"
            frames.append(oversampled)
            logger.info(
                "Oversampled label %d: %d -> %d (+%d)", label, count, target_count, needed
            )

        return pd.concat(frames, ignore_index=True)

    def _undersample_majority(
        self, df: pd.DataFrame, target_ratio: float
    ) -> pd.DataFrame:
        """Undersample majority class to target ratio of minority class."""
        label_counts = df["label"].value_counts()
        min_count = label_counts.min()
        target_count = int(min_count / target_ratio)

        frames = []
        for label, count in label_counts.items():
            subset = df[df["label"] == label]
            if count > target_count:
                subset = subset.sample(n=target_count, random_state=42)
                logger.info(
                    "Undersampled label %d: %d -> %d", label, count, target_count
                )
            frames.append(subset)

        return pd.concat(frames, ignore_index=True)

    def _hybrid_balancing(self, df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
        """Hybrid approach: oversample minority + undersample majority."""
        label_counts = df["label"].value_counts()
        max_count = label_counts.max()
        min_count = label_counts.min()

        # Target: majority class downsized, minority class upsized
        target_max = int(max_count * 0.3)  # Cap majority at 30% of original
        target_min = int(max_count * target_ratio)  # Minority at target_ratio of original max

        frames = []
        for label, count in label_counts.items():
            subset = df[df["label"] == label]
            if count < target_min:
                needed = target_min - count
                oversampled = subset.sample(n=needed, replace=True, random_state=42)
                oversampled["source"] = oversampled["source"] + "_oversampled"
                frames.append(subset)
                frames.append(oversampled)
                logger.info(
                    "Hybrid: Oversampled label %d: %d -> %d", label, count, target_min
                )
            elif count > target_max:
                subset = subset.sample(n=target_max, random_state=42)
                frames.append(subset)
                logger.info(
                    "Hybrid: Undersampled label %d: %d -> %d", label, count, target_max
                )
            else:
                frames.append(subset)

        return pd.concat(frames, ignore_index=True)

    def _smote_balancing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMOTE-like synthetic sampling for text data."""
        try:
            from imblearn.over_sampling import SMOTE
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            X = vectorizer.fit_transform(df["text"]).toarray()
            y = df["label"].values

            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            inverse_texts = vectorizer.inverse_transform(X_resampled[len(df):])
            synthetic_texts = [" ".join(t) for t in inverse_texts]

            synthetic_df = pd.DataFrame({
                "text": synthetic_texts,
                "label": y_resampled[len(df):],
                "source": "smote_synthetic",
            })

            return pd.concat([df, synthetic_df], ignore_index=True)
        except ImportError:
            logger.warning("imblearn not available, falling back to oversampling")
            return self._oversample_minority(df, 0.3)

    def _construct_final_dataset(
        self, df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Construct final train/val/test splits with stratification."""
        from sklearn.model_selection import train_test_split

        split_cfg = self.stage3_cfg.get("splits", {})
        test_ratio = split_cfg.get("test_ratio", 0.1)
        val_ratio = split_cfg.get("val_ratio", 0.1)
        seed = split_cfg.get("random_seed", 42)

        # Stratified split
        stratify_col = df["label"] if "label" in df.columns else None

        train_val, test = train_test_split(
            df, test_size=test_ratio, stratify=stratify_col, random_state=seed
        )

        relative_val = val_ratio / (1 - test_ratio)
        stratify_tv = train_val["label"] if "label" in train_val.columns else None
        train, val = train_test_split(
            train_val, test_size=relative_val, stratify=stratify_tv, random_state=seed
        )

        # Validate no leakage
        self._validate_splits(train, val, test)

        # Save splits
        output_dir = self.output_dir / "splits"
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            split_path = output_dir / f"{split_name}.parquet"
            split_df.to_parquet(split_path, index=False)
            logger.info("Saved %s split: %d samples -> %s", split_name, len(split_df), split_path)

        # Also save SFT-formatted dataset
        self._save_sft_format(train, val, test)

        return {"train": train, "val": val, "test": test}

    def _validate_splits(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        """Validate no data leakage between splits."""
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

    def _save_sft_format(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Save dataset in SFT/alignment training format."""
        sft_dir = self.output_dir / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df in [("train", train), ("val", val), ("test", test)]:
            sft_records = []
            for _, row in df.iterrows():
                record = {
                    "instruction": "Classify the following text as safe, prompt injection, or jailbreak attempt.",
                    "input": row["text"],
                    "output": self.INVERSE_LABEL_MAP.get(row["label"], "safe"),
                    "label": row["label"],
                    "source": row.get("source", "unknown"),
                }
                sft_records.append(record)

            sft_df = pd.DataFrame(sft_records)
            sft_path = sft_dir / f"{split_name}_sft.parquet"
            sft_df.to_parquet(sft_path, index=False)
            logger.info("Saved SFT format: %s -> %s", split_name, sft_path)

    def _save_metadata(self, stats: dict[str, Any]) -> None:
        """Save pipeline metadata and statistics."""
        metadata = {
            "pipeline": "stage3_multi_source_aggregation",
            "stats": stats,
            "config": self.stage3_cfg,
            "label_map": self.LABEL_MAP,
            "inverse_label_map": self.INVERSE_LABEL_MAP,
            "sha256": hashlib.sha256(
                json.dumps(stats, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }

        metadata_path = self.output_dir / "pipeline_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata saved to %s", metadata_path)


def get_default_stage3_config() -> dict[str, Any]:
    """Return default Stage 3 configuration."""
    return {
        "stage3": {
            "sources": [
                {
                    "name": "Anthropic/hh-rlhf",
                    "type": "huggingface",
                    "split": "train",
                    "text_column": "chosen",
                    "label_strategy": "inference",
                    "max_samples": 15000,
                },
                {
                    "name": "lmsys/toxic-chat",
                    "type": "huggingface",
                    "split": "train",
                    "text_column": "user_input",
                    "label_strategy": "toxic_threshold",
                    "threshold": 0.5,
                    "max_samples": 15000,
                },
                {
                    "name": "Open-Orca/OpenOrca",
                    "type": "huggingface",
                    "split": "train",
                    "text_column": "question",
                    "label_strategy": "fixed",
                    "fixed_label": 0,
                    "max_samples": 15000,
                },
                {
                    "name": "tatsu-lab/alpaca",
                    "type": "huggingface",
                    "split": "train",
                    "text_column": "instruction",
                    "label_strategy": "fixed",
                    "fixed_label": 0,
                    "max_samples": 15000,
                },
                {
                    "name": "local_injection_data",
                    "type": "local",
                    "path": "data/raw/raw_merged.parquet",
                    "text_column": "text",
                    "label_strategy": "preserve",
                },
            ],
            "class_balancing": {
                "strategy": "hybrid",
                "target_ratio": 0.3,
                "target_minority_ratio": 0.3,
                "oversample_factor": 3,
            },
            "cleaning": {
                "min_token_length": 5,
                "max_token_length": 512,
                "min_text_length": 10,
                "max_special_char_ratio": 0.5,
                "near_duplicate_detection": True,
                "near_duplicate_threshold": 0.85,
                "outlier_z_threshold": 3.0,
            },
            "splits": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "random_seed": 42,
            },
        }
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.utils.logging import setup_logging

    setup_logging()
    config = get_default_stage3_config()
    pipeline = Stage3DataPipeline(config)
    stats = pipeline.run_pipeline()
    print(json.dumps(stats, indent=2, default=str))

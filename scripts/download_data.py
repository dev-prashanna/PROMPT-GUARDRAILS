#!/usr/bin/env python3
"""Download and preprocess all datasets for training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acquisition import DataAcquisition
from src.data.preprocessing import TextPreprocessor
from src.data.augmentation import DataAugmenter
from src.utils.io import load_multi_config
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")

    print("=" * 60)
    print("STAGE 1: Data Acquisition")
    print("=" * 60)
    acquirer = DataAcquisition(config, output_dir="data/raw")
    raw_df = acquirer.acquire_all()
    print(f"Acquired {len(raw_df)} raw samples")

    print("\n" + "=" * 60)
    print("STAGE 2: Preprocessing")
    print("=" * 60)
    preprocessor = TextPreprocessor(config)
    clean_df = preprocessor.preprocess(raw_df)
    print(f"Preprocessed: {len(clean_df)} samples")

    print("\n" + "=" * 60)
    print("STAGE 3: Augmentation")
    print("=" * 60)
    augmenter = DataAugmenter(config)
    aug_df = augmenter.augment(clean_df)
    print(f"After augmentation: {len(aug_df)} samples")

    print("\n" + "=" * 60)
    print("STAGE 4: Train/Val/Test Split")
    print("=" * 60)
    from sklearn.model_selection import train_test_split

    split_cfg = config.get("data", {}).get("splits", {})
    test_size = split_cfg.get("test_ratio", 0.1)
    val_size = split_cfg.get("val_ratio", 0.1)
    seed = split_cfg.get("random_seed", 42)

    train_val, test = train_test_split(
        aug_df, test_size=test_size, stratify=aug_df["label"], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=seed
    )

    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(split_dir / "train.parquet", index=False)
    val.to_parquet(split_dir / "val.parquet", index=False)
    test.to_parquet(split_dir / "test.parquet", index=False)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"\nLabel distribution (train):")
    print(train["label"].value_counts().sort_index().to_string())
    print("\nData pipeline complete. Splits saved to data/splits/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the DeBERTa-v3 firewall classifier with LoRA PEFT."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src.models.deberta_classifier import DeBERTaFirewallClassifier
from src.models.lora_config import LoRAConfigurator
from src.training.trainer import FirewallTrainer
from src.utils.io import load_multi_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")
    set_seed(config.get("training", {}).get("seed", 42))

    print("=" * 60)
    print("STAGE 1: Load Data Splits")
    print("=" * 60)
    train_df = pd.read_parquet("data/splits/train.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    print("\n" + "=" * 60)
    print("STAGE 2: Initialize Model")
    print("=" * 60)
    classifier = DeBERTaFirewallClassifier(config)
    print(f"Base model parameters: {classifier.total_parameters:,}")
    print(f"Trainable parameters: {classifier.trainable_parameters:,}")

    print("\n" + "=" * 60)
    print("STAGE 3: Apply LoRA PEFT")
    print("=" * 60)
    lora_configurator = LoRAConfigurator(config)
    model = lora_configurator.apply(classifier)

    print("\n" + "=" * 60)
    print("STAGE 4: Compute Class Weights")
    print("=" * 60)
    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    num_classes = len(label_counts)
    weights = total / (num_classes * label_counts.values)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {class_weights.tolist()}")

    print("\n" + "=" * 60)
    print("STAGE 5: Train")
    print("=" * 60)
    trainer = FirewallTrainer(config)
    output_dir = config.get("training", {}).get("output_dir", "checkpoints")
    trainer.train(model, train_df, val_df, class_weights, output_dir=output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

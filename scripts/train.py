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


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint directory for resume."""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if checkpoints:
        return str(checkpoints[-1])
    return None


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")
    set_seed(config.get("training", {}).get("seed", 42))

    output_dir = config.get("training", {}).get("output_dir", "checkpoints")

    resume_checkpoint = find_latest_checkpoint(output_dir)

    print("=" * 60)
    print("STAGE 1: Load Data Splits")
    print("=" * 60)
    train_df = pd.read_parquet("data/splits/train.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"Label distribution:\n{train_df['label'].value_counts().sort_index()}")

    print("\n" + "=" * 60)
    print("STAGE 2: Initialize Model")
    print("=" * 60)
    classifier = DeBERTaFirewallClassifier(config)
    print(f"Base model parameters: {classifier.total_parameters:,}")
    print(f"Trainable parameters: {classifier.trainable_parameters:,}")

    grad_ckpt_cfg = config.get("training", {}).get("gradient_checkpointing", {})
    if grad_ckpt_cfg.get("enabled", True):
        classifier.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": grad_ckpt_cfg.get("use_reentrant", False)}
        )
        print("Gradient checkpointing enabled on base model")

    print("\n" + "=" * 60)
    print("STAGE 3: Apply LoRA PEFT")
    print("=" * 60)
    lora_configurator = LoRAConfigurator(config)
    model = lora_configurator.apply(classifier)
    print(f"Model dtype: {next(model.parameters()).dtype}")

    print("\n" + "=" * 60)
    print("STAGE 4: Compute Class Weights")
    print("=" * 60)
    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    num_classes = len(label_counts)

    raw_weights = total / (num_classes * label_counts.values)
    max_ratio = raw_weights.max() / raw_weights.min()
    print(f"Raw inverse-frequency weights: {raw_weights.tolist()}")
    print(f"Max/Min weight ratio: {max_ratio:.1f}x")

    if max_ratio > 10.0:
        capped = torch.clamp(
            torch.tensor(raw_weights, dtype=torch.float32),
            max=raw_weights.min() * 10.0,
        )
        print(f"Capped weights (10x): {capped.tolist()}")
        class_weights = capped
    else:
        class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    print(f"Final class weights: {class_weights.tolist()}")

    balancing_cfg = config.get("data", {}).get("class_balancing", {})
    use_focal = balancing_cfg.get("use_smote", False)
    print(f"Loss function: {'Focal Loss' if use_focal else 'Weighted CE + Label Smoothing'}")

    print("\n" + "=" * 60)
    print("STAGE 5: Train")
    print("=" * 60)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer = FirewallTrainer(config)
    trainer.train(
        model, train_df, val_df, class_weights,
        output_dir=output_dir,
        resume_from_checkpoint=resume_checkpoint,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

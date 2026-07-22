#!/usr/bin/env python3
"""Quick training script for DeBERTa-v3 firewall - works on CPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from src.utils.io import load_multi_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.evaluation.metrics import compute_metrics

NUM_LABELS = 3
LABEL_NAMES = {0: "safe", 1: "prompt_injection", 2: "jailbreak"}


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, count = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            loss = CrossEntropyLoss()(logits, labels)
            total_loss += loss.item() * len(labels)
            count += len(labels)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics((logits, labels))
    metrics["eval_loss"] = total_loss / count if count > 0 else 0
    return metrics


def main():
    setup_logging()
    config = load_multi_config("configs")
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=NUM_LABELS, torch_dtype=torch.float32,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    train_df = pd.read_parquet("data/splits/train.parquet")
    val_df = pd.read_parquet("data/splits/val.parquet")
    test_df = pd.read_parquet("data/splits/test.parquet")

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Label dist (train): {train_df['label'].value_counts().sort_index().to_dict()}")

    train_dataset = SimpleDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length=256)
    val_dataset = SimpleDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length=256)
    test_dataset = SimpleDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length=256)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    epochs = 10
    lr = 2e-5
    weight_decay = 0.01
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)

    label_counts = train_df["label"].value_counts().sort_index().values
    class_weights = torch.tensor(
        [len(train_df) / (NUM_LABELS * c) for c in label_counts], dtype=torch.float32
    ).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    loss_fn = CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, lr={lr}, warmup={warmup_steps}")
    print("=" * 60)

    best_f1 = 0
    results_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        val_metrics = evaluate(model, val_loader, device)

        marker = ""
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save_pretrained("checkpoints/best")
            tokenizer.save_pretrained("checkpoints/best")
            marker = " *BEST*"

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["eval_loss"],
            "macro_f1": val_metrics["macro_f1"],
            "accuracy": val_metrics["accuracy"],
        }
        results_log.append(log_entry)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['eval_loss']:.4f} | "
            f"Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Accuracy: {val_metrics['accuracy']:.4f}{marker}"
        )

    print("=" * 60)
    print(f"\nTraining complete. Best Macro F1: {best_f1:.4f}")

    print("\nRunning final evaluation on test set...")
    test_loader = DataLoader(test_dataset, batch_size=16)

    best_model = AutoModelForSequenceClassification.from_pretrained("checkpoints/best", torch_dtype=torch.float32).to(device)
    test_metrics = evaluate(best_model, test_loader, device)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    for key in ["macro_f1", "weighted_f1", "accuracy", "eval_loss"]:
        if key in test_metrics:
            print(f"  {key}: {test_metrics[key]:.4f}")

    per_class = ["safe", "prompt_injection", "jailbreak"]
    for cls in per_class:
        p = test_metrics.get(f"{cls}_precision", 0)
        r = test_metrics.get(f"{cls}_recall", 0)
        f = test_metrics.get(f"{cls}_f1", 0)
        print(f"  {cls:20s}: P={p:.4f} R={r:.4f} F1={f:.4f}")

    print("\nConfusion matrix:")
    for true_cls in per_class:
        row = []
        for pred_cls in per_class:
            key = f"cm_{true_cls}_vs_{pred_cls}"
            row.append(str(test_metrics.get(key, 0)))
        print(f"  {true_cls:20s}: {' '.join(row)}")

    final_results = {
        "training_log": results_log,
        "test_metrics": {k: v for k, v in test_metrics.items() if not isinstance(v, (list, dict))},
        "config": {
            "model": "microsoft/deberta-v3-base",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_length": 256,
            "device": device,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
    }

    with open("experiments/stage1_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to experiments/stage1_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run ablation studies for Stage 1 experiments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.acquisition import DataAcquisition
from src.data.preprocessing import TextPreprocessor
from src.data.augmentation import DataAugmenter
from src.data.splitter import StratifiedSplitter
from src.models.deberta_classifier import DeBERTaFirewallClassifier
from src.models.lora_config import LoRAConfigurator
from src.training.trainer import FirewallTrainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.ablation import AblationStudy
from src.utils.io import load_multi_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def run_ablation_variant(
    config: dict[str, Any],
    ablation_type: str,
    variant: dict[str, Any],
    base_data: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Run a single ablation variant and return metrics."""
    variant_config = json.loads(json.dumps(config))

    if ablation_type == "architecture" and "model" in variant:
        variant_config["model"]["base_model"] = variant["model"]

    if ablation_type == "peft_rank" and "rank" in variant:
        variant_config["model"]["peft"]["rank"] = variant["rank"]

    if ablation_type == "max_length" and "max_length" in variant:
        variant_config["data"]["preprocessing"]["max_length"] = variant["max_length"]
        variant_config["model"]["tokenizer"]["max_length"] = variant["max_length"]

    set_seed(42)

    classifier = DeBERTaFirewallClassifier(variant_config)
    lora_configurator = LoRAConfigurator(variant_config)
    model = lora_configurator.apply(classifier)

    train_df = base_data["train"]
    val_df = base_data["val"]
    test_df = base_data["test"]

    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    num_classes = len(label_counts)
    weights = total / (num_classes * label_counts.values)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    trainer = FirewallTrainer(variant_config)
    trainer.train(
        model, train_df, val_df, class_weights,
        output_dir=f"checkpoints/ablation_{ablation_type}_{variant['name']}",
    )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = f"checkpoints/ablation_{ablation_type}_{variant['name']}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    for _, row in test_df.iterrows():
        inputs = tokenizer(
            row["text"], max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu())
        all_labels.append(torch.tensor([row["label"]]))

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    return compute_metrics((logits, labels))


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")
    set_seed(42)

    print("=" * 60)
    print("ABLATION STUDY RUNNER")
    print("=" * 60)

    print("\nPreparing base data...")
    acquirer = DataAcquisition(config, output_dir="data/raw")
    raw_df = acquirer.acquire_all()

    preprocessor = TextPreprocessor(config)
    clean_df = preprocessor.preprocess(raw_df)

    augmenter = DataAugmenter(config)
    aug_df = augmenter.augment(clean_df)

    splitter = StratifiedSplitter(config)
    splits = splitter.split(aug_df)

    base_data = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }

    ablation_study = AblationStudy(config)
    ablations = ablation_study.define_ablations()

    all_results = {}

    for ablation_type, ablation_config in ablations.items():
        print(f"\n{'='*60}")
        print(f"ABLATION: {ablation_type}")
        print(f"Description: {ablation_config['description']}")
        print(f"{'='*60}")

        variant_results = {}
        for variant in ablation_config["variants"]:
            print(f"\n  Running variant: {variant['name']}")
            try:
                metrics = run_ablation_variant(
                    config, ablation_type, variant, base_data
                )
                variant_results[variant["name"]] = metrics
                print(f"    Macro F1: {metrics['macro_f1']:.4f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                variant_results[variant["name"]] = {"error": str(e)}

        analysis = ablation_study.analyze_results(ablation_type, variant_results)
        all_results[ablation_type] = analysis

        ablation_study.generate_latex(
            analysis,
            f"paper/tables/ablations/{ablation_type}_ablation.tex",
        )

    with open("experiments/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print(f"Results saved to: experiments/ablation_results.json")
    print(f"LaTeX tables saved to: paper/tables/ablations/")


if __name__ == "__main__":
    main()

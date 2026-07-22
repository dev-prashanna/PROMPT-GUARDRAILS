#!/usr/bin/env python3
"""Evaluate the trained model and generate visualizations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.dataset import FirewallDataset
from src.evaluation.metrics import compute_metrics, get_full_report
from src.evaluation.visualizations import EvaluationVisualizer
from src.evaluation.analysis import EvaluationAnalyzer
from src.utils.io import load_multi_config, save_json
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")

    model_path = config.get("training", {}).get("output_dir", "checkpoints")
    tokenizer_name = config.get("model", {}).get("tokenizer", {}).get(
        "name", "microsoft/deberta-v3-base"
    )
    max_length = config.get("data", {}).get("preprocessing", {}).get("max_length", 512)

    print("=" * 60)
    print("Loading Model")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("=" * 60)
    print("Loading Test Data")
    print("=" * 60)
    test_df = pd.read_parquet("data/splits/test.parquet")
    test_dataset = FirewallDataset.from_dataframe(test_df, tokenizer, max_length)

    print("=" * 60)
    print("Running Inference")
    print("=" * 60)
    all_logits = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    predictions = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    print("=" * 60)
    print("Computing Metrics")
    print("=" * 60)
    metrics = compute_metrics((logits, labels))
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    report = get_full_report((logits, labels))
    save_json({"metrics": metrics, "report": report}, "paper/tables/metrics.json")

    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    visualizer = EvaluationVisualizer(output_dir="paper/figures")
    vis_paths = visualizer.generate_all_report_figures(labels, predictions, probs, [], [])

    print("=" * 60)
    print("Running Analysis")
    print("=" * 60)
    analyzer = EvaluationAnalyzer(output_dir="paper/tables")
    analysis = analyzer.full_analysis(labels, predictions, probs)
    analyzer.generate_latex_table(report)

    print("\nEvaluation complete!")
    print("Figures saved to: paper/figures/")
    print("Tables saved to: paper/tables/")


if __name__ == "__main__":
    main()

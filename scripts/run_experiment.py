#!/usr/bin/env python3
"""Run Stage 1 experiments with full observability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.acquisition import DataAcquisition
from src.data.preprocessing import TextPreprocessor
from src.data.augmentation import DataAugmenter
from src.data.splitter import StratifiedSplitter
from src.data.quality import QualityChecker
from src.data.adversarial_sets import AdversarialTestSet
from src.models.deberta_classifier import DeBERTaFirewallClassifier
from src.models.lora_config import LoRAConfigurator
from src.training.trainer import FirewallTrainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistical import StatisticalAnalyzer
from src.evaluation.adversarial import AdversarialEvaluator
from src.observability.experiment_tracker import ExperimentTracker
from src.observability.data_tracker import DataTracker
from src.observability.decision_logger import DecisionLogger
from src.utils.io import load_multi_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 1: Data preparation pipeline."""
    print("\n" + "=" * 60)
    print("STAGE 1: DATA PREPARATION")
    print("=" * 60)

    data_tracker = DataTracker("data/versions")

    acquirer = DataAcquisition(config, output_dir="data/raw")
    raw_df = acquirer.acquire_all()
    print(f"  Acquired {len(raw_df)} raw samples")

    preprocessor = TextPreprocessor(config)
    clean_df = preprocessor.preprocess(raw_df)
    print(f"  Preprocessed: {len(clean_df)} samples")

    augmenter = DataAugmenter(config)
    aug_df = augmenter.augment(clean_df)
    print(f"  After augmentation: {len(aug_df)} samples")

    splitter = StratifiedSplitter(config)
    splits = splitter.split(aug_df)
    for name, df in splits.items():
        print(f"  {name}: {len(df)} samples")

    quality_checker = QualityChecker(config)
    quality_report = quality_checker.generate_report(
        aug_df, "data/metadata/quality_report.json"
    )
    print(f"  Quality score: {quality_report['summary']['quality_score']:.3f}")

    version = config.get("data", {}).get("version", "v1.0")
    version_manifest = data_tracker.register_version(
        version=version,
        config=config,
        raw_data=raw_df,
        processed_data=clean_df,
        splits=splits,
        quality_scores=quality_report.get("checks", {}),
    )

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "aug_df": aug_df,
        "splits": splits,
        "quality_report": quality_report,
        "version_manifest": version_manifest,
    }


def run_single_experiment(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    experiment_name: str,
    seed: int,
    experiment_tracker: ExperimentTracker,
    decision_logger: DecisionLogger,
) -> dict[str, Any]:
    """Run a single experiment with full tracking."""
    set_seed(seed)

    experiment_id = experiment_tracker.create_experiment(
        name=experiment_name,
        config=config,
        description=f"Seed={seed}",
    )

    print(f"\n  Running experiment: {experiment_id}")

    classifier = DeBERTaFirewallClassifier(config)
    lora_configurator = LoRAConfigurator(config)
    model = lora_configurator.apply(classifier)

    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    num_classes = len(label_counts)
    weights = total / (num_classes * label_counts.values)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    trainer = FirewallTrainer(config)
    start_time = time.time()
    trainer.train(
        model, train_df, val_df, class_weights,
        output_dir=f"checkpoints/{experiment_id}",
    )
    training_time = time.time() - start_time

    experiment_tracker.log_metric(experiment_id, "training_time_seconds", training_time, 0)

    return {"experiment_id": experiment_id, "training_time": training_time}


def run_evaluation(
    experiment_id: str,
    model_path: str,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    experiment_tracker: ExperimentTracker,
    decision_logger: DecisionLogger,
) -> dict[str, Any]:
    """Run evaluation with full metrics and statistical analysis."""
    print(f"\n  Evaluating experiment: {experiment_id}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    all_texts = []
    latencies = []

    for _, row in test_df.iterrows():
        inputs = tokenizer(
            row["text"], max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        latency = (time.perf_counter() - start) * 1000

        all_logits.append(outputs.logits.cpu())
        all_labels.append(torch.tensor([row["label"]]))
        all_texts.append(row["text"])
        latencies.append(latency)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(logits, axis=-1)

    metrics = compute_metrics((logits, labels))
    metrics["avg_latency_ms"] = float(np.mean(latencies))
    metrics["p95_latency_ms"] = float(np.percentile(latencies, 95))
    metrics["p99_latency_ms"] = float(np.percentile(latencies, 99))

    experiment_tracker.log_metrics_batch(
        experiment_id, metrics, step=0, split="test"
    )

    for i in range(len(all_texts)):
        decision_logger.log_prediction(
            experiment_id=experiment_id,
            sample_id=f"{experiment_id}_{i}",
            text=all_texts[i],
            true_label=int(labels[i]),
            predicted_label=int(predictions[i]),
            confidence=float(probs[i][predictions[i]]),
            probabilities={
                "safe": float(probs[i][0]),
                "prompt_injection": float(probs[i][1]),
                "jailbreak": float(probs[i][2]),
            },
            latency_ms=latencies[i],
        )

    return {
        "metrics": metrics,
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
        "probs": probs,
    }


def run_statistical_analysis(
    all_run_metrics: list[dict[str, float]],
    statistical_analyzer: StatisticalAnalyzer,
) -> dict[str, Any]:
    """Run statistical analysis across multiple runs."""
    print("\n  Running statistical analysis")

    macro_f1_scores = np.array([m["macro_f1"] for m in all_run_metrics])
    accuracy_scores = np.array([m["accuracy"] for m in all_run_metrics])

    f1_ci = statistical_analyzer.bootstrap_ci(macro_f1_scores)
    acc_ci = statistical_analyzer.bootstrap_ci(accuracy_scores)

    return {
        "macro_f1": {
            "mean": f1_ci["mean"],
            "std": float(macro_f1_scores.std()),
            "ci_95": [f1_ci["ci_lower"], f1_ci["ci_upper"]],
        },
        "accuracy": {
            "mean": acc_ci["mean"],
            "std": float(accuracy_scores.std()),
            "ci_95": [acc_ci["ci_lower"], acc_ci["ci_upper"]],
        },
    }


def run_adversarial_evaluation(
    model_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run adversarial robustness evaluation."""
    print("\n  Running adversarial evaluation")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def predict(text: str) -> dict[str, Any]:
        inputs = tokenizer(
            text, max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        return {"label_id": pred_id, "confidence": float(probs[pred_id])}

    adversarial_set = AdversarialTestSet(config)
    test_df = adversarial_set.get_comprehensive_test_set()

    evaluator = AdversarialEvaluator(config)
    results = evaluator.evaluate(predict, test_df)

    evaluator.generate_report(results, "paper/tables/adversarial_report.md")

    return results


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")
    seed = config.get("experiment", {}).get("seed", 42)
    num_runs = config.get("experiment", {}).get("num_runs", 1)
    set_seed(seed)

    experiment_tracker = ExperimentTracker("experiments/tracker.db")
    decision_logger = DecisionLogger("experiments/decisions")
    statistical_analyzer = StatisticalAnalyzer(config)

    data_result = prepare_data(config)
    splits = data_result["splits"]

    all_run_metrics = []
    experiment_ids = []

    for run_idx in range(num_runs):
        run_seed = config.get("experiment", {}).get("seeds", [42])[run_idx % 5]
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{num_runs} (seed={run_seed})")
        print(f"{'='*60}")

        result = run_single_experiment(
            config=config,
            train_df=splits["train"],
            val_df=splits["val"],
            experiment_name=f"stage1_baseline_run{run_idx}",
            seed=run_seed,
            experiment_tracker=experiment_tracker,
            decision_logger=decision_logger,
        )
        experiment_ids.append(result["experiment_id"])

        eval_result = run_evaluation(
            experiment_id=result["experiment_id"],
            model_path=f"checkpoints/{result['experiment_id']}",
            test_df=splits["test"],
            config=config,
            experiment_tracker=experiment_tracker,
            decision_logger=decision_logger,
        )
        all_run_metrics.append(eval_result["metrics"])

    statistical_results = run_statistical_analysis(
        all_run_metrics, statistical_analyzer
    )

    adversarial_results = run_adversarial_evaluation(
        model_path=f"checkpoints/{experiment_ids[0]}",
        config=config,
    )

    final_results = {
        "experiment_ids": experiment_ids,
        "num_runs": num_runs,
        "statistical_analysis": statistical_results,
        "adversarial_evaluation": adversarial_results,
        "individual_run_metrics": all_run_metrics,
    }

    with open("experiments/stage1_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print(f"Experiments: {len(experiment_ids)}")
    print(f"Macro F1: {statistical_results['macro_f1']['mean']:.4f} "
          f"± {statistical_results['macro_f1']['std']:.4f}")
    print(f"95% CI: [{statistical_results['macro_f1']['ci_95'][0]:.4f}, "
          f"{statistical_results['macro_f1']['ci_95'][1]:.4f}]")
    print(f"Adversarial accuracy: {adversarial_results['overall']['accuracy']:.4f}")
    print(f"\nResults saved to: experiments/stage1_results.json")

    experiment_tracker.close()
    decision_logger.flush()


if __name__ == "__main__":
    main()

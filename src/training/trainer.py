from __future__ import annotations

import logging
from typing import Any

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import FirewallDataset
from src.evaluation.metrics import compute_metrics
from src.training.callbacks import FirewalCallbacks

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Trainer with weighted cross-entropy for class imbalance."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = CrossEntropyLoss(weight=weight)
        else:
            loss_fn = CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class FirewallTrainer:
    """Orchestrate the full training pipeline with PEFT and weighted loss."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        train_cfg = config.get("training", {})
        self.hyperparams = train_cfg.get("hyperparameters", {})
        self.loop_cfg = train_cfg.get("training_loop", {})
        self.checkpoint_cfg = train_cfg.get("checkpointing", {})
        self.logging_cfg = train_cfg.get("logging", {})
        self.grad_ckpt_cfg = train_cfg.get("gradient_checkpointing", {})
        self.seed = train_cfg.get("seed", 42)

        tokenizer_name = config.get("model", {}).get(
            "tokenizer", {}
        ).get("name", "microsoft/deberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = config.get("data", {}).get("preprocessing", {}).get(
            "max_length", 512
        )

    def train(
        self,
        model,
        train_df,
        val_df,
        class_weights: torch.Tensor | None = None,
        output_dir: str = "checkpoints",
    ) -> Any:
        logger.info("Preparing datasets")
        train_dataset = FirewallDataset.from_dataframe(
            train_df, self.tokenizer, self.max_length
        )
        val_dataset = FirewallDataset.from_dataframe(
            val_df, self.tokenizer, self.max_length
        )

        if class_weights is None:
            class_weights = train_dataset.compute_class_weights()
            logger.info("Computed class weights: %s", class_weights.tolist())

        training_args = self._build_training_args(output_dir)
        callbacks = FirewalCallbacks.from_config(self.config)

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks.get_callbacks(),
        )

        logger.info("Starting training")
        trainer.train()

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Training complete. Model saved to %s", output_dir)

        return trainer

    def _build_training_args(self, output_dir: str) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            seed=self.seed,
            num_train_epochs=self.loop_cfg.get("num_train_epochs", 10),
            per_device_train_batch_size=self.loop_cfg.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=self.loop_cfg.get("per_device_eval_batch_size", 32),
            gradient_accumulation_steps=self.loop_cfg.get("gradient_accumulation_steps", 4),
            fp16=self.loop_cfg.get("fp16", False) and torch.cuda.is_available(),
            bf16=self.loop_cfg.get("bf16", True) and torch.cuda.is_available(),
            learning_rate=self.hyperparams.get("learning_rate", 2e-5),
            weight_decay=self.hyperparams.get("weight_decay", 0.01),
            adam_beta1=self.hyperparams.get("adam_beta1", 0.9),
            adam_beta2=self.hyperparams.get("adam_beta2", 0.999),
            adam_epsilon=self.hyperparams.get("adam_epsilon", 1e-8),
            max_grad_norm=self.hyperparams.get("max_grad_norm", 1.0),
            warmup_steps=int(self.hyperparams.get("warmup_ratio", 0.1) * 1000),
            lr_scheduler_type=self.hyperparams.get("lr_scheduler_type", "cosine"),
            optim=self.loop_cfg.get("optim", "adamw_torch"),
            eval_strategy=self.config.get("training", {}).get("validation", {}).get(
                "eval_strategy", "epoch"
            ),
            save_strategy=self.checkpoint_cfg.get("save_strategy", "epoch"),
            save_total_limit=self.checkpoint_cfg.get("save_total_limit", 3),
            load_best_model_at_end=self.checkpoint_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=self.checkpoint_cfg.get("metric_for_best_model", "macro_f1"),
            greater_is_better=self.checkpoint_cfg.get("greater_is_better", True),
            logging_dir=self.logging_cfg.get("logging_dir", "logs"),
            logging_steps=self.logging_cfg.get("logging_steps", 10),
            report_to=self.logging_cfg.get("report_to", ["tensorboard"]),
            run_name=self.logging_cfg.get("run_name", "deberta-v3-firewall"),
            gradient_checkpointing=False,
            dataloader_num_workers=self.loop_cfg.get("dataloader_num_workers", 4),
            dataloader_pin_memory=self.loop_cfg.get("dataloader_pin_memory", True),
            remove_unused_columns=False,
        )

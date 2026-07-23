from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.nn import CrossEntropyLoss, functional as F
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import FirewallDataset
from src.evaluation.metrics import compute_metrics
from src.training.callbacks import FirewalCallbacks

logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance with numerical stability.

    Reduces loss contribution from easy (well-classified) examples,
    focusing training on hard misclassified samples. Acts as a soft
    dynamic re-weighting that avoids the extreme weight magnitudes
    of static inverse-frequency weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class StableWeightedTrainer(Trainer):
    """Trainer with numerically stable weighted loss and NaN guards.

    Addresses the training collapse by:
      1. Casting class weights to match model output dtype (bf16/fp16).
      2. Using focal loss (default) or clamped weighted CE to prevent
         extreme gradient magnitudes.
      3. Detecting NaN/Inf in loss and gradients, skipping corrupted
         optimizer steps to prevent weight corruption.
      4. Logging gradient norms for diagnostics.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.1,
        max_weight_ratio: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._nan_skip_count = 0
        self._total_steps = 0

        self.class_weights_raw = class_weights

        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.max_weight_ratio = max_weight_ratio

    def _get_loss_fn(self, logits: torch.Tensor):
        device = logits.device
        dtype = logits.dtype

        if self.use_focal_loss:
            alpha = None
            if self.class_weights_raw is not None:
                normalized = self.class_weights_raw / self.class_weights_raw.sum()
                alpha = normalized.to(device=device, dtype=dtype)
            return FocalLoss(
                gamma=self.focal_gamma,
                alpha=alpha,
                label_smoothing=self.label_smoothing,
            )

        if self.class_weights_raw is not None:
            weight = self.class_weights_raw.to(device=device, dtype=dtype)
            w_max = weight.max()
            w_min = weight.clamp(min=1e-8).min()
            if w_max / w_min > self.max_weight_ratio:
                weight = torch.clamp(weight, max=w_min * self.max_weight_ratio)
                logger.warning(
                    "Clamped class weights from ratio %.1f to %.1f",
                    w_max / w_min, self.max_weight_ratio,
                )
            return CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)

        return CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._total_steps += 1
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get(0))
        else:
            logits = outputs[0]

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning(
                "Step %d: NaN/Inf detected in model outputs, skipping",
                self._total_steps,
            )
            self._nan_skip_count += 1
            safe_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (safe_loss, outputs) if return_outputs else safe_loss

        loss_fn = self._get_loss_fn(logits)
        loss = loss_fn(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "Step %d: NaN/Inf loss (%.4f), skipping optimizer step",
                self._total_steps, loss.item() if not torch.isnan(loss) else float("nan"),
            )
            self._nan_skip_count += 1
            safe_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (safe_loss, outputs) if return_outputs else safe_loss

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self._total_steps % 50 == 0 and self._nan_skip_count > 0:
            logger.info(
                "Training health: %d/%d steps skipped due to NaN/Inf",
                self._nan_skip_count, self._total_steps,
            )

        return loss

    def log(self, logs, start_time=None):
        if self._nan_skip_count > 0:
            logs["nan_skipped_steps"] = self._nan_skip_count
        super().log(logs, start_time=start_time)

    def get_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return optimizer


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

        data_cfg = config.get("data", {})
        balancing_cfg = data_cfg.get("class_balancing", {})
        self.use_focal_loss = balancing_cfg.get("use_smote", False)
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1

        tokenizer_name = config.get("model", {}).get(
            "tokenizer", {}
        ).get("name", "microsoft/deberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = data_cfg.get("preprocessing", {}).get("max_length", 512)

    def train(
        self,
        model,
        train_df,
        val_df,
        class_weights: torch.Tensor | None = None,
        output_dir: str = "checkpoints",
        resume_from_checkpoint: str | None = None,
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

        trainer = StableWeightedTrainer(
            class_weights=class_weights,
            use_focal_loss=self.use_focal_loss,
            focal_gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks.get_callbacks(),
        )

        logger.info("Starting training")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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

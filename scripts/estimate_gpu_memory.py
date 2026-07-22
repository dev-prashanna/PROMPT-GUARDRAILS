#!/usr/bin/env python3
"""Estimate GPU memory requirements for training on RTX 4060."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model


def estimate_memory(
    model_name: str = "microsoft/deberta-v3-base",
    batch_size: int = 8,
    max_length: int = 256,
    lora_rank: int = 16,
    gradient_checkpointing: bool = True,
    fp16: bool = True,
):
    """Estimate GPU memory usage for training."""
    print("=" * 60)
    print("GPU MEMORY ESTIMATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_length}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"Mixed precision: {'fp16' if fp16 else 'bf16' if not fp16 else 'none'}")
    print()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = num_params
    print(f"Base model parameters: {num_params:,}")

    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA): {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / num_params * 100:.2f}%")
    print()

    model_size_mb = num_params * 2 / 1024 / 1024
    if fp16:
        model_size_mb = num_params * 2 / 1024 / 1024
    else:
        model_size_mb = num_params * 4 / 1024 / 1024
    print(f"Model weights ({'fp16' if fp16 else 'fp32'}): {model_size_mb:.1f} MB")

    trainable_size_mb = trainable_params * 4 / 1024 / 1024
    optimizer_size_mb = trainable_size_mb * 2
    print(f"Optimizer states (AdamW): {optimizer_size_mb:.1f} MB")

    gradients_mb = trainable_size_mb
    print(f"Gradients: {gradients_mb:.1f} MB")

    hidden_size = 768
    num_layers = 12
    if gradient_checkpointing:
        activations_mb = batch_size * max_length * hidden_size * 2 / 1024 / 1024 * 0.3
    else:
        activations_mb = batch_size * max_length * hidden_size * 2 * num_layers / 1024 / 1024
    print(f"Activations (gradient ckpt={gradient_checkpointing}): {activations_mb:.1f} MB")

    total_mb = model_size_mb + optimizer_size_mb + gradients_mb + activations_mb
    overhead_mb = total_mb * 0.15
    total_with_overhead = total_mb + overhead_mb

    print()
    print("=" * 60)
    print("MEMORY SUMMARY")
    print("=" * 60)
    print(f"Model weights:        {model_size_mb:8.1f} MB")
    print(f"Optimizer states:     {optimizer_size_mb:8.1f} MB")
    print(f"Gradients:            {gradients_mb:8.1f} MB")
    print(f"Activations:          {activations_mb:8.1f} MB")
    print(f"CUDA overhead:        {overhead_mb:8.1f} MB")
    print(f"{'─' * 40}")
    print(f"TOTAL ESTIMATED:      {total_with_overhead:8.1f} MB")
    print(f"                      {total_with_overhead / 1024:.2f} GB")
    print()

    gpu_memory_gb = 8.0
    if total_with_overhead / 1024 > gpu_memory_gb:
        print(f"WARNING: Estimated {total_with_overhead / 1024:.2f} GB exceeds RTX 4060 ({gpu_memory_gb} GB)")
        print("Recommendations:")
        print("  - Reduce batch size")
        print("  - Reduce max_length")
        print("  - Enable gradient checkpointing")
        print("  - Use LoRA (already enabled)")
    else:
        available = gpu_memory_gb - total_with_overhead / 1024
        print(f"SUCCESS: Fits in RTX 4060 ({gpu_memory_gb} GB)")
        print(f"Available headroom: {available:.2f} GB")

    print()
    print("=" * 60)
    print("RECOMMENDED RTX 4060 SETTINGS")
    print("=" * 60)
    print("training:")
    print("  training_loop:")
    print("    per_device_train_batch_size: 8")
    print("    per_device_eval_batch_size: 16")
    print("    gradient_accumulation_steps: 8")
    print("    effective_batch_size: 64")
    print("    bf16: true")
    print("  gradient_checkpointing:")
    print("    enabled: true")
    print()
    print("model:")
    print("  peft:")
    print("    rank: 16")
    print("    alpha: 32")
    print("  tokenizer:")
    print("    max_length: 256")


if __name__ == "__main__":
    estimate_memory()

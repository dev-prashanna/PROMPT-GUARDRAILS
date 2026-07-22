#!/usr/bin/env python3
"""Merge LoRA adapter weights into the base DeBERTa-v3 model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.merge import ModelMerger
from src.utils.io import load_multi_config
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    config = load_multi_config("configs")

    peft_path = config.get("training", {}).get("output_dir", "checkpoints")
    output_path = config.get("deployment", {}).get("merged_model_path", "deployment/merged_model")

    print(f"Merging LoRA from {peft_path} into base model")
    merger = ModelMerger(config)
    saved_path = merger.merge(peft_path, output_path)
    print(f"Merged model saved to: {saved_path}")


if __name__ == "__main__":
    main()

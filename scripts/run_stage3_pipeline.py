#!/usr/bin/env python3
"""Run Stage 3: Multi-Source Data Aggregation Pipeline.

This script executes the complete Stage 3 pipeline for data collection,
preprocessing, class balancing, and dataset construction.

Usage:
    python scripts/run_stage3_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.stage3_pipeline import Stage3DataPipeline, get_default_stage3_config
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()

    print("=" * 70)
    print("STAGE 3: Multi-Source Data Aggregation Pipeline")
    print("Class Imbalance Mitigation for Model Training")
    print("=" * 70)

    # Load configuration
    config = get_default_stage3_config()

    # Initialize pipeline
    pipeline = Stage3DataPipeline(config, output_dir="data/stage3")

    # Run pipeline
    print("\nExecuting pipeline...")
    stats = pipeline.run_pipeline()

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nAggregation Statistics:")
    agg = stats.get("aggregation", {})
    print(f"  Total samples: {agg.get('total_samples', 0)}")
    print(f"  Label distribution: {agg.get('label_distribution', {})}")
    print(f"  Source distribution: {agg.get('source_distribution', {})}")

    print("\nCleaning Statistics:")
    clean = stats.get("cleaning", {})
    print(f"  Before: {clean.get('before', 0)}")
    print(f"  After: {clean.get('after', 0)}")
    print(f"  Removed: {clean.get('removed', 0)}")

    print("\nBalancing Statistics:")
    bal = stats.get("balancing", {})
    print(f"  Strategy: {bal.get('strategy', 'unknown')}")
    print(f"  Before: {bal.get('before', {})}")
    print(f"  After: {bal.get('after', {})}")

    print("\nFinal Dataset:")
    final = stats.get("final", {})
    for split, info in final.items():
        print(f"  {split}: {info['size']} samples")
        print(f"    Labels: {info['label_distribution']}")

    print("\nOutput Files:")
    print("  - data/stage3/splits/{train,val,test}.parquet")
    print("  - data/stage3/sft/{train,val,test}_sft.parquet")
    print("  - data/stage3/pipeline_metadata.json")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

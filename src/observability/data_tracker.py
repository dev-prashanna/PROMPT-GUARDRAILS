from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataTracker:
    """Track dataset versions, provenance, and quality metrics."""

    def __init__(self, versions_dir: str = "data/versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def register_version(
        self,
        version: str,
        config: dict[str, Any],
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        quality_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

        label_dist = raw_data["label"].value_counts().to_dict() if "label" in raw_data.columns else {}
        source_dist = raw_data["source"].value_counts().to_dict() if "source" in raw_data.columns else {}

        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "config_hash": config_hash,
            "total_samples": len(raw_data),
            "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
            "source_distribution": {str(k): int(v) for k, v in source_dist.items()},
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "quality_scores": quality_scores or {},
            "provenance": {
                "raw_samples": len(raw_data),
                "processed_samples": len(processed_data),
                "processing_delta": len(raw_data) - len(processed_data),
            },
        }

        raw_data.to_parquet(version_dir / "raw.parquet", index=False)
        processed_data.to_parquet(version_dir / "processed.parquet", index=False)
        for split_name, split_df in splits.items():
            split_df.to_parquet(version_dir / f"{split_name}.parquet", index=False)

        with open(version_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(
            "Registered dataset version %s: %d samples, hash=%s",
            version,
            len(raw_data),
            config_hash,
        )
        return manifest

    def load_version(self, version: str) -> dict[str, Any]:
        version_dir = self.versions_dir / version
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset version {version} not found")

        with open(manifest_path) as f:
            manifest = json.load(f)

        manifest["raw"] = pd.read_parquet(version_dir / "raw.parquet")
        manifest["processed"] = pd.read_parquet(version_dir / "processed.parquet")
        manifest["splits"] = {}
        for split_file in version_dir.glob("*.parquet"):
            split_name = split_file.stem
            if split_name not in ("raw", "processed"):
                manifest["splits"][split_name] = pd.read_parquet(split_file)

        return manifest

    def list_versions(self) -> list[dict[str, Any]]:
        versions = []
        for version_dir in sorted(self.versions_dir.iterdir()):
            if version_dir.is_dir():
                manifest_path = version_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        versions.append(json.load(f))
        return versions

    def compare_versions(self, v1: str, v2: str) -> dict[str, Any]:
        m1 = self.load_version(v1)
        m2 = self.load_version(v2)

        return {
            "version_1": v1,
            "version_2": v2,
            "sample_diff": m2["total_samples"] - m1["total_samples"],
            "label_distribution_diff": {
                k: m2["label_distribution"].get(k, 0) - m1["label_distribution"].get(k, 0)
                for k in set(m1["label_distribution"]) | set(m2["label_distribution"])
            },
            "quality_diff": {
                k: m2.get("quality_scores", {}).get(k, 0) - m1.get("quality_scores", {}).get(k, 0)
                for k in set(m1.get("quality_scores", {})) | set(m2.get("quality_scores", {}))
            },
        }

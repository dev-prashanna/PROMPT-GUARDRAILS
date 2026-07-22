from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def create_app(config: dict[str, Any]):
    """Create a FastAPI application for serving the firewall."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    from src.deployment.gates import ThreeGateArchitecture
    from src.deployment.pipeline import FirewallPipeline

    app = FastAPI(
        title="DeBERTa-v3 Firewall API",
        description="Prompt Injection and Jailbreak Detection Service",
        version="0.1.0",
    )

    pipeline_instance = None
    gates_instance = None

    class TextInput(BaseModel):
        text: str
        scan_type: str = "input"
        threshold: float = 0.85

    class PipelineInput(BaseModel):
        user_input: str
        context: str | None = None
        llm_output: str | None = None

    class ScanResponse(BaseModel):
        label: str
        confidence: float
        safe: bool
        latency_ms: float
        details: dict[str, Any] = {}

    @app.on_event("startup")
    async def load_models():
        nonlocal pipeline_instance, gates_instance
        model_path = config.get("deployment", {}).get(
            "merged_model_path", "deployment/merged_model"
        )
        pipeline_instance = FirewallPipeline(model_path, config)
        pipeline_instance.load()

        gates_instance = ThreeGateArchitecture(config)
        gates_instance.load_all()
        logger.info("All models loaded for serving")

    @app.post("/scan", response_model=ScanResponse)
    async def scan_text(input_data: TextInput):
        if pipeline_instance is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        result = pipeline_instance.predict(input_data.text, input_data.threshold)
        return ScanResponse(
            label=result["label"],
            confidence=result["confidence"],
            safe=result["safe"],
            latency_ms=result["latency_ms"],
            details=result["probabilities"],
        )

    @app.post("/pipeline/scan")
    async def scan_pipeline(input_data: PipelineInput):
        if gates_instance is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        results = gates_instance.scan_full_pipeline(
            input_data.user_input, input_data.context, input_data.llm_output
        )
        return {
            "blocked": results.get("pipeline_blocked", False),
            "blocked_at": results.get("blocked_at"),
            "gates": {
                k: {
                    "label": v.label,
                    "confidence": v.confidence,
                    "blocked": v.blocked,
                    "latency_ms": v.latency_ms,
                }
                for k, v in results.items()
                if hasattr(v, "label")
            },
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model_loaded": pipeline_instance is not None,
            "gates_loaded": gates_instance is not None,
        }

    return app

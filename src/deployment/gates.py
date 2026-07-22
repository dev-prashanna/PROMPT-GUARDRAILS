from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "safe", 1: "prompt_injection", 2: "jailbreak"}


@dataclass
class GateResult:
    gate_name: str
    label: str
    confidence: float
    blocked: bool
    latency_ms: float
    details: dict = field(default_factory=dict)


class Gate:
    """Single scanning gate in the 3-Gate Architecture."""

    def __init__(
        self,
        name: str,
        model_path: str,
        threshold: float,
        action: str,
        latency_budget_ms: float,
        device: str = "cuda",
    ):
        self.name = name
        self.threshold = threshold
        self.action = action
        self.latency_budget_ms = latency_budget_ms
        self.device = device
        self._pipeline = None
        self._model_path = model_path

    def load(self) -> None:
        logger.info("Loading gate '%s' from %s", self.name, self._model_path)
        self._pipeline = pipeline(
            "text-classification",
            model=self._model_path,
            tokenizer=self._model_path,
            device=0 if self.device == "cuda" else -1,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("Gate '%s' loaded successfully", self.name)

    def scan(self, text: str) -> GateResult:
        if self._pipeline is None:
            raise RuntimeError(f"Gate '{self.name}' not loaded. Call load() first.")

        start = time.perf_counter()
        results = self._pipeline(text)
        latency = (time.perf_counter() - start) * 1000

        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]

        top_result = max(results, key=lambda x: x["score"])
        label_str = top_result["label"].lower()
        label_id = self._label_to_id(label_str)
        confidence = top_result["score"]

        blocked = (
            label_id != 0
            and confidence >= self.threshold
            and self.action == "block"
        )

        return GateResult(
            gate_name=self.name,
            label=LABEL_MAP.get(label_id, "unknown"),
            confidence=confidence,
            blocked=blocked,
            latency_ms=latency,
            details={"all_scores": {r["label"]: r["score"] for r in results}},
        )

    def _label_to_id(self, label: str) -> int:
        label = label.lower().replace("label_", "").strip()
        for k, v in LABEL_MAP.items():
            if v in label:
                return k
        return 0


class ThreeGateArchitecture:
    """
    3-Gate Security Architecture:
        Gate 1: Input Scanning - Detects prompt injection in user input
        Gate 2: RAG/Document Scanning - Detects embedded injections in retrieved context
        Gate 3: Output Scanning - Detects unsafe or leaked content in LLM output
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        gates_cfg = config.get("deployment", {}).get("gates", {})
        model_path = config.get("deployment", {}).get(
            "merged_model_path", "deployment/merged_model"
        )
        device = config.get("deployment", {}).get("serving", {}).get(
            "model_loading", {}
        ).get("device", "cuda")

        self.gates = {}
        gate_configs = {
            "gate_1_input_scanning": ("Input Scanner", gates_cfg.get("gate_1_input_scanning", {})),
            "gate_2_rag_scanning": ("RAG Scanner", gates_cfg.get("gate_2_rag_scanning", {})),
            "gate_3_output_scanning": ("Output Scanner", gates_cfg.get("gate_3_output_scanning", {})),
        }

        for key, (default_name, gate_cfg) in gate_configs.items():
            if gate_cfg.get("enabled", True):
                self.gates[key] = Gate(
                    name=gate_cfg.get("name", default_name),
                    model_path=gate_cfg.get("model", model_path),
                    threshold=gate_cfg.get("threshold", 0.85),
                    action=gate_cfg.get("action_on_detect", "block"),
                    latency_budget_ms=gate_cfg.get("latency_budget_ms", 50),
                    device=device,
                )

    def load_all(self) -> None:
        for gate in self.gates.values():
            gate.load()
        logger.info("All %d gates loaded", len(self.gates))

    def scan_input(self, user_input: str) -> GateResult:
        return self.gates["gate_1_input_scanning"].scan(user_input)

    def scan_context(self, context: str) -> GateResult:
        return self.gates["gate_2_rag_scanning"].scan(context)

    def scan_output(self, llm_output: str) -> GateResult:
        return self.gates["gate_3_output_scanning"].scan(llm_output)

    def scan_full_pipeline(
        self,
        user_input: str,
        context: str | None = None,
        llm_output: str | None = None,
    ) -> dict[str, GateResult]:
        results = {}

        results["input"] = self.scan_input(user_input)
        if results["input"].blocked:
            logger.warning("Input blocked: %s", results["input"].label)
            results["pipeline_blocked"] = True
            results["blocked_at"] = "input"
            return results

        if context:
            results["context"] = self.scan_context(context)
            if results["context"].blocked:
                logger.warning("Context blocked: %s", results["context"].label)
                results["pipeline_blocked"] = True
                results["blocked_at"] = "context"
                return results

        if llm_output:
            results["output"] = self.scan_output(llm_output)
            if results["output"].blocked:
                logger.warning("Output blocked: %s", results["output"].label)
                results["pipeline_blocked"] = True
                results["blocked_at"] = "output"
                return results

        results["pipeline_blocked"] = False
        results["blocked_at"] = None
        return results

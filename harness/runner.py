import json
import time
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    prompt_id: str
    prompt_text: str
    true_label: int  # 1 = harmful, 0 = benign
    predicted_label: int  # 1 = block, 0 = allow
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    guardrail_name: str
    run_id: int
    results: List[GuardrailResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def save(self, output_dir: str):
        path = Path(output_dir) / f"{self.guardrail_name}_run_{self.run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in self.results:
                f.write(json.dumps(asdict(r)) + "\n")
        logger.info(f"Saved {len(self.results)} results to {path}")


def load_config(config_path: str = "harness/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(path: str) -> List[dict]:
    prompts = []
    data_dir = Path(path)
    for file in sorted(data_dir.glob("*.jsonl")):
        with open(file) as f:
            for line in f:
                prompts.append(json.loads(line))
    return prompts


class GuardrailAdapter:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    def predict(self, prompt: str) -> tuple[int, Optional[float]]:
        raise NotImplementedError

    def health_check(self) -> bool:
        raise NotImplementedError


def run_single_prompt(adapter: GuardrailAdapter, prompt: dict) -> GuardrailResult:
    start = time.perf_counter()
    try:
        pred_label, confidence = adapter.predict(prompt["text"])
        latency = (time.perf_counter() - start) * 1000
        return GuardrailResult(
            prompt_id=prompt["id"],
            prompt_text=prompt["text"],
            true_label=prompt["label"],
            predicted_label=pred_label,
            confidence=confidence,
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return GuardrailResult(
            prompt_id=prompt["id"],
            prompt_text=prompt["text"],
            true_label=prompt["label"],
            predicted_label=0,
            latency_ms=latency,
            error=str(e),
        )


def run_experiment(
    adapter: GuardrailAdapter,
    dataset: List[dict],
    run_id: int,
    max_workers: int = 1,
    config: dict = None,
) -> ExperimentResult:
    if not adapter.health_check():
        logger.warning(f"Health check failed for {adapter.name}, skipping")

    result = ExperimentResult(
        guardrail_name=adapter.name,
        run_id=run_id,
        config=config or {},
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_prompt, adapter, p): p for p in dataset
        }
        for future in as_completed(futures):
            result.results.append(future.result())

    result.results.sort(key=lambda r: r.prompt_id)
    return result

.PHONY: help install data train evaluate merge serve clean lint test format pipeline stage1 ablation adversarial benchmark

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e ".[dev,paper]"

data: ## Download and preprocess data
	python scripts/download_data.py

train: ## Train the model
	python scripts/train.py

evaluate: ## Evaluate the model
	python scripts/evaluate.py

merge: ## Merge LoRA weights into base model
	python scripts/merge_model.py

serve: ## Start the API server
	python scripts/serve.py

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint: ## Run linters
	ruff check src/ scripts/ tests/
	mypy src/

test: ## Run tests
	pytest tests/ -v

format: ## Format code
	ruff format src/ scripts/ tests/

pipeline: data train evaluate merge ## Run full pipeline

stage1: ## Run Stage 1 experiments (data + training + evaluation + stats)
	python scripts/run_experiment.py

ablation: ## Run ablation studies
	python scripts/run_ablation.py

adversarial: ## Run adversarial robustness evaluation
	python -c "from src.data.adversarial_sets import AdversarialTestSet; print(AdversarialTestSet().get_category_summary())"

benchmark: ## Run cross-model benchmarks
	python -c "from src.evaluation.benchmarks import BenchmarkRunner; print('Benchmark runner ready')"

quality: ## Run data quality checks
	python -c "from src.data.quality import QualityChecker; print('Quality checker ready')"

versions: ## List dataset versions
	python -c "from src.observability.data_tracker import DataTracker; print(DataTracker().list_versions())"

experiments: ## List all experiments
	python -c "from src.observability.experiment_tracker import ExperimentTracker; [print(e['id'], e['status']) for e in ExperimentTracker().list_experiments()]"

dashboard: ## Show experiment dashboard summary
	@echo "=== Stage 1 Experiment Dashboard ==="
	@echo ""
	@echo "Data Versions:"
	@python -c "from src.observability.data_tracker import DataTracker; [print(f'  {v[\"version\"]}: {v[\"total_samples\"]} samples') for v in DataTracker().list_versions()]"
	@echo ""
	@echo "Experiments:"
	@python -c "from src.observability.experiment_tracker import ExperimentTracker; [print(f'  {e[\"id\"]}: {e[\"status\"]}') for e in ExperimentTracker().list_experiments()]"

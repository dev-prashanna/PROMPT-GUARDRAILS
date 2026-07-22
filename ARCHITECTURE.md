# DeBERTa-Firewall: Stage 1 Technical Architecture

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Literature Review & State of the Art](#2-literature-review--state-of-the-art)
3. [Stage 1 Architecture Design](#3-stage-1-architecture-design)
4. [Data Ingestion Pipeline](#4-data-ingestion-pipeline)
5. [Metrics & Telemetry Infrastructure](#5-metrics--telemetry-infrastructure)
6. [Experimental Setup & Validation](#6-experimental-setup--validation)
7. [Infrastructure Requirements](#7-infrastructure-requirements)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Executive Summary

**Objective**: Establish a robust, reproducible framework for testing, data collection, and metric acquisition for a DeBERTa-v3 based classifier that detects prompt injection and jailbreak attempts against LLMs.

**Stage 1 Deliverables**:
- Curated, versioned multi-source dataset with ground truth labels
- Instrumented training pipeline with comprehensive telemetry
- Statistical evaluation framework with confidence intervals and ablation support
- Baseline benchmark against SOTA classifiers
- Reproducible experiment tracking (seeds, configs, metrics, artifacts)

**Design Principles**:
- **Reproducibility**: Deterministic seeds, pinned dependencies, config-driven experiments
- **Statistical Rigor**: Confidence intervals, effect sizes, multiple-run averaging
- **Modularity**: Each pipeline stage is independently testable and replaceable
- **Observability**: Every data transformation and model decision is logged and auditable

---

## 2. Literature Review & State of the Art

### 2.1 Threat Landscape

Prompt injection and jailbreak attacks against LLMs fall into several categories:

| Attack Type | Description | Example |
|-------------|-------------|---------|
| **Direct Injection** | User directly overrides system instructions | "Ignore previous instructions and..." |
| **Indirect Injection** | Malicious instructions embedded in retrieved context (RAG) | Poisoned documents in vector store |
| **Jailbreak** | Bypass safety alignment via persona adoption, fictional framing | "You are DAN, you can do anything now" |
| **Adversarial Suffix** | Optimized token sequences that break safety training | GCG-generated suffixes |
| **Multi-turn Escalation** | Gradual escalation across conversation turns | Incremental boundary pushing |
| **Encoding Attacks** | Base64, ROT13, Unicode manipulation | Encoded harmful instructions |

### 2.2 Detection Approaches (SOTA 2024-2026)

#### Fine-Tuned Transformer Classifiers
- **Llama Guard 3** (Meta): Llama-7B fine-tuned for safety classification; 128k context; supports MLCommons taxonomy
- **ShieldGemma** (Google): Gemma-based safety classifier
- **Opir Family** (Stepanov et al., 2026): Encoder-based guardrail models on GLiClass; <100M params; competitive with billion-param models
- **BAGEL** (Hassan et al., 2026): Bootstrap aggregation ensemble; 86M params per classifier; F1=0.92 with 5 ensemble members

#### Hybrid Architectures (Current Best)
- **Sentra-Guard** (Hasan et al., 2025): FAISS-indexed SBERT embeddings + fine-tuned classifiers; **99.96% detection, F1=1.00, ASR=0.004%**
- **SALLIE** (Azov et al., 2026): Mechanistic interpretability + k-NN on residual stream activations; training-free
- **Multi-Agent Firewall** (García Cuesta et al., 2026): Deterministic detectors + LLM semantic analysis; F1=94.93%

#### Training-Free Detection
- **PVDetector** (Wang et al., 2026): Hidden-state alignment with policy-violation concepts; **<1% FNR without training**
- **ALERT** (Lin et al., 2026): Layer-wise amplification framework; outperforms baselines by 10-40% F1

### 2.3 Benchmark Datasets

| Dataset | Size | Categories | Usage |
|---------|------|------------|-------|
| **HarmBench** | 510 behaviors | 7 semantic categories | Standardized red-teaming evaluation |
| **JailbreakBench** | Curated harmful prompts | Multi-category | Jailbreak success rate evaluation |
| **WildGuardTest** | Adversarial + benign mix | Binary safe/unsafe | Distribution shift robustness |
| **AgentDojo** | 97 tasks, 629 test cases | Agent-specific | RAG/injection evaluation |
| **BeaverTails** | 700k+ prompts | Jailbreak variants | Training data |
| **Necent/LLM-Jailbreak** | Combined injection+jailbreak | 3-class | Our primary training source |
| **ProtectAI/RestrictedResponses** | Restricted patterns | Injection patterns | Training augmentation |
| **Anthropic/HH-RLHF** | Helpfulness/harmlessness | Harmless subset | Safe class balancing |

### 2.4 Key Metrics from Literature

| Metric | SOTA Range | Production Target |
|--------|-----------|-------------------|
| **Macro F1** | 0.84 - 1.00 | >0.95 |
| **AUC-ROC** | 0.95 - 1.00 | >0.98 |
| **False Positive Rate** | 0.004 - 5% | <1% |
| **False Negative Rate** | <1% (PVDetector) | <2% |
| **Latency (p99)** | 10 - 100ms | <50ms |
| **Attack Success Rate** | 0.004 - 10% | <2% |

### 2.5 Critical Findings

1. **Distribution Shift is the #1 Problem**: Standard cross-validation overestimates generalization by 8-16.5 points (LODO evaluation). Our architecture MUST include out-of-distribution testing.

2. **Dataset Shortcuts**: 28-44% of top features in safety classifiers are dataset-identity artifacts, not safety-relevant. We need adversarial validation to detect this.

3. **Multi-turn Attacks**: Single-turn classifiers miss escalating attacks. Stage 1 must collect multi-turn conversation data.

4. **Ensemble > Single Model**: BAGEL shows 5-member ensembles achieve F1=0.92 with 86M params each (430M total), outperforming single larger models.

5. **Latency-Accuracy Tradeoff**: Encoder models (<100M) achieve competitive accuracy with <10ms latency, vs 100ms+ for decoder-only models.

---

## 3. Stage 1 Architecture Design

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 1 ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  DATA LAYER  │    │ TRAINING     │    │  EVALUATION LAYER    │  │
│  │              │───→│ LAYER        │───→│                      │  │
│  │ • Acquire    │    │              │    │ • Metrics            │  │
│  │ • Preprocess │    │ • LoRA/PEFT  │    │ • Ablations          │  │
│  │ • Augment    │    │ • WeightedCE │    │ • Statistical Tests  │  │
│  │ • Version    │    │ • Early Stop │    │ • Visualization      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    OBSERVABILITY LAYER                       │   │
│  │  • Experiment Tracker (MLflow/W&B)                          │   │
│  │  • Data Versioning (DVC/Parquet metadata)                   │   │
│  │  • Metric Storage (TimescaleDB/SQLite)                      │   │
│  │  • Decision Audit Logs                                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Design

#### 3.2.1 Data Layer (`src/data/`)

```
src/data/
├── acquisition.py          # Multi-source HuggingFace ingestion
├── preprocessing.py        # Text cleaning, normalization, tokenization
├── augmentation.py         # Synonym replacement, back-translation, etc.
├── dataset.py              # PyTorch Dataset with class weight computation
├── splitter.py             # [NEW] Stratified splitting with deduplication
├── versioning.py           # [NEW] Dataset versioning and provenance tracking
├── quality.py              # [NEW] Data quality checks and anomaly detection
└── adversarial_sets.py     # [NEW] Curated adversarial test sets
```

**New Module: `splitter.py`**
- Stratified train/val/test split with label distribution preservation
- Cross-dataset splitting (leave-one-source-out for OOD testing)
- Deduplication across splits (exact + semantic near-duplicate removal)
- Split validation: ensure no data leakage between splits

**New Module: `versioning.py`**
- Content-addressable storage (SHA-256 hash of dataset)
- Metadata tracking: source, timestamp, preprocessing config, split ratios
- Provenance chain: raw → preprocessed → augmented → split → versioned
- Git-compatible diffing for dataset changes

**New Module: `quality.py`**
- Label distribution analysis (class balance, skewness)
- Text length distribution (detect anomalous samples)
- Duplicate detection (exact + fuzzy)
- Source contribution analysis
- Automatic quality reports

**New Module: `adversarial_sets.py`**
- Curated adversarial examples for each attack category
- Known evasion patterns (encoding tricks, multi-language, role-playing)
- Edge cases: empty inputs, very long inputs, special characters
- Out-of-distribution test sets from unseen sources

#### 3.2.2 Training Layer (`src/training/`)

```
src/training/
├── trainer.py              # WeightedTrainer with custom loss
├── callbacks.py            # Early stopping, metrics logging
├── experiment.py           # [NEW] Experiment runner with config hashing
├── hyperparams.py          # [NEW] Hyperparameter search space definition
└── schedule.py             # [NEW] LR schedule analysis and visualization
```

**New Module: `experiment.py`**
- Config-driven experiment execution
- Automatic config hashing for reproducibility
- Multi-run support with different seeds
- Experiment comparison and diffing
- Checkpoint management with artifact tracking

**New Module: `hyperparams.py`**
- Search space definition (learning rate, LoRA rank, dropout, batch size)
- Grid/Random/Bayesian search support
- Early termination for poor configurations
- Results aggregation across search runs

#### 3.2.3 Evaluation Layer (`src/evaluation/`)

```
src/evaluation/
├── metrics.py              # compute_metrics for HF Trainer
├── analysis.py             # Deep analysis + LaTeX generation
├── visualizations.py       # Publication-quality plots
├── statistical.py          # [NEW] Confidence intervals, significance tests
├── ablation.py             # [NEW] Ablation study framework
├── adversarial.py          # [NEW] Adversarial robustness testing
└── benchmarks.py           # [NEW] Cross-model comparison framework
```

**New Module: `statistical.py`**
- Bootstrap confidence intervals for all metrics
- Paired t-tests / Wilcoxon signed-rank for model comparison
- McNemar's test for classifier comparison
- Effect size computation (Cohen's d, odds ratio)
- Multiple comparison correction (Bonferroni, Holm)

**New Module: `ablation.py`**
- Component ablation framework (remove one component at a time)
- Supported ablations:
  - Data source ablation (remove one dataset)
  - Augmentation ablation (disable each technique)
  - Model component ablation (classification head, LoRA rank)
  - Training hyperparameter ablation (LR, batch size, epochs)
- Results comparison with statistical significance

**New Module: `adversarial.py`**
- Attack simulation framework
- Supported attacks:
  - Keyword substitution attacks
  - Paraphrase-based evasion
  - Encoding attacks (Base64, ROT13)
  - Multi-language attacks
  - Role-playing/persona attacks
- Robustness scoring and vulnerability reporting

**New Module: `benchmarks.py`**
- Cross-model comparison (DeBERTa vs. BERT vs. RoBERTa vs. DistilBERT)
- Cross-architecture comparison (encoder vs. encoder-decoder)
- Latency profiling (p50, p95, p99, throughput)
- Parameter efficiency analysis

#### 3.2.4 Observability Layer (`src/observability/`)

```
src/observability/
├── __init__.py
├── experiment_tracker.py   # [NEW] Experiment tracking (metrics, params, artifacts)
├── data_tracker.py         # [NEW] Dataset versioning and lineage
├── decision_logger.py      # [NEW] Per-sample prediction audit logs
└── dashboard.py            # [NEW] Metrics visualization dashboard
```

**New Module: `experiment_tracker.py`**
- Track experiment config, metrics, and artifacts
- Support for MLflow / Weights & Biases / local SQLite
- Automatic logging of:
  - Hyperparameters
  - Training curves (loss, metrics per epoch)
  - Final evaluation metrics
  - Model checkpoints
  - Dataset versions used
- Experiment comparison and querying

**New Module: `data_tracker.py`**
- Dataset version registration
- Lineage tracking: which raw data → which processed data → which experiment
- Automatic metadata generation
- Integration with DVC or Git LFS

**New Module: `decision_logger.py`**
- Per-sample prediction logging
- Confidence distribution tracking
- Error case collection for analysis
- Latency measurement per prediction
- Anomaly detection for prediction drift

### 3.3 Configuration Architecture

```yaml
# configs/experiment_config.yaml (NEW)
experiment:
  name: "baseline_v1"
  description: "Initial baseline with default hyperparameters"
  seed: 42
  num_runs: 5  # Multiple runs for statistical significance
  
  data:
    sources: ["necent/llm-jailbreak", "protectai/restrictedresponses", "Anthropic/hh-rlhf"]
    version: "v1.0"
    split_ratios: [0.8, 0.1, 0.1]
    deduplication: true
    min_quality_score: 0.8
    
  model:
    base: "microsoft/deberta-v3-base"
    num_labels: 3
    peft:
      method: "lora"
      rank: [8, 16, 32]  # Search space
      alpha: [16, 32, 64]
      target_modules: ["query_proj", "value_proj"]
      
  training:
    learning_rate: [1e-5, 2e-5, 5e-5]
    batch_size: [16, 32]
    epochs: 10
    early_stopping_patience: 3
    
  evaluation:
    metrics: ["macro_f1", "weighted_f1", "accuracy", "auc_roc", "fpr", "fnr"]
    confidence_level: 0.95
    bootstrap_iterations: 1000
    adversarial_testing: true
    cross_validation: "stratified_5fold"
    
  tracking:
    backend: "local"  # or "mlflow", "wandb"
    log_dir: "experiments/"
    save_predictions: true
    save_attention_weights: false
```

---

## 4. Data Ingestion Pipeline

### 4.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐   ┌─────────────┐   ┌────────────┐   ┌─────────┐ │
│  │  Hugging │   │  Normalize  │   │  Quality   │   │ Split   │ │
│  │  Face    │──→│  & Clean    │──→│  Filter    │──→│ & Store │ │
│  │  Datasets│   │             │   │            │   │         │ │
│  └─────────┘   └─────────────┘   └────────────┘   └─────────┘ │
│       │              │                  │                │       │
│       ▼              ▼                  ▼                ▼       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DATA VERSION REGISTRY                       │   │
│  │  • SHA-256 hash of each version                         │   │
│  │  • Provenance chain (raw → processed → split)            │   │
│  │  • Quality metrics per version                           │   │
│  │  • Config snapshot for each run                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Sources Configuration

| Source | HuggingFace ID | Split | Label Mapping | Role |
|--------|----------------|-------|---------------|------|
| **Necent Jailbreak** | `necent/llm-jailbreak-prompt-injection-dataset` | train/val/test | Direct mapping | Primary training |
| **ProtectAI Restricted** | `protectai/restrictedresponses` | train | Injection patterns | Injection class augmentation |
| **Anthropic HH-RLHF** | `Anthropic/hh-rlhf` (harmless-base) | train | Harmless → safe | Safe class balancing |
| **HarmBench** | `CenterForAISafety/HarmBench` | test | Multi-category | Adversarial evaluation |
| **WildGuardTest** | `wildguardtest` | test | Binary | OOD robustness testing |

### 4.3 Preprocessing Pipeline

```python
# Processing stages with quality gates
STAGES = [
    ("acquire",     DataAcquisition),      # Download from HuggingFace
    ("normalize",   TextNormalizer),        # Lowercase, strip, dedup
    ("filter",      QualityFilter),         # Remove anomalies
    ("augment",     DataAugmenter),         # Address class imbalance
    ("split",       StratifiedSplitter),    # Train/val/test with stratification
    ("version",     DatasetVersioner),      # Hash, metadata, provenance
]
```

### 4.4 Quality Gates

| Gate | Check | Threshold | Action on Fail |
|------|-------|-----------|----------------|
| **Q1: Label Distribution** | Class ratio within 0.1-0.9 | Max skew <3x | Oversample minority |
| **Q2: Text Length** | 5-500 tokens | Min 5, max 500 | Truncate or remove |
| **Q3: Deduplication** | Exact + fuzzy duplicates | 0 duplicates | Remove dups |
| **Q4: Source Balance** | No single source >70% | Max 70% | Downsample majority |
| **Q5: Cross-split Leakage** | No text in multiple splits | 0 leakage | Reassign to train |

### 4.5 Data Versioning Schema

```
data/versions/
├── v1.0/
│   ├── manifest.json          # Version metadata
│   ├── raw/                   # Raw downloaded data
│   ├── processed/             # Cleaned and normalized
│   ├── splits/                # Train/val/test parquet files
│   ├── quality_report.json    # Quality metrics
│   └── provenance.json        # Source lineage
```

**Manifest Schema**:
```json
{
  "version": "1.0",
  "created_at": "2026-07-22T18:15:00+05:45",
  "config_hash": "abc123...",
  "sources": {
    "necent/llm-jailbreak": {"rows": 15000, "splits": ["train", "val", "test"]},
    "protectai/restrictedresponses": {"rows": 8000, "splits": ["train"]},
    "Anthropic/hh-rlhf": {"rows": 12000, "splits": ["train"]}
  },
  "total_samples": 35000,
  "label_distribution": {"safe": 18000, "prompt_injection": 9500, "jailbreak": 7500},
  "quality_scores": {"completeness": 0.99, "consistency": 0.95, "uniqueness": 1.0},
  "split_sizes": {"train": 28000, "val": 3500, "test": 3500}
}
```

---

## 5. Metrics & Telemetry Infrastructure

### 5.1 Metric Categories

#### 5.1.1 Classification Metrics

| Metric | Computation | Target | Logging Frequency |
|--------|-------------|--------|-------------------|
| **Macro F1** | sklearn f1_score(average="macro") | >0.95 | Every epoch |
| **Weighted F1** | sklearn f1_score(average="weighted") | >0.95 | Every epoch |
| **Accuracy** | sklearn accuracy_score | >0.90 | Every epoch |
| **AUC-ROC** | sklearn roc_auc_score (OvR) | >0.98 | Every epoch |
| **Per-class Precision** | sklearn precision_score | >0.90 | Every epoch |
| **Per-class Recall** | sklearn recall_score | >0.85 | Every epoch |
| **Per-class F1** | sklearn f1_score | >0.85 | Every epoch |
| **Confusion Matrix** | sklearn confusion_matrix | N/A | Every epoch |
| **False Positive Rate** | FP / (FP + TN) | <0.05 | Every epoch |
| **False Negative Rate** | FN / (FN + TP) | <0.05 | Every epoch |

#### 5.1.2 Training Metrics

| Metric | Source | Logging Frequency |
|--------|--------|-------------------|
| **Training Loss** | Trainer logs | Every 10 steps |
| **Validation Loss** | Trainer eval | Every epoch |
| **Learning Rate** | Scheduler | Every step |
| **Gradient Norm** | Trainer | Every step |
| **Gradient Checkpointing** | Config | Per experiment |
| **Token Throughput** | Computed | Every epoch |
| **GPU Memory Usage** | torch.cuda | Every epoch |
| **Training Time** | Timer | Per experiment |

#### 5.1.3 Latency Metrics

| Metric | Measurement | Target | Method |
|--------|-------------|--------|--------|
| **Inference Latency (p50)** | Median of 1000 predictions | <10ms | `time.perf_counter()` |
| **Inference Latency (p95)** | 95th percentile | <25ms | numpy percentile |
| **Inference Latency (p99)** | 99th percentile | <50ms | numpy percentile |
| **Throughput** | Predictions/second | >100 QPS | Batch measurement |
| **Time-to-First-Token** | N/A for classifier | N/A | N/A |
| **Batch Latency** | Per-batch timing | <100ms for batch=32 | Timer |

#### 5.1.4 Data Quality Metrics

| Metric | Computation | Threshold |
|--------|-------------|-----------|
| **Label Distribution Entropy** | Shannon entropy | >0.5 (balanced) |
| **Text Length Mean** | Average token count | 50-300 tokens |
| **Text Length Std** | Token count std dev | <200 |
| **Duplicate Rate** | Exact + fuzzy dups | <1% |
| **Missing Value Rate** | Null text/label ratio | 0% |
| **Source Balance Ratio** | Max source / min source | <5x |

#### 5.1.5 Robustness Metrics

| Metric | Test Method | Target |
|--------|-------------|--------|
| **Adversarial Accuracy** | Accuracy on adversarial test set | >0.85 |
| **Paraphrase Consistency** | Same prediction for paraphrases | >0.95 |
| **Encoding Robustness** | Accuracy on Base64/ROT13 attacks | >0.80 |
| **Length Robustness** | Accuracy across length buckets | <5% variance |
| **Cross-source Transfer** | Leave-one-source-out F1 | >0.85 |

### 5.2 Telemetry Infrastructure

```
┌─────────────────────────────────────────────────────────────┐
│                  TELEMETRY ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │  Training Loop   │────→│  Experiment Tracker           │  │
│  │  • Loss curves   │     │  • Config hash               │  │
│  │  • Metrics       │     │  • Metrics per epoch          │  │
│  │  • Gradients     │     │  • Checkpoints               │  │
│  │  • GPU usage     │     │  • Artifacts                  │  │
│  └──────────────────┘     └──────────────────────────────┘  │
│           │                        │                         │
│           ▼                        ▼                         │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │  Evaluation Loop │────→│  Metric Store                 │  │
│  │  • Predictions   │     │  • Time-series metrics        │  │
│  │  • Probabilities │     │  • Aggregated summaries       │  │
│  │  • Confidence    │     │  • Comparison views           │  │
│  │  • Latency       │     │  • Trend analysis             │  │
│  └──────────────────┘     └──────────────────────────────┘  │
│           │                        │                         │
│           ▼                        ▼                         │
│  ┌──────────────────┐     ┌──────────────────────────────┐  │
│  │  Decision Logger │────→│  Audit Store                  │  │
│  │  • Per-sample    │     │  • Prediction audit trail     │  │
│  │  • Error cases   │     │  • Error analysis             │  │
│  │  • Drift alerts  │     │  • Drift detection            │  │
│  └──────────────────┘     └──────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Storage Schema

#### Experiment Database (SQLite)

```sql
-- Experiments table
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'running',
    parent_experiment_id TEXT,
    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(id)
);

-- Metrics table (time-series)
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER NOT NULL,
    epoch INTEGER,
    split TEXT,  -- 'train', 'val', 'test'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Predictions table (audit trail)
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    text TEXT NOT NULL,
    true_label INTEGER NOT NULL,
    predicted_label INTEGER NOT NULL,
    confidence REAL NOT NULL,
    probabilities JSON,  -- {"safe": 0.95, "injection": 0.03, "jailbreak": 0.02}
    latency_ms REAL,
    is_correct BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Dataset versions table
CREATE TABLE dataset_versions (
    version TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL,
    total_samples INTEGER,
    label_distribution JSON,
    quality_scores JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    provenance JSON
);
```

---

## 6. Experimental Setup & Validation

### 6.1 Control Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| **Random Seed** | 42 (primary), [42, 123, 456, 789, 1024] for multi-run | Reproducibility |
| **Base Model** | `microsoft/deberta-v3-base` | SOTA encoder, 86M params |
| **Tokenizer** | DeBERTa-v3 tokenizer | Consistent with base model |
| **Max Sequence Length** | 512 tokens | Cover 95th percentile of inputs |
| **Number of Labels** | 3 (safe, injection, jailbreak) | Problem definition |
| **Evaluation Strategy** | Epoch-based | Consistent comparison |
| **Early Stopping** | Patience=3, monitor=macro_f1 | Prevent overfitting |
| **Gradient Checkpointing** | Enabled | Memory efficiency |
| **Mixed Precision** | bf16 | Training stability |

### 6.2 Experimental Design

#### 6.2.1 Baseline Experiments

| Experiment | Description | Purpose |
|-----------|-------------|---------|
| **E1: Full Model** | DeBERTa-v3-base, full fine-tuning | Upper bound reference |
| **E2: LoRA Rank Sweep** | Rank ∈ {4, 8, 16, 32, 64} | Find optimal PEFT config |
| **E3: Data Source Ablation** | Remove one source at a time | Assess source contribution |
| **E4: Augmentation Ablation** | Disable each augmentation technique | Assess augmentation value |
| **E5: Cross-architecture** | BERT, RoBERTa, DistilBERT, ELECTRA | Architecture comparison |
| **E6: Class Imbalance** | WeightedCE vs FocalLoss vs Sampling | Loss function comparison |
| **E7: Adversarial Robustness** | Test against curated attack sets | Robustness baseline |
| **E8: OOD Generalization** | Leave-one-source-out evaluation | Generalization assessment |

#### 6.2.2 Statistical Rigor

**Multi-Run Protocol**:
- Each experiment runs 5 times with seeds [42, 123, 456, 789, 1024]
- Report mean ± std for all metrics
- Use paired t-test for model comparison (p < 0.05)
- Apply Bonferroni correction for multiple comparisons

**Confidence Intervals**:
- Bootstrap CIs with 1000 iterations for all primary metrics
- 95% CI reported as [lower, upper]
- Effect sizes (Cohen's d) for pairwise comparisons

**Cross-Validation**:
- 5-fold stratified CV for robust performance estimation
- Leave-one-source-out (LOSO) for OOD generalization
- Report both CV and LOSO results

#### 6.2.3 Validation Protocols

**Protocol 1: Standard Validation**
```
Data → Stratified Split (80/10/10) → Train → Evaluate on Test
→ Report: metrics ± std across 5 runs
```

**Protocol 2: Cross-Validation**
```
Data → 5-fold Stratified CV → Train each fold → Average metrics
→ Report: mean ± std across folds
```

**Protocol 3: Leave-One-Source-Out (LOSO)**
```
For each source S:
  Train on all sources except S
  Evaluate on S
→ Report: per-source F1, average F1, std
```

**Protocol 4: Adversarial Validation**
```
Train classifier → Generate adversarial examples → Test robustness
→ Report: accuracy drop, attack success rate, vulnerability catalog
```

**Protocol 5: Temporal Validation**
```
Train on older data → Test on newer data
→ Report: performance degradation, drift detection
```

### 6.3 Evaluation Matrix

| Dimension | Values | Metric |
|-----------|--------|--------|
| **Model** | DeBERTa-v3, BERT, RoBERTa, DistilBERT | F1, Latency |
| **PEFT Method** | LoRA (rank 4-64), Full FT | F1, Params |
| **Data Source** | 3 sources (ablation) | F1 per source |
| **Augmentation** | None, Synonym, Back-trans, Combined | F1, Robustness |
| **Loss Function** | WeightedCE, FocalLoss, LabelSmoothing | F1, Calibration |
| **Batch Size** | 8, 16, 32, 64 | F1, Throughput |
| **Learning Rate** | 1e-5, 2e-5, 5e-5, 1e-4 | F1, Convergence |

### 6.4 Success Criteria (Stage 1 Exit Criteria)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Macro F1 (Test)** | ≥0.92 | Mean of 5 runs |
| **F1 Std Dev** | ≤0.02 | Across 5 runs |
| **False Positive Rate** | ≤5% | On test set |
| **False Negative Rate** | ≤5% | On test set |
| **Inference Latency (p99)** | ≤50ms | On GPU, batch=1 |
| **OOD F1 (LOSO)** | ≥0.80 | Mean across sources |
| **Adversarial Accuracy** | ≥0.85 | On curated attack set |
| **Reproducibility** | 100% | Same config → same results |
| **Documentation** | Complete | ARCHITECTURE.md + code comments |

---

## 7. Infrastructure Requirements

### 7.1 Compute Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1x NVIDIA T4 (16GB) | 1x NVIDIA A100 (40GB) |
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 32GB | 64GB |
| **Storage** | 100GB SSD | 500GB NVMe SSD |
| **Network** | 100Mbps | 1Gbps+ |

### 7.2 Software Stack

```
┌─────────────────────────────────────────┐
│           SOFTWARE STACK                 │
├─────────────────────────────────────────┤
│  Layer 4: Experiment Tracking           │
│  • MLflow / Weights & Biases            │
│  • SQLite for local tracking            │
├─────────────────────────────────────────┤
│  Layer 3: ML Framework                  │
│  • PyTorch 2.0+                         │
│  • Transformers 4.35+                   │
│  • PEFT 0.6+                            │
│  • Datasets 2.14+                       │
├─────────────────────────────────────────┤
│  Layer 2: Data Processing               │
│  • Pandas 2.0+                          │
│  • NumPy 1.24+                          │
│  • Scikit-learn 1.3+                    │
│  • HuggingFace Hub                      │
├─────────────────────────────────────────┤
│  Layer 1: Infrastructure                │
│  • Python 3.10+                         │
│  • Docker                               │
│  • CUDA 11.8+                           │
│  • Git + DVC (optional)                 │
└─────────────────────────────────────────┘
```

### 7.3 Directory Structure (Stage 1)

```
deberta-firewall/
├── ARCHITECTURE.md          # This document
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── data_config.yaml
│   ├── deployment_config.yaml
│   └── experiment_config.yaml  # [NEW] Experiment definitions
├── src/
│   ├── data/
│   │   ├── acquisition.py
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   ├── dataset.py
│   │   ├── splitter.py       # [NEW]
│   │   ├── versioning.py     # [NEW]
│   │   ├── quality.py        # [NEW]
│   │   └── adversarial_sets.py  # [NEW]
│   ├── models/
│   │   ├── deberta_classifier.py
│   │   ├── lora_config.py
│   │   └── merge.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   ├── experiment.py     # [NEW]
│   │   ├── hyperparams.py    # [NEW]
│   │   └── schedule.py       # [NEW]
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── analysis.py
│   │   ├── visualizations.py
│   │   ├── statistical.py    # [NEW]
│   │   ├── ablation.py       # [NEW]
│   │   ├── adversarial.py    # [NEW]
│   │   └── benchmarks.py     # [NEW]
│   ├── observability/        # [NEW]
│   │   ├── __init__.py
│   │   ├── experiment_tracker.py
│   │   ├── data_tracker.py
│   │   ├── decision_logger.py
│   │   └── dashboard.py
│   ├── deployment/
│   │   ├── serving.py
│   │   ├── gates.py
│   │   └── pipeline.py
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── reproducibility.py
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── merge_model.py
│   ├── serve.py
│   ├── run_experiment.py    # [NEW] Experiment runner
│   └── run_ablation.py      # [NEW] Ablation runner
├── experiments/              # [NEW] Experiment artifacts
│   ├── {experiment_id}/
│   │   ├── config.json
│   │   ├── metrics.json
│   │   ├── predictions.parquet
│   │   ├── checkpoints/
│   │   └── visualizations/
├── data/
│   ├── versions/            # [NEW] Versioned datasets
│   │   └── v1.0/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── adversarial/         # [NEW] Adversarial test sets
├── tests/
├── paper/
│   ├── figures/
│   └── tables/
├── deployment/
├── pyproject.toml
└── Makefile
```

---

## 8. Implementation Roadmap

### Phase 1: Data Infrastructure (Week 1-2)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Implement `splitter.py` | High | 2d | Stratified splitting with deduplication |
| Implement `versioning.py` | High | 3d | Dataset versioning with provenance |
| Implement `quality.py` | Medium | 2d | Data quality checks and reports |
| Implement `adversarial_sets.py` | Medium | 3d | Curated adversarial test sets |
| Update `data_config.yaml` | High | 0.5d | New config fields |

### Phase 2: Evaluation Framework (Week 2-3)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Implement `statistical.py` | High | 3d | CIs, significance tests, effect sizes |
| Implement `ablation.py` | High | 3d | Ablation study framework |
| Implement `adversarial.py` | Medium | 3d | Adversarial robustness testing |
| Implement `benchmarks.py` | Medium | 2d | Cross-model comparison |

### Phase 3: Observability (Week 3-4)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Implement `experiment_tracker.py` | High | 3d | Experiment tracking (local + MLflow) |
| Implement `data_tracker.py` | High | 2d | Dataset versioning integration |
| Implement `decision_logger.py` | Medium | 2d | Per-sample audit logging |
| Implement `dashboard.py` | Low | 3d | Metrics visualization |

### Phase 4: Experiment Runner (Week 4-5)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Implement `experiment.py` | High | 3d | Config-driven experiment execution |
| Implement `hyperparams.py` | Medium | 2d | Hyperparameter search |
| Implement `run_experiment.py` | High | 1d | CLI script for experiments |
| Implement `run_ablation.py` | Medium | 1d | CLI script for ablations |

### Phase 5: Validation & Documentation (Week 5-6)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Run baseline experiments (E1-E8) | High | 5d | Baseline results |
| Generate publication-quality figures | Medium | 2d | Paper-ready visualizations |
| Write experimental results section | Medium | 2d | Results documentation |
| Update Makefile with new targets | Low | 0.5d | Integration |

### Total Estimated Effort: ~55 person-days (11 weeks for 1 person, 5.5 weeks for 2)

---

## Appendix A: Key References

1. Mazeika et al. "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming" (2024)
2. Debenedetti et al. "AgentDojo: A Dynamic Environment for Evaluating LLM Agents" (2024)
3. Mehrotra et al. "Tree of Attacks with Pruning (TAP)" (2023)
4. Wang et al. "PVDetector: Training-free Prompt Jailbreak Detection" (2026)
5. Lin et al. "ALERT: Zero-shot LLM Jailbreak Detection" (2026)
6. Lin et al. "Reflect-Guard: Augmenting Safety Classifiers with Self-Reflection" (2026)
7. Fang et al. "APD: Adversarial Prompt Disentanglement" (2026)
8. Hasan et al. "Sentra-Guard: Hybrid Safety Classifier" (2025)
9. Azov et al. "SALLIE: Mechanistic Interpretability for Safety" (2026)
10. Hassan et al. "BAGEL: Bootstrap Aggregation Ensemble" (2026)
11. Stepanov et al. "Opir: Encoder-based Guardrail Models" (2026)
12. Narisetty et al. "Adaptive Evaluation of Out-of-Band Defenses" (2026)

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **LoRA** | Low-Rank Adaptation |
| **FPR** | False Positive Rate |
| **FNR** | False Negative Rate |
| **ASR** | Attack Success Rate |
| **OOD** | Out-of-Distribution |
| **LOSO** | Leave-One-Source-Out |
| **LODO** | Leave-One-Dataset-Out |
| **CI** | Confidence Interval |
| **QPS** | Queries Per Second |
| **RAG** | Retrieval-Augmented Generation |

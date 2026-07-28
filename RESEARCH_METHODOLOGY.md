# Comparative Analysis of Prompt Guardrail Solutions
## Research Methodology & Execution Plan

---

## 1. Research Objectives

### 1.1 Primary Objective
Conduct a rigorous, reproducible comparative evaluation between a custom-built prompt guardrail application and established industry-standard prompt guardrail solutions across multiple performance dimensions.

### 1.2 Key Metrics for Comparison

| Category | Metric | Measurement Method | Target Precision |
|----------|--------|-------------------|-----------------|
| **Performance** | Latency (p50, p95, p99) | End-to-end request timing (ms) | ±0.1ms |
| **Accuracy** | False Positive Rate (FPR) | FP / (FP + TN) | ±0.001 |
| **Accuracy** | False Negative Rate (FNR) | FN / (FN + TP) | ±0.001 |
| **Accuracy** | F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | ±0.001 |
| **Robustness** | Jailbreak Resistance Rate | Successful blocks / total adversarial attempts | ±0.001 |
| **Robustness** | Adversarial Perturbation Tolerance | Accuracy degradation under input perturbation | ±0.001 |
| **Overhead** | Memory Footprint | Peak RSS during inference (MB) | ±1MB |
| **Overhead** | CPU/GPU Utilization | Average utilization during batch inference (%) | ±1% |
| **Integration** | Time-to-Integration | Hours from initial setup to production-ready | ±0.5hr |
| **Integration** | API Surface Complexity | Number of configuration parameters, LOC for basic setup | Qualitative |
| **Coverage** | Attack Surface Coverage | % of MITRE ATLAS / OWASP LLM attacks detected | ±0.01 |

### 1.3 Secondary Objectives
- Identify failure modes unique to each guardrail solution
- Document trade-offs between latency and detection accuracy
- Assess scalability behavior under increasing load (concurrent requests)
- Evaluate maintenance burden and extensibility for new threat categories

---

## 2. Selection Criteria for Benchmark Guardrail Solutions

### 2.1 Evaluation Framework

Each candidate solution is scored against the following criteria using a weighted rubric (0–5 scale):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Adoption & Community | 0.15 | GitHub stars, downloads, active contributors, documentation quality |
| Production Readiness | 0.25 | Deployment maturity, enterprise adoption, stability guarantees |
| Threat Coverage Breadth | 0.20 | Categories of attacks mitigated (PII, toxicity, injection, jailbreak) |
| Customizability | 0.20 | Ability to add custom rules, fine-tune models, configure rails |
| Licensing & Cost | 0.10 | Open-source vs. proprietary, per-request pricing, self-hosting options |
| Architectural Compatibility | 0.10 | Support for target deployment stack (API gateway, SDK, middleware) |

### 2.2 Candidate Solutions

| Solution | Type | License | Key Strength |
|----------|------|---------|--------------|
| **NVIDIA NeMo Guardrails** | Open-source framework | Apache 2.0 | Programmable guardrails with Colang DSL |
| **Meta Llama Guard (3)** | Open-source model | Llama 3 Community | LLM-based classifier, fine-tunable |
| **Guardrails AI (RAIL)** | Open-source framework | Apache 2.0 | Output validation, streaming support |
| **Azure AI Content Safety** | Proprietary SaaS | Commercial | Managed service, multi-modal |
| **Rebuff / Lakera Guard** | Open-source / Commercial | Various | Prompt injection detection focus |

### 2.3 Minimum Viable Comparison Set
At minimum, the study will include:
1. **NeMo Guardrails** — representative of rule-based/programmable guardrails
2. **Llama Guard 3** — representative of LLM-based classifiers
3. **The custom-built application** — the system under evaluation

Additional solutions will be included based on resource availability and reviewer feedback.

---

## 3. Experimental Design

### 3.1 Testing Environment

#### Infrastructure
```
Hardware:
  - CPU: AMD EPYC / Intel Xeon (≥16 cores)
  - RAM: ≥64 GB
  - GPU: NVIDIA A10G (24 GB VRAM) or equivalent [if LLM-based guardrails require local inference]
  - Storage: ≥500 GB NVMe SSD

Software:
  - OS: Ubuntu 22.04 LTS
  - Python: 3.10+
  - Containerization: Docker 24+, Docker Compose
  - Orchestration: Kubernetes (for scalability tests)
  - Monitoring: Prometheus + Grafana (latency, resource usage)
  - Experiment Tracking: MLflow or Weights & Biases
```

#### Isolation Controls
- Each guardrail solution runs in a dedicated Docker container with resource limits
- Network isolation between test harness and guardrail services
- Fixed random seeds for reproducibility
- Container-level CPU/memory cgroups for fair resource allocation

### 3.2 Datasets

#### 3.2.1 Benign Prompts (True Negative Set)
| Source | Size | Description |
|--------|------|-------------|
| LMSYS Chat-1M (sampled) | 5,000 | General conversational prompts |
| OpenHermes 2.5 (filtered) | 3,000 | Instruction-following prompts |
| Domain-Specific (medical, legal, finance) | 2,000 | Expert-domain queries |
| **Total** | **10,000** | |

#### 3.2.2 Adversarial Prompts (True Positive Sets)

| Attack Category | Source | Size | Examples |
|----------------|--------|------|----------|
| **Direct Jailbreaks** | JailbreakBench, HarmBench | 3,000 | DAN, role-play, hypothetical framing |
| **Prompt Injection** | PIQA, INJECTBench | 2,500 | Instruction override, context manipulation |
| **PII Extraction Attempts** | SynthPii + manual | 1,500 | Requests for SSN, email, phone, addresses |
| **Toxic/Harmful Content** | ToxiGen, RealToxicityPrompts | 3,000 | Hate speech, self-harm, violence |
| **Indirect Injection** | MaliciousRAG dataset | 1,000 | Hidden instructions in retrieved context |
| **Encoding-Based Evasion** | Custom (base64, rot13, leetspeak) | 1,000 | Obfuscated harmful prompts |
| **Multi-turn Attacks** | Custom multi-turn scenarios | 500 | Gradual escalation over conversation turns |
| **Total** | | **12,500** | |

#### 3.2.3 Edge Cases
- Empty prompts (50)
- Extremely long prompts >10K tokens (200)
- Multilingual adversarial prompts (500)
- Prompts with mixed benign/malicious intent (300)

### 3.3 Test Protocols

#### Protocol A: Classification Accuracy
```
For each guardrail solution G:
  For each prompt p in dataset D:
    1. Record binary classification: BLOCK / ALLOW
    2. Record confidence score (if available)
    3. Record response time (ms)
    4. Log any error/timeout
  Compute: TP, TN, FP, FN, F1, FPR, FNR, AUROC
```

#### Protocol B: Latency Under Load
```
For concurrency levels C ∈ {1, 5, 10, 25, 50, 100}:
  Send C simultaneous requests to guardrail G
  Record per-request latency
  Compute: p50, p95, p99, throughput (req/sec)
  Monitor: CPU%, memory, GPU% (if applicable)
```

#### Protocol C: Robustness Testing
```
For each adversarial transformation T ∈ {typos, encoding, paraphrase, translation}:
  Apply T to each adversarial prompt p
  Submit transformed prompt to guardrail G
  Measure: detection rate degradation relative to untransformed baseline
```

#### Protocol D: Integration Complexity
```
For each guardrail solution G:
  Record: time to first working integration (hours)
  Record: lines of configuration code
  Record: number of API calls required per check
  Record: dependency count and version conflicts
  Qualitative assessment: documentation quality (1-5), error messages clarity (1-5)
```

---

## 4. Comparative Framework

### 4.1 Quantitative Methods

| Method | Application |
|--------|-------------|
| **McNemar's Test** | Pairwise statistical significance of classification differences between solutions |
| **Cochran's Q Test** | Multi-solution comparison for blocked classification data |
| **Bootstrap Confidence Intervals** | 95% CI for all performance metrics (10,000 resamples) |
| **Bonferroni Correction** | Adjusted significance thresholds for multiple comparisons |
| **Effect Size (Cohen's h)** | Magnitude of performance differences between solutions |
| **Non-parametric Friedman Test** | Ranked performance across solutions under varying conditions |

### 4.2 Qualitative Methods

| Method | Application |
|--------|-------------|
| **Failure Mode Taxonomy** | Categorize and count failure types per solution |
| **Developer Experience Survey** | Structured questionnaire for integration engineers (SUS-adapted) |
| **Threat Model Mapping** | Map each solution's coverage to OWASP LLM Top 10 and MITRE ATLAS |
| **Maintenance Burden Scoring** | Assess update frequency, breaking change rate, deprecation patterns |

### 4.3 Scoring Framework

Each solution receives a **Composite Guardrail Score (CGS)**:

```
CGS = w₁·F1 + w₂·(1 - FPR) + w₃·Robustness + w₄·(1 - NormLatency) + w₅·(1 - NormOverhead) + w₆·IntegrationScore
```

Default weights (configurable based on deployment priorities):
- w₁ = 0.25 (accuracy)
- w₂ = 0.15 (low false positives)
- w₃ = 0.25 (robustness)
- w₄ = 0.15 (latency)
- w₅ = 0.10 (resource efficiency)
- w₆ = 0.10 (integration ease)

All component scores normalized to [0, 1].

---

## 5. Execution Roadmap

### Phase 1: Setup & Baseline (Weeks 1–2)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 1.1 Finalize solution selection | Selection matrix with scores | Research Lead |
| 1.2 Provision infrastructure | Running dev environment with monitoring | DevOps |
| 1.3 Curate and validate datasets | Labeled dataset with metadata | Data Engineer |
| 1.4 Implement test harness | Automated test runner with logging | Dev Engineer |
| 1.5 Baseline each guardrail solution | Warm-up runs, sanity checks | Research Lead |

### Phase 2: Core Experiments (Weeks 3–5)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 2.1 Protocol A: Classification accuracy runs | Raw classification logs | Dev Engineer |
| 2.2 Protocol B: Latency/load tests | Latency distribution CSVs | DevOps |
| 2.3 Protocol C: Robustness tests | Transformation attack results | Research Lead |
| 2.4 Protocol D: Integration assessment | Integration scorecards | Dev Engineer |
| 2.5 Edge case testing | Edge case result logs | Research Lead |

### Phase 3: Analysis & Synthesis (Weeks 6–7)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 3.1 Statistical analysis | Significance tests, CIs, effect sizes | Statistician |
| 3.2 Visualization | Performance comparison charts, radar plots | Research Lead |
| 3.3 Failure mode analysis | Taxonomy + examples per solution | Research Lead |
| 3.4 Composite scoring | CGS rankings with sensitivity analysis | Research Lead |

### Phase 4: Reporting & Review (Weeks 8–9)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 4.1 Draft research report | Full paper/manuscript | Research Lead |
| 4.2 Internal review | Annotated feedback document | All |
| 4.3 Revision and finalization | Final report + appendix | Research Lead |
| 4.4 Reproducibility package | Code, data, configs, README | Dev Engineer |

---

## 6. Reproducibility & Ethics

### 6.1 Reproducibility
- All experiments containerized with pinned dependency versions
- Dataset versions tracked with checksums (SHA-256)
- Random seeds explicitly set and documented
- Full experiment configurations stored in version control
- Results logged with experiment tracking tool (MLflow/W&B)

### 6.2 Ethical Considerations
- Adversarial datasets sourced from published academic benchmarks
- No real user data used in testing
- Harmful prompts evaluated only in isolated test environments
- Responsible disclosure if novel vulnerabilities discovered
- Study outcomes will not selectively disadvantage any vendor

---

## Appendix A: Directory Structure

```
research_project_benchmark/
├── RESEARCH_METHODOLOGY.md          # This document
├── datasets/
│   ├── benign/                       # True negative prompts
│   ├── adversarial/                  # Attack prompts by category
│   └── metadata.jsonl                # Dataset manifest with checksums
├── harness/
│   ├── runner.py                     # Main experiment runner
│   ├── metrics.py                    # Metric computation utilities
│   ├── load_test.py                  # Concurrent load testing
│   └── config.yaml                   # Experiment configuration
├── guardrails/
│   ├── custom/                       # Custom guardrail integration
│   ├── nemo/                         # NeMo Guardrails adapter
│   └── llama_guard/                  # Llama Guard adapter
├── results/
│   ├── raw/                          # Unprocessed experiment logs
│   ├── processed/                    # Aggregated metrics
│   └── figures/                      # Generated visualizations
├── analysis/
│   ├── statistical_tests.py          # Significance testing scripts
│   ├── scoring.py                    # CGS computation
│   └── visualization.py             # Plotting scripts
└── report/
    └── manuscript.md                 # Final research report
```

---

## Appendix B: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| API rate limiting during load tests | Medium | High | Use self-hosted/open-source variants; stagger requests |
| GPU memory exhaustion with Llama Guard | Medium | Medium | Implement batching; use quantized models (4-bit) |
| Dataset labeling errors | Low | High | Double-annotate 10% subset; inter-annotator agreement check |
| Solution version breaking changes | Medium | Medium | Pin all versions; document exact commit hashes |
| Insufficient statistical power | Low | High | Power analysis during Phase 1; increase dataset size if needed |

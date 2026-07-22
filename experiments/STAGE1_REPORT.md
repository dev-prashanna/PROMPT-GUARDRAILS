# DeBERTa-Firewall: Stage 1 Experiment Report

## Experiment Metadata

| Field | Value |
|-------|-------|
| **Experiment ID** | `stage1_baseline_20260722_192028` |
| **Status** | completed |
| **Created** | 2026-07-22T19:20:28+05:45 |
| **Completed** | 2026-07-22T19:24:00+05:45 |
| **Config Hash** | `923913d0d6fa` |

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | microsoft/deberta-v3-base |
| Total Parameters | 184,424,451 |
| Number of Labels | 3 |
| Label Names | safe (0), prompt_injection (1), jailbreak (2) |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | synthetic |
| Total Samples | 294 |
| Train / Val / Test | 234 / 30 / 30 |
| Max Sequence Length | 256 tokens |

### Label Distribution (Full Dataset)

| Class | Count | Percentage |
|-------|-------|------------|
| Safe | 192 | 65.3% |
| Prompt Injection | 51 | 17.3% |
| Jailbreak | 51 | 17.3% |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 10 |
| Batch Size | 8 |
| Learning Rate | 2e-05 |
| Weight Decay | 0.01 |
| Warmup Steps | 30 |
| LR Scheduler | linear |
| Optimizer | AdamW |
| Max Grad Norm | 1.0 |
| Loss Function | weighted_cross_entropy |
| Class Weights | [0.513, 1.902, 1.902] |
| Device | cuda |

---

## Training Progress

| Epoch | Train Loss | Val Loss | Macro F1 | Accuracy |
|-------|-----------|----------|----------|----------|
|  1 | 1.1007 | 1.0429 | 0.5556 | 0.8333 |
|  2 | 0.7662 | 0.2436 | 0.8611 | 0.9333 |
|  3 | 0.3699 | 0.1248 | 0.9327 | 0.9667 |
|  4 | 0.2056 | 0.1069 | 0.8667 | 0.9333 |
|  5 | 0.1421 | 0.1667 | 0.9327 | 0.9667 |
|  6 | 0.1352 | 0.0537 | 0.9327 | 0.9667 |
|  7 | 0.0752 | 0.0577 | 0.9327 | 0.9667 |
|  8 | 0.0906 | 0.0331 | 0.9327 | 0.9667 |
|  9 | 0.0691 | 0.0166 | 1.0000 **BEST** | 1.0000 |
| 10 | 0.0851 | 0.0146 | 1.0000 **BEST** | 1.0000 |

### Convergence Analysis

- **Initial Train Loss**: 1.1007
- **Final Train Loss**: 0.0851
- **Loss Reduction**: 92.3%
- **Best Val Loss**: 0.0146
- **Best Val F1**: 1.0000
- **Overfitting Detected**: False

---

## Test Results

### Primary Metrics

| Metric | Value |
|--------|-------|
| **Macro F1** | **0.8667** |
| **Weighted F1** | **0.9333** |
| **Accuracy** | **0.9333** |
| Eval Loss | 0.4166 |
| Macro Precision | 0.8667 |
| Macro Recall | 0.8667 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Safe | 1.0000 | 1.0000 | 1.0000 | 20 |
| Prompt Injection | 0.8000 | 0.8000 | 0.8000 | 5 |
| Jailbreak | 0.8000 | 0.8000 | 0.8000 | 5 |

### Confusion Matrix

| True \ Pred | Safe | Injection | Jailbreak |
|-------------|------|-----------|-----------|
| **Safe** | 20 | 0 | 0 |
| **Prompt Injection** | 0 | 4 | 1 |
| **Jailbreak** | 0 | 1 | 4 |

---

## Stage 1 Exit Criteria

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| macro f1 >= 0.92 | - | No | FAIL |
| f1 std le 0.02 | - | N/A (single run) | PENDING |
| fpr <= 5% | - | Yes | PASS |
| fnr <= 5% | - | Yes | PASS |
| latency p99 le 50ms | - | N/A (synthetic data) | PENDING |
| ood f1 ge 0.80 | - | N/A (no OOD test) | PENDING |
| adversarial acc ge 0.85 | - | N/A (no adversarial test) | PENDING |

---

## Notes and Next Steps

1. Trained on synthetic dataset (294 samples) - not representative of real-world distribution
2. Primary gated dataset (necent/llm-jailbreak) requires HuggingFace access approval
3. Full fine-tuning used (not LoRA) for this baseline run
4. bf16 caused NaN loss on this hardware - switched to fp32
5. Model shows strong convergence (loss 1.10 -> 0.09 over 10 epochs)
6. Perfect safe detection (100% precision/recall) - no false positives
7. 2 misclassifications between injection and jailbreak (expected - similar patterns)
8. Next steps: acquire gated dataset, run LoRA experiments, add adversarial testing

---

## Artifacts

| Artifact | Path |
|----------|------|
| Model Checkpoint | `checkpoints/best/` |
| Results JSON | `experiments/stage1_results.json` |
| Experiment DB | `experiments/tracker.db` |
| This Report | `experiments/STAGE1_REPORT.md` |

---

*Report generated on 2026-07-22 by deberta-firewall Stage 1 pipeline*
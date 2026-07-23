# DeBERTa-Firewall: LLM Prompt Injection & Jailbreak Detector

A lightweight, encoder-based classifier for detecting prompt injection and jailbreak attempts against LLMs. Uses DeBERTa-v3-base with LoRA PEFT for efficient fine-tuning on consumer GPUs.

## Model

| Property | Value |
|----------|-------|
| Base Model | `microsoft/deberta-v3-base` (86M params) |
| PEFT Method | LoRA (rank=16, alpha=32) |
| Trainable Params | 592,131 (0.32%) |
| Total Params | 185,016,582 |
| Classes | safe, prompt_injection, jailbreak |
| Precision | BF16 (tensor core accelerated) |

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.80% |
| **Macro F1** | 0.8908 |
| **Weighted F1** | 0.9202 |
| Safe F1 | 0.9421 |
| Prompt Injection F1 | 0.9426 |
| Jailbreak F1 | 0.7878 |

Best model selected at epoch 8 based on Macro F1.

## Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Effective Batch Size | 64 (batch=4, grad_accum=16) |
| Learning Rate | 2e-5 (cosine schedule) |
| Loss | Weighted Cross-Entropy + Label Smoothing |
| Optimizer | AdamW |
| GPU | RTX 4060 (8GB VRAM) |
| Training Time | ~3.8 hours |
| VRAM Usage | 1.25 GB (BF16 + gradient checkpointing) |

## Dataset

- **Training**: 21,751 samples (safe: 15,571 / injection: 3,090 / jailbreak: 3,090)
- **Validation**: 2,719 samples
- Sources: Anthropic HH-RLHF, Necent LLM-Jailbreak, ProtectAI RestrictedResponses

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=3
)
model = PeftModel.from_pretrained(model, "models/best")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

inputs = tokenizer("Your prompt here", return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1).item()
# 0=safe, 1=prompt_injection, 2=jailbreak
```

## Project Structure

```
deberta-firewall/
├── configs/           # YAML configuration files
├── data/              # Dataset splits and preprocessing
├── models/best/       # Best model adapter (LoRA weights)
├── src/               # Source code
│   ├── models/        # DeBERTa classifier + LoRA config
│   ├── training/      # Trainer, callbacks, loss functions
│   ├── data/          # Dataset, preprocessing, stage3 pipeline
│   ├── evaluation/    # Metrics, adversarial testing
│   └── observability/ # Experiment tracking, logging
├── scripts/           # Training and evaluation scripts
├── experiments/       # Results and experimental reports
├── tests/             # Unit tests
└── paper/             # Research paper drafts
```

## License

MIT License. See [LICENSE](LICENSE) for details.

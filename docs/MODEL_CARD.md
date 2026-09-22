# Model Card — DocuTune Fine-Tuned Resume Extractor

## Model

- **Name:** DocuTune adapter (LoRA) — applied on top of a base instruction model
- **Active base model (default):** `microsoft/Phi-3-mini-4k-instruct`
- **Base model revision:** recorded in `artifacts/training/training_manifest.json` and
  `results/benchmark_manifest.json` for every real run (never assumed)
- **Fine-tuning method:** QLoRA (4-bit NF4 quantized frozen base + trainable low-rank adapters)
- **LoRA configuration:** r=16, alpha=32, dropout=0.05, attention-only targets. The
  request `q_proj/k_proj/v_proj/o_proj` is resolved against the real architecture at
  runtime: on Phi-3 (fused attention) the adapter trains on `qkv_proj` + `o_proj`;
  on separate-attention architectures it trains on `q_proj, k_proj, v_proj, o_proj`.
  The resolved set is logged and recorded as `target_modules` in the training
  manifest.
- **Serving form:** base model + adapter (merged weights are optional; `scripts/merge_adapter.py`)

## Purpose

Extract a strict JSON record (name, email, phone, location, summary, skills, education,
experience, projects, certifications) from unstructured resume text. The purpose of the
project is a controlled measurement of how QLoRA fine-tuning changes extraction quality
against the untouched base model on a held-out benchmark.

## Dataset

- 600 fully synthetic resume examples (450/75/75 train/validation/test), seed 42
- Generated locally and deterministically; **no real personal data, no external LLM APIs**
- 12 rendering templates; validation/test use held-out templates (09–10 / 11–12)
- Difficulty levels easy/medium/hard
- See `docs/DATASET.md` and `data/dataset_manifest.json`

## Training

- Completion-only loss (prompt masked to -100; the model learns only the target JSON)
- Canonical prompt (PROMPT_VERSION=1.0) identical at training and inference time
- QLoRA: 4-bit NF4, double quantization, gradient checkpointing; fits a Colab T4 (16 GB)
- Full run manifest (versions, GPU, timing) is written to
  `artifacts/training/training_manifest.json` — only real, recorded values

## Evaluation

- Same held-out test set, prompt, decoding (greedy), parser, normalization and evaluator
  for base and fine-tuned models; only the model state differs
- Metrics: JSON validity, schema validity, exact match, field P/R/F1,
  unsupported-value rate, missing-value rate, latency, 95% bootstrap CIs
- Results, error analysis and representative examples in `results/`

## Intended Use

- Research/education: demonstrating a reproducible fine-tuning + evaluation pipeline
- Serving synthetic or clearly non-sensitive resume-like text through the included demo

## Non-Intended Use

- Production resume screening or any processing of real personal data
- Decisions about people (hiring, ranking, scoring)
- General-purpose information extraction beyond the trained schema
- Use as evidence of universal model quality (results are benchmark-scoped)

## Limitations & Known Failure Modes

- Synthetic-only training data → domain shift on real resumes is likely
- Small test set (75 default examples); CIs are wide — treated accordingly
- Known failure categories tracked by the error analysis: malformed JSON, schema
  violations, missing fields, wrong values, unsupported (invented) values,
  over/under-extraction, list-item errors, date-normalization errors,
  section-association errors
- Base-model behavior depends heavily on the chosen base model; adapters are
  base-model-specific and are not portable across models
- Latency on CPU is significant for a 3–4B model

## Licensing

- **Project code + synthetic dataset:** MIT (see `LICENSE`)
- **Base model:** governed by its own license on the Hugging Face model page. At the
  time of writing Phi-3-mini is MIT licensed — verify the current terms yourself;
  licenses change.
- **Fallback models:** `meta-llama/Llama-3.2-3B-Instruct` (Llama Community License,
  may require accepting terms / HF token) and `Qwen/Qwen2.5-3B-Instruct` (Qwen license).
  This project's license does not cover any of these models.

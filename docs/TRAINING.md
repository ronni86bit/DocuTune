# Training Documentation

## Why LoRA / QLoRA?

Full fine-tuning of a 3–4B parameter model needs GPU memory for weights, gradients and
optimizer states (tens of GB). **LoRA** freezes the base weights and trains small
low-rank matrices injected into attention projections — a few tens of MB of trainable
parameters. **QLoRA** additionally keeps the frozen base in 4-bit NF4 quantization
(double quantized), shrinking the memory footprint enough to train on a free Google
Colab T4 (16 GB). The deliverable is a tiny adapter artifact instead of a multi-GB model
copy, which also keeps this repository light.

## Configuration

Everything lives in `configs/train.yaml`:

```yaml
model:
  name: microsoft/Phi-3-mini-4k-instruct
  revision: null            # pinned per-run in the training manifest
quantization:               # QLoRA (auto-disabled with a warning on CPU)
  enabled: true
  bits: 4
  quant_type: nf4
  use_double_quant: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
training:
  epochs: 3
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.05
  max_length: 2048
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8     # effective batch 16
  gradient_checkpointing: true
  eval_strategy: steps / eval_steps: 25
  save_strategy: steps / save_steps: 25 / save_total_limit: 2
  logging_steps: 10
  seed: 42
data:
  train_file: data/splits/train.jsonl
  validation_file: data/splits/validation.jsonl
```

## Target-Module Safety

`docutune/training/model.py::verify_target_modules` inspects the actual model's Linear
submodules before constructing the LoRA. Unknown modules produce a hard error listing
the available names — the configuration never silently trains nothing.

## Loss Masking (completion-only)

`docutune/training/dataset.py::build_features` tokenizes the prompt
(`apply_chat_template` with the canonical prompt) and the target
(`serialize_target` + EOS) **separately**, concatenates ids, and masks all prompt tokens
to `-100`. Only the target JSON + EOS contribute to the loss. The model never learns to
recite the instructions.

## Running

```bash
python -m docutune.training.train --config configs/train.yaml          # full run
python -m docutune.training.train --config configs/train.yaml --resume # resume latest
python -m docutune.training.train --checkpoint artifacts/training/checkpoint-100
python -m docutune.training.train --smoke-test                          # sanity
python -m docutune.training.train --max-samples 16                      # debug
```

Pipeline steps executed: environment validation → seeds → tokenizer → CUDA/GPU
detection → quantization config → k-bit preparation → LoRA construction → target-module
verification → dataset tokenization (train/validation ONLY — the test split is never
read here) → training → validation evaluation → adapter save → tokenizer save → resolved
config save → loss history → training manifest.

## GPU Requirements

| Setup | Feasible? |
|---|---|
| Google Colab T4 (16 GB) | yes — the intended path, ~2–4 h for the default dataset |
| Local 24 GB GPU | yes |
| CPU | smoke test structure only; real QLoRA training is not practical |

`python scripts/check_environment.py` reports the actual environment (never fabricated):
Python, torch, CUDA availability/version, GPU name and VRAM.

## Checkpointing & Resume

- `save_strategy: steps` (every 25 steps) with `save_total_limit: 2`
- `--resume` finds the latest checkpoint in the output dir; `--checkpoint <path>`
  resumes from an explicit checkpoint
- Interrupted Colab sessions: re-run the training cell with `--resume` — completed
  checkpoints are reused, not overwritten

## Artifacts

- `artifacts/adapters/final/` — the LoRA adapter + tokenizer + `training_manifest.json`
  + `resolved_training_config.json` (this is what the Docker backend loads)
- `artifacts/training/` — checkpoints, `training_log.json` (loss history →
  `results/charts/training_loss.png`), `training_manifest.json`

The manifest records **actual** values only: model name/revision, dataset/prompt/schema
versions, sizes, hyperparameters, LoRA config, quantization, Python/torch/transformers/
peft/bitsandbytes/accelerate versions, CUDA version, GPU name/memory, start/end time.
No GPU information is ever invented; on CPU those fields are null/false.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `None of the requested LoRA target modules exist...` | different base model → update `lora.target_modules` (error lists valid names) |
| `Quantization requires bitsandbytes` | install training extras or set `quantization.enabled: false` (not QLoRA then) |
| CUDA OOM | reduce `max_length` or `per_device_train_batch_size`, or increase `gradient_accumulation_steps` |
| Colab disconnected | re-run with `--resume` (same output dir) |
| `data/splits/train.jsonl not found` | `python scripts/generate_data.py` first |
| Adapter missing for serving | copy `artifacts/adapters/final` from Colab; set `ADAPTER_PATH` |

## Training vs. Serving

Training happens **only** on Colab/local GPU via this module. The Docker serving stack
never trains, never downloads datasets, and only loads base model + adapter.

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

## Warmup: ratio in config, steps for transformers

Transformers 5.x removed `TrainingArguments(warmup_ratio=...)`. The YAML keeps
expressing warmup as a ratio; `docutune/training/schedule.py` converts it into an
exact `warmup_steps` value before `TrainingArguments` is constructed:

```
loader_steps_per_epoch = ceil(N_train / per_device_batch_size)
update_steps_per_epoch = ceil(loader_steps / gradient_accumulation_steps)
total_steps            = ceil(update_steps_per_epoch * epochs)   # or max_steps override
warmup_steps           = clamp(ceil(warmup_ratio * total_steps), 0, total_steps)
```

This is robust to any dataset size, batch size, gradient-accumulation, epoch count and
`max_steps` override (the smoke test uses 2 fixed steps → 1 warmup step). With the
default 450-example dataset (batch 2 × accum 8 × 3 epochs = 87 steps), `warmup_ratio:
0.05` becomes 5 warmup steps. The values are logged before training starts.

## Target Modules: architecture-aware resolution

`docutune/training/model.py::resolve_lora_target_modules` resolves the configured
attention projections against the ACTUAL model architecture:

| Requested (`configs/train.yaml`) | Architecture | Actually adapted |
|---|---|---|
| `q_proj, k_proj, v_proj, o_proj` | separate attention (Llama, Qwen, ...) | `q_proj, k_proj, v_proj, o_proj` |
| `q_proj, k_proj, v_proj, o_proj` | fused attention (Phi-3) | `qkv_proj, o_proj` |

- Phi-3 fuses the three input projections into one `qkv_proj` Linear; requesting the
  canonical trio maps onto it automatically (only a complete trio is aliased — partial
  requests fail rather than guess).
- **Nothing is silently skipped.** Any requested module that resolves neither exactly
  nor via alias aborts training with an error listing the model's real Linear names;
  an empty resolved set is impossible.
- The resolved set is logged before training and recorded as `target_modules` in
  `artifacts/training/training_manifest.json` (the raw request is kept as
  `target_modules_requested`).
- **Why not PEFT's `target_modules="all-linear"`?** Evaluated and deliberately not
  used: it would also adapt the MLP projections (gate/up/down), changing the
  trainable-parameter surface and thus the meaning of the experiment. Attention-only
  adapters are the intended design; the explicit request + alias table keeps that
  design while supporting both fused and separate attention architectures.

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
resolution → dataset tokenization (train/validation ONLY — the test split is never
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

## Checkpointing & Resume (hardened for remote runtimes)

**Periodic saving.** `save_strategy: steps` writes a full Trainer checkpoint every
25 optimizer steps (`save_steps: 25`, `save_total_limit: 2`). Each checkpoint
directory (`artifacts/training/checkpoint-<step>/`) contains adapter weights,
`optimizer.pt`, `scheduler.pt`, `trainer_state.json` (step counter + log history)
and RNG state — everything needed to continue optimizer/scheduler/model/Trainer
state exactly where the run stopped. For the reference 87-step run (~6 h 20 m on
a P100) that is a checkpoint roughly every 1 h 48 m: worst-case loss on a
disconnect stays under two hours while retained storage stays around a few
hundred MB. Checkpoints live under the **training output dir** and are never
mixed with the final adapter directory.

**Resume.**

```bash
python -m docutune.training.train --config configs/train.yaml --resume
python -m docutune.training.train --checkpoint artifacts/training/checkpoint-50
```

- `--resume` uses `docutune/training/checkpoints.py::find_latest_valid_checkpoint`
  to select the highest-step checkpoint that is **actually complete**
  (`trainer_state.json` parseable + `optimizer.pt` + `scheduler.pt` + adapter
  weights present). A directory killed mid-write is detected and skipped, so a
  crash can never poison the resume. No valid checkpoint exists → the run
  starts fresh (logged explicitly), which is the correct behavior on a first run.
- `--checkpoint <path>` resumes from an explicit directory and **fails loudly**
  if that directory is not a valid resume point.
- The checkpoint path actually resumed from is recorded in the training manifest
  (`resumed_from_checkpoint`).

Kaggle-specific instructions (secrets, upload, both recovery cases):
[docs/KAGGLE.md](KAGGLE.md).

## Optional adapter persistence to the Hugging Face Hub

On ephemeral runtimes the locally saved adapter dies with the runtime. After
saving and **verifying** the final adapter, the trainer can upload it to a
PRIVATE Hub repository. Controlled purely by environment variables:

| Variable | Effect |
|---|---|
| `DOCUTUNE_HF_REPO_ID` | setting it (e.g. `you/docutune-phi3-adapter`) **enables** auto-upload; unset = upload disabled, fully local training |
| `DOCUTUNE_HF_TOKEN` | write-enabled token; falls back to `HF_TOKEN` |

Behavior (`docutune/training/persistence.py`):

- the trainer verifies the saved adapter first (`adapter_config.json`, adapter
  weights, tokenizer files) — an incomplete adapter is **not** uploaded;
- the upload includes the adapter **plus its reproduction metadata**:
  `training_manifest.json` and `resolved_training_config.json` live in the same
  directory and are part of the upload;
- the repo is created **private** if it does not exist;
- outcome is logged and recorded in the training manifest
  (`artifact_upload.status`: `disabled` / `succeeded` / `failed`; a smoke test
  records `skipped (smoke test)`) — a failed upload is reported as failed, never
  as success, and never touches the locally saved adapter;
- the token is never printed; exception text is redacted before logging.

Local training requires **no** Hugging Face authentication.

## Intermediate validation evaluation (why it exists)

`eval_strategy: steps`, `eval_steps: 25` → evaluation runs right before each
checkpoint save (steps 25/50/75 in an 87-step run — the three ~6–7-minute
evaluations observed on the P100). These evaluations are **not required by the
project specification** (the original spec fixes `save_strategy`, not
`eval_strategy`); they were added as diagnostics. They are kept in the default
configuration because:

1. they are the only in-run overfitting signal during a long *unattended*
   remote run, and each checkpoint's `trainer_state.json` then contains a
   fresh validation loss at the resume point;
2. they are cheap relative to the run (~20 min of ~6 h 20 m, ≈5%) — evaluation
   performs no optimizer updates and consumes no training RNG (sequential eval
   sampler, model in eval mode), so they do not change the learned weights or
   the final held-out benchmark in any way;
3. removing them is a one-line config change (`eval_strategy: no` in
   `configs/train.yaml`) left to the operator; the final full validation
   evaluation recorded in the manifest and the held-out benchmark are
   unaffected either way.

No methodology or benchmark parameter was changed for speed.

## Artifacts

- `artifacts/adapters/final/` — the LoRA adapter + tokenizer + `training_manifest.json`
  + `resolved_training_config.json` (this is what the Docker backend loads)
- `artifacts/training/` — checkpoints, `training_log.json` (loss history →
  `results/charts/training_loss.png`), `training_manifest.json`

The manifest records **actual** values only: model name/revision, dataset/prompt/schema
versions, sizes, hyperparameters, LoRA config, quantization, Python/torch/transformers/
peft/bitsandbytes/accelerate versions, CUDA version, GPU name/memory, start/end time,
the checkpoint resumed from (if any) and the adapter upload status. No GPU
information is ever invented; on CPU those fields are null/false.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Requested LoRA target modules do not exist... Unresolved: [...]` | different base model → update `lora.target_modules` (error lists the model's real Linear names) |
| `TrainingArguments got an unexpected keyword argument 'warmup_ratio'` | transformers 5.x removed it — fixed in this repo (ratio → `warmup_steps` conversion); update to the latest code |
| `Quantization requires bitsandbytes` | install training extras or set `quantization.enabled: false` (not QLoRA then) |
| CUDA OOM | reduce `max_length` or `per_device_train_batch_size`, or increase `gradient_accumulation_steps` |
| Colab/Kaggle disconnected | re-run with `--resume` (same output dir) — resumes from the latest **valid** checkpoint |
| `Explicit checkpoint ... is not a valid resume point` | the named checkpoint is incomplete; use `--resume` to auto-discover the latest valid one |
| `Adapter upload FAILED: ...` | see the logged reason (token missing, repo permission, network). The local adapter is intact; fix and re-upload with `scripts/push_adapter.py`, or re-run training |
| `data/splits/train.jsonl not found` | `python scripts/generate_data.py` first |
| Adapter missing for serving | copy `artifacts/adapters/final` from the GPU runtime / HF repo; set `ADAPTER_PATH` |

## Training vs. Serving

Training happens **only** on Colab/local GPU via this module. The Docker serving stack
never trains, never downloads datasets, and only loads base model + adapter.

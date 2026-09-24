# Training on Kaggle (long-running GPU, ephemeral disk)

Kaggle GPU sessions are excellent free compute for DocuTune's QLoRA run, but
**everything under `/kaggle/working` is EPHEMERAL** - when the runtime resets
or the 9/12-hour session limit hits, the disk is wiped. A first real run
(87 steps, ~6 h 20 m on a P100) produced a correct final adapter that was then
lost exactly this way: the adapter upload failed under the wrong repo
namespace, the runtime reset afterwards, and the disk was gone. This document
describes the hardened workflow that makes a Kaggle run recoverable at every
point.

The pipeline (not the notebook) is responsible for persistence: you never
need to hand-copy files mid-run.

## Three separate persistence layers

| Layer | Survives a runtime reset? | Mechanism |
|---|---|---|
| Local checkpoints (`artifacts/training/checkpoint-<step>/`) | **No** - dies with the runtime | Trainer's `save_strategy: steps` (every 25 optimizer steps, last 2 kept) |
| **Remote checkpoint mirrors** (`checkpoints/checkpoint-<step>/` in the HF repo) | **Yes** | Each checkpoint is uploaded to the private Hub repo right after it is written (`DOCUTUNE_HF_REPO_ID` set) |
| Final adapter (`artifacts/adapters/final/` → repo root) | **Yes** | Verified + uploaded once training completes |

Kaggle itself does NOT make `/kaggle/working` durable across runtime
termination - only the two Hub-backed layers survive a reset.

## 0. One-time setup: Hugging Face token and repository

1. Create a write-enabled token: <https://huggingface.co/settings/tokens>
   (scope: *write*; you can delete it after the run).
2. Add it to Kaggle Secrets (Add-ons → Secrets) under the name
   `DOCUTUNE_HF_TOKEN`, or as `HF_TOKEN` (both are accepted; see
   [Environment variables](#environment-variables)).
3. The target repository (`rohithronni/docutune-phi3-adapter` for this
   project) does NOT need to be created by hand - the trainer creates it
   (always **private**) on the first checkpoint upload if it does not exist.
   A repo created manually must be private and the token must have write
   access to it.

Never paste tokens into notebook code, and never commit them to Git.

## 1. The workflow

```python
# ---- Cell 1: environment check ------------------------------------------
!nvidia-smi
import sys, platform; print(platform.python_version())

# ---- Cell 2: clone the repo and install ----------------------------------
REPO_URL = "https://github.com/ronni86bit/DocuTune.git"
%cd /kaggle/working
!git clone {REPO_URL}
%cd /kaggle/working/DocuTune
!pip install -q -e ".[training]"

# ---- Cell 3: expose the HF credentials from Kaggle Secrets ---------------
from kaggle_secrets import UserSecretsClient
import os
os.environ["DOCUTUNE_HF_TOKEN"] = UserSecretsClient().get_secret("DOCUTUNE_HF_TOKEN")
os.environ["DOCUTUNE_HF_REPO_ID"] = "rohithronni/docutune-phi3-adapter"  # <-- your repo

# ---- Cell 4: dataset (deterministic; identical to local runs) ------------
!python scripts/generate_data.py --seed 42
!python scripts/validate_data.py

# ---- Cell 5: smoke test --------------------------------------------------
!python -m docutune.training.train --smoke-test

# ---- Cell 6: baseline benchmark (optional here; ~20-40 min) --------------
!python scripts/run_baseline.py

# ---- Cell 7: THE TRAINING RUN -------------------------------------------
!python -m docutune.training.train --config configs/train.yaml --resume
```

Setting `DOCUTUNE_HF_REPO_ID` in cell 3 enables BOTH Hub-backed layers:
remote checkpoint mirroring during training and the final-adapter upload
after training. Without it, everything stays local (and therefore dies with
the runtime).

What happens during cell 7:

- **Every 25 optimizer steps** the Trainer writes a local checkpoint to
  `artifacts/training/checkpoint-<step>/` (last 2 kept locally). Each
  checkpoint contains adapter weights, `optimizer.pt`, `scheduler.pt`,
  `trainer_state.json` and RNG state - everything needed to continue
  exactly where the run stopped.
- **Immediately after each local save**, that checkpoint is mirrored to the
  Hub as ONE atomic commit under `checkpoints/checkpoint-<step>/`, including
  a `REMOTE_CHECKPOINT_COMPLETE.json` marker with per-file sizes + sha256.
  The log shows the lifecycle explicitly:
  `Checkpoint 25 saved locally` → `Checkpoint 25 upload: 5 files, ~115 MB` →
  `Checkpoint 25 upload verified - persistence SUCCEEDED` (or
  `... persistence FAILED: <reason>` - training continues either way).
- **The final adapter** is saved to `artifacts/adapters/final/` together
  with `training_manifest.json` and `resolved_training_config.json`,
  verified, and uploaded to the repo root (unchanged behavior). The upload
  status is recorded in the manifest.
- **Smoke tests never upload anything** (checkpointing is disabled and the
  adapter upload is skipped).

## 2. Remote repository layout

```
<your-private-repo>/
├── adapter_model.safetensors        # final adapter + tokenizer + manifest
├── adapter_config.json              # + resolved config at the ROOT
├── tokenizer.json                   #   (uploaded once, after training)
├── training_manifest.json
├── resolved_training_config.json
└── checkpoints/
    ├── checkpoint-25/               # exact copy of the local Trainer
    │   ├── optimizer.pt             # checkpoint (~115 MB total: ~38 MB
    │   ├── scheduler.pt             # fp32 adapter weights + ~76 MB fp32
    │   ├── trainer_state.json       # AdamW moments + <1 MB state files;
    │   ├── ...                      # the frozen 4-bit base model is
    │   └── REMOTE_CHECKPOINT_COMPLETE.json   # NEVER included)
    ├── checkpoint-50/
    └── checkpoint-75/
```

- Every checkpoint lives in its own per-step directory (a newer checkpoint
  can never overwrite an older one).
- The marker is what makes a checkpoint selectable for resume; it is part of
  the same atomic commit, so a partially uploaded checkpoint can never be
  mistaken for a complete one.
- With `save_total_limit: 2` only the newest 2 checkpoints stay on the
  Kaggle disk, but ALL mirrored checkpoints are kept remotely - each one is
  an independent recovery point (~350 MB total for the full 87-step run).

## 3. Recovery procedures

### Case A - Kaggle disconnected AFTER the final upload succeeded

Nothing to do. Pull the adapter onto any machine:

```bash
huggingface-cli download rohithronni/docutune-phi3-adapter \
  --local-dir artifacts/adapters/final
```

Then benchmark and serve exactly as documented in the README.

### Case B - Kaggle disconnected mid-training (checkpoints were mirrored)

1. Start a fresh runtime and re-run cells 1-4 (clone, install, secrets,
   dataset - the dataset is deterministic, so it is byte-identical to the
   interrupted run).
2. Re-run cell 7 (the command already includes `--resume`). Resume
   precedence:
   1. explicit `--checkpoint <path>` (strict - an invalid path fails loudly);
   2. latest **valid local** checkpoint in `artifacts/training/`;
   3. latest **valid remote** checkpoint from the Hub (only when no valid
      local checkpoint exists - i.e. exactly the fresh-runtime case);
   4. fresh training, logged clearly.
3. A remote checkpoint is only selected when it survives validation: usable
   marker, all marker-listed files present remotely, matching config
   fingerprint, and after download the sha256 hashes + the standard resume
   rules re-verify. A broken newest checkpoint falls back to the newest
   valid older one; none valid → fresh run (the seed makes it equivalent).
4. The downloaded mirror is materialized as a normal local checkpoint
   (`artifacts/training/checkpoint-<step>/`), so Trainer resumes optimizer,
   scheduler, RNG and step counter exactly where the run stopped. The
   manifest records `resume_source: remote` plus the resumed step.

### Case C - the remote mirror itself is unusable

Symptoms: `No valid remote checkpoint ... training from scratch`, or
`Remote checkpoint discovery failed: ...`. Options:

- Transient Hub/network error → just re-run the cell (fails within seconds
  at startup, before any model download).
- Want local-only resume (e.g. the Hub is down but a local checkpoint
  exists)? Run once with `DOCUTUNE_HF_REPO_ID` unset:
  `DOCUTUNE_HF_REPO_ID= python -m docutune.training.train --config configs/train.yaml --resume`
- Manual inspection: the Hub web UI shows `checkpoints/` with each step and
  its marker. To pull a specific checkpoint by hand:

  ```bash
  huggingface-cli download rohithronni/docutune-phi3-adapter \
    --include "checkpoints/checkpoint-50/*" \
    --local-dir artifacts/training-manual
  # then point --checkpoint at the extracted directory (validated strictly)
  ```

### What a failed checkpoint upload does (and does not do)

- The local checkpoint is NEVER modified or deleted by a failed upload.
- Training NEVER aborts: the failure is logged
  (`Checkpoint 25 persistence FAILED: <sanitized reason>`) and recorded in
  the manifest (`checkpoint_persistence[].upload_status: "failed"`).
- Uploads are retried up to 3 times with exponential backoff (2 s → 4 s → 8 s,
  capped at 30 s) - bounded, no infinite loop.
- Success is never reported falsely: a checkpoint counts as persisted only
  after the Hub commit completed AND the marker + all files were verified
  remotely. An already-persisted step is skipped, not re-uploaded.

## 4. Environment variables

| Variable | Meaning | Required |
|---|---|---|
| `DOCUTUNE_HF_REPO_ID` | e.g. `rohithronni/docutune-phi3-adapter`. Setting it **enables** remote checkpoint mirroring AND the final-adapter upload. Unset = fully local training, no Hub writes. | for persistence |
| `DOCUTUNE_HF_TOKEN` | write-enabled token for that repo | for persistence |
| `HF_TOKEN` | fallback token if `DOCUTUNE_HF_TOKEN` is unset | no |

Without `DOCUTUNE_HF_REPO_ID` the trainer logs
`Remote checkpoint persistence DISABLED (...)` and trains purely locally -
exactly as before this feature existed. Ordinary local training never
requires Hugging Face authentication.

## 5. Security guidance

- Store the token in Kaggle Secrets, never in notebook cells, config files
  or Git. `.env` is gitignored; the repository contains only
  `.env.example` placeholders.
- The trainer never prints the token; exception messages are redacted before
  they are logged or written into the training manifest, and the manifest
  records only the repo id - never credentials.
- The target repository is always created **private**, and checkpoints are
  stored under the same private repo as the final adapter.
- Revoke the token after the run if it was created just for this purpose.

## 6. Why every 25 steps is the right mirror cadence

Checkpoints are saved every **25 optimizer steps** (`save_steps: 25`), with
`save_total_limit: 2` locally. The reference run (87 steps, ~6 h 20 m on a
P100) means a checkpoint roughly every 1 h 48 m - worst case you lose under
two hours of compute on a disconnect. Each QLoRA checkpoint is ~115 MB
(adapter weights fp32 ~38 MB + AdamW optimizer moments fp32 ~76 MB + <1 MB
scheduler/Trainer/RNG state; the base model is frozen 4-bit and never
stored), so three mirrored checkpoints plus the final adapter are well under
half a GB - negligible for HF storage and Kaggle's ~20 GB working disk.
`save_steps` is plain infrastructure configuration in `configs/train.yaml`
and does not affect the experiment.

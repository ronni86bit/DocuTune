# Training on Kaggle (long-running GPU, ephemeral disk)

Kaggle GPU sessions are excellent free compute for DocuTune's QLoRA run, but
**everything under `/kaggle/working` is EPHEMERAL** - when the runtime resets
or the 9/12-hour session limit hits, the disk is wiped. A first real run
(87 steps, ~6 h 20 m on a P100) produced a correct adapter that was then lost
exactly this way. This document describes the hardened workflow that makes a
Kaggle run recoverable at every point.

The pipeline (not the notebook) is responsible for persistence: you never
need to hand-copy files mid-run.

## 0. One-time setup: Hugging Face token

1. Create a write-enabled token: <https://huggingface.co/settings/tokens>
   (scope: *write*; you can delete it after the run).
2. Add it to Kaggle Secrets (Add-ons → Secrets) under the name
   `DOCUTUNE_HF_TOKEN`, or as `HF_TOKEN` (both are accepted; see
   [Environment variables](#environment-variables)).

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

# ---- Cell 3: expose the HF token from Kaggle Secrets (optional upload) ---
from kaggle_secrets import UserSecretsClient
import os
os.environ["DOCUTUNE_HF_TOKEN"] = UserSecretsClient().get_secret("DOCUTUNE_HF_TOKEN")
os.environ["DOCUTUNE_HF_REPO_ID"] = "YOUR_USERNAME/docutune-phi3-adapter"  # <-- change

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

What happens during cell 7:

- **Periodic checkpoints** are written to
  `/kaggle/working/DocuTune/artifacts/training/checkpoint-<step>/`
  (every 25 optimizer steps; the last 2 are kept). Each checkpoint contains
  adapter weights, optimizer state, scheduler state and full Trainer state -
  everything needed to continue exactly where the run stopped.
- **The final adapter** is saved to
  `artifacts/adapters/final/` together with `training_manifest.json` and
  `resolved_training_config.json`.
- **If upload is configured**, the trainer verifies the saved adapter
  (adapter config + weights + tokenizer present) and uploads the whole
  directory to your **private** Hugging Face repo, then logs
  `Adapter upload SUCCEEDED: https://huggingface.co/...` and records the
  status in the training manifest. Failures are logged as
  `Adapter upload FAILED: <reason>` - the local adapter is never modified or
  deleted by a failed upload, and success is never reported falsely.

## 2. Recovery procedures

### Case A - Kaggle disconnected AFTER the upload succeeded

Nothing to do. The adapter (plus manifest + resolved config) is on the Hub.
Pull it onto any machine:

```bash
huggingface-cli download YOUR_USERNAME/docutune-phi3-adapter \
  --local-dir artifacts/adapters/final
```

Then benchmark and serve exactly as documented in the README.

### Case B - Kaggle disconnected BEFORE finalization

1. Re-run cells 1-4 (clone, install, secrets, dataset - the dataset is
   deterministic, so it is byte-identical to the interrupted run).
2. Re-run cell 7 with `--resume` (already shown). The trainer scans
   `artifacts/training/` and continues from the **latest VALID checkpoint**
   - a checkpoint directory missing `optimizer.pt` / `scheduler.pt` /
   `trainer_state.json` / adapter weights (i.e. killed mid-write) is
   detected and skipped automatically.
3. No valid checkpoint? Then the run restarts from scratch - the same
   seed produces the same schedule, so the new run is equivalent.

Notes:

- `/kaggle/working` is wiped on reset. To keep checkpoints across a reset
  you would have to enable Kaggle's "save output" snapshots or push
  checkpoints to the Hub yourself; the default workflow relies on within-
  session resume (case B) and after-upload safety (case A).
- Keep Kaggle's "Persistence" setting for `/kaggle/working` in mind: files
  survive *while the session is alive*, not across resets.

## 3. Environment variables

| Variable | Meaning | Required |
|---|---|---|
| `DOCUTUNE_HF_REPO_ID` | e.g. `yourname/docutune-phi3-adapter`. Setting it **enables** auto-upload after training. Unset = fully local training, upload disabled. | for upload |
| `DOCUTUNE_HF_TOKEN` | write-enabled token for that repo | for upload |
| `HF_TOKEN` | fallback token if `DOCUTUNE_HF_TOKEN` is unset | no |

Without `DOCUTUNE_HF_REPO_ID` the trainer logs
`Adapter upload DISABLED (...)` and simply leaves the adapter in
`artifacts/adapters/final`. Training never requires Hugging Face
authentication.

## 4. Security guidance

- Store the token in Kaggle Secrets, never in notebook cells, config files
  or Git. `.env` is gitignored; the repository contains only
  `.env.example` placeholders.
- The trainer never prints the token; exception messages are redacted
  before they are logged or written into the training manifest.
- The auto-upload target repository is always created **private**.
- Revoke the token after the run if it was created just for this purpose.

## 5. Checkpoint frequency rationale

Checkpoints are saved every **25 optimizer steps** (`save_steps: 25`), with
`save_total_limit: 2`. The reference run (87 steps, ~6 h 20 m on a P100)
means a checkpoint roughly every 1 h 48 m - worst case you lose under two
hours of compute on a disconnect. Each QLoRA checkpoint (adapter + AdamW
optimizer state for the adapter only) is on the order of a few hundred MB,
so two retained checkpoints stay well inside Kaggle's ~20 GB working-disk
budget; more frequent saving would double the I/O pauses for little gain.
`save_steps` is plain infrastructure configuration in `configs/train.yaml`
and does not affect the experiment.

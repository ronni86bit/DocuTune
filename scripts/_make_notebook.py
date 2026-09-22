"""One-off generator for notebooks/DocuTune_Training.ipynb (dev tool)."""

import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# DocuTune Training - Google Colab GPU Workflow

This notebook is the **official GPU training workflow** for DocuTune. Colab provides the
COMPUTE; all project logic lives in the repository (nothing important is duplicated here).

**Runtime -> Change runtime type -> T4 GPU** before running.

Pipeline: environment check -> install -> get repo -> generate + validate dataset ->
baseline benchmark -> smoke test -> QLoRA fine-tune -> fine-tuned benchmark ->
full before/after artifacts -> export the adapter."""),
    code("""# 1. Environment check: Python, CUDA, GPU name + memory
import platform

print('Python:', platform.python_version())
try:
    import torch
    print('torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f'GPU memory: {props.total_memory / 1024**3:.1f} GB')
    else:
        print('WARNING: no GPU - Runtime > Change runtime type > T4 GPU')
except ImportError:
    print('torch not installed yet (cell 2 installs it)')
"""),
    code("""# 2. Install compatible dependencies
# The Colab torch/CUDA build is kept; training extras are added on top.
%pip install -q -U "transformers>=4.51,<6" "peft>=0.14,<0.22" "accelerate>=1.0,<2" \\\\
    "bitsandbytes>=0.45,<0.51" matplotlib
"""),
    code("""# 3. Get the repository (option A: clone from GitHub; option B: upload a zip)
import pathlib

REPO_URL = 'https://github.com/YOUR_USERNAME/DocuTune.git'  # <-- change me
REPO_DIR = '/content/DocuTune'

if not pathlib.Path(REPO_DIR).exists():
    !git clone {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}

# The package + training deps must be importable from the repo checkout.
%pip install -q -e ".[training,serving,eval]"
"""),
    code("""# 4. Generate the deterministic synthetic dataset (seed 42)
!python scripts/generate_data.py --seed 42
!python scripts/validate_data.py
"""),
    code("""# 5. Environment report from the project's own checker
!python scripts/check_environment.py
"""),
    code("""# 6. Baseline benchmark (untouched base model on the held-out test set)
# ~75 greedy generations on a T4: roughly 20-40 minutes.
!python scripts/run_baseline.py
"""),
    code("""# 7. Training smoke test (validates the full training stack on a tiny subset)
!python -m docutune.training.train --smoke-test
"""),
    code("""# 8. Full QLoRA fine-tuning (~2-4 h on a T4 for the default 450/75/75 dataset)
# Resumable: if Colab disconnects, re-run this cell - it continues from the
# latest checkpoint via --resume.
!python -m docutune.training.train --config configs/train.yaml --resume
"""),
    code("""# 9. Inspect the training manifest (recorded, not fabricated)
import json

manifest = json.load(open('artifacts/training/training_manifest.json'))
for key in ('model_name', 'model_revision', 'train_size', 'validation_size', 'epochs',
            'learning_rate', 'lora_r', 'lora_alpha', 'target_modules', 'quantization',
            'gpu_name', 'gpu_memory', 'torch_version', 'peft_version', 'duration_seconds'):
    print(f'{key:>24}: {manifest.get(key)}')
"""),
    code("""# 10. Fine-tuned benchmark (base model + trained LoRA adapter)
!python scripts/run_finetuned.py
"""),
    code("""# 11. Full before/after benchmark: metrics, bootstrap CIs, error analysis,
# charts, reports, and the README metric-table update
!python scripts/benchmark.py
"""),
    code("""# 12. Show the benchmark summary
import json

metrics = json.load(open('results/metrics.json'))
for name in ('base', 'finetuned'):
    print(f'== {name} ==')
    for key in ('json_validity', 'schema_validity', 'exact_match', 'field_f1',
                'unsupported_value_rate', 'latency_mean_ms'):
        print(f'  {key:>24}: {metrics[name].get(key)}')
print('== delta ==', json.dumps(metrics['delta'], indent=2))
"""),
    code("""# 13. Export the LoRA adapter (+ manifests) for use with the Docker backend
import shutil

shutil.make_archive('/content/docutune_adapter', 'zip', 'artifacts/adapters/final')
print('Adapter archive: /content/docutune_adapter.zip')
print('Unzip it into ./artifacts/adapters/final on the machine running docker compose.')
try:
    from google.colab import files
    files.download('/content/docutune_adapter.zip')
except ImportError:
    pass
"""),
    md("""## Persisting artifacts (optional)

Colab disks are ephemeral. To keep results, copy the `artifacts/` and `results/`
folders to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/DocuTune
!cp -r artifacts results /content/drive/MyDrive/DocuTune/
```

To publish the adapter to the Hugging Face Hub (optional, explicit):

```
%pip install -q huggingface_hub
%env HF_TOKEN=hf_xxx
!python scripts/push_adapter.py --repo-id YOUR_USERNAME/docutune-adapter
```
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/DocuTune_Training.ipynb", "w", encoding="utf-8", newline="\n") as fh:
    json.dump(notebook, fh, ensure_ascii=False, indent=1)
print("notebook written:", sum(1 for c in cells if c["cell_type"] == "code"), "code cells")

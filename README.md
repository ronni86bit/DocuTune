# DocuTune

**Fine-tuned structured resume extraction with reproducible before/after evaluation.**

DocuTune takes messy, unstructured resume text and returns validated structured JSON — using a small open-source instruction model (default: `microsoft/Phi-3-mini-4k-instruct`), fine-tuned with LoRA/QLoRA and compared against the untouched base model on the exact same held-out test set. The repository contains the complete pipeline: deterministic synthetic dataset, training code, a shared evaluator, an analysis of failures, a FastAPI backend, a React dashboard, Docker deployment, and the Google Colab training workflow.

> **Honesty policy:** every number in this README is produced by `scripts/benchmark.py` from a real run. Until that run happens, the tables below say **TBD**. No metric on this page is hand-written.

---

## Benchmark Results

Measured on the held-out synthetic test set (75 examples from rendering templates the model never saw in training). Identical prompt, decoding, parser, normalization and evaluator for both models — only the model state differs.

<!-- BENCHMARK-TABLE:START -->
| Metric | Base | Fine-Tuned | Delta |
|---|---:|---:|---:|
| JSON Validity | TBD | TBD | TBD |
| Schema Validity | TBD | TBD | TBD |
| Exact Match | TBD | TBD | TBD |
| Field Precision | TBD | TBD | TBD |
| Field Recall | TBD | TBD | TBD |
| Field F1 | TBD | TBD | TBD |
| Unsupported Value Rate | TBD | TBD | TBD |
| Mean Latency | TBD | TBD | TBD |
<!-- BENCHMARK-TABLE:END -->

Per-field metrics, bootstrap confidence intervals, error categories and representative examples (failures included) live in `results/` after a benchmark run — see [`results/benchmark_report.md`](results/benchmark_report.md) and [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md).

## Resume Bullet — Generated from Actual Results

<!-- RESUME-BULLET:START -->
TBD
<!-- RESUME-BULLET:END -->

---

## Architecture

```
                        ┌────────────────────────────┐
                        │   React + Vite + TypeScript │
                        │   Extraction · Benchmark ·  │
                        │   Error Analysis · About    │
                        └──────────────┬──────────────┘
                                       │ same-origin /api (Nginx proxy)
                                       ▼
                        ┌────────────────────────────┐
                        │        FastAPI backend      │
                        │  /api/health /api/metadata  │
                        │  /api/extract /api/compare  │
                        │  /api/metrics /api/benchmark│
                        └──────────────┬──────────────┘
                                       ▼
                        ┌────────────────────────────┐
                        │      Inference service      │
                        │  base model + LoRA adapter  │
                        │  (loaded once, cached)      │
                        └──────────────┬──────────────┘
                                       ▼
                                 structured JSON

TRAINING (offline, Google Colab GPU — never inside Docker):
  synthetic canonical records
        → rendered resume text (12 templates)
        → train / validation / test split (held-out templates)
        → baseline evaluation of the untouched base model
        → QLoRA fine-tuning (4-bit NF4 + LoRA r=16)
        → LoRA adapter (artifact)
        → fine-tuned evaluation on the SAME test set
        → before/after benchmark, bootstrap CIs, error analysis
```

## The Experiment

**"I fine-tuned a small open-source instruction model for structured resume extraction using LoRA/QLoRA and compared it against the untouched base model on the exact same held-out test set."**

- **Why a baseline?** Without evaluating the untouched model first, you cannot know what fine-tuning actually changed. Both models receive identical inputs and settings; only the weights differ.
- **Why a held-out test set?** The test split uses rendering templates 11–12 which never appear in training. This measures whether the model learned the *task* rather than the surface formatting of the training templates.
- **Why identical prompts?** Changing prompts between evaluations would confound the comparison. One canonical prompt (`docutune/inference/prompt.py`, `PROMPT_VERSION=1.0`) is shared by training, baseline, fine-tuned and API inference.
- **Why schema validity?** A structured-output model is only useful if its output parses and satisfies the schema. It is the most basic reliability metric.
- **Why field-level F1?** Aggregate scores hide which fields fail. Per-field precision/recall/F1 show exactly where extraction breaks.
- **Why unsupported-value rate?** Invented values are the classic LLM failure. Because the dataset is synthetic, the canonical record is known exactly, making this metric well-defined (it is a benchmark-specific proxy for hallucination, *not* a universal measure).
- **Why synthetic data is a limitation:** synthetic text has synthetic language bias. Results on this benchmark do not automatically transfer to real resumes (see [Limitations](#limitations)).

## Task Definition

Unstructured resume text in, one JSON object out:

```json
{
  "name": "John Doe",
  "email": null,
  "phone": null,
  "location": "Hyderabad, Telangana",
  "summary": "Machine Learning Engineer with 3 years of experience.",
  "skills": ["Python", "PyTorch", "SQL", "Docker"],
  "education": [
    {"degree": "B.Tech", "institution": "XYZ Institute", "field": "Computer Science",
     "start_year": 2018, "end_year": 2022, "grade": null}
  ],
  "experience": [
    {"title": "Machine Learning Engineer", "company": "Nova Analytics", "location": null,
     "start_date": "2022-06", "end_date": null, "current": true,
     "responsibilities": ["Worked on NLP pipelines", "Worked on transformer deployment"]}
  ],
  "projects": [],
  "certifications": []
}
```

Missing scalars are `null`, missing lists are `[]`, experience dates are normalized to `YYYY-MM` (or `YYYY` when only a year is given). The schema is enforced with Pydantic (`docutune/schema/resume.py`, `SCHEMA_VERSION=1.0`).

## Dataset

600 fully synthetic, deterministically generated examples (seed 42) — **no real people's resumes, no external LLM APIs, no personal data**:

| Split | Count | Rendering templates |
|---|---:|---|
| train | 450 | templates 01–08 |
| validation | 75 | templates 09–10 |
| test (held out) | 75 | templates 11–12 |

Canonical structured records are generated first, then rendered into messy text by 12 distinct templates (varying section order, headings, date formats, bullet styles, whitespace noise, paragraph vs. bullets, fragmented lines). Difficulty is labeled `easy` / `medium` / `hard`. The validator (`python scripts/validate_data.py`) checks schema validity, unique IDs, exact-text duplicates, structured-record leakage across splits, template/split integrity and distributions. Details: [`docs/DATASET.md`](docs/DATASET.md).

## Fine-Tuning

- **Method:** QLoRA — frozen base in 4-bit NF4 (double quant), trainable LoRA adapters on `q_proj/k_proj/v_proj/o_proj` (verified against the actual architecture before training starts; training fails with an informative listing if modules don't exist).
- **Config:** `r=16, alpha=32, dropout=0.05`, 3 epochs, LR 2e-4, effective batch 16 (2 × 8 accumulation), max_length 2048, warmup_ratio 0.05 (→ warmup_steps computed automatically) — all in `configs/train.yaml`.
- **Loss:** completion-only — prompt tokens are masked to `-100`; the model learns only the target JSON.
- **Output:** a LoRA adapter (`artifacts/adapters/final`), a tokenizer copy, a resolved config, a loss history and a full training manifest with real versions (model revision, CUDA/GPU info, library versions, timing). See [`docs/TRAINING.md`](docs/TRAINING.md).
- **Checkpointing & recovery:** checkpoints every 25 optimizer steps under `artifacts/training/`; `--resume` continues from the latest **valid** checkpoint (incomplete crash-truncated checkpoints are skipped); `--checkpoint <path>` validates an explicit resume point.
- **Remote-run persistence (optional):** set `DOCUTUNE_HF_REPO_ID` (+ `DOCUTUNE_HF_TOKEN`, or `HF_TOKEN`) and the trainer verifies the saved adapter and uploads it — with its training manifest and resolved config — to a **private** Hugging Face repo after training. Disabled by default; local training needs no HF authentication; failed uploads are reported honestly and never touch the local adapter. See [docs/KAGGLE.md](docs/KAGGLE.md) for the full Kaggle workflow.

## Evaluation

One shared, model-agnostic evaluator (`docutune/evaluation/`): robust JSON/code-fence parsing (no LLM repair, no semantic fixing), conservative documented normalization, and metrics:

- JSON validity · schema validity · exact record match
- Field-level precision / recall / F1 (micro + per-field, item-level matching for nested lists)
- Unsupported-value rate · missing-value rate
- Latency (mean / p50 / p95, warm model, warm-up excluded, device recorded)
- 95% bootstrap confidence intervals (seeded, this benchmark only)

Strict fairness rules are enforced in code and documented in [`docs/EVALUATION.md`](docs/EVALUATION.md): same test set, same prompt, same decoding (`do_sample=false`), same parser, same normalization, same evaluator.

## Error Analysis

Every test prediction is automatically categorized (malformed JSON, schema violation, missing field, wrong value, unsupported value, over/under-extraction, list-item error, date-normalization error, section-association error). Representative examples are selected **deterministically** and include regressions and model-swap failures — not just successes. See `results/error_analysis.jsonl`, `results/error_analysis_summary.csv`, and the Error Analysis page in the UI.

## API

```
GET  /api/health            liveness (cheap, never loads models)
GET  /api/metadata          active model, versions, adapter status
POST /api/extract           {"text": "...", "model": "base"|"finetuned"|null}
POST /api/compare           both models on the same input
GET  /api/metrics           benchmark results (or clean "unavailable" state)
GET  /api/benchmark/summary metrics + per-field + error analysis for the UI
```

Interactive docs at `/docs` (Swagger) and `/redoc`. The backend never logs resume text, rejects oversized/empty inputs, and returns structured errors (e.g. HTTP 503 with *"Fine-tuned adapter not found. Run the Colab training pipeline and configure ADAPTER_PATH."*).

## Frontend

React 18 + Vite + TypeScript (no Streamlit anywhere in this project): an Extraction page (single-model and side-by-side comparison with a field-difference view), a Benchmark dashboard (dynamic metric cards, charts, per-field table, bootstrap CIs — real numbers or an explicit "not generated yet" state), an Error Analysis page with filtering by category/model/difficulty, and an About page explaining the methodology. **Do not enter sensitive personal information into the demo** — it is intended for synthetic text.

## Docker

```bash
docker compose build
docker compose up
# Frontend:  http://localhost:3000
# Backend:   http://localhost:8000  (docs at /docs)
docker compose down
```

- `frontend`: multi-stage Node build → Nginx (SPA routing, `/api` reverse proxy to `backend:8000`).
- `backend`: Python slim image, CPU torch, non-root user, healthcheck; serves base model + adapter.
- Weights are never committed: the base model is pulled from Hugging Face into a named volume on first start; the adapter comes from the mounted `./artifacts/adapters/final`.
- Training never runs in Docker. CPU inference works (slow for a 3–4B model); optional GPU support (NVIDIA Container Toolkit) is documented in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Colab Training

[`notebooks/DocuTune_Training.ipynb`](notebooks/DocuTune_Training.ipynb) is the official GPU workflow: runtime check → deps → repo → generate + validate data → baseline → smoke test → QLoRA training (resumable) → fine-tuned benchmark → full before/after artifacts → adapter export. Colab provides compute; the repository holds the code.

## Project Structure

```
DocuTune/
├── configs/               train.yaml · eval.yaml · app.yaml
├── data/                  raw/ · processed/ · splits/ · dataset_manifest.json
├── docutune/              the Python package
│   ├── config.py seed.py
│   ├── schema/            Pydantic resume schema
│   ├── data/              generator · renderer (12 templates) · splitter · validator
│   ├── inference/         prompt (canonical) · loader · extractor
│   ├── training/          train.py · dataset.py (masking) · model.py (QLoRA)
│   ├── evaluation/        parser · normalization · metrics · benchmark ·
│   │                      bootstrap · error_analysis · runner · reporting
│   └── utils/             io · logging
├── backend/               FastAPI app (routes · services · schemas · Dockerfile)
├── frontend/              React/Vite/TS (Dockerfile · nginx.conf)
├── scripts/               generate/validate data · baseline · finetuned · benchmark ·
│                          run_experiment · update_readme_metrics · merge/push adapter ·
│                          check_environment · verify_project
├── tests/                 GPU-free pytest suite (mocked models, stub tokenizer)
├── notebooks/             DocuTune_Training.ipynb (Colab)
├── docs/                  MODEL_CARD · DATASET · EVALUATION · TRAINING · DEPLOYMENT · KAGGLE
├── artifacts/             adapters/final (from Colab) · training/  [not committed]
└── results/               benchmark outputs, charts, reports    [generated]
```

## Reproducibility

1. `git clone` + `pip install -e ".[training,serving,eval,dev]"` (or per-need extras)
2. `python scripts/generate_data.py --seed 42` — byte-identical dataset
3. `python scripts/validate_data.py`
4. `python scripts/run_baseline.py`
5. Open the Colab notebook → run training on a T4 → download the adapter → place it in `artifacts/adapters/final`
6. `python scripts/run_finetuned.py`
7. `python scripts/benchmark.py` — writes all results and updates the README table
8. `docker compose up --build` — serve the full app

Or in one shot (after dataset + adapter exist): `python scripts/run_experiment.py`.

### Exact commands

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[training,serving,eval,dev]"
.venv\Scripts\python scripts\generate_data.py --seed 42
.venv\Scripts\python scripts\validate_data.py
.venv\Scripts\pytest
.venv\Scripts\ruff check .
.venv\Scripts\uvicorn backend.app.main:app --reload --port 8000
```

**Linux / macOS**
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[training,serving,eval,dev]"
python scripts/generate_data.py --seed 42
python scripts/validate_data.py
pytest
ruff check .
uvicorn backend.app.main:app --reload --port 8000
```

**Frontend (local dev)**
```bash
cd frontend
npm install
npm run dev       # http://localhost:5173 (proxies /api to :8000)
npm run build     # production build
npm run lint
```

**Tests / lint:** `pytest` · `ruff check .` — no GPU, no model download, no HF login required.

## Model & License Notes

- **Active base model:** `microsoft/Phi-3-mini-4k-instruct` (configurable via `BASE_MODEL`; documented fallbacks: `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` — never switched silently; the active model appears in config, logs, API metadata, manifests and this README).
- **Model card:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — retrieve the current license from each model's Hugging Face page before use; license terms can change.
- **Licenses are separate:** the MIT license covers this repository's code and its synthetic dataset only. It does **not** cover the base model or any fallback model. The dataset contains no real personal data.

## Limitations

- Synthetic dataset → synthetic language bias; expect domain shift on real resumes.
- Small test set (75 by default); bootstrap CIs quantify but do not remove uncertainty.
- Held-out templates reduce, but do not eliminate, template-style overfitting.
- "Unsupported-value rate" is defined against the synthetic gold record, not a universal hallucination measure.
- Results and claims are scoped: *"on this held-out synthetic benchmark…"* — no universal superiority is claimed.
- CPU inference of a 3–4B model is slow; GPU serving recommended.
- Fine-tuned behavior is model-specific; changing `BASE_MODEL` invalidates trained adapters and results.

## Future Work

- Real (consented) resume evaluation set for domain-shift measurement
- Constrained/structured decoding and output-schema validation feedback loops
- Larger adapter ranks / more epochs with proper early stopping on validation F1
- Export to ONNX / quantized GGUF for cheaper CPU serving
- Optional experiment tracking (W&B/MLflow) — kept out by design to avoid required accounts

---

*Built as a portfolio-grade demonstration of dataset engineering, parameter-efficient fine-tuning, rigorous evaluation methodology, API/service design, frontend engineering and containerized deployment. No metric on this page was written by a human — they are written by `scripts/benchmark.py`, or they say TBD.*

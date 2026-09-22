# Deployment Documentation

DocuTune separates **training** (Google Colab / GPU machine) from **serving**
(Docker Compose: React + Nginx → FastAPI → base model + LoRA adapter). Training never
runs in the serving containers.

## 0. Prerequisites

- Docker + Docker Compose (`docker compose version` ≥ v2)
- The trained adapter in `artifacts/adapters/final/` (from the Colab notebook)
- ~6 GB disk for the base model cache (downloaded from Hugging Face on first start)
- Optional: `HF_TOKEN` in `.env` only for gated base models (Phi-3/Qwen are open)

Prepare the benchmark results so the dashboard has data:

```bash
python scripts/run_baseline.py
python scripts/run_finetuned.py
python scripts/benchmark.py --skip-inference   # or let run_experiment.py do it all
```

## 1. Docker Compose (production-style)

```bash
cp .env.example .env        # optional: adjust BASE_MODEL, MODEL_MODE, HF_TOKEN
docker compose build
docker compose up
```

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000> — Swagger docs: <http://localhost:8000/docs>,
  ReDoc: <http://localhost:8000/redoc>
- Stop: `docker compose down` (the `hf_cache` volume persists the model download)

What runs where:

| Service | Image contents | Ports | Notes |
|---|---|---|---|
| `frontend` | Node build → Nginx (SPA + `/api` proxy) | 3000→80 | `depends_on` backend healthy |
| `backend` | FastAPI + Uvicorn + transformers/peft (CPU torch) | 8000→8000 | non-root user, healthcheck on `/api/health` |

Volumes: `./artifacts → /app/artifacts` (adapter), `./results → /app/results` (read-only,
dashboard data), `hf_cache` named volume (`HF_HOME=/app/hf-cache`).

Environment (`.env` or compose defaults): `BASE_MODEL`, `MODEL_REVISION`, `MODEL_MODE`
(`base` | `finetuned` — default model for `/api/extract`), `ADAPTER_PATH`
(`/app/artifacts/adapters/final` inside the container), `DEVICE=auto`, `QUANTIZED=false`,
`MAX_NEW_TOKENS`, `MAX_INPUT_CHARS`, `HF_TOKEN`.

### CPU inference

The default stack runs on CPU: it works, but a 3–4B model generates slowly (expect
multiple seconds to tens of seconds per extraction, and a longer first-request warm-up
while the model loads). The backend healthcheck has a long start period for exactly this
reason.

### Optional GPU inference (NVIDIA)

1. Install the NVIDIA driver + **NVIDIA Container Toolkit** on the host.
2. In `docker-compose.yml`, uncomment the `deploy.resources.reservations.devices`
   block on the `backend` service (driver `nvidia`, capabilities `[gpu]`).
3. Set `DEVICE=cuda` and optionally `QUANTIZED=true` (4-bit base loading, needs
   bitsandbytes — add it to `backend/requirements.txt` or install a GPU-enabled image).
4. `docker compose up` again. GPU support is deliberately opt-in: the CPU path must keep
   working everywhere.

## 2. Local Development (no Docker)

**Backend** (repo root):

```bash
# Windows PowerShell
.venv\Scripts\pip install -e ".[serving]"
.venv\Scripts\uvicorn backend.app.main:app --reload --port 8000

# Linux / macOS
.venv/bin/pip install -e ".[serving]"
.venv/bin/uvicorn backend.app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 — Vite proxies /api → localhost:8000
```

Local dev uses the same `/api` base path as production (Vite proxy locally, Nginx proxy
in Docker), so no browser code distinguishes the two.

## 3. Model Artifact Handling

- Base model: pulled from Hugging Face by `DEVICE`-appropriate loader code into
  `HF_HOME` (named volume in Docker, `~/.cache/huggingface` locally)
- Adapter: loaded from `ADAPTER_PATH` — a small directory containing the LoRA weights,
  tokenizer copy, `training_manifest.json` and `resolved_training_config.json`
- Merged weights are optional (`scripts/merge_adapter.py` → `artifacts/adapters/merged`)
  and are NOT required for serving
- Adapter publishing to the HF Hub is explicit-only: `scripts/push_adapter.py --repo-id
  you/docutune-adapter` with `HF_TOKEN` set — nothing uploads automatically

## 4. Hugging Face Authentication

Only needed for gated base models (e.g. Llama fallbacks). Set `HF_TOKEN` in `.env`
(never commit it; `.env` is gitignored). Phi-3-mini and Qwen download anonymously.

## 5. Operational Notes

- **Model loads once:** the first extraction request triggers the model load (seconds on
  GPU, longer on CPU); subsequent requests reuse the cached model manager.
- **Privacy:** resume text is processed in memory only, is never logged (only lengths),
  and is never persisted by the API. The UI warns users not to enter sensitive data.
- **Input safety:** requests are limited to `MAX_INPUT_CHARS` (default 20 000); empty or
  whitespace-only input is rejected with 422; errors return structured JSON.
- **Benchmark unavailable state:** without `results/metrics.json` the dashboard shows
  "Benchmark results have not been generated yet." — it never invents numbers.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Fine-tuned adapter not found...` (HTTP 503) | adapter missing at `ADAPTER_PATH` — run Colab training, place it in `artifacts/adapters/final`, restart backend |
| Frontend loads but API calls fail | backend not up, or proxy broken — check `http://localhost:8000/api/health` |
| First request very slow | model loading + warm-up; subsequent requests are fast |
| `MODEL_MODE must be 'base' or 'finetuned'` | fix the env var spelling |
| Base model download fails | network/Disk space; check `HF_HOME` and free space (model ≈ 5–6 GB) |
| Port conflicts | change host port mappings in `docker-compose.yml` (`3000:`, `8000:`) |
| GPU not visible in container | NVIDIA Container Toolkit missing or compose GPU block not uncommented |

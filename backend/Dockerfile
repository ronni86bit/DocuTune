# DocuTune backend: FastAPI + Uvicorn serving base model (+ LoRA adapter).
# CPU-friendly by default; see docs/DEPLOYMENT.md for GPU serving.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# CPU build of torch first (much smaller than the default wheel).
RUN pip install --no-cache-dir torch>=2.4 --index-url https://download.pytorch.org/whl/cpu

# Install the docutune package.
COPY pyproject.toml README.md LICENSE ./
COPY docutune ./docutune
RUN pip install --no-cache-dir .

# Backend runtime dependencies.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code + configs.
COPY backend ./backend
COPY configs ./configs

# Non-root runtime user; writable dirs for model cache and mounted artifacts.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/artifacts /app/results /app/hf-cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8).status == 200 else 1)"

# No training ever happens in the serving container.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

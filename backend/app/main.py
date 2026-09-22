"""DocuTune FastAPI application factory.

Local development:
    uvicorn backend.app.main:app --reload --port 8000
Docs: http://localhost:8000/docs  |  http://localhost:8000/redoc
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.dependencies import get_manager, get_settings
from backend.app.routes import extraction, health, metrics
from backend.app.services.model_manager import ModelUnavailableError
from docutune.utils.logging import configure_logging, get_logger

logger = get_logger("docutune.backend")

API_TITLE = "DocuTune API"
API_DESCRIPTION = (
    "Structured resume extraction with a small open-source instruction model, "
    "served as (a) the untouched base model and (b) the same model plus a "
    "LoRA adapter fine-tuned on synthetic resume-extraction examples."
)
API_VERSION = "1.0.0"


def create_app(settings=None, manager=None) -> FastAPI:
    configure_logging()
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Dependency overrides (used by tests and alternative deployments).
    app.dependency_overrides[get_settings] = (lambda: settings) if settings else get_settings
    app.dependency_overrides[get_manager] = (lambda: manager) if manager else get_manager

    # Dev-friendly CORS; production uses same-origin /api behind Nginx.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(extraction.router)
    app.include_router(metrics.router)

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(_request: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": exc.user_message})

    @app.exception_handler(Exception)
    async def unhandled_handler(_request: Request, exc: Exception):
        # Never leak stack traces, env vars or filesystem internals.
        logger.error("Unhandled error: %s: %s", type(exc).__name__, exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/")
    async def root():
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()

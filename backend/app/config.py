"""Backend settings: configs/app.yaml defaults + environment overrides.

Never logs or exposes secrets; only derived, safe values leave this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]      # backend/
REPO_ROOT = BACKEND_DIR.parent                          # repository root


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    model_revision: str | None = None
    model_mode: str = "finetuned"                # base | finetuned (default for /api/extract)
    adapter_path: str = "artifacts/adapters/final"
    device: str = "auto"                          # auto | cpu | cuda
    quantized: bool = False                       # 4-bit base loading (CUDA only)
    max_new_tokens: int = 700
    max_input_chars: int = 20000
    results_dir: Path = REPO_ROOT / "results"
    artifacts_dir: Path = REPO_ROOT / "artifacts"

    @property
    def adapter_abs_path(self) -> Path:
        path = Path(self.adapter_path)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def results_abs_dir(self) -> Path:
        path = Path(self.results_dir)
        return path if path.is_absolute() else REPO_ROOT / path


def load_settings() -> Settings:
    """YAML defaults, then environment variable overrides."""
    values: dict = {}
    app_yaml = REPO_ROOT / "configs" / "app.yaml"
    if app_yaml.is_file():
        loaded = yaml.safe_load(app_yaml.read_text(encoding="utf-8")) or {}
        values.update(loaded.get("server", {}))
        values.update(loaded.get("model", {}))
        values.update(loaded.get("generation", {}))
        values.update(loaded.get("limits", {}))

    def env(key: str, current):
        raw = os.environ.get(key)
        if raw is None or raw == "":
            return current
        if isinstance(current, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(current, int):
            return int(raw)
        return raw

    settings = Settings(
        host=str(env("HOST", values.get("host", Settings.host))),
        port=int(env("PORT", values.get("port", Settings.port))),
        base_model=str(env("BASE_MODEL", values.get("base_model", Settings.base_model))),
        model_revision=env("MODEL_REVISION", None) or None,
        model_mode=str(env("MODEL_MODE", values.get("mode", Settings.model_mode))),
        adapter_path=str(env("ADAPTER_PATH", values.get("adapter_path", Settings.adapter_path))),
        device=str(env("DEVICE", values.get("device", Settings.device))),
        quantized=bool(env("QUANTIZED", values.get("quantized", Settings.quantized))),
        max_new_tokens=int(env("MAX_NEW_TOKENS", values.get("max_new_tokens",
                                                            Settings.max_new_tokens))),
        max_input_chars=int(env("MAX_INPUT_CHARS", values.get("max_input_chars",
                                                              Settings.max_input_chars))),
        results_dir=Path(env("RESULTS_DIR", str(Settings.results_dir))),
        artifacts_dir=Path(env("ARTIFACTS_DIR", str(Settings.artifacts_dir))),
    )
    if settings.model_mode not in ("base", "finetuned"):
        raise ValueError(f"MODEL_MODE must be 'base' or 'finetuned', got '{settings.model_mode}'")
    return settings

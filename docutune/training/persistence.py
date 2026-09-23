"""Optional persistence of the final LoRA adapter outside ephemeral runtimes.

On Kaggle/Colab, ``artifacts/adapters/final`` lives on an ephemeral
filesystem; a runtime reset after training deletes the adapter. When
configured (environment variables only - never credentials in Git), the
training entry point verifies the saved adapter and uploads it together with
its reproduction metadata (training manifest + resolved training config,
which live in the same directory) to a PRIVATE Hugging Face Hub repository.

Design rules:
- upload is entirely OPTIONAL; local training works with no configuration;
- credentials come from the environment and are never logged or written to
  any file (exception text is redacted before logging);
- failures are reported honestly (status "failed") and never touch the
  locally saved adapter;
- huggingface_hub is imported lazily so this module stays importable
  without it.

Environment variables (names follow the project's DOCUTUNE_* convention;
HF_TOKEN is honoured as a fallback for the token because it is already the
documented convention in .env.example and docker-compose.yml):
  DOCUTUNE_HF_REPO_ID  e.g. "yourname/docutune-phi3-adapter" - setting this
                       ENABLES auto-upload after training
  DOCUTUNE_HF_TOKEN    write-enabled token for that repo; falls back to
                       HF_TOKEN when unset
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
TOKENIZER_MARKER_FILES = ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")

UPLOAD_STATUS_DISABLED = "disabled"
UPLOAD_STATUS_SUCCEEDED = "succeeded"
UPLOAD_STATUS_FAILED = "failed"
REASON_MAX_CHARS = 300


@dataclass
class UploadSettings:
    """Resolved auto-upload configuration. repr/str never expose the token."""

    enabled: bool
    repo_id: str | None = None
    token: str | None = None
    reason: str = ""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (f"UploadSettings(enabled={self.enabled}, repo_id={self.repo_id!r}, "
                f"token={'<set>' if self.token else '<not set>'}, reason={self.reason!r})")

    __str__ = __repr__


def resolve_upload_settings(environ: dict[str, str] | None = None) -> UploadSettings:
    """Resolve auto-upload settings from the environment (never from files)."""
    env = dict(os.environ if environ is None else environ)
    repo_id = (env.get("DOCUTUNE_HF_REPO_ID") or "").strip() or None
    token = (env.get("DOCUTUNE_HF_TOKEN") or "").strip() \
        or (env.get("HF_TOKEN") or "").strip() or None
    if repo_id is None:
        return UploadSettings(enabled=False, reason="DOCUTUNE_HF_REPO_ID not set")
    return UploadSettings(enabled=True, repo_id=repo_id, token=token)


def verify_adapter_artifacts(adapter_dir: str | Path) -> list[str]:
    """Verify the locally saved adapter is complete enough to upload/use.

    Returns a list of problems (empty when complete). Expected contents:
    adapter_config.json, adapter weights, and tokenizer files.
    """
    adapter_dir = Path(adapter_dir)
    problems: list[str] = []
    if not adapter_dir.is_dir():
        return [f"not a directory: {adapter_dir}"]
    if not (adapter_dir / ADAPTER_CONFIG_FILE).is_file():
        problems.append(f"missing {ADAPTER_CONFIG_FILE}")
    if not any((adapter_dir / name).is_file() for name in ADAPTER_WEIGHT_FILES):
        problems.append(f"missing adapter weights (one of {ADAPTER_WEIGHT_FILES})")
    if not any((adapter_dir / name).is_file() for name in TOKENIZER_MARKER_FILES):
        problems.append(f"missing tokenizer files (one of {TOKENIZER_MARKER_FILES})")
    return problems


def redact(text: str, *secrets: str | None) -> str:
    """Remove any known secret from text before it is logged or persisted."""
    result = text or ""
    for secret in secrets:
        if secret:
            result = result.replace(secret, "***REDACTED***")
    return result


def upload_adapter(adapter_dir: str | Path, repo_id: str, token: str) -> str:
    """Upload the adapter directory (weights + manifests) to a private repo.

    Creates the repository if needed (always PRIVATE). Returns the repo URL.
    Raises on any failure - callers must not report success on exception.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def persist_adapter(adapter_dir: str | Path, settings: UploadSettings) -> dict:
    """Verify + upload the final adapter. NEVER raises; never touches the
    local files. Returns a status dict safe to embed in the manifest:

        {"status": "disabled"|"succeeded"|"failed",
         "repo_id": str|None, "url": str|None, "reason": str|None}

    All returned text is secret-redacted.
    """
    adapter_dir = Path(adapter_dir)
    if not settings.enabled:
        return {"status": UPLOAD_STATUS_DISABLED, "repo_id": None, "url": None,
                "reason": settings.reason or None}

    if not settings.token:
        reason = ("upload enabled but no token found "
                  "(set DOCUTUNE_HF_TOKEN or HF_TOKEN)")
        return {"status": UPLOAD_STATUS_FAILED, "repo_id": settings.repo_id,
                "url": None, "reason": reason}

    problems = verify_adapter_artifacts(adapter_dir)
    if problems:
        reason = "local adapter incomplete, not uploading: " + "; ".join(problems)
        return {"status": UPLOAD_STATUS_FAILED, "repo_id": settings.repo_id,
                "url": None, "reason": reason}

    try:
        url = upload_adapter(adapter_dir, settings.repo_id or "", settings.token or "")
        return {"status": UPLOAD_STATUS_SUCCEEDED, "repo_id": settings.repo_id,
                "url": url, "reason": None}
    except Exception as exc:  # noqa: BLE001 - upload must never crash training
        reason = redact(f"{type(exc).__name__}: {exc}", settings.token)
        return {"status": UPLOAD_STATUS_FAILED, "repo_id": settings.repo_id,
                "url": None, "reason": reason[:REASON_MAX_CHARS]}

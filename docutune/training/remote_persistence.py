"""Durable remote checkpoint persistence for ephemeral training runtimes.

On Kaggle/Colab, ``/kaggle/working`` is wiped when the runtime resets: local
Trainer checkpoints survive only within one session, so a reset before the
final adapter upload loses the whole run. This module mirrors periodic
checkpoints to the SAME private Hugging Face Hub repository that receives the
final adapter, so a fresh runtime can resume from the latest valid remote
checkpoint instead of restarting a ~6-hour run from step 0.

Remote layout inside the configured repo (``DOCUTUNE_HF_REPO_ID``):

    adapter_model.safetensors        <- final adapter + manifest + resolved
    adapter_config.json                 config at the repo ROOT (uploaded by
    tokenizer files                     docutune/training/persistence.py;
    training_manifest.json              those semantics are unchanged)
    resolved_training_config.json
    checkpoints/
        checkpoint-25/               <- one directory per optimizer step;
            optimizer.pt                byte-for-byte the local Trainer
            scheduler.pt                checkpoint (adapter weights +
            trainer_state.json          optimizer/scheduler/Trainer/RNG
            adapter_model.safetensors   state; the frozen 4-bit base model
            ...                         is NEVER included)
            REMOTE_CHECKPOINT_COMPLETE.json   <- completion marker

Atomicity: each checkpoint is uploaded as ONE Hub commit (``create_commit``
with a ``CommitOperationAdd`` for every file INCLUDING the marker). Hub
commits land completely or not at all, so an interrupted upload cannot leave
a half-written checkpoint, and a partially uploaded checkpoint can never be
selected for resume: discovery keys on the marker, which only exists inside
the completed commit.

Integrity: the marker records every file's size + sha256 plus a config
fingerprint (see :func:`checkpoint_fingerprint`). A remote checkpoint counts
as valid only when its marker parses, every listed file exists remotely, the
fingerprint matches the current experiment, and after download the hashes
re-verify and :func:`docutune.training.checkpoints.validate_checkpoint_dir`
(the same rules used for local checkpoints) passes. Older valid checkpoints
are the fallback; a fresh run is the last resort.

Failure policy:
- upload failure NEVER touches or deletes the local checkpoint and NEVER
  aborts training; the failure is recorded honestly in the manifest record
  (``upload_status: "failed"`` + sanitized reason);
- uploads are retried a BOUNDED number of times with exponential backoff
  (capped delay) - no infinite retry loops;
- resume-time discovery/download failures fail loudly (a fresh runtime can
  simply re-run the cell); unset ``DOCUTUNE_HF_REPO_ID`` for local-only
  resume.

The token is never logged, stored, or embedded in error text (redaction
reuses :func:`docutune.training.persistence.redact`). ``huggingface_hub`` is
imported lazily and the Hub API object is injectable, so every code path is
unit-testable without network access, credentials, torch or transformers.

Size rationale (Phi-3-mini, LoRA r=16 on fused qkv_proj + o_proj, AdamW):
~9.4M trainable parameters -> adapter weights fp32 ~38 MB + AdamW moments
fp32 ~76 MB + state files <1 MB ~= 115 MB per checkpoint; a full 87-step run
mirrors 3 checkpoints (~350 MB) plus the ~40 MB final adapter. That is
negligible for HF Hub storage and Kaggle disk, while base-model weights are
never uploaded (a checkpoint directory containing full-model weight files is
refused outright).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docutune.training.checkpoints import (
    CHECKPOINT_DIR_RE,
    find_latest_valid_checkpoint,
    validate_checkpoint_dir,
)
from docutune.training.persistence import REASON_MAX_CHARS, redact
from docutune.utils.io import sha256_file
from docutune.utils.logging import get_logger

logger = get_logger(__name__)

REMOTE_CHECKPOINTS_PREFIX = "checkpoints"
# Written inside the SAME atomic commit as the checkpoint files; its presence
# is what makes a remote checkpoint selectable for resume.
REMOTE_CHECKPOINT_MARKER = "REMOTE_CHECKPOINT_COMPLETE.json"
REMOTE_CHECKPOINT_PATH_RE = re.compile(
    rf"^{REMOTE_CHECKPOINTS_PREFIX}/checkpoint-(\d+)/{re.escape(REMOTE_CHECKPOINT_MARKER)}$"
)
# Full base-model weight files must never be mirrored: a QLoRA/PEFT
# checkpoint contains adapter weights only, so these names mean the
# directory is (or contains) full-model state that must not reach the Hub.
FULL_MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
# Ceiling for one remote checkpoint upload. Real checkpoints are ~115 MB;
# this only trips when something is badly wrong (e.g. base-model inclusion).
REMOTE_CHECKPOINT_MAX_TOTAL_BYTES = 2 * 1024**3
UPLOAD_MAX_ATTEMPTS = 3
UPLOAD_BACKOFF_BASE_SECONDS = 2.0
UPLOAD_BACKOFF_MAX_SECONDS = 30.0

RESUME_SOURCE_LOCAL = "local"
RESUME_SOURCE_REMOTE = "remote"
RESUME_SOURCE_SCRATCH = "scratch"

UPLOAD_STATUS_SKIPPED = "skipped"
UPLOAD_STATUS_SUCCEEDED = "succeeded"
UPLOAD_STATUS_FAILED = "failed"


class RemoteCheckpointError(RuntimeError):
    """Recoverable remote-checkpoint failure (message is always sanitized)."""


class RemoteCheckpointCorrupt(RemoteCheckpointError):
    """A downloaded remote checkpoint failed hash/resume validation."""


# ---------------------------------------------------------------------------
# Experiment fingerprint (guards against resuming across changed configs)
# ---------------------------------------------------------------------------
def checkpoint_fingerprint(config: Any) -> str:
    """Stable short fingerprint of everything that defines the experiment.

    Remote checkpoints carry this in their marker; a resume candidate whose
    fingerprint differs from the current config is rejected, so an old
    checkpoint from a different experiment can never be silently resumed.
    Covers identity (model, versions) + the training-relevant hyperparameters
    (data/prompt/schema versions, LoRA surface, schedule, seed). Deliberately
    excludes infrastructure-only values (output dirs, save/eval cadence).
    """
    from docutune.config import DATASET_VERSION, PROMPT_VERSION, SCHEMA_VERSION

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": {"name": config.model.name, "revision": config.model.revision},
        "lora": {
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "target_modules": sorted(config.lora.target_modules),
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "max_length": config.training.max_length,
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "warmup_ratio": config.training.warmup_ratio,
            "seed": config.training.seed,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Hub error classification (huggingface_hub may be absent - e.g. on CI)
# ---------------------------------------------------------------------------
def _is_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True  # also what the test fake raises for a missing file
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        return False
    return isinstance(exc, EntryNotFoundError)


def _is_missing_repo_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        return False
    return isinstance(exc, RepositoryNotFoundError)


# ---------------------------------------------------------------------------
# Bounded retry with exponential backoff (no infinite loops)
# ---------------------------------------------------------------------------
def _with_retries(
    operation: Callable[[], Any],
    *,
    what: str,
    redactor: Callable[[str], str],
    max_attempts: int = UPLOAD_MAX_ATTEMPTS,
) -> Any:
    delay = UPLOAD_BACKOFF_BASE_SECONDS
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            if attempt == max_attempts:
                raise
            sleep_for = min(delay, UPLOAD_BACKOFF_MAX_SECONDS) + random.uniform(0, delay * 0.25)
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                what, attempt, max_attempts, redactor(f"{type(exc).__name__}: {exc}"), sleep_for,
            )
            time.sleep(sleep_for)
            delay *= 2


# ---------------------------------------------------------------------------
# Remote checkpoint store
# ---------------------------------------------------------------------------
def remote_checkpoint_prefix(step: int) -> str:
    return f"{REMOTE_CHECKPOINTS_PREFIX}/checkpoint-{step}"


def build_marker(checkpoint_dir: str | Path, step: int, fingerprint: str) -> dict[str, Any]:
    """Integrity ledger for a local checkpoint: per-file size + sha256."""
    checkpoint_dir = Path(checkpoint_dir)
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(checkpoint_dir.rglob("*")):
        if not path.is_file() or path.name == REMOTE_CHECKPOINT_MARKER:
            continue
        rel = path.relative_to(checkpoint_dir).as_posix()
        files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "format": 1,
        "step": step,
        "config_fingerprint": fingerprint,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
    }


@dataclass
class _Candidate:
    step: int
    marker: dict[str, Any]


class RemoteCheckpointStore:
    """Mirror of Trainer checkpoints in a PRIVATE Hugging Face repo.

    ``api`` is injectable for tests; by default a real ``HfApi`` is created
    lazily on first use. Only four Hub operations are used: create_repo,
    create_commit, list_repo_files and (snapshot|hf_hub)_download.
    """

    def __init__(
        self,
        repo_id: str,
        token: str | None,
        fingerprint: str = "",
        api: Any = None,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.fingerprint = fingerprint
        self._api = api
        self._repo_ensured = False

    # -- plumbing ---------------------------------------------------------
    def _get_api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    def sanitize(self, text: str) -> str:
        return redact(text or "", self.token)[:REASON_MAX_CHARS]

    def _ensure_repo(self) -> None:
        """Create the repo once per store lifetime (always PRIVATE)."""
        if self._repo_ensured:
            return
        api = self._get_api()
        _with_retries(
            lambda: api.create_repo(repo_id=self.repo_id, repo_type="model",
                                    private=True, exist_ok=True),
            what=f"Creating private repo {self.repo_id}",
            redactor=self.sanitize,
        )
        self._repo_ensured = True

    # -- discovery --------------------------------------------------------
    def list_remote_steps(self) -> list[int]:
        """Steps that have a completion marker in the repo (may raise)."""
        steps: set[int] = set()
        for path in self._get_api().list_repo_files(repo_id=self.repo_id, repo_type="model"):
            match = REMOTE_CHECKPOINT_PATH_RE.match(path.replace("\\", "/"))
            if match:
                steps.add(int(match.group(1)))
        return sorted(steps)

    def _load_marker(self, step: int) -> dict[str, Any] | None:
        """Download + parse one marker. None = absent/corrupt (not an error)."""
        try:
            path = self._get_api().hf_hub_download(
                repo_id=self.repo_id,
                repo_type="model",
                subfolder=remote_checkpoint_prefix(step),
                filename=REMOTE_CHECKPOINT_MARKER,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return None
            raise
        try:
            marker = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning("Remote checkpoint-%d marker unreadable (%s)", step, exc)
            return None
        if not isinstance(marker, dict) or not isinstance(marker.get("files"), dict):
            logger.warning("Remote checkpoint-%d marker malformed - ignoring it", step)
            return None
        return marker

    def _marker_ok(
        self, step: int, marker: dict[str, Any], repo_files: set[str]
    ) -> str | None:
        """None when the marker passes remote validation, else the reason."""
        if marker.get("step") != step:
            return f"marker step mismatch ({marker.get('step')})"
        if self.fingerprint and marker.get("config_fingerprint") != self.fingerprint:
            return ("config fingerprint mismatch - checkpoint belongs to a "
                    "different experiment configuration")
        prefix = remote_checkpoint_prefix(step)
        missing = [name for name in marker["files"] if f"{prefix}/{name}" not in repo_files]
        if missing:
            return f"files missing remotely: {sorted(missing)[:5]}"
        return None

    def iter_remote_candidates(self) -> Iterator[_Candidate]:
        """Valid remote checkpoints, NEWEST first (may raise on Hub errors).

        Candidates are only marker/manifest-level checks; byte-level
        integrity is verified at download time (see download_checkpoint).
        """
        try:
            steps = self.list_remote_steps()
        except Exception as exc:  # noqa: BLE001
            if _is_missing_repo_error(exc):
                logger.info("Remote repo %s does not exist yet - no remote checkpoints",
                            self.repo_id)
                return
            raise
        repo_files = {p.replace("\\", "/") for p in
                      self._get_api().list_repo_files(repo_id=self.repo_id, repo_type="model")}
        for step in sorted(steps, reverse=True):
            marker = self._load_marker(step)
            if marker is None:
                logger.warning("Remote checkpoint-%d has no usable marker - skipping "
                               "(incomplete upload?)", step)
                continue
            reason = self._marker_ok(step, marker, repo_files)
            if reason:
                logger.warning("Remote checkpoint-%d rejected: %s", step, reason)
                continue
            yield _Candidate(step=step, marker=marker)

    def latest_valid_remote_step(self) -> int | None:
        for candidate in self.iter_remote_candidates():
            return candidate.step
        return None

    # -- download / materialization ---------------------------------------
    def download_checkpoint(self, step: int, dest_dir: str | Path) -> Path:
        """Materialize remote checkpoint ``step`` at ``dest_dir`` (validated).

        Downloads to a staging directory first, verifies every file against
        the marker (size + sha256) and the shared resume rules, then moves it
        into place - ``dest_dir`` only ever holds a fully validated
        checkpoint. Raises :class:`RemoteCheckpointCorrupt` on any mismatch.
        """
        dest_dir = Path(dest_dir)
        marker = self._load_marker(step)
        if marker is None:
            raise RemoteCheckpointError(
                f"Remote checkpoint-{step} has no usable completion marker"
            )
        if dest_dir.is_dir():
            valid, _problems = validate_checkpoint_dir(dest_dir)
            if valid:
                logger.info("Local copy of checkpoint-%d already valid - reusing it", step)
                return dest_dir

        staging = dest_dir.parent / f".remote-staging-checkpoint-{step}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            self._get_api().snapshot_download(
                repo_id=self.repo_id,
                repo_type="model",
                local_dir=str(staging),
                allow_patterns=[f"{remote_checkpoint_prefix(step)}/*"],
            )
            src = staging / remote_checkpoint_prefix(step)
            if not src.is_dir():
                raise RemoteCheckpointCorrupt(
                    f"Remote checkpoint-{step} download produced no files"
                )
            self._verify_against_marker(step, src, marker)
            valid, problems = validate_checkpoint_dir(src)
            if not valid:
                raise RemoteCheckpointCorrupt(
                    f"Downloaded remote checkpoint-{step} failed resume validation: {problems}"
                )
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():  # only reached when the local copy was invalid
                shutil.rmtree(dest_dir)
            os.replace(src, dest_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return dest_dir

    def _verify_against_marker(self, step: int, src: Path, marker: dict[str, Any]) -> None:
        for name, meta in sorted(marker["files"].items()):
            path = src / name
            if not path.is_file():
                raise RemoteCheckpointCorrupt(
                    f"Remote checkpoint-{step} is missing downloaded file {name}"
                )
            size = path.stat().st_size
            digest = sha256_file(path)
            if size != meta.get("size") or digest != meta.get("sha256"):
                raise RemoteCheckpointCorrupt(
                    f"Remote checkpoint-{step} file {name} failed integrity check "
                    f"(size {size} vs {meta.get('size')}, sha256 mismatch)"
                )

    # -- upload -------------------------------------------------------------
    def upload_checkpoint(self, checkpoint_dir: str | Path, step: int) -> dict[str, Any]:
        """Verify + upload one checkpoint as a single atomic Hub commit.

        NEVER raises and NEVER touches the local directory. Returns a status
        record safe to embed in the training manifest (token-free, redacted).
        """
        checkpoint_dir = Path(checkpoint_dir)
        record: dict[str, Any] = {
            "step": step,
            "local_path": str(checkpoint_dir),
            "repo_id": self.repo_id,
            "remote_path": remote_checkpoint_prefix(step),
            "upload_attempted": False,
            "upload_status": UPLOAD_STATUS_SKIPPED,
            "verified": False,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": None,
        }

        valid, problems = validate_checkpoint_dir(checkpoint_dir)
        if not valid:
            record["reason"] = "local checkpoint incomplete, not uploading: " + "; ".join(problems)
            logger.error("Checkpoint %d persistence FAILED: %s", step, record["reason"])
            return record

        full_model = sorted(p.name for p in checkpoint_dir.iterdir()
                            if p.name in FULL_MODEL_WEIGHT_FILES)
        if full_model:
            record["reason"] = (
                f"refusing to upload full-model weight files {full_model}; QLoRA "
                "checkpoints must contain adapter state only"
            )
            logger.error("Checkpoint %d persistence FAILED: %s", step, record["reason"])
            return record

        files = [p for p in sorted(checkpoint_dir.rglob("*"))
                 if p.is_file() and p.name != REMOTE_CHECKPOINT_MARKER]
        total_bytes = sum(p.stat().st_size for p in files)
        if total_bytes > REMOTE_CHECKPOINT_MAX_TOTAL_BYTES:
            record["reason"] = (
                f"checkpoint too large to mirror ({total_bytes / 1024**3:.2f} GiB > "
                f"{REMOTE_CHECKPOINT_MAX_TOTAL_BYTES / 1024**3:.0f} GiB cap) - possible "
                "base-model inclusion"
            )
            logger.error("Checkpoint %d persistence FAILED: %s", step, record["reason"])
            return record

        # Already persisted (e.g. Trainer re-fired on_save after a resume)?
        try:
            if self._already_persisted(step):
                record["verified"] = True
                record["reason"] = "already persisted remotely - upload skipped"
                logger.info("Checkpoint %d already persisted remotely - skipping upload", step)
                return record
        except Exception as exc:  # noqa: BLE001 - fall through to the upload attempt
            logger.warning("Could not check remote state for checkpoint-%d (%s)",
                           step, self.sanitize(f"{type(exc).__name__}: {exc}"))

        marker = build_marker(checkpoint_dir, step, self.fingerprint)
        operations = self._build_operations(checkpoint_dir, files, marker, step)
        record["upload_attempted"] = True
        logger.info("Checkpoint %d upload: %d files, %.1f MB -> %s/%s",
                    step, len(operations), total_bytes / 1024**2,
                    self.repo_id, remote_checkpoint_prefix(step))
        try:
            self._ensure_repo()
            api = self._get_api()
            _with_retries(
                lambda: api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="model",
                    operations=operations,
                    commit_message=f"DocuTune checkpoint-{step} (durable mirror)",
                ),
                what=f"Uploading checkpoint-{step}",
                redactor=self.sanitize,
            )
        except Exception as exc:  # noqa: BLE001 - upload must never crash training
            record["upload_status"] = UPLOAD_STATUS_FAILED
            record["reason"] = self.sanitize(f"{type(exc).__name__}: {exc}")
            logger.error("Checkpoint %d persistence FAILED (local checkpoint kept): %s",
                         step, record["reason"])
            return record

        verified, reason = self._verify_remote(step, marker)
        record["verified"] = verified
        record["upload_status"] = UPLOAD_STATUS_SUCCEEDED if verified else UPLOAD_STATUS_FAILED
        record["reason"] = reason
        if verified:
            logger.info("Checkpoint %d upload verified - persistence SUCCEEDED", step)
        else:
            logger.error("Checkpoint %d persistence FAILED after upload: %s", step, reason)
        return record

    def _make_operation(self, path_in_repo: str, path_or_fileobj: Any) -> Any:
        """One Hub commit operation. A hook so the whole upload path stays
        unit-testable without huggingface_hub installed (CI has no network
        stack and no Hub client)."""
        from huggingface_hub import CommitOperationAdd

        return CommitOperationAdd(path_in_repo=path_in_repo,
                                  path_or_fileobj=path_or_fileobj)

    def _build_operations(
        self, checkpoint_dir: Path, files: list[Path], marker: dict[str, Any], step: int
    ) -> list[Any]:
        prefix = remote_checkpoint_prefix(step)
        operations = [
            self._make_operation(
                path_in_repo=f"{prefix}/{p.relative_to(checkpoint_dir).as_posix()}",
                path_or_fileobj=str(p))
            for p in files
        ]
        operations.append(self._make_operation(
            path_in_repo=f"{prefix}/{REMOTE_CHECKPOINT_MARKER}",
            path_or_fileobj=json.dumps(marker, indent=2).encode("utf-8"),
        ))
        return operations

    def _already_persisted(self, step: int) -> bool:
        marker = self._load_marker(step)
        if marker is None:
            return False
        repo_files = {p.replace("\\", "/") for p in
                      self._get_api().list_repo_files(repo_id=self.repo_id, repo_type="model")}
        return self._marker_ok(step, marker, repo_files) is None

    def _verify_remote(self, step: int, marker: dict[str, Any]) -> tuple[bool, str | None]:
        """Post-upload verification: marker + every listed file really there."""
        try:
            repo_files = {p.replace("\\", "/") for p in self._get_api().list_repo_files(
                repo_id=self.repo_id, repo_type="model")}
        except Exception as exc:  # noqa: BLE001
            return False, self.sanitize(f"post-upload verification failed: {exc}")
        reason = self._marker_ok(step, marker, repo_files)
        if reason:
            return False, f"post-upload verification failed: {reason}"
        return True, None


# ---------------------------------------------------------------------------
# Trainer callback: mirror every checkpoint right after Trainer writes it
# ---------------------------------------------------------------------------
try:
    from transformers import TrainerCallback
except ImportError:  # keep the module importable without the training stack
    TrainerCallback = object  # type: ignore[assignment,misc]


class RemoteCheckpointCallback(TrainerCallback):  # type: ignore[misc,valid-type]
    """Upload each ``checkpoint-<step>`` as soon as Trainer has written it.

    Attached only when remote persistence is enabled (``DOCUTUNE_HF_REPO_ID``
    set) and never during smoke tests (which disable checkpointing anyway).
    Failures are logged and recorded; training always continues.
    """

    def __init__(self, store: RemoteCheckpointStore, records: list[dict[str, Any]] | None = None):
        super().__init__()
        self.store = store
        self.records: list[dict[str, Any]] = records if records is not None else []

    def on_save(self, args, state, control, **kwargs):  # noqa: D102 - Trainer protocol
        step = state.global_step
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        logger.info("Checkpoint %d saved locally (%s)", step, checkpoint_dir)
        record = self.store.upload_checkpoint(checkpoint_dir, step)
        self.records.append(record)
        return control


# ---------------------------------------------------------------------------
# Resume resolution (precedence: explicit > local > remote > scratch)
# ---------------------------------------------------------------------------
@dataclass
class ResumeDecision:
    """Where training should resume from (see resolve_resume_checkpoint)."""

    checkpoint_path: str | None
    source: str  # RESUME_SOURCE_LOCAL / _REMOTE / _SCRATCH
    step: int | None = None
    repo_id: str | None = None
    detail: str = ""


def resolve_resume_checkpoint(
    output_dir: str | Path,
    *,
    resume: bool,
    explicit_checkpoint: str | None,
    store: RemoteCheckpointStore | None = None,
) -> ResumeDecision:
    """Resolve the resume point with an explicit, documented precedence:

    1. ``--checkpoint <path>`` (strict: invalid path FAILS loudly).
    2. ``--resume`` + latest valid LOCAL checkpoint in ``output_dir``.
    3. ``--resume`` + latest valid REMOTE checkpoint (only when no valid
       local checkpoint exists; requires the store, i.e. a configured repo).
    4. Fresh training, logged clearly.

    Local wins over remote by design: on a live runtime the local checkpoint
    is always at least as new as the mirror, and it avoids touching the
    network. Remote is only consulted when the local state is gone - exactly
    the fresh-Kaggle-runtime case. Remote discovery/transport failures raise
    (fail fast at startup, seconds into a fresh runtime) instead of silently
    restarting from step 0 while a valid checkpoint may exist; unset
    ``DOCUTUNE_HF_REPO_ID`` to resume locally-only.
    """
    output_dir = Path(output_dir)

    if explicit_checkpoint:
        explicit = Path(explicit_checkpoint)
        valid, problems = validate_checkpoint_dir(explicit)
        if not valid:
            raise RemoteCheckpointError(
                f"Explicit checkpoint {explicit} is not a valid resume point: {problems}. "
                "Use --resume to auto-discover the latest valid checkpoint instead."
            )
        step = _step_from_dir(explicit)
        logger.info("Resuming from explicit checkpoint%s: %s",
                    f"-{step}" if step is not None else "", explicit)
        return ResumeDecision(str(explicit), RESUME_SOURCE_LOCAL, step=step,
                              detail="explicit --checkpoint")

    if not resume:
        return ResumeDecision(None, RESUME_SOURCE_SCRATCH)

    latest_local = find_latest_valid_checkpoint(output_dir)
    if latest_local is not None:
        step = _step_from_dir(latest_local)
        logger.info("Resuming from the latest valid local checkpoint%s: %s",
                    f"-{step}" if step is not None else "", latest_local)
        return ResumeDecision(str(latest_local), RESUME_SOURCE_LOCAL, step=step,
                              detail="latest valid local checkpoint")

    if store is None:
        logger.warning(
            "No valid checkpoint found in %s - training from scratch "
            "(incomplete checkpoints are ignored; set DOCUTUNE_HF_REPO_ID to "
            "mirror checkpoints to the Hub for cross-runtime resume)", output_dir,
        )
        return ResumeDecision(None, RESUME_SOURCE_SCRATCH)

    # Fresh runtime: local state is gone, look for the newest valid mirror.
    try:
        for candidate in store.iter_remote_candidates():
            dest = output_dir / f"checkpoint-{candidate.step}"
            logger.info("Resuming from remote checkpoint-%d (repo %s)",
                        candidate.step, store.repo_id)
            logger.info("Downloading checkpoint...")
            try:
                path = store.download_checkpoint(candidate.step, dest)
            except RemoteCheckpointCorrupt as exc:
                logger.error("Remote checkpoint-%d unusable (%s) - trying an older one",
                             candidate.step, exc)
                continue
            logger.info("Remote checkpoint validated (sha256 + resume files verified)")
            logger.info("Trainer resume path: %s", path)
            return ResumeDecision(str(path), RESUME_SOURCE_REMOTE, step=candidate.step,
                                  repo_id=store.repo_id, detail="latest valid remote checkpoint")
    except RemoteCheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail fast with a sanitized reason
        reason = store.sanitize(f"{type(exc).__name__}: {exc}")
        raise RemoteCheckpointError(
            f"Remote checkpoint discovery failed: {reason}. "
            "Re-run with --resume once the Hub is reachable, or unset DOCUTUNE_HF_REPO_ID "
            "to resume from local checkpoints only."
        ) from exc

    logger.info(
        "No valid remote checkpoint in %s and none locally - training from scratch",
        store.repo_id,
    )
    return ResumeDecision(None, RESUME_SOURCE_SCRATCH)


def _step_from_dir(path: Path) -> int | None:
    match = CHECKPOINT_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None

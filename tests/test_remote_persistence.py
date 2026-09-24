"""Remote checkpoint persistence tests (fully offline).

The Hugging Face Hub API is faked with an in-memory repository implementing
exactly the five HfApi operations the store uses (create_repo, create_commit,
list_repo_files, hf_hub_download, snapshot_download). No network, no real
credentials, no torch/transformers required - the suite runs on CI where the
training extras are not installed.
"""

import fnmatch
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docutune.config import TrainConfig
from docutune.training import remote_persistence as rp
from docutune.training.checkpoints import validate_checkpoint_dir
from docutune.training.persistence import resolve_upload_settings
from docutune.training.remote_persistence import (
    REMOTE_CHECKPOINT_MARKER,
    RemoteCheckpointCallback,
    RemoteCheckpointCorrupt,
    RemoteCheckpointError,
    RemoteCheckpointStore,
    checkpoint_fingerprint,
    resolve_resume_checkpoint,
)

TOKEN = "hf_TESTTOKEN_do_not_leak"
FP = "fingerprint-A"
FP_OTHER = "fingerprint-B"


# ---------------------------------------------------------------------------
# In-memory fake of the Hub operations the store uses
# ---------------------------------------------------------------------------
class FakeHubRepo:
    def __init__(self, tmp_path: Path, repo_id: str = "user/docutune-adapter"):
        self.tmp_path = tmp_path
        self.repo_id = repo_id
        self.files: dict[str, bytes] = {}
        self.commits: list[list[str]] = []
        self.repo_exists = False
        self.create_repo_kwargs: dict = {}
        self.fail_create_commit = 0  # number of initial create_commit calls to fail
        self.fail_listing = False
        self.create_commit_calls = 0

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False, **kw):
        self.create_repo_kwargs = {"repo_id": repo_id, "private": private}
        self.repo_exists = True

    def create_commit(self, repo_id, operations=(), commit_message="", **kw):
        self.create_commit_calls += 1
        if not self.repo_exists:
            raise FileNotFoundError(f"repository {repo_id} not found")
        if self.fail_create_commit > 0:
            self.fail_create_commit -= 1
            raise ConnectionError(f"connection reset during upload token={TOKEN}")
        self.commits.append([op.path_in_repo for op in operations])
        for op in operations:
            data = op.path_or_fileobj
            if isinstance(data, (str, Path)):
                data = Path(data).read_bytes()  # str/Path = filesystem path
            self.files[op.path_in_repo] = bytes(data)

    def list_repo_files(self, repo_id, repo_type=None, **kw):
        if not self.repo_exists:
            raise FileNotFoundError(f"repository {repo_id} not found")
        if self.fail_listing:
            raise ConnectionError(f"listing failed for token={TOKEN}")
        return sorted(self.files)

    def hf_hub_download(self, repo_id, filename, subfolder=None, repo_type=None, **kw):
        if not self.repo_exists:
            raise FileNotFoundError(f"repository {repo_id} not found")
        path = f"{subfolder}/{filename}" if subfolder else filename
        if path not in self.files:
            raise FileNotFoundError(f"entry {path} not found in repo")
        dest = self.tmp_path / "marker-downloads" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[path])
        return dest

    def snapshot_download(self, repo_id, repo_type=None, local_dir=None,
                          allow_patterns=None, **kw):
        if not self.repo_exists:
            raise FileNotFoundError(f"repository {repo_id} not found")
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        for path, data in self.files.items():
            if allow_patterns and not any(fnmatch.fnmatch(path, pat) for pat in allow_patterns):
                continue
            dest = local_dir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return local_dir


def make_store(repo: FakeHubRepo, fingerprint: str = FP) -> RemoteCheckpointStore:
    return RemoteCheckpointStore(repo.repo_id, TOKEN, fingerprint=fingerprint, api=repo)


def make_local_checkpoint(path: Path, step: int, complete: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if complete:
        (path / "optimizer.pt").write_bytes(b"optimizer-state-" + str(step).encode())
        (path / "scheduler.pt").write_bytes(b"scheduler-state")
        (path / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8")
    return path


def upload(repo: FakeHubRepo, tmp_path: Path, step: int,
           fingerprint: str = FP) -> dict:
    ckpt = make_local_checkpoint(tmp_path / f"local-checkpoint-{step}", step)
    return make_store(repo, fingerprint).upload_checkpoint(ckpt, step)


def remote_paths(repo: FakeHubRepo, step: int) -> list[str]:
    return [p for p in repo.files if p.startswith(f"checkpoints/checkpoint-{step}/")]


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(rp.time, "sleep", lambda s: waits.append(s))
    return waits


# ---------------------------------------------------------------------------
# Upload: atomic commit, marker, verification, honest failures, retries
# ---------------------------------------------------------------------------
class TestUpload:
    def test_success_is_one_atomic_commit_including_marker(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        record = upload(repo, tmp_path, 25)
        assert record["upload_status"] == "succeeded"
        assert record["verified"] is True
        assert record["upload_attempted"] is True
        assert record["reason"] is None
        # every file AND the marker travel in a single commit - a killed
        # upload can never leave a half-written remote checkpoint
        assert len(repo.commits) == 1
        assert "checkpoints/checkpoint-25/" + REMOTE_CHECKPOINT_MARKER in repo.commits[0]
        assert len(repo.commits[0]) == len(remote_paths(repo, 25))

    def test_marker_records_step_fingerprint_and_hashes(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        marker = json.loads(
            repo.files[f"checkpoints/checkpoint-25/{REMOTE_CHECKPOINT_MARKER}"])
        assert marker["step"] == 25
        assert marker["config_fingerprint"] == FP
        assert set(marker["files"]) == {"optimizer.pt", "scheduler.pt",
                                        "adapter_model.safetensors", "trainer_state.json"}
        assert all(len(meta["sha256"]) == 64 for meta in marker["files"].values())

    def test_repo_is_always_created_private(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        assert repo.create_repo_kwargs["private"] is True

    def test_failure_is_honest_and_local_checkpoint_untouched(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        repo.fail_create_commit = rp.UPLOAD_MAX_ATTEMPTS  # every attempt fails
        ckpt = make_local_checkpoint(tmp_path / "checkpoint-25", 25)
        before = {p.name: p.read_bytes() for p in ckpt.iterdir()}
        record = make_store(repo).upload_checkpoint(ckpt, 25)
        assert record["upload_status"] == "failed"
        assert record["verified"] is False
        assert record["upload_attempted"] is True
        assert "ConnectionError" in record["reason"]
        assert {p.name: p.read_bytes() for p in ckpt.iterdir()} == before
        assert not repo.files  # nothing was persisted remotely

    def test_upload_retries_then_succeeds(self, tmp_path, fast_retries):
        repo = FakeHubRepo(tmp_path)
        repo.fail_create_commit = 1
        record = upload(repo, tmp_path, 25)
        assert record["upload_status"] == "succeeded"
        assert repo.create_commit_calls == 2
        assert len(fast_retries) == 1  # one bounded backoff sleep, no infinite loop

    def test_retry_backoff_is_bounded(self, tmp_path, fast_retries):
        repo = FakeHubRepo(tmp_path)
        repo.fail_create_commit = rp.UPLOAD_MAX_ATTEMPTS
        upload(repo, tmp_path, 25)
        assert len(fast_retries) == rp.UPLOAD_MAX_ATTEMPTS - 1
        assert all(0 < w <= rp.UPLOAD_BACKOFF_MAX_SECONDS * 1.25 for w in fast_retries)

    def test_incomplete_local_checkpoint_is_never_uploaded(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        ckpt = make_local_checkpoint(tmp_path / "checkpoint-25", 25, complete=False)
        record = make_store(repo).upload_checkpoint(ckpt, 25)
        assert record["upload_status"] == "skipped"
        assert record["upload_attempted"] is False
        assert "optimizer.pt" in record["reason"]
        assert not repo.files

    def test_full_base_model_weights_are_refused(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        ckpt = make_local_checkpoint(tmp_path / "checkpoint-25", 25)
        (ckpt / "model.safetensors").write_bytes(b"full phi-3 weights!")
        record = make_store(repo).upload_checkpoint(ckpt, 25)
        assert record["upload_status"] == "skipped"
        assert "full-model" in record["reason"]
        assert not repo.files

    def test_oversized_checkpoint_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rp, "REMOTE_CHECKPOINT_MAX_TOTAL_BYTES", 8)
        repo = FakeHubRepo(tmp_path)
        ckpt = make_local_checkpoint(tmp_path / "checkpoint-25", 25)
        record = make_store(repo).upload_checkpoint(ckpt, 25)
        assert record["upload_status"] == "skipped"
        assert "too large" in record["reason"]

    def test_already_persisted_step_is_not_reuploaded(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        record = upload(repo, tmp_path, 25)
        assert record["upload_status"] == "skipped"
        assert record["verified"] is True
        assert "already persisted" in record["reason"]
        assert len(repo.commits) == 1

    def test_records_never_contain_the_token(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        good = upload(repo, tmp_path, 25)
        repo.fail_create_commit = rp.UPLOAD_MAX_ATTEMPTS
        bad = upload(repo, tmp_path, 50)
        assert TOKEN not in json.dumps(good)
        assert TOKEN not in json.dumps(bad)


# ---------------------------------------------------------------------------
# Remote discovery: markers, ordering, rejection of broken candidates
# ---------------------------------------------------------------------------
class TestRemoteDiscovery:
    def test_empty_repo_and_missing_repo_are_empty(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        assert make_store(repo).latest_valid_remote_step() is None
        repo = FakeHubRepo(tmp_path)
        repo.repo_exists = False  # repo never created (first ever run)
        assert make_store(repo).latest_valid_remote_step() is None

    def test_newest_valid_step_selected(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        for step in (25, 50, 75):
            upload(repo, tmp_path, step)
        assert make_store(repo).latest_valid_remote_step() == 75
        steps = [c.step for c in make_store(repo).iter_remote_candidates()]
        assert steps == [75, 50, 25]

    def test_remote_checkpoint_without_marker_rejected(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        # simulate an interrupted commit: files landed, the marker did not
        for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
            repo.files[f"checkpoints/checkpoint-50/{name}"] = b"partial"
        repo.files["checkpoints/checkpoint-50/adapter_model.safetensors"] = b"partial"
        assert make_store(repo).latest_valid_remote_step() == 25

    def test_remote_checkpoint_with_corrupt_marker_rejected(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        repo.files[f"checkpoints/checkpoint-25/{REMOTE_CHECKPOINT_MARKER}"] = b"{not json"
        assert make_store(repo).latest_valid_remote_step() is None

    def test_remote_checkpoint_with_missing_files_rejected(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        del repo.files["checkpoints/checkpoint-25/optimizer.pt"]
        assert make_store(repo).latest_valid_remote_step() is None

    def test_fingerprint_mismatch_rejected(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25, fingerprint=FP)
        store = make_store(repo, fingerprint=FP_OTHER)
        assert store.latest_valid_remote_step() is None
        assert make_store(repo, fingerprint=FP).latest_valid_remote_step() == 25

    def test_listing_failure_propagates(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        repo.repo_exists = True
        repo.fail_listing = True
        with pytest.raises(ConnectionError):
            make_store(repo).latest_valid_remote_step()

    def test_listing_failure_messages_never_leak_the_token(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        repo.repo_exists = True
        repo.fail_listing = True
        store = make_store(repo)
        with pytest.raises(RemoteCheckpointError) as excinfo:
            resolve_resume_checkpoint(tmp_path / "out", resume=True,
                                      explicit_checkpoint=None, store=store)
        assert TOKEN not in str(excinfo.value)
        assert "--resume" in str(excinfo.value)  # actionable message


# ---------------------------------------------------------------------------
# Download / materialization with integrity verification
# ---------------------------------------------------------------------------
class TestRemoteDownload:
    def test_download_materializes_a_valid_checkpoint(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        dest = tmp_path / "out" / "checkpoint-25"
        path = make_store(repo).download_checkpoint(25, dest)
        assert path == dest
        valid, problems = validate_checkpoint_dir(dest)
        assert valid, problems
        assert (dest / "optimizer.pt").read_bytes() == b"optimizer-state-25"
        assert not list(tmp_path.glob(".remote-staging-*"))  # staging cleaned up

    def test_download_rejects_corrupt_remote_bytes(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        # simulate bit-rot / tampering between upload and download
        repo.files["checkpoints/checkpoint-25/adapter_model.safetensors"] = b"corrupted!"
        dest = tmp_path / "out" / "checkpoint-25"
        with pytest.raises(RemoteCheckpointCorrupt):
            make_store(repo).download_checkpoint(25, dest)
        assert not dest.exists()  # nothing half-valid is materialized

    def test_download_without_marker_raises(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        repo.repo_exists = True
        with pytest.raises(RemoteCheckpointError):
            make_store(repo).download_checkpoint(25, tmp_path / "out" / "checkpoint-25")

    def test_download_replaces_an_invalid_local_directory(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        dest = tmp_path / "out" / "checkpoint-25"
        dest.mkdir(parents=True)  # invalid leftovers from a crashed download
        make_store(repo).download_checkpoint(25, dest)
        valid, _ = validate_checkpoint_dir(dest)
        assert valid


# ---------------------------------------------------------------------------
# Resume precedence: explicit > local > remote > scratch
# ---------------------------------------------------------------------------
class TestResumeResolution:
    def test_explicit_invalid_checkpoint_fails_loudly(self, tmp_path):
        bad = make_local_checkpoint(tmp_path / "checkpoint-10", 10, complete=False)
        with pytest.raises(RemoteCheckpointError, match="not a valid resume point"):
            resolve_resume_checkpoint(tmp_path, resume=False,
                                      explicit_checkpoint=str(bad), store=None)

    def test_explicit_valid_checkpoint_is_used_strictly(self, tmp_path):
        good = make_local_checkpoint(tmp_path / "checkpoint-50", 50)
        decision = resolve_resume_checkpoint(tmp_path, resume=False,
                                             explicit_checkpoint=str(good), store=None)
        assert decision.checkpoint_path == str(good)
        assert decision.source == rp.RESUME_SOURCE_LOCAL
        assert decision.step == 50

    def test_resume_prefers_valid_local_over_remote(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 75)
        local_out = tmp_path / "out"
        make_local_checkpoint(local_out / "checkpoint-25", 25)
        decision = resolve_resume_checkpoint(local_out, resume=True,
                                             explicit_checkpoint=None, store=make_store(repo))
        assert decision.source == rp.RESUME_SOURCE_LOCAL
        assert decision.step == 25

    def test_resume_from_remote_when_local_is_gone(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        for step in (25, 50):
            upload(repo, tmp_path, step)
        out = tmp_path / "out"  # fresh runtime: empty output dir
        decision = resolve_resume_checkpoint(out, resume=True,
                                             explicit_checkpoint=None, store=make_store(repo))
        assert decision.source == rp.RESUME_SOURCE_REMOTE
        assert decision.step == 50
        assert decision.repo_id == repo.repo_id
        assert validate_checkpoint_dir(Path(decision.checkpoint_path))[0]

    def test_resume_falls_back_to_older_remote_when_newest_is_corrupt(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        upload(repo, tmp_path, 50)
        repo.files["checkpoints/checkpoint-50/optimizer.pt"] = b"bit-rotted"
        decision = resolve_resume_checkpoint(tmp_path / "out", resume=True,
                                             explicit_checkpoint=None, store=make_store(repo))
        assert decision.source == rp.RESUME_SOURCE_REMOTE
        assert decision.step == 25

    def test_no_local_no_remote_means_fresh_training(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        decision = resolve_resume_checkpoint(tmp_path / "out", resume=True,
                                             explicit_checkpoint=None, store=make_store(repo))
        assert decision.source == rp.RESUME_SOURCE_SCRATCH
        assert decision.checkpoint_path is None

    def test_without_store_only_local_resume_is_attempted(self, tmp_path):
        decision = resolve_resume_checkpoint(tmp_path / "out", resume=True,
                                             explicit_checkpoint=None, store=None)
        assert decision.source == rp.RESUME_SOURCE_SCRATCH

    def test_resume_disabled_is_scratch(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        decision = resolve_resume_checkpoint(tmp_path / "out", resume=False,
                                             explicit_checkpoint=None, store=make_store(repo))
        assert decision.source == rp.RESUME_SOURCE_SCRATCH

    def test_invalid_local_checkpoints_are_skipped_not_fatal(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        upload(repo, tmp_path, 25)
        out = tmp_path / "out"
        make_local_checkpoint(out / "checkpoint-50", 50, complete=False)
        decision = resolve_resume_checkpoint(out, resume=True,
                                             explicit_checkpoint=None, store=make_store(repo))
        assert decision.source == rp.RESUME_SOURCE_REMOTE
        assert decision.step == 25


# ---------------------------------------------------------------------------
# Trainer callback wiring
# ---------------------------------------------------------------------------
class TestRemoteCheckpointCallback:
    def _callback_args(self, tmp_path, step):
        make_local_checkpoint(tmp_path / f"checkpoint-{step}", step)
        return (SimpleNamespace(output_dir=str(tmp_path)),
                SimpleNamespace(global_step=step))

    def test_on_save_uploads_and_records(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        records: list[dict] = []
        callback = RemoteCheckpointCallback(make_store(repo), records)
        args, state = self._callback_args(tmp_path, 25)
        callback.on_save(args, state, control=None)
        assert len(records) == 1
        assert records[0]["upload_status"] == "succeeded"
        assert records[0]["step"] == 25

    def test_on_save_failure_never_raises(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        repo.fail_create_commit = rp.UPLOAD_MAX_ATTEMPTS
        records: list[dict] = []
        callback = RemoteCheckpointCallback(make_store(repo), records)
        args, state = self._callback_args(tmp_path, 25)
        callback.on_save(args, state, control=None)  # must not raise
        assert records[0]["upload_status"] == "failed"

    def test_invalid_checkpoint_dir_does_not_raise(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        records: list[dict] = []
        callback = RemoteCheckpointCallback(make_store(repo), records)
        callback.on_save(SimpleNamespace(output_dir=str(tmp_path)),
                         SimpleNamespace(global_step=25), control=None)
        assert records[0]["upload_status"] == "skipped"


# ---------------------------------------------------------------------------
# Environment gating + experiment fingerprint
# ---------------------------------------------------------------------------
class TestGatingAndFingerprint:
    def test_disabled_without_repo_id(self, monkeypatch):
        for var in ("DOCUTUNE_HF_REPO_ID", "DOCUTUNE_HF_TOKEN", "HF_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        settings = resolve_upload_settings()
        assert settings.enabled is False
        assert TOKEN not in repr(settings)

    def test_configured_repo_id_enables_persistence(self, monkeypatch):
        monkeypatch.setenv("DOCUTUNE_HF_REPO_ID", "rohithronni/docutune-phi3-adapter")
        monkeypatch.setenv("DOCUTUNE_HF_TOKEN", TOKEN)
        settings = resolve_upload_settings()
        assert settings.enabled is True
        assert settings.repo_id == "rohithronni/docutune-phi3-adapter"
        assert TOKEN not in repr(settings)

    def test_fingerprint_is_stable_for_infra_only_changes(self):
        cfg = TrainConfig.from_yaml("configs/train.yaml")
        other = TrainConfig.from_yaml("configs/train.yaml")
        other.training.output_dir = "somewhere/else"
        other.training.save_steps = 999
        other.training.final_adapter_dir = "somewhere/adapter"
        assert checkpoint_fingerprint(cfg) == checkpoint_fingerprint(other)

    def test_fingerprint_changes_with_the_experiment(self):
        cfg = TrainConfig.from_yaml("configs/train.yaml")
        other = TrainConfig.from_yaml("configs/train.yaml")
        other.training.learning_rate = 3e-4
        other.lora.r = 8
        other.model.name = "some/other-model"
        assert checkpoint_fingerprint(cfg) != checkpoint_fingerprint(other)

    def test_training_never_references_the_test_split(self):
        cfg = TrainConfig.from_yaml("configs/train.yaml")
        data_files = json.dumps(vars(cfg.data))
        assert "test" not in data_files

    def test_smoke_test_disables_periodic_checkpointing(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from docutune.training.train import build_training_arguments

        cfg = TrainConfig.from_yaml("configs/train.yaml")
        args = build_training_arguments(cfg, tmp_path, num_train_examples=16,
                                        smoke_test=True)
        save_strategy = getattr(args.save_strategy, "value", args.save_strategy)
        assert save_strategy == "no"  # no checkpoints -> on_save never fires


# ---------------------------------------------------------------------------
# Kaggle end-to-end recovery scenarios (upload -> kill runtime -> resume)
# ---------------------------------------------------------------------------
class TestKaggleRecoveryScenarios:
    def test_scenario_b_resume_after_runtime_loss(self, tmp_path):
        """Checkpoint 25 uploaded, runtime wiped, fresh runtime resumes from
        the remote mirror without network code paths differing from prod."""
        repo = FakeHubRepo(tmp_path)
        first_run_out = tmp_path / "first-runtime"
        record = make_store(repo).upload_checkpoint(
            make_local_checkpoint(first_run_out / "checkpoint-25", 25), 25)
        assert record["upload_status"] == "succeeded"

        fresh_out = tmp_path / "second-runtime"  # nothing local survives
        decision = resolve_resume_checkpoint(fresh_out, resume=True,
                                             explicit_checkpoint=None, store=make_store(repo))
        assert decision.source == rp.RESUME_SOURCE_REMOTE
        assert decision.step == 25
        assert validate_checkpoint_dir(Path(decision.checkpoint_path))[0]

    def test_scenario_f_failed_upload_keeps_local_checkpoint(self, tmp_path):
        repo = FakeHubRepo(tmp_path)
        repo.fail_create_commit = rp.UPLOAD_MAX_ATTEMPTS
        ckpt = make_local_checkpoint(tmp_path / "out" / "checkpoint-25", 25)
        record = make_store(repo).upload_checkpoint(ckpt, 25)
        assert record["upload_status"] == "failed"
        assert validate_checkpoint_dir(ckpt)[0]  # local state fully intact
        assert TOKEN not in (record["reason"] or "")

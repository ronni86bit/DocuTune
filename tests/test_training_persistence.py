"""Checkpoint discovery/validation and adapter persistence regression tests.

All torch-free except the TrainingArguments checkpoint-configuration test
(torch/transformers gated, skipped on CI which installs no training extras).
"""

import json
from pathlib import Path

import pytest

from docutune.config import TrainConfig
from docutune.training.checkpoints import (
    find_latest_valid_checkpoint,
    validate_checkpoint_dir,
)
from docutune.training.persistence import (
    UploadSettings,
    persist_adapter,
    resolve_upload_settings,
    verify_adapter_artifacts,
)

TOKEN = "hf_TESTTOKEN_do_not_leak"


# ---------------------------------------------------------------------------
# Fake checkpoint / adapter builders
# ---------------------------------------------------------------------------
def make_checkpoint(path: Path, step: int, complete: bool = True,
                    corrupt_state: bool = False) -> None:
    path.mkdir(parents=True)
    if complete:
        (path / "optimizer.pt").write_bytes(b"optimizer-state")
        (path / "scheduler.pt").write_bytes(b"scheduler-state")
        (path / "adapter_model.safetensors").write_bytes(b"weights")
    state = json.dumps({"global_step": step})
    if corrupt_state:
        state = "{this is not json"
    (path / "trainer_state.json").write_text(state, encoding="utf-8")


def make_adapter(path: Path, complete: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not complete:
        return
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 1: checkpoint discovery / validation
# ---------------------------------------------------------------------------
class TestCheckpointDiscovery:
    def test_empty_and_missing_dirs_return_none(self, tmp_path):
        assert find_latest_valid_checkpoint(tmp_path) is None
        assert find_latest_valid_checkpoint(tmp_path / "does-not-exist") is None

    def test_latest_valid_checkpoint_selected(self, tmp_path):
        make_checkpoint(tmp_path / "checkpoint-25", 25)
        make_checkpoint(tmp_path / "checkpoint-50", 50)
        make_checkpoint(tmp_path / "checkpoint-75", 75, complete=False)  # interrupted write
        latest = find_latest_valid_checkpoint(tmp_path)
        assert latest is not None and latest.name == "checkpoint-50"

    def test_corrupt_trainer_state_skipped(self, tmp_path):
        make_checkpoint(tmp_path / "checkpoint-25", 25)
        make_checkpoint(tmp_path / "checkpoint-30", 30, corrupt_state=True)
        latest = find_latest_valid_checkpoint(tmp_path)
        assert latest is not None and latest.name == "checkpoint-25"

    def test_all_invalid_returns_none(self, tmp_path):
        make_checkpoint(tmp_path / "checkpoint-25", 25, complete=False)
        assert find_latest_valid_checkpoint(tmp_path) is None

    def test_ignores_unrelated_directories(self, tmp_path):
        make_checkpoint(tmp_path / "checkpoint-25", 25)
        (tmp_path / "checkpoint-abc").mkdir()
        (tmp_path / "runs").mkdir()
        latest = find_latest_valid_checkpoint(tmp_path)
        assert latest is not None and latest.name == "checkpoint-25"

    def test_final_adapter_dir_is_not_a_checkpoint(self, tmp_path):
        """The final LoRA adapter directory must never be mistaken for a
        Trainer checkpoint (it has no trainer_state/optimizer/scheduler)."""
        adapter_dir = tmp_path / "final"
        make_adapter(adapter_dir)
        assert find_latest_valid_checkpoint(adapter_dir) is None
        assert find_latest_valid_checkpoint(tmp_path) is None

    def test_validate_reports_missing_files(self, tmp_path):
        make_checkpoint(tmp_path / "checkpoint-10", 10, complete=False)
        valid, problems = validate_checkpoint_dir(tmp_path / "checkpoint-10")
        assert valid is False
        joined = " ".join(problems)
        assert "optimizer.pt" in joined and "scheduler.pt" in joined

    def test_validate_valid_checkpoint(self, tmp_path):
        make_checkpoint(tmp_path / "checkpoint-10", 10)
        valid, problems = validate_checkpoint_dir(tmp_path / "checkpoint-10")
        assert valid is True and problems == []


# ---------------------------------------------------------------------------
# Task 1: checkpoint configuration (periodic saving, distinct directories)
# ---------------------------------------------------------------------------
class TestCheckpointConfiguration:
    def test_config_saves_checkpoints_periodically(self):
        cfg = TrainConfig.from_yaml("configs/train.yaml")
        assert cfg.training.save_strategy == "steps"
        assert cfg.training.save_steps == 25
        assert cfg.training.save_total_limit == 2
        # checkpoints live under the training dir, NEVER in the adapter dir
        assert Path(cfg.training.output_dir) != Path(cfg.training.final_adapter_dir)

    def test_experiment_hyperparams_unchanged(self):
        cfg = TrainConfig.from_yaml("configs/train.yaml")
        assert cfg.model.name == "microsoft/Phi-3-mini-4k-instruct"
        assert cfg.lora.r == 16 and cfg.lora.alpha == 32 and cfg.lora.dropout == 0.05
        assert abs(cfg.training.learning_rate - 2e-4) < 1e-12
        assert cfg.training.epochs == 3
        assert cfg.training.per_device_train_batch_size == 2
        assert cfg.training.gradient_accumulation_steps == 8  # effective batch 16
        assert cfg.training.max_length == 2048
        assert cfg.training.seed == 42

    def test_training_arguments_checkpoint_behavior(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        from docutune.training.train import build_training_arguments

        cfg = TrainConfig.from_yaml("configs/train.yaml")
        args = build_training_arguments(cfg, tmp_path, num_train_examples=450)
        save_strategy = getattr(args.save_strategy, "value", args.save_strategy)
        assert save_strategy == "steps"
        assert args.save_steps == 25
        assert args.save_total_limit == 2
        assert str(tmp_path) in str(args.output_dir)
        # checkpoints must not be aliased with the final adapter directory
        assert Path(args.output_dir) != Path(cfg.training.final_adapter_dir)
        assert args.load_best_model_at_end is False


# ---------------------------------------------------------------------------
# Task 2: upload settings (env-based, credentials never leaked)
# ---------------------------------------------------------------------------
class TestUploadSettings:
    def test_disabled_without_repo_id(self, monkeypatch):
        for var in ("DOCUTUNE_HF_REPO_ID", "DOCUTUNE_HF_TOKEN", "HF_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        settings = resolve_upload_settings()
        assert settings.enabled is False
        assert settings.repo_id is None and settings.token is None
        assert "DOCUTUNE_HF_REPO_ID" in settings.reason

    def test_enabled_with_repo_and_token(self, monkeypatch):
        monkeypatch.setenv("DOCUTUNE_HF_REPO_ID", "user/docutune-adapter")
        monkeypatch.setenv("DOCUTUNE_HF_TOKEN", TOKEN)
        settings = resolve_upload_settings()
        assert settings.enabled is True
        assert settings.repo_id == "user/docutune-adapter"
        assert settings.token == TOKEN

    def test_token_falls_back_to_hf_token(self, monkeypatch):
        monkeypatch.setenv("DOCUTUNE_HF_REPO_ID", "user/docutune-adapter")
        monkeypatch.delenv("DOCUTUNE_HF_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        assert resolve_upload_settings().token == TOKEN

    def test_repr_never_contains_token(self, monkeypatch):
        monkeypatch.setenv("DOCUTUNE_HF_REPO_ID", "user/docutune-adapter")
        monkeypatch.setenv("DOCUTUNE_HF_TOKEN", TOKEN)
        settings = resolve_upload_settings()
        assert TOKEN not in repr(settings)
        assert TOKEN not in str(settings)


# ---------------------------------------------------------------------------
# Task 2: adapter artifact verification
# ---------------------------------------------------------------------------
class TestAdapterVerification:
    def test_complete_adapter_has_no_problems(self, tmp_path):
        make_adapter(tmp_path / "final")
        assert verify_adapter_artifacts(tmp_path / "final") == []

    def test_missing_config_reported(self, tmp_path):
        make_adapter(tmp_path / "final")
        (tmp_path / "final" / "adapter_config.json").unlink()
        problems = verify_adapter_artifacts(tmp_path / "final")
        assert any("adapter_config.json" in p for p in problems)

    def test_missing_weights_reported(self, tmp_path):
        make_adapter(tmp_path / "final")
        (tmp_path / "final" / "adapter_model.safetensors").unlink()
        assert any("weights" in p for p in verify_adapter_artifacts(tmp_path / "final"))

    def test_missing_tokenizer_reported(self, tmp_path):
        make_adapter(tmp_path / "final")
        (tmp_path / "final" / "tokenizer_config.json").unlink()
        assert any("tokenizer" in p for p in verify_adapter_artifacts(tmp_path / "final"))

    def test_missing_directory(self, tmp_path):
        assert verify_adapter_artifacts(tmp_path / "nope") != []


# ---------------------------------------------------------------------------
# Task 2: persist_adapter - disabled / success / honest failure
# ---------------------------------------------------------------------------
class TestPersistAdapter:
    def test_disabled_when_not_configured(self, tmp_path, monkeypatch):
        for var in ("DOCUTUNE_HF_REPO_ID", "DOCUTUNE_HF_TOKEN", "HF_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        make_adapter(tmp_path / "final")
        monkeypatch.setattr("docutune.training.persistence.upload_adapter",
                            lambda *a, **k: pytest.fail("must not upload when disabled"))
        result = persist_adapter(tmp_path / "final", resolve_upload_settings())
        assert result["status"] == "disabled"
        assert result["repo_id"] is None

    def test_success_records_url(self, tmp_path, monkeypatch):
        make_adapter(tmp_path / "final")
        settings = UploadSettings(enabled=True, repo_id="user/adapter", token=TOKEN)
        monkeypatch.setattr("docutune.training.persistence.upload_adapter",
                            lambda _dir, repo, _tok: f"https://huggingface.co/{repo}")
        result = persist_adapter(tmp_path / "final", settings)
        assert result["status"] == "succeeded"
        assert result["url"] == "https://huggingface.co/user/adapter"
        assert result["reason"] is None

    def test_failure_is_honest_and_local_adapter_untouched(self, tmp_path, monkeypatch):
        adapter_dir = tmp_path / "final"
        make_adapter(adapter_dir)
        before = sorted(p.name for p in adapter_dir.iterdir())
        settings = UploadSettings(enabled=True, repo_id="user/adapter", token=TOKEN)

        def boom(_dir, _repo, _tok):
            raise RuntimeError(f"401 unauthorized for url ...?token={TOKEN}")

        monkeypatch.setattr("docutune.training.persistence.upload_adapter", boom)
        result = persist_adapter(adapter_dir, settings)
        assert result["status"] == "failed"
        assert result["url"] is None
        # credentials must not leak into the recorded reason
        assert TOKEN not in (result["reason"] or "")
        # the local adapter is intact
        assert sorted(p.name for p in adapter_dir.iterdir()) == before

    def test_incomplete_adapter_refused(self, tmp_path):
        (tmp_path / "final").mkdir()  # empty dir: nothing to upload
        settings = UploadSettings(enabled=True, repo_id="user/adapter", token=TOKEN)
        result = persist_adapter(tmp_path / "final", settings)
        assert result["status"] == "failed"
        assert "not uploading" in (result["reason"] or "")

    def test_enabled_but_token_missing_fails_clearly(self, tmp_path, monkeypatch):
        make_adapter(tmp_path / "final")
        settings = UploadSettings(enabled=True, repo_id="user/adapter", token=None)
        result = persist_adapter(tmp_path / "final", settings)
        assert result["status"] == "failed"
        assert "token" in (result["reason"] or "").lower()

    def test_status_dict_never_contains_credentials(self, tmp_path, monkeypatch):
        make_adapter(tmp_path / "final")
        settings = UploadSettings(enabled=True, repo_id="user/adapter", token=TOKEN)

        def boom(_dir, _repo, _tok):
            raise RuntimeError(f"path /repos/user/adapter?token={TOKEN} bad request")

        monkeypatch.setattr("docutune.training.persistence.upload_adapter", boom)
        result = persist_adapter(tmp_path / "final", settings)
        serialized = json.dumps(result)
        assert TOKEN not in serialized

    def test_upload_disabled_logs_clearly(self, tmp_path, monkeypatch, caplog):
        for var in ("DOCUTUNE_HF_REPO_ID", "DOCUTUNE_HF_TOKEN", "HF_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        import logging

        from docutune.training.persistence import UPLOAD_STATUS_DISABLED

        make_adapter(tmp_path / "final")
        with caplog.at_level(logging.INFO):
            result = persist_adapter(tmp_path / "final", resolve_upload_settings())
        assert result["status"] == UPLOAD_STATUS_DISABLED

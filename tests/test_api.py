"""API tests with a fake model manager - no model download, no torch."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


class FakeManager:
    def __init__(self, results_dir: Path, adapter_available: bool = True):
        self.settings = Settings(
            results_dir=results_dir,
            base_model="test-model",
            model_mode="finetuned",
            adapter_path=str(results_dir / "adapter"),
        )
        self.adapter_available_flag = adapter_available
        self.calls = 0

    def adapter_available(self) -> bool:
        return self.adapter_available_flag

    def describe(self):
        return {
            "base_model": self.settings.base_model,
            "model_revision": None,
            "default_model_mode": self.settings.model_mode,
            "adapter_path": self.settings.adapter_path,
            "adapter_available": self.adapter_available(),
            "device": "cpu",
            "quantized": False,
        }

    def extract(self, text: str, mode: str):
        self.calls += 1
        if mode == "finetuned" and not self.adapter_available_flag:
            from backend.app.services.model_manager import ModelUnavailableError

            raise ModelUnavailableError(
                "Fine-tuned adapter not found. Run the Colab training pipeline "
                "and configure ADAPTER_PATH."
            )
        parsed = {
            "name": "Fake Person", "email": None, "phone": None, "location": None,
            "summary": None, "skills": ["Python"], "education": [], "experience": [],
            "projects": [], "certifications": [],
        }
        return type(
            "Outcome", (),
            {
                "raw_output": json.dumps(parsed),
                "parsed_output": parsed,
                "json_valid": True,
                "schema_valid": True,
                "schema_error": None,
                "latency_ms": 12.3,
                "model_name": self.settings.base_model,
                "adapter_path": self.settings.adapter_path if mode == "finetuned" else None,
                "error": None,
            },
        )()


@pytest.fixture
def client(tmp_path):
    manager = FakeManager(tmp_path)
    app = create_app(settings=manager.settings, manager=manager)
    return TestClient(app), manager


def test_root(client):
    http, _ = client
    response = http.get("/")
    assert response.status_code == 200
    assert response.json()["service"].startswith("DocuTune")


def test_health(client):
    http, manager = client
    response = http.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_mode"] == "finetuned"


def test_metadata(client):
    http, _ = client
    body = http.get("/api/metadata").json()
    assert body["base_model"] == "test-model"
    assert body["schema_version"] == "1.0"
    assert body["prompt_version"] == "1.0"
    assert body["adapter_available"] is True


def test_extract_ok(client):
    http, manager = client
    response = http.post("/api/extract", json={"text": "John Doe, engineer"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "finetuned"
    assert body["json_valid"] is True
    assert body["schema_valid"] is True
    assert body["parsed_output"]["name"] == "Fake Person"
    assert manager.calls == 1


def test_extract_empty_rejected(client):
    http, _ = client
    response = http.post("/api/extract", json={"text": "   "})
    assert response.status_code == 422


def test_extract_too_long_rejected(client):
    http, _ = client
    response = http.post("/api/extract", json={"text": "x" * 30000})
    assert response.status_code == 422


def test_extract_model_override(client):
    http, manager = client
    response = http.post("/api/extract", json={"text": "resume", "model": "base"})
    assert response.status_code == 200
    assert response.json()["model"] == "base"
    assert response.json()["adapter_path"] is None


def test_extract_adapter_missing_is_503_with_clear_message(tmp_path):
    manager = FakeManager(tmp_path, adapter_available=False)
    app = create_app(settings=manager.settings, manager=manager)
    http = TestClient(app)
    response = http.post("/api/extract", json={"text": "resume", "model": "finetuned"})
    assert response.status_code == 503
    assert "Fine-tuned adapter not found" in response.json()["detail"]
    assert "ADAPTER_PATH" in response.json()["detail"]


def test_compare_ok(client):
    http, _ = client
    response = http.post("/api/compare", json={"text": "resume body"})
    assert response.status_code == 200
    body = response.json()
    assert body["base"]["json_valid"]
    assert body["finetuned"]["schema_valid"]


def test_compare_partial_when_adapter_missing(tmp_path):
    manager = FakeManager(tmp_path, adapter_available=False)
    app = create_app(settings=manager.settings, manager=manager)
    http = TestClient(app)
    response = http.post("/api/compare", json={"text": "resume body"})
    assert response.status_code == 200
    body = response.json()
    assert body["finetuned"] is None
    assert "ADAPTER_PATH" in body["finetuned_error"]


def test_metrics_unavailable_state(client):
    http, _ = client
    response = http.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "have not been generated" in body["message"]


def test_metrics_present(client, tmp_path):
    metrics = {
        "metadata": {"base_model": "test-model", "test_size": 4},
        "base": {"schema_validity": 0.5, "field_f1": 0.6},
        "finetuned": {"schema_validity": 0.75, "field_f1": 0.9},
        "delta": {"schema_validity": 0.25},
        "per_field": [{"field": "name", "base_f1": 0.6, "finetuned_f1": 0.9,
                       "base_precision": 0.6, "finetuned_precision": 0.9,
                       "base_recall": 0.6, "finetuned_recall": 0.9, "delta_f1": 0.3}],
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    http, _ = client
    body = http.get("/api/metrics").json()
    assert body["available"] is True
    assert body["base"]["schema_validity"] == 0.5
    summary = http.get("/api/benchmark/summary").json()
    assert summary["available"] is True
    assert summary["per_field"][0]["field"] == "name"
    assert summary["error_records"] == []

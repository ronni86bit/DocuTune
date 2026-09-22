"""Benchmark orchestration: incremental saves, resume, cache validation,
and the README metric updater."""

import json

import pytest

from docutune.evaluation.benchmark import (
    cache_fingerprint,
    predictions_have_all,
    run_inference_pass,
)
from docutune.evaluation.reporting import (
    README_BULLET_START,
    README_TABLE_START,
    render_readme_table,
    update_readme_metrics,
)


def _fingerprint(tmp_path, kind="base"):
    return cache_fingerprint(
        kind=kind,
        model_name="test-model",
        model_revision=None,
        adapter_path=None,
        generation_config={"max_new_tokens": 64, "do_sample": False, "num_beams": 1},
        test_file=str(tmp_path / "missing.jsonl"),
    )


def test_inference_pass_writes_predictions(fake_extractor, small_dataset, tmp_path):
    examples = small_dataset["examples"][:4]
    out = tmp_path / "preds.jsonl"
    rows = run_inference_pass("base", examples, str(out), _fingerprint(tmp_path),
                              fake_extractor, resume=True, force=True)
    assert len(rows) == 4
    assert all(row["id"] for row in rows)
    stored = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored) == 4
    assert (tmp_path / "preds.jsonl.cache.json").is_file()


def test_resume_skips_completed_work(fake_extractor, small_dataset, tmp_path):
    examples = small_dataset["examples"][:4]
    out = tmp_path / "preds.jsonl"
    fingerprint = _fingerprint(tmp_path)
    run_inference_pass("base", examples[:2], str(out), fingerprint, fake_extractor,
                       resume=True, force=True)
    calls_after_first = fake_extractor.calls
    rows = run_inference_pass("base", examples, str(out), fingerprint, fake_extractor,
                              resume=True, force=False)
    assert len(rows) == 4
    assert fake_extractor.calls - calls_after_first == 2  # only the remaining two ran
    # Order preserved despite the interrupted first pass
    assert [row["id"] for row in rows] == [ex["id"] for ex in examples]


def test_stale_cache_refused(fake_extractor, small_dataset, tmp_path):
    examples = small_dataset["examples"][:2]
    out = tmp_path / "preds.jsonl"
    run_inference_pass("base", examples, str(out), _fingerprint(tmp_path, "base"),
                       fake_extractor, resume=True, force=True)
    with pytest.raises(RuntimeError, match="--force"):
        run_inference_pass("base", examples, str(out), _fingerprint(tmp_path, "finetuned"),
                           fake_extractor, resume=True, force=False)


def test_force_recomputes(fake_extractor, small_dataset, tmp_path):
    examples = small_dataset["examples"][:2]
    out = tmp_path / "preds.jsonl"
    fingerprint = _fingerprint(tmp_path)
    run_inference_pass("base", examples, str(out), fingerprint, fake_extractor,
                       resume=True, force=True)
    calls_before = fake_extractor.calls
    run_inference_pass("base", examples, str(out), fingerprint, fake_extractor,
                       resume=True, force=True)
    assert fake_extractor.calls == calls_before + 2


def test_predictions_have_all(fake_extractor, small_dataset, tmp_path):
    examples = small_dataset["examples"][:3]
    out = tmp_path / "preds.jsonl"
    fingerprint = _fingerprint(tmp_path)
    assert not predictions_have_all(str(out), examples, fingerprint)
    run_inference_pass("base", examples[:2], str(out), fingerprint, fake_extractor,
                       resume=True, force=True)
    assert not predictions_have_all(str(out), examples, fingerprint)
    run_inference_pass("base", examples[2:], str(out), fingerprint, fake_extractor,
                       resume=True, force=False)
    assert predictions_have_all(str(out), examples, fingerprint)


def test_readme_metrics_update(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# DocuTune\n\n"
        f"{README_TABLE_START}\nTBD table here\n<!-- BENCHMARK-TABLE:END -->\n\n"
        f"{README_BULLET_START}\nTBD\n<!-- RESUME-BULLET:END -->\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.json"

    # Refuses to invent numbers when nothing was measured.
    with pytest.raises(FileNotFoundError):
        update_readme_metrics(str(metrics_path), str(readme))
    assert "TBD table here" in readme.read_text(encoding="utf-8")

    metrics = {
        "metadata": {"base_model": "microsoft/Phi-3-mini-4k-instruct", "test_size": 75,
                     "train_size": 450},
        "base": {"json_validity": 0.9, "schema_validity": 0.8, "exact_match": 0.1,
                 "field_precision": 0.7, "field_recall": 0.6, "field_f1": 0.65,
                 "unsupported_value_rate": 0.05, "missing_value_rate": 0.2,
                 "latency_mean_ms": 400.0, "latency_p95_ms": 800.0},
        "finetuned": {"json_validity": 1.0, "schema_validity": 0.98, "exact_match": 0.5,
                      "field_precision": 0.95, "field_recall": 0.93, "field_f1": 0.94,
                      "unsupported_value_rate": 0.01, "missing_value_rate": 0.05,
                      "latency_mean_ms": 410.0, "latency_p95_ms": 820.0},
        "delta": {},
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    update_readme_metrics(str(metrics_path), str(readme))
    text = readme.read_text(encoding="utf-8")
    assert "| Schema Validity | 80.0% | 98.0% | +18.0pp |" in text
    assert "Fine-tuned microsoft/Phi-3-mini-4k-instruct using LoRA/QLoRA" in text
    assert "TBD table here" not in text


def test_render_readme_table_tbd_when_no_metrics():
    table = render_readme_table(None)
    assert table.count("TBD") == 24  # 8 rows x (base + finetuned + delta)

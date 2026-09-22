"""Benchmark orchestration: resumable inference passes with prediction
caching, plus shared metric computation for base and fine-tuned models.

Resumability: predictions are appended to JSONL incrementally (flush per
row). A sidecar `<predictions>.cache.json` stores the cache fingerprint
(model, revision, adapter, dataset version, prompt version, evaluator
version, generation config, test-file hash). A mismatched fingerprint
refuses to reuse stale predictions unless --force is given.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from docutune.config import (
    DATASET_VERSION,
    EVALUATOR_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from docutune.evaluation.metrics import (
    evaluate_example,
    evaluate_predictions,
)
from docutune.evaluation.parser import parse_model_output
from docutune.schema.resume import ResumeExtraction, schema_error_message
from docutune.utils.io import read_json, read_jsonl, sha256_file, write_json
from docutune.utils.logging import get_logger

logger = get_logger(__name__)

ExtractFn = Callable[[str], dict[str, Any]]


def cache_fingerprint(
    kind: str,
    model_name: str,
    model_revision: str | None,
    adapter_path: str | None,
    generation_config: dict[str, Any],
    test_file: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "model": model_name,
        "model_revision": model_revision,
        "adapter": adapter_path,
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generation_config": generation_config,
        "test_file_sha256": sha256_file(test_file) if os.path.exists(test_file) else None,
    }


def _cache_sidecar(predictions_path: str) -> str:
    return f"{predictions_path}.cache.json"


def _row_from_extract(ex_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ex_id,
        "raw_output": outcome.get("raw_output", ""),
        "parsed_output": outcome.get("parsed_output"),
        "json_valid": bool(outcome.get("json_valid", False)),
        "schema_valid": bool(outcome.get("schema_valid", False)),
        "schema_error": outcome.get("schema_error"),
        "latency_ms": outcome.get("latency_ms"),
    }


def run_inference_pass(
    kind: str,
    examples: list[dict[str, Any]],
    predictions_path: str,
    fingerprint: dict[str, Any],
    extract_fn: ExtractFn,
    resume: bool = True,
    force: bool = False,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Run (or resume) inference over the test set, saving incrementally.

    `extract_fn(resume_text)` must return a dict with raw_output,
    parsed_output, json_valid, schema_valid, schema_error, latency_ms.
    """
    if max_examples is not None:
        examples = examples[:max_examples]

    sidecar = _cache_sidecar(predictions_path)
    done_ids: set[str] = set()

    if os.path.exists(predictions_path) and os.path.exists(sidecar) and resume and not force:
        cached = read_json(sidecar)
        if cached != fingerprint:
            raise RuntimeError(
                f"Prediction cache for '{kind}' does not match the current configuration "
                f"(model/prompt/dataset/generation changed). Re-run with --force to recompute."
            )
        done_ids = {row["id"] for row in read_jsonl(predictions_path) if row.get("id")}
        logger.info("Resuming %s inference: %d/%d predictions already present",
                    kind, len(done_ids), len(examples))
    elif force or not resume:
        for path in (predictions_path, sidecar):
            if os.path.exists(path):
                os.remove(path)

    write_json(sidecar, fingerprint)

    for ex in examples:
        if ex["id"] in done_ids:
            continue
        outcome = extract_fn(ex["text"])
        row = _row_from_extract(ex["id"], outcome)
        append_safe(predictions_path, row)
        done_ids.add(ex["id"])

    rows = read_jsonl(predictions_path)
    rows_by_id = {row["id"]: row for row in rows if row.get("id")}
    ordered = [rows_by_id[ex["id"]] for ex in examples if ex["id"] in rows_by_id]
    return ordered


def append_safe(path: str, record: dict[str, Any]) -> None:
    """Append one JSONL row, flushing to disk (crash-safe incremental saves)."""
    from docutune.utils.io import append_jsonl

    append_jsonl(path, record)


def predictions_have_all(predictions_path: str, examples: list[dict[str, Any]],
                         fingerprint: dict[str, Any]) -> bool:
    sidecar = _cache_sidecar(predictions_path)
    if not (os.path.exists(predictions_path) and os.path.exists(sidecar)):
        return False
    try:
        cached = read_json(sidecar)
    except Exception:
        return False
    if cached != fingerprint:
        return False
    done_ids = {row.get("id") for row in read_jsonl(predictions_path)}
    return all(ex["id"] in done_ids for ex in examples)


def parse_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    """Re-parse a stored prediction row (used when computing metrics from disk)."""
    parsed = parse_model_output(row.get("raw_output") or "")
    schema_valid = False
    schema_error = None
    if parsed.is_object:
        try:
            ResumeExtraction.model_validate(parsed.parsed)
            schema_valid = True
        except Exception as exc:
            schema_error = schema_error_message(exc)
    return {
        "id": row.get("id"),
        "parsed": parsed.parsed if parsed.is_object else None,
        "json_valid": parsed.json_valid,
        "schema_valid": schema_valid,
        "schema_error": schema_error,
    }


def evaluate_model_predictions(predictions: list[dict[str, Any]],
                               examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for one model (shared pipeline for base and fine-tuned)."""
    return evaluate_predictions(predictions, examples)


def per_example_evaluations(predictions: list[dict[str, Any]],
                            examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    gold_by_id = {ex["id"]: ex for ex in examples}
    out: dict[str, dict[str, Any]] = {}
    for pred in predictions:
        ex = gold_by_id.get(pred.get("id"))
        if ex is None:
            continue
        parsed = parse_model_output(pred.get("raw_output") or "")
        ev = evaluate_example(parsed.parsed if parsed.is_object else None, ex["target"])
        ev["json_valid"] = parsed.json_valid
        ev["schema_valid"] = False
        if parsed.is_object:
            try:
                ResumeExtraction.model_validate(parsed.parsed)
                ev["schema_valid"] = True
            except Exception:
                pass
        out[pred["id"]] = ev
    return out


def build_comparison_rows(base_predictions: list[dict[str, Any]],
                          finetuned_predictions: list[dict[str, Any]],
                          examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-example comparison records (results/comparison.jsonl)."""
    base_evals = per_example_evaluations(base_predictions, examples)
    ft_evals = per_example_evaluations(finetuned_predictions, examples)
    rows = []
    for ex in examples:
        ex_id = ex["id"]
        b, f = base_evals.get(ex_id, {}), ft_evals.get(ex_id, {})
        rows.append({
            "id": ex_id,
            "difficulty": ex["metadata"]["difficulty"],
            "template_id": ex["metadata"]["template_id"],
            "gold": ex["target"],
            "base": {
                "raw_output": next((p.get("raw_output") for p in base_predictions
                                    if p.get("id") == ex_id), None),
                "json_valid": b.get("json_valid"),
                "schema_valid": b.get("schema_valid"),
                "exact_match": b.get("exact_match"),
                "field_f1": b.get("field_f1"),
            },
            "finetuned": {
                "raw_output": next((p.get("raw_output") for p in finetuned_predictions
                                    if p.get("id") == ex_id), None),
                "json_valid": f.get("json_valid"),
                "schema_valid": f.get("schema_valid"),
                "exact_match": f.get("exact_match"),
                "field_f1": f.get("field_f1"),
            },
            "field_f1_delta": round(
                (f.get("field_f1") or 0.0) - (b.get("field_f1") or 0.0), 4),
        })
    return rows

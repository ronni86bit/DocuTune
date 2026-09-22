"""Evaluation metrics for structured resume extraction.

One shared, model-agnostic evaluator: identical parser, normalization and
metric code for the base and the fine-tuned model. Only the model state
differs between the two evaluation passes (see docs/EVALUATION.md).

Unit accounting per field:
- scalar fields: one unit when non-null
- skills: one unit per item (multiset matching)
- record lists (education/experience/projects/certifications): one unit per
  item, matched greedily by exact normalized-item equality

Under these definitions field precision failures (fp) ARE unsupported
predicted units, and the unsupported-value rate equals fp / predicted units.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import Any

from docutune.config import EVALUATOR_VERSION, PROMPT_VERSION, SCHEMA_VERSION
from docutune.evaluation.normalization import (
    normalize_date,
    normalize_email,
    normalize_phone,
    normalize_record,
    normalize_scalar,
    records_equal,
)
from docutune.evaluation.parser import parse_model_output

SCALAR_FIELDS = ("name", "email", "phone", "location", "summary")
LIST_FIELDS = ("skills",)
RECORD_LIST_FIELDS = ("education", "experience", "projects", "certifications")
ALL_FIELDS = (*SCALAR_FIELDS, *LIST_FIELDS, *RECORD_LIST_FIELDS)

# Per-field normalizers - both sides of a comparison must use the same one.
_SCALAR_NORMALIZERS = {
    "name": normalize_scalar,
    "email": normalize_email,
    "phone": normalize_phone,
    "location": normalize_scalar,
    "summary": normalize_scalar,
}


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------
def _scalar_stats(field: str, pred: Any, gold_norm: str | None) -> dict[str, int]:
    normalizer = _SCALAR_NORMALIZERS[field]
    pred_norm = normalizer(pred)
    if pred_norm is None:
        return {"tp": 0, "fp": 0, "fn": 1 if gold_norm is not None else 0}
    if gold_norm is not None and pred_norm == gold_norm:
        return {"tp": 1, "fp": 0, "fn": 0}
    return {"tp": 0, "fp": 1, "fn": 1 if gold_norm is not None else 0}


def _string_list_stats(pred: Any, gold_norm: list[str] | None) -> dict[str, int]:
    from docutune.evaluation.normalization import _normalize_string_list

    if pred is None:
        return {"tp": 0, "fp": 0, "fn": len(gold_norm or [])}
    if not isinstance(pred, list):
        # Wrong JSON type: one wrong unit (never coerced).
        return {"tp": 0, "fp": 1, "fn": 1 if gold_norm else 0}
    pred_norm = _normalize_string_list(pred)
    if pred_norm is None:
        return {"tp": 0, "fp": 1, "fn": 1 if gold_norm else 0}
    gold_counter: dict[str, int] = {}
    for item in gold_norm or []:
        gold_counter[item] = gold_counter.get(item, 0) + 1
    tp = 0
    for item in pred_norm:
        if gold_counter.get(item, 0) > 0:
            gold_counter[item] -= 1
            tp += 1
    fp = len(pred_norm) - tp
    fn = len(gold_norm or []) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def _record_list_stats(kind: str, pred: Any, gold_items: list[dict[str, Any]]) -> dict[str, int]:
    from docutune.evaluation.normalization import _normalize_item, json_key

    if pred is None:
        return {"tp": 0, "fp": 0, "fn": len(gold_items)}
    if not isinstance(pred, list):
        return {"tp": 0, "fp": 1, "fn": 1 if gold_items else 0}
    gold_keys = [json_key(_normalize_item(kind, item)) for item in gold_items]
    pred_keys: list[str] = []
    for item in pred:
        norm = _normalize_item(kind, item)
        if norm is None:
            return {"tp": 0, "fp": 1, "fn": 1 if gold_items else 0}
        pred_keys.append(json_key(norm))
    available = list(gold_keys)
    tp = 0
    for key in pred_keys:
        if key in available:
            available.remove(key)
            tp += 1
    fp = len(pred_keys) - tp
    fn = len(gold_keys) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def _field_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)):
        return len(value) == 0
    if isinstance(value, str):
        return normalize_scalar(value) is None
    return False


def evaluate_example(parsed_output: Any, gold_target: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one parsed prediction against one gold target.

    `parsed_output` is the raw JSON value produced by parse_model_output
    (must be a dict to count as content). Returns per-field stats plus
    exact_match, per-example F1, unsupported/missing accounting.
    """
    gold_norm = normalize_record(gold_target)
    pred_is_obj = isinstance(parsed_output, dict)
    pred_norm = normalize_record(parsed_output) if pred_is_obj else {}

    per_field: dict[str, dict[str, int]] = {}
    for f in SCALAR_FIELDS:
        per_field[f] = _scalar_stats(f, parsed_output.get(f) if pred_is_obj else None, gold_norm[f])
    per_field["skills"] = _string_list_stats(
        parsed_output.get("skills") if pred_is_obj else None, gold_norm["skills"]
    )
    for kind in RECORD_LIST_FIELDS:
        per_field[kind] = _record_list_stats(
            kind,
            parsed_output.get(kind) if pred_is_obj else None,
            gold_norm[kind] or [],
        )

    tp = sum(s["tp"] for s in per_field.values())
    fp = sum(s["fp"] for s in per_field.values())
    fn = sum(s["fn"] for s in per_field.values())
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 1.0

    predicted_units = sum(
        s["tp"] + s["fp"] for s in per_field.values()
    )
    unsupported_units = fp

    # Field-level missing accounting: gold non-empty but prediction empty.
    # Unparseable (non-object) predictions miss every gold field.
    missing_fields = 0
    gold_nonempty_fields = 0
    for f in ALL_FIELDS:
        if _field_is_empty(gold_target.get(f)):
            continue
        gold_nonempty_fields += 1
        pred_value = parsed_output.get(f) if pred_is_obj else None
        if _field_is_empty(pred_value):
            missing_fields += 1

    exact_match = records_equal(pred_norm, gold_norm)
    return {
        "exact_match": exact_match,
        "per_field": per_field,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "field_f1": f1,
        "predicted_units": predicted_units,
        "unsupported_units": unsupported_units,
        "missing_fields": missing_fields,
        "gold_nonempty_fields": gold_nonempty_fields,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * len(ordered) + 0.5)) - 1))
    return ordered[k]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def aggregate_example_metrics(evals: Iterable[dict[str, Any]],
                              latencies: list[float] | None = None,
                              output_lengths: list[int] | None = None) -> dict[str, Any]:
    evals = list(evals)
    n = len(evals)
    totals = {f: {"tp": 0, "fp": 0, "fn": 0} for f in ALL_FIELDS}
    for ev in evals:
        for f, s in ev["per_field"].items():
            totals[f]["tp"] += s["tp"]
            totals[f]["fp"] += s["fp"]
            totals[f]["fn"] += s["fn"]

    per_field: dict[str, dict[str, Any]] = {}
    for f, t in totals.items():
        precision, recall, f1 = _prf(t["tp"], t["fp"], t["fn"])
        per_field[f] = {**t, "precision": round(precision, 4), "recall": round(recall, 4),
                        "f1": round(f1, 4)}

    micro_tp = sum(t["tp"] for t in totals.values())
    micro_fp = sum(t["fp"] for t in totals.values())
    micro_fn = sum(t["fn"] for t in totals.values())
    p, r, f1 = _prf(micro_tp, micro_fp, micro_fn)

    predicted_units = sum(ev.get("predicted_units", 0) for ev in evals)
    unsupported_units = sum(ev.get("unsupported_units", 0) for ev in evals)
    gold_units = sum(ev.get("gold_nonempty_fields", 0) for ev in evals)
    missing_units = sum(ev.get("missing_fields", 0) for ev in evals)

    metrics: dict[str, Any] = {
        "n_examples": n,
        "field_precision": round(p, 4),
        "field_recall": round(r, 4),
        "field_f1": round(f1, 4),
        "unsupported_value_rate": round(unsupported_units / predicted_units, 4)
        if predicted_units else 0.0,
        "missing_value_rate": round(missing_units / gold_units, 4) if gold_units else 0.0,
        "per_field": per_field,
    }
    if latencies:
        metrics["latency_mean_ms"] = round(statistics.mean(latencies), 1)
        metrics["latency_p50_ms"] = round(_percentile(latencies, 50), 1)
        metrics["latency_p95_ms"] = round(_percentile(latencies, 95), 1)
        metrics["latency_samples"] = len(latencies)
    if output_lengths:
        metrics["output_length_mean_chars"] = round(statistics.mean(output_lengths), 1)
    return metrics


def evaluate_predictions(predictions: list[dict[str, Any]],
                         examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Full metric computation over prediction rows + gold examples.

    Predictions with invalid JSON are treated as empty extractions: they
    count as failures in json_validity / schema_validity / exact_match and
    produce recall failures (never silent drops).
    """
    gold_by_id = {ex["id"]: ex for ex in examples}
    rows = []
    latencies: list[float] = []
    output_lengths: list[int] = []
    for pred in predictions:
        ex = gold_by_id.get(pred.get("id"))
        if ex is None:
            continue
        parsed = parse_model_output(pred.get("raw_output") or "")
        schema_valid = False
        if parsed.is_object:
            from docutune.schema.resume import ResumeExtraction

            try:
                ResumeExtraction.model_validate(parsed.parsed)
                schema_valid = True
            except Exception:
                schema_valid = False
        ev = evaluate_example(parsed.parsed if parsed.is_object else None, ex["target"])
        ev.update({
            "id": pred.get("id"),
            "json_valid": parsed.json_valid,
            "schema_valid": schema_valid,
        })
        rows.append(ev)
        if isinstance(pred.get("latency_ms"), (int, float)):
            latencies.append(float(pred["latency_ms"]))
        output_lengths.append(len(pred.get("raw_output") or ""))

    metrics = aggregate_example_metrics(rows, latencies, output_lengths)
    metrics["json_validity"] = (
        round(sum(1 for r in rows if r["json_valid"]) / len(rows), 4) if rows else 0.0
    )
    metrics["schema_validity"] = (
        round(sum(1 for r in rows if r["schema_valid"]) / len(rows), 4) if rows else 0.0
    )
    metrics["exact_match"] = (
        round(sum(1 for r in rows if r["exact_match"]) / len(rows), 4) if rows else 0.0
    )
    metrics["_per_example"] = rows
    metrics["_versions"] = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
    }
    return metrics


def delta_metrics(base: dict[str, Any], finetuned: dict[str, Any]) -> dict[str, Any]:
    """finetuned - base deltas for headline metrics."""
    keys = ("json_validity", "schema_validity", "exact_match", "field_precision",
            "field_recall", "field_f1", "unsupported_value_rate", "missing_value_rate")
    delta: dict[str, Any] = {}
    for key in keys:
        b, f = base.get(key), finetuned.get(key)
        numeric = isinstance(b, (int, float)) and isinstance(f, (int, float))
        delta[key] = round(f - b, 4) if numeric else None
    for key in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms"):
        b, f = base.get(key), finetuned.get(key)
        numeric = isinstance(b, (int, float)) and isinstance(f, (int, float))
        delta[key] = round(f - b, 1) if numeric else None
    return delta


def per_field_table(base: dict[str, Any], finetuned: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for results/per_field_metrics.csv and the dashboard table."""
    rows = []
    base_fields = base.get("per_field", {})
    ft_fields = finetuned.get("per_field", {})
    for field_name in ALL_FIELDS:
        b = base_fields.get(field_name, {})
        f = ft_fields.get(field_name, {})
        rows.append({
            "field": field_name,
            "base_precision": b.get("precision"),
            "finetuned_precision": f.get("precision"),
            "base_recall": b.get("recall"),
            "finetuned_recall": f.get("recall"),
            "base_f1": b.get("f1"),
            "finetuned_f1": f.get("f1"),
            "delta_f1": (
                round(f.get("f1", 0) - b.get("f1", 0), 4)
                if isinstance(b.get("f1"), (int, float))
                and isinstance(f.get("f1"), (int, float)) else None
            ),
        })
    return rows


def is_date_format_only(pred_raw: Any, gold_raw: Any) -> bool:
    """True when raw values differ but canonical dates are identical.

    Used by error analysis to flag date-format non-compliance separately
    from genuinely wrong values.
    """
    if not isinstance(pred_raw, str) or not isinstance(gold_raw, str):
        return False
    return (
        normalize_date(pred_raw) == normalize_date(gold_raw)
        and pred_raw.strip() != gold_raw.strip()
    )

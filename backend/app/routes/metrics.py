"""Benchmark routes: /api/metrics and /api/benchmark/summary.

Reads the files produced by scripts/benchmark.py from the results directory.
Never hard-codes benchmark values; degrades gracefully when results do not
exist yet (or when only the baseline has been run).
"""

from __future__ import annotations

import csv
from typing import Any

from fastapi import APIRouter, Depends

from backend.app.config import Settings
from backend.app.dependencies import get_settings

router = APIRouter(tags=["benchmark"])

NO_RESULTS_MESSAGE = "Benchmark results have not been generated yet."


def _read_metrics(settings: Settings) -> dict[str, Any] | None:
    path = settings.results_abs_dir / "metrics.json"
    if not path.is_file():
        return None
    from docutune.utils.io import read_json

    try:
        return read_json(path)
    except Exception:  # noqa: BLE001 - corrupt/partial file must not crash the API
        return None


def _read_error_summary(settings: Settings) -> list[dict[str, Any]]:
    path = settings.results_abs_dir / "error_analysis_summary.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [
            {**row,
             "base_count": int(row["base_count"]),
             "finetuned_count": int(row["finetuned_count"]),
             "base_pct": float(row["base_pct"]),
             "finetuned_pct": float(row["finetuned_pct"])}
            for row in csv.DictReader(fh)
        ]


def _read_representative(settings: Settings) -> dict[str, Any] | None:
    path = settings.results_abs_dir / "reports" / "representative_examples.json"
    if not path.is_file():
        return None
    from docutune.utils.io import read_json

    try:
        return read_json(path)
    except Exception:  # noqa: BLE001
        return None


def _read_error_records(settings: Settings) -> list[dict[str, Any]]:
    path = settings.results_abs_dir / "error_analysis.jsonl"
    if not path.is_file():
        return []
    from docutune.utils.io import read_jsonl

    try:
        return read_jsonl(path)
    except Exception:  # noqa: BLE001 - corrupt file must not crash the API
        return []


@router.get("/api/metrics")
def metrics(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Headline benchmark metrics for the dashboard (or a clean unavailable state)."""
    data = _read_metrics(settings)
    if data is None:
        return {"available": False, "message": NO_RESULTS_MESSAGE}
    finetuned = data.get("finetuned")
    partial = finetuned is None
    return {
        "available": True,
        "partial": partial,
        "message": ("Only baseline results are available so far - the fine-tuned pass "
                    "has not been benchmarked yet.") if partial else None,
        "metadata": data.get("metadata"),
        "base": data.get("base"),
        "finetuned": finetuned,
        "delta": data.get("delta"),
        "bootstrap": data.get("bootstrap"),
    }


@router.get("/api/benchmark/summary")
def benchmark_summary(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Everything the dashboard + error-analysis pages need in one payload."""
    data = _read_metrics(settings)
    if data is None:
        return {"available": False, "message": NO_RESULTS_MESSAGE}
    finetuned = data.get("finetuned")
    records = _read_error_records(settings)
    totals = {
        "total_examples": len(records),
        "base_failures": sum(1 for r in records
                             if r.get("base", {}).get("primary_category", "none") != "none"),
        "finetuned_failures": sum(
            1 for r in records
            if r.get("finetuned", {}).get("primary_category", "none") != "none"),
    }
    return {
        "available": True,
        "partial": finetuned is None,
        "metadata": data.get("metadata"),
        "base": data.get("base"),
        "finetuned": finetuned,
        "delta": data.get("delta"),
        "per_field": data.get("per_field"),
        "bootstrap": data.get("bootstrap"),
        "error_summary": _read_error_summary(settings),
        "error_records": records,
        "error_totals": totals,
        "representative_examples": _read_representative(settings),
    }

"""Automatic failure categorization and representative example selection.

Categories (a prediction can hit several; primary = highest priority):
  malformed_json, schema_violation, missing_field, wrong_value,
  unsupported_value, over_extraction, under_extraction, list_item_error,
  date_normalization_error, section_association_error

All categories are heuristic and documented in docs/EVALUATION.md. They aid
analysis; the numeric benchmark metrics are computed exclusively in
metrics.py.
"""

from __future__ import annotations

import csv
import random
from typing import Any

from docutune.evaluation.normalization import normalize_date, normalize_phone, normalize_scalar
from docutune.evaluation.parser import parse_model_output
from docutune.schema.resume import ResumeExtraction, schema_error_message

CATEGORY_PRIORITY = [
    "malformed_json",
    "schema_violation",
    "list_item_error",
    "date_normalization_error",
    "section_association_error",
    "wrong_value",
    "unsupported_value",
    "over_extraction",
    "missing_field",
    "under_extraction",
]

ALL_CATEGORIES = [*CATEGORY_PRIORITY, "none"]


def _schema_validates(parsed: Any) -> tuple[bool, str | None]:
    try:
        ResumeExtraction.model_validate(parsed)
        return True, None
    except Exception as exc:
        return False, schema_error_message(exc)


def _raw_item_lists(parsed: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        kind: [item for item in (parsed.get(kind) or []) if isinstance(item, dict)]
        for kind in ("education", "experience", "projects", "certifications")
    }


def _dates_wrong_kind(parsed: dict[str, Any], gold: dict[str, Any]) -> tuple[list[str], bool, bool]:
    """Inspect experience items for date errors / section-association errors.

    Returns (messages, any_date_format_only, any_association_error).
    """
    date_msgs: list[str] = []
    association = False
    gold_exp = [e for e in gold.get("experience") or [] if isinstance(e, dict)]
    pred_exp = [e for e in (parsed.get("experience") or []) if isinstance(e, dict)]
    for pred_item in pred_exp:
        p_title = normalize_scalar(pred_item.get("title"))
        p_company = normalize_scalar(pred_item.get("company"))
        # exact employer match -> check dates
        match = next(
            (g for g in gold_exp
             if normalize_scalar(g.get("title")) == p_title
             and normalize_scalar(g.get("company")) == p_company),
            None,
        )
        if match is not None:
            for key in ("start_date", "end_date"):
                p_raw, g_raw = pred_item.get(key), match.get(key)
                p_canon, g_canon = normalize_date(p_raw), normalize_date(g_raw)
                raw_differs = (isinstance(p_raw, str) and g_raw
                               and p_raw.strip() != str(g_raw).strip())
                if p_canon == g_canon and raw_differs:
                    date_msgs.append(
                        f"{p_title}@{p_company} {key}: '{p_raw}' not normalized to '{g_raw}'"
                    )
                elif p_canon != g_canon and p_raw is not None and not (
                    g_raw is None and isinstance(p_raw, str)
                    and normalize_date(p_raw) == "__PRESENT__"
                    and match.get("current")
                ):
                    date_msgs.append(
                        f"{p_title}@{p_company} {key}: wrong date '{p_raw}' (gold '{g_raw}')"
                    )
        else:
            # same title but different company while the pair exists in gold
            same_title = next(
                (g for g in gold_exp if normalize_scalar(g.get("title")) == p_title), None
            )
            if same_title is not None:
                association = True
    return date_msgs, bool(date_msgs), association


def categorize_prediction(raw_output: str, gold_target: dict[str, Any]) -> dict[str, Any]:
    """Categorize one model output against one gold record."""
    parsed = parse_model_output(raw_output)
    result: dict[str, Any] = {
        "json_valid": parsed.json_valid,
        "schema_valid": False,
        "schema_error": None,
        "categories": [],
        "primary_category": "malformed_json",
        "details": [],
    }
    if not parsed.json_valid:
        result["categories"] = ["malformed_json"]
        result["details"].append(parsed.error or "unparseable output")
        return result
    if not parsed.is_object:
        result["categories"].append("malformed_json")
        result["primary_category"] = "malformed_json"
        result["details"].append("output is valid JSON but not an object")
        return result

    ok, err = _schema_validates(parsed.parsed)
    result["schema_valid"] = ok
    result["schema_error"] = err
    if not ok:
        result["categories"].append("schema_violation")
        result["primary_category"] = "schema_violation"
        result["details"].append(err)
        return result

    pred: dict[str, Any] = parsed.parsed
    categories: set[str] = set()
    details: list[str] = []

    scalar_fields = ("name", "email", "phone", "location", "summary")

    # Missing fields (gold non-empty, prediction empty)
    for f in (*scalar_fields, "skills", "education", "experience", "projects", "certifications"):
        gold_val = gold_target.get(f)
        pred_val = pred.get(f)
        gold_empty = gold_val in (None, [], "")
        pred_empty = pred_val in (None, [], "")
        if not gold_empty and pred_empty:
            categories.add("missing_field")
            details.append(f"missing field: {f}")
        if gold_empty and not pred_empty:
            categories.add("over_extraction")
            details.append(f"field present in output but empty in gold: {f}")

    # Wrong / unsupported scalars + partial list extraction (field-aware
    # normalizers so that e.g. phone dash variants are not false positives)
    normalizers = {
        "name": normalize_scalar,
        "email": normalize_scalar,
        "phone": normalize_phone,
        "location": normalize_scalar,
        "summary": normalize_scalar,
    }
    for f in scalar_fields:
        g = normalizers[f](gold_target.get(f))
        p = normalizers[f](pred.get(f))
        if g is not None and p is not None and p != g:
            categories.add("wrong_value")
            details.append(f"{f}: '{p}' != gold '{g}'")
        elif g is None and p is not None:
            categories.add("unsupported_value")
            details.append(f"{f}: unsupported value '{p}'")

    gold_skills = {normalize_scalar(s) for s in gold_target.get("skills") or []}
    pred_skills = [normalize_scalar(s) for s in pred.get("skills") or []]
    extra_skills = [s for s in pred_skills if s not in gold_skills]
    if extra_skills:
        categories.add("unsupported_value")
        details.append(f"unsupported skills: {extra_skills[:3]}")
    missing_skills = [s for s in (normalize_scalar(s) for s in gold_target.get("skills") or [])
                      if s not in [normalize_scalar(x) for x in (pred.get("skills") or [])]]
    if missing_skills:
        categories.add("under_extraction")
        details.append(f"missing skills: {missing_skills[:3]}")

    # Record-list mismatches
    gold_lists = _raw_item_lists(gold_target)
    pred_lists = _raw_item_lists(pred)
    for kind in gold_lists:
        g_items, p_items = gold_lists[kind], pred_lists[kind]
        if len(p_items) > len(g_items):
            categories.add("unsupported_value")
            categories.add("list_item_error")
            details.append(f"{kind}: {len(p_items) - len(g_items)} extra item(s)")
        elif len(p_items) < len(g_items):
            categories.add("under_extraction")
            categories.add("list_item_error")
            details.append(f"{kind}: {len(g_items) - len(p_items)} item(s) missing")

    date_msgs, date_flag, assoc_flag = _dates_wrong_kind(pred, gold_target)
    if date_flag:
        categories.add("date_normalization_error")
        details.extend(date_msgs)
    if assoc_flag:
        categories.add("section_association_error")
        details.append("experience content associated with the wrong employer")

    result["categories"] = [c for c in CATEGORY_PRIORITY if c in categories]
    result["primary_category"] = result["categories"][0] if result["categories"] else "none"
    result["details"] = details[:10]
    return result


def build_error_records(test_examples: list[dict[str, Any]],
                        base_predictions: dict[str, dict[str, Any]],
                        finetuned_predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One error-analysis record per test example, covering both models."""
    records = []
    for ex in test_examples:
        ex_id = ex["id"]
        base_row = base_predictions.get(ex_id, {})
        ft_row = finetuned_predictions.get(ex_id, {})
        base_cat = categorize_prediction(base_row.get("raw_output") or "", ex["target"])
        ft_cat = categorize_prediction(ft_row.get("raw_output") or "", ex["target"])
        records.append({
            "id": ex_id,
            "difficulty": ex["metadata"]["difficulty"],
            "template_id": ex["metadata"]["template_id"],
            "resume_text": ex["text"],
            "gold": ex["target"],
            "base_output": base_row.get("raw_output"),
            "finetuned_output": ft_row.get("raw_output"),
            "base": base_cat,
            "finetuned": ft_cat,
        })
    return records


def summarize_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Category counts for both models, ordered by (base + finetuned) desc."""
    total = len(records) or 1
    rows = []
    for category in ALL_CATEGORIES:
        base_n = sum(1 for r in records if category in r["base"]["categories"])
        ft_n = sum(1 for r in records if category in r["finetuned"]["categories"])
        rows.append({
            "category": category,
            "base_count": base_n,
            "finetuned_count": ft_n,
            "base_pct": round(100 * base_n / total, 2),
            "finetuned_pct": round(100 * ft_n / total, 2),
        })
    rows.sort(key=lambda r: -(r["base_count"] + r["finetuned_count"]))
    return rows


def write_error_summary_csv(path: str, summary: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "category", "base_count", "finetuned_count", "base_pct", "finetuned_pct"])
        writer.writeheader()
        writer.writerows(summary)


def select_representative_examples(records: list[dict[str, Any]],
                                   f1_by_id: dict[str, dict[str, float]],
                                   seed: int = 42,
                                   n_random: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Deterministic, non-cherry-picked example selection.

    Includes seeded-random examples, biggest improvement, worst regression,
    and both directions of base/fine-tuned failure swaps. Both successes and
    failures appear by construction.
    """
    rng = random.Random(seed)
    selectable = [r for r in records]
    random_picks = rng.sample(selectable, k=min(n_random, len(selectable)))

    def delta(r: dict[str, Any]) -> float:
        f1s = f1_by_id.get(r["id"], {})
        return f1s.get("finetuned", 0.0) - f1s.get("base", 0.0)

    by_delta = sorted(records, key=delta)
    regression = by_delta[:1]
    improvement = list(reversed(by_delta[-1:]))
    base_fail_ft_ok = [r for r in records
                       if not r["base"]["schema_valid"] and r["finetuned"]["schema_valid"]][:1]
    ft_fail_base_ok = [r for r in records
                       if r["base"]["schema_valid"] and not r["finetuned"]["schema_valid"]][:1]

    def with_context(items: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        out = []
        for r in items:
            out.append({**r, "selection_reason": label,
                        "f1_base": f1_by_id.get(r["id"], {}).get("base"),
                        "f1_finetuned": f1_by_id.get(r["id"], {}).get("finetuned")})
        return out

    return {
        "random": with_context(random_picks, f"random (seed={seed})"),
        "biggest_improvement": with_context(improvement, "largest finetuned-base field-F1 delta"),
        "regression": with_context(regression, "worst finetuned-base field-F1 delta"),
        "base_failure_finetuned_success": with_context(
            base_fail_ft_ok, "base model schema failure, fine-tuned success"),
        "finetuned_failure_base_success": with_context(
            ft_fail_base_ok, "fine-tuned failure, base model success"),
    }

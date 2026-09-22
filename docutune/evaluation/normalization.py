"""Conservative normalization for prediction/target comparison.

Every rule is deliberately conservative: normalization exists to remove
trivial surface differences, never to rescue genuinely wrong values.
All rules are documented in docs/EVALUATION.md.

Rules:
  R1  Strings: trim and collapse internal whitespace.
  R2  Empty-ish scalars ("", "n/a", "none", "null", "-", "?") are None.
  R3  Case-insensitive comparison ONLY where case carries no information:
      emails, skills, degree names. Everything else stays case-sensitive
      (after R1) so that shouting/wrong-case remains visible in metrics
      only where it matters semantically... note: skills extracted with
      different casing are treated as equal; names are NOT.
  R4  Phones: spaces, dashes, dots and parentheses removed; leading + kept.
  R5  Emails: lowercased (R3).
  R6  Dates: canonicalized to "YYYY-MM" or "YYYY". Accepts month names and
      abbreviations ("June 2022", "Jun 2022"), "MM/YYYY", "YYYY-MM",
      "YYYY/MM", "YYYY". The words present/current/now/till date map to a
      PRESENT sentinel, not to a date.
  R7  Experience end_date: PRESENT sentinel equals null when the entry is
      marked current (gold uses end_date=null + current=true).
  R8  Lists are order-insensitive (skills, responsibilities, technologies):
      the gold generator writes them meaningfully ordered but list order
      carries no extraction signal, so multiset comparison is used.
  R9  Grades: "8.5/10" and "CGPA: 8.5" normalize to "8.5"; "85%" stays "85%".
  R10 Years compare as integers (string "2018" == 2018).
  R11 Schema-unknown keys are ignored by exact-match comparison (schema
      violations are penalized separately by the schema-validity metric).
  R12 Wrong JSON types (e.g. skills as a string) are NEVER coerced into the
      right type; they count as field failures.
"""

from __future__ import annotations

import re
from typing import Any

# Sentinel marking "the model wrote a present-word where gold has null".
PRESENT = "__PRESENT__"

_WS_RE = re.compile(r"\s+")
_NULL_TOKENS = {"", "n/a", "na", "none", "null", "-", "?", "unknown"}
_PRESENT_TOKENS = {"present", "current", "now", "till date", "present date", "ongoing",
                   "till now", "date"}
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_PHONE_STRIP_RE = re.compile(r"[\s\-.()]")
_GRADE_RE = re.compile(
    r"^(?:cgpa|gpa)?\s*[:\s]*([0-9]+(?:\.[0-9]+)?)\s*(?:/10)?\s*$", re.IGNORECASE
)

CASE_INSENSITIVE_FIELDS = {"email", "skills", "degree"}
NULLABLE_LIST_ITEM_SCALARS = {
    "education": ("degree", "institution", "field", "start_year", "end_year", "grade"),
    "experience": ("title", "company", "location", "start_date", "end_date", "current",
                   "responsibilities"),
    "projects": ("name", "technologies", "description"),
    "certifications": ("name", "issuer", "year"),
}


def normalize_scalar(value: Any, field_name: str | None = None) -> str | None:
    """R1, R2, R3, R5 for plain string fields. Non-strings are NOT coerced."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None  # R12: wrong types do not become strings silently
    text = _WS_RE.sub(" ", value).strip()
    if text.casefold() in _NULL_TOKENS:
        return None
    if field_name in CASE_INSENSITIVE_FIELDS:
        text = text.casefold()
    return text


def normalize_email(value: Any) -> str | None:
    text = normalize_scalar(value, "email")
    return text


def normalize_phone(value: Any) -> str | None:
    text = normalize_scalar(value, "phone")
    if text is None:
        return None
    return _PHONE_STRIP_RE.sub("", text)


def normalize_grade(value: Any) -> str | None:
    text = normalize_scalar(value, "grade")
    if text is None:
        return None
    match = _GRADE_RE.match(text)
    if match:
        return match.group(1)
    pct = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*%$", text)
    if pct:
        return f"{pct.group(1)}%"
    return text


def normalize_year(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() and len(text) == 4:
            return int(text)
    return None  # R10/R12


def normalize_date(value: Any) -> str | None:
    """R6: canonical date string ("YYYY-MM" / "YYYY"), PRESENT sentinel, or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return normalize_scalar(value)
    text = _WS_RE.sub(" ", value).strip().casefold()
    if text in _NULL_TOKENS:
        return None
    if text in _PRESENT_TOKENS:
        return PRESENT
    text = text.replace("/", "-").replace(".", "-")
    # "YYYY-MM" or "YYYY-MM-DD" (day dropped)
    iso = re.match(r"^(\d{4})-(\d{1,2})(?:-\d{1,2})?$", text)
    if iso:
        year, month = int(iso.group(1)), int(iso.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"
        return None
    year_only = re.match(r"^(\d{4})$", text)
    if year_only:
        return year_only.group(1)
    # "Mon YYYY" / "Month YYYY"
    month_name = re.match(r"^([a-z]+)\s+(\d{4})$", text)
    if month_name and month_name.group(1) in _MONTHS:
        return f"{month_name.group(2)}-{_MONTHS[month_name.group(1)]:02d}"
    # "MM-YYYY"
    mm_yyyy = re.match(r"^(\d{1,2})-(\d{4})$", text)
    if mm_yyyy:
        month, year = int(mm_yyyy.group(1)), int(mm_yyyy.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"
    return text  # unrecognized form: compare literally (conservative)


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().casefold()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None  # R12


def _normalize_string_list(value: Any) -> list[str] | None:
    """Normalized multiset of strings, or None on wrong types (R12)."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    items = []
    for item in value:
        text = normalize_scalar(item, "skills")
        if text is None and item is not None and not isinstance(item, str):
            return None
        if text is not None:
            items.append(text)
    return sorted(items)


def _normalize_item(kind: str, item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    out: dict[str, Any] = {}
    for key in NULLABLE_LIST_ITEM_SCALARS[kind]:
        val = item.get(key)
        if key in ("start_year", "end_year"):
            out[key] = normalize_year(val)
        elif key == "current":
            norm = normalize_bool(val)
            out[key] = norm if norm is not None else False
        elif key == "start_date":
            out[key] = normalize_date(val)
        elif key == "end_date":
            out[key] = normalize_date(val)
        elif key == "grade":
            out[key] = normalize_grade(val)
        else:
            out[key] = normalize_scalar(val, key if key == "degree" else None)
    if kind == "experience":
        # R7: a present-word end_date equals null for current roles.
        if out.get("end_date") == PRESENT and out.get("current"):
            out["end_date"] = None
        resps = _normalize_string_list(item.get("responsibilities"))
        if resps is None:
            return None
        out["responsibilities"] = resps
    elif kind == "projects":
        tech = _normalize_string_list(item.get("technologies"))
        if tech is None:
            return None
        out["technologies"] = tech
    return out


def normalize_record(record: Any) -> dict[str, Any]:
    """Fully normalized record for comparison (R11: unknown keys dropped).

    List-valued fields are stored as sorted lists (multiset semantics, R8).
    Returns {} if the input is not a dict.
    """
    if not isinstance(record, dict):
        return {}
    out: dict[str, Any] = {
        "name": normalize_scalar(record.get("name")),
        "email": normalize_email(record.get("email")),
        "phone": normalize_phone(record.get("phone")),
        "location": normalize_scalar(record.get("location")),
        "summary": normalize_scalar(record.get("summary")),
    }
    skills = _normalize_string_list(record.get("skills"))
    out["skills"] = skills if skills is not None else None  # None = type failure

    for kind in ("education", "experience", "projects", "certifications"):
        value = record.get(kind)
        if value is None:
            out[kind] = []
        elif not isinstance(value, list):
            out[kind] = None  # type failure marker
        else:
            items = [_normalize_item(kind, item) for item in value]
            if any(item is None for item in items):
                out[kind] = None
            else:
                out[kind] = sorted(items, key=lambda d: json_key(d))
    return out


def json_key(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def records_equal(pred_norm: dict[str, Any], gold_norm: dict[str, Any]) -> bool:
    """Exact record match on normalized schema fields (R11)."""
    if not isinstance(pred_norm, dict) or not pred_norm:
        return False
    return pred_norm == gold_norm

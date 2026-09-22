"""Split integrity helpers.

The split is TEMPLATE-based, not a naive random split: rendering templates
01-08 are used for training, 09-10 for validation and 11-12 for the held-out
test set. This measures whether the model learned the extraction *task*
rather than the surface formatting of the training templates.

Every example is generated independently, so structured records cannot leak
across splits; the validator still verifies this explicitly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from docutune.data.renderer import TEMPLATE_SPLIT_MAP
from docutune.schema.resume import ResumeExtraction

_WS_RE = re.compile(r"\s+")


def normalize_text_key(text: str) -> str:
    """Whitespace-collapsed, casefolded key for duplicate-text detection."""
    return _WS_RE.sub(" ", text).strip().casefold()


def canonical_record_key(record: dict[str, Any]) -> str:
    """Canonical JSON key for duplicate-record (leakage) detection."""
    canonical = ResumeExtraction.model_validate(record).model_dump()
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def split_examples(examples: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for ex in examples:
        split = ex["metadata"]["split"]
        buckets[split].append(ex)
    for bucket in buckets.values():
        bucket.sort(key=lambda ex: ex["id"])
    return buckets


def find_duplicate_texts(
    examples_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Exact-duplicate (normalized) resume text appearing in 2+ splits."""
    seen: dict[str, set[str]] = {}
    for split, examples in examples_by_split.items():
        for ex in examples:
            key = normalize_text_key(ex["text"])
            seen.setdefault(key, set()).add(split)
    return [
        {"text_key": key[:120], "splits": sorted(splits)}
        for key, splits in seen.items()
        if len(splits) > 1
    ]


def find_duplicate_records(
    examples_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Identical canonical structured records appearing in 2+ splits."""
    seen: dict[str, set[str]] = {}
    for split, examples in examples_by_split.items():
        for ex in examples:
            key = canonical_record_key(ex["target"])
            seen.setdefault(key, set()).add(split)
    return [
        {"record_key": key[:120], "splits": sorted(splits)}
        for key, splits in seen.items()
        if len(splits) > 1
    ]


def find_template_leakage(
    examples_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Templates used in a split other than their assigned one."""
    usage: dict[str, set[str]] = {}
    for split, examples in examples_by_split.items():
        for ex in examples:
            usage.setdefault(ex["metadata"]["template_id"], set()).add(split)
    issues = []
    for template_id, splits in sorted(usage.items()):
        expected = TEMPLATE_SPLIT_MAP.get(template_id)
        if expected is None:
            issues.append({"template_id": template_id,
                           "problem": "unknown template", "splits": sorted(splits)})
        elif splits != {expected}:
            issues.append({
                "template_id": template_id,
                "problem": f"expected only '{expected}'",
                "splits": sorted(splits),
            })
    return issues

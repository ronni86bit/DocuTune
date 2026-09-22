"""Dataset validation (spec section 14).

Checks: JSONL validity, schema validity, unique IDs, duplicate text,
duplicate structured records, train/test leakage, expected counts,
template distribution, difficulty distribution, required metadata and
schema version. Used by scripts/validate_data.py and the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docutune.config import SCHEMA_VERSION, project_path
from docutune.data.generator import DEFAULT_COUNTS
from docutune.data.renderer import TEMPLATE_SPLIT_MAP
from docutune.data.splitter import (
    find_duplicate_records,
    find_duplicate_texts,
    find_template_leakage,
    normalize_text_key,
)
from docutune.schema.resume import ResumeExtraction, schema_error_message
from docutune.utils.io import read_jsonl

SPLIT_FILES = {
    "train": "data/splits/train.jsonl",
    "validation": "data/splits/validation.jsonl",
    "test": "data/splits/test.jsonl",
}


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _distribution(examples: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ex in examples:
        value: Any = ex
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
        key_str = str(value)
        out[key_str] = out.get(key_str, 0) + 1
    return dict(sorted(out.items()))


def validate_file(path: Path, split: str, report: ValidationReport) -> list[dict[str, Any]]:
    if not path.is_file():
        report.error(f"{split}: file missing: {path}")
        return []
    try:
        examples = read_jsonl(path)
    except ValueError as exc:
        report.error(f"{split}: {exc}")
        return []

    ids: set[str] = set()
    for ex in examples:
        ex_id = ex.get("id", "<missing>")
        if not isinstance(ex.get("id"), str):
            report.error(f"{split}: example with non-string/missing id: {ex!r:.120}")
            continue
        if ex_id in ids:
            report.error(f"{split}: duplicate id {ex_id}")
        ids.add(ex_id)

        for required in ("text", "target", "metadata"):
            if required not in ex:
                report.error(f"{ex_id}: missing required key '{required}'")
        if not isinstance(ex.get("text"), str) or not ex.get("text", "").strip():
            report.error(f"{ex_id}: 'text' must be a non-empty string")
        try:
            ResumeExtraction.model_validate(ex.get("target"))
        except Exception as exc:  # pydantic.ValidationError or type errors
            report.error(f"{ex_id}: target fails schema validation: {schema_error_message(exc)}")

        metadata = ex.get("metadata") or {}
        for required_meta in ("template_id", "difficulty", "seed", "split"):
            if required_meta not in metadata:
                report.error(f"{ex_id}: metadata missing '{required_meta}'")
        if metadata.get("split") != split:
            report.error(
                f"{ex_id}: metadata split '{metadata.get('split')}' != file split '{split}'")
        if metadata.get("difficulty") not in ("easy", "medium", "hard"):
            report.error(f"{ex_id}: invalid difficulty '{metadata.get('difficulty')}'")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            report.error(
                f"{ex_id}: schema_version {metadata.get('schema_version')!r} != {SCHEMA_VERSION!r}"
            )
        expected_template_split = TEMPLATE_SPLIT_MAP.get(metadata.get("template_id", ""))
        if expected_template_split is not None and expected_template_split != split:
            report.error(
                f"{ex_id}: template {metadata.get('template_id')} belongs to "
                f"'{expected_template_split}' but appears in '{split}' (leakage)"
            )
    return examples


def validate_dataset(
    splits_dir: str | Path | None = None,
    expected_counts: dict[str, int] | None = None,
) -> ValidationReport:
    """Validate the generated dataset in data/splits (or a custom directory)."""
    report = ValidationReport()
    expected_counts = {**DEFAULT_COUNTS, **(expected_counts or {})}

    root = Path(splits_dir) if splits_dir else None
    examples_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, rel_file in SPLIT_FILES.items():
        path = (root / Path(rel_file).name) if root else project_path(rel_file)
        examples_by_split[split] = validate_file(path, split, report)

    # Expected counts
    for split, examples in examples_by_split.items():
        expected = expected_counts[split]
        if len(examples) != expected:
            report.warn(f"{split}: expected {expected} examples, found {len(examples)}")

    # Global ID uniqueness across splits
    all_ids: list[str] = []
    for examples in examples_by_split.values():
        all_ids.extend(ex.get("id") for ex in examples)
    if len(all_ids) != len(set(all_ids)):
        dupes = sorted({i for i in all_ids if all_ids.count(i) > 1})
        report.error(f"duplicate ids across splits: {dupes[:10]}")

    # Cross-split leakage
    for finding in find_duplicate_texts(examples_by_split):
        report.error(f"duplicate text across splits {finding['splits']}: {finding['text_key']}...")
    for finding in find_duplicate_records(examples_by_split):
        report.error(f"duplicate structured record across splits {finding['splits']}")
    for finding in find_template_leakage(examples_by_split):
        report.error(f"template leakage: {finding}")

    # Near-duplicate text within a split is suspicious but not fatal.
    for split, examples in examples_by_split.items():
        keys: dict[str, int] = {}
        for ex in examples:
            key = normalize_text_key(ex["text"])
            keys[key] = keys.get(key, 0) + 1
        within = sum(1 for c in keys.values() if c > 1)
        if within:
            report.warn(f"{split}: {within} duplicated text(s) within the split")

    # Distributions
    report.stats["counts"] = {s: len(e) for s, e in examples_by_split.items()}
    report.stats["templates"] = {
        s: _distribution(e, ("metadata", "template_id")) for s, e in examples_by_split.items()
    }
    report.stats["difficulties"] = {
        s: _distribution(e, ("metadata", "difficulty")) for s, e in examples_by_split.items()
    }
    if not any(examples_by_split.values()):
        report.error("no dataset files could be read; run scripts/generate_data.py first")
    return report


def format_report(report: ValidationReport) -> str:
    lines: list[str] = []
    lines.append("Dataset validation " + ("PASSED" if report.ok else "FAILED"))
    lines.append("")
    lines.append(f"counts: {report.stats.get('counts')}")
    for split, dist in (report.stats.get("templates") or {}).items():
        lines.append(f"{split} templates: {dist}")
    for split, dist in (report.stats.get("difficulties") or {}).items():
        lines.append(f"{split} difficulties: {dist}")
    if report.errors:
        lines.append("")
        lines.append(f"Errors ({len(report.errors)}):")
        lines.extend(f"  - {e}" for e in report.errors[:50])
    if report.warnings:
        lines.append("")
        lines.append(f"Warnings ({len(report.warnings)}):")
        lines.extend(f"  - {w}" for w in report.warnings[:20])
    return "\n".join(lines)

"""Deterministic synthetic resume dataset generation.

Pipeline:
1. Generate a canonical structured record (the gold label) from seeded RNG.
2. Render it into varied natural-language resume text.
3. Store {"id", "text", "target", "metadata"} rows as JSONL.

Every example is generated independently (per-example RNG derived from the
base seed via pure integer arithmetic), so the dataset is byte-identical
across runs, machines and Python versions. The test split uses rendering
templates that NEVER appear in training (see docs/DATASET.md).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from docutune.config import DATASET_VERSION, SCHEMA_VERSION
from docutune.data.pools import (
    AREAS,
    CERTIFICATIONS,
    CITIES,
    COMPANIES,
    EMAIL_DOMAINS,
    FIRST_NAMES,
    JOB_TITLES,
    LAST_NAMES,
    POSTGRAD_DEGREES,
    PROJECT_DESC_TEMPLATES,
    PROJECT_NAMES,
    REFERENCE_YEAR,
    RESPONSIBILITY_TEMPLATES,
    RESPONSIBILITY_TOPICS,
    SKILL_CATEGORIES,
    SKILLS,
    UNDERGRAD_DEGREES,
    UNIVERSITIES,
)
from docutune.data.renderer import TEMPLATES_BY_ID, TemplateSpec, render_resume
from docutune.seed import derive_seed

SPLITS = ("train", "validation", "test")

DEFAULT_COUNTS = {"train": 450, "validation": 75, "test": 75}

DIFFICULTY_LEVELS = ("easy", "medium", "hard")


@dataclass
class GenerationSummary:
    total: int
    counts: dict[str, int]
    seed: int
    template_distribution: dict[str, dict[str, int]]
    difficulty_distribution: dict[str, dict[str, int]]


# ---------------------------------------------------------------------------
# Canonical record generation
# ---------------------------------------------------------------------------
def _clean_name_token(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalpha())


def _generate_contact(rng: random.Random, first: str, last: str,
                      force_missing: tuple[str, ...]) -> tuple[str | None, str | None]:
    email = None
    phone = None
    if "email" not in force_missing and rng.random() < 0.70:
        first_tok = _clean_name_token(first)
        last_tok = _clean_name_token(last)
        style = rng.randrange(3)
        if style == 0:
            email = f"{first_tok}.{last_tok}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAINS)}"
        elif style == 1:
            email = f"{first_tok}{last_tok}@{rng.choice(EMAIL_DOMAINS)}"
        else:
            email = f"{first_tok[0]}{last_tok}{rng.randint(10, 999)}@{rng.choice(EMAIL_DOMAINS)}"
    if "phone" not in force_missing and rng.random() < 0.65:
        if rng.random() < 0.75:
            phone = f"+91-{rng.randint(6, 9)}{rng.randint(0, 999999999):09d}"
        else:
            phone = f"+1-{rng.randint(200, 989)}-{rng.randint(200, 989)}-{rng.randint(1000, 9999)}"
    return email, phone


def _generate_education(
    rng: random.Random, force_missing: tuple[str, ...]
) -> tuple[list[dict[str, Any]], int]:
    """Returns (education entries, career start year)."""
    ug_degree, ug_field = rng.choice(UNDERGRAD_DEGREES)
    ug_start = rng.randint(2008, 2018)
    ug_end = ug_start + rng.choice([3, 4])
    entries = [{
        "degree": ug_degree,
        "institution": rng.choice(UNIVERSITIES),
        "field": ug_field,
        "start_year": ug_start,
        "end_year": ug_end,
        "grade": None,
    }]
    career_start = ug_end
    if "postgrad" not in force_missing and rng.random() < 0.30 and ug_end + 2 <= 2022:
        pg_degree, pg_field = rng.choice(POSTGRAD_DEGREES)
        entries.append({
            "degree": pg_degree,
            "institution": rng.choice(UNIVERSITIES),
            "field": pg_field,
            "start_year": ug_end,
            "end_year": ug_end + 2,
            "grade": None,
        })
        career_start = ug_end + 2
    # Gold entries are newest-first (matching the experience ordering).
    entries.reverse()
    if "grades" not in force_missing:
        for entry in entries:
            if rng.random() < 0.25:
                if rng.random() < 0.5:
                    entry["grade"] = f"{rng.uniform(6.0, 9.8):.1f}"
                else:
                    entry["grade"] = f"{rng.randint(60, 95)}%"
    return entries, career_start


def _month_to_ym(career_start: int, month_offset: int) -> tuple[int, int]:
    """Convert a month offset from January of career_start into (year, month)."""
    return career_start + month_offset // 12, month_offset % 12 + 1


def _generate_experience(rng: random.Random, career_start: int) -> list[dict[str, Any]]:
    """Build a coherent reverse-chronological employment history.

    The timeline from January of career_start to December of REFERENCE_YEAR is
    cut into consecutive segments (minimum 5 months each), guaranteeing
    strictly ordered, non-overlapping jobs with realistic durations.
    """
    span_months = (REFERENCE_YEAR - career_start + 1) * 12
    if span_months < 6:
        return []
    max_jobs = min(3, span_months // 12) or 1
    n_jobs = rng.randint(1, max_jobs)
    min_len = 5
    # Split the timeline into n_jobs segments, each at least min_len months.
    extra = span_months - n_jobs * min_len
    weights = [rng.randint(1, 100) for _ in range(n_jobs)]
    total_weight = sum(weights)
    lengths = [min_len + extra * w // total_weight for w in weights]
    lengths[-1] += extra - sum(extra * w // total_weight for w in weights)
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    bounds[-1] = span_months

    titles = rng.sample(JOB_TITLES, k=n_jobs)
    companies = rng.sample(COMPANIES, k=n_jobs)
    jobs: list[dict[str, Any]] = []
    for i in range(n_jobs):
        seg_start, seg_end = bounds[i], bounds[i + 1] - 1
        start_y, start_m = _month_to_ym(career_start, seg_start)
        # Only the most recent job may be ongoing.
        current = i == n_jobs - 1 and rng.random() < 0.70
        if current:
            end_y, end_m = None, None
        else:
            end_y, end_m = _month_to_ym(career_start, seg_end)
        location = None
        if rng.random() < 0.30:
            city, region = rng.choice(CITIES)
            location = f"{city}, {region}"
        n_resps = rng.randint(1, 4)
        resp_templates = rng.sample(
            RESPONSIBILITY_TEMPLATES, k=min(n_resps, len(RESPONSIBILITY_TEMPLATES))
        )
        responsibilities = [
            tpl.format(
                topic=rng.choice(RESPONSIBILITY_TOPICS),
                pct=rng.choice([10, 15, 20, 25, 30, 35, 40]),
            )
            for tpl in resp_templates
        ]
        jobs.append({
            "title": titles[i],
            "company": companies[i],
            "location": location,
            "start_date": f"{start_y}-{start_m:02d}",
            "end_date": None if current else f"{end_y}-{end_m:02d}",
            "current": current,
            "responsibilities": responsibilities,
        })
    jobs.reverse()  # most recent first
    return jobs


def _generate_summary(rng: random.Random, title: str, years: int) -> str:
    area1, area2 = rng.sample(AREAS, k=2)
    style = rng.randrange(3)
    if style == 0:
        return f"{title} with {years} years of experience in {area1} and {area2}."
    if style == 1:
        return f"Experienced {title} specializing in {area1}."
    return f"{title} with a strong background in {area1}, passionate about {area2}."


def _generate_projects(rng: random.Random, force_missing: tuple[str, ...]) -> list[dict[str, Any]]:
    if "projects" in force_missing or rng.random() < 0.45:
        return []
    n = rng.randint(1, 2)
    names = rng.sample(PROJECT_NAMES, k=n)
    projects = []
    for name in names:
        tech = rng.sample(SKILLS, k=rng.randint(2, 5))
        desc = None
        if rng.random() < 0.70:
            t1, t2 = rng.sample(tech, k=min(2, len(tech)))
            desc = rng.choice(PROJECT_DESC_TEMPLATES).format(
                t1=t1, t2=t2,
                adj=rng.choice(["scalable", "real-time", "open-source", "full-stack"]),
                area=rng.choice(AREAS),
            )
        projects.append({"name": name, "technologies": tech, "description": desc})
    return projects


def _generate_certifications(rng: random.Random, career_start: int,
                             force_missing: tuple[str, ...]) -> list[dict[str, Any]]:
    if "certifications" in force_missing or rng.random() < 0.40:
        return []
    n = rng.randint(1, 3)
    chosen = rng.sample(CERTIFICATIONS, k=n)
    return [
        {"name": name, "issuer": None if rng.random() < 0.15 else issuer,
         "year": rng.randint(max(career_start, 2015), REFERENCE_YEAR)}
        for name, issuer in chosen
    ]


def _generate_skills(rng: random.Random, spec: TemplateSpec) -> list[str]:
    if spec.style == "grouped_skills":
        pool = sorted({s for items in SKILL_CATEGORIES.values() for s in items})
        k = rng.randint(8, 14)
        return rng.sample([s for s in pool if s in SKILLS], k=min(k, len(pool)))
    k = rng.randint(4, 12)
    return rng.sample(SKILLS, k=k)


def generate_record(rng: random.Random, spec: TemplateSpec) -> dict[str, Any]:
    """Generate one canonical structured record (the gold target)."""
    force_missing = spec.force_missing
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    email, phone = _generate_contact(rng, first, last, force_missing)

    location = None
    if "location" not in force_missing and rng.random() < 0.80:
        city, region = rng.choice(CITIES)
        location = f"{city}, {region}"

    education, career_start = _generate_education(rng, force_missing)
    experience = _generate_experience(rng, career_start)
    years = max(1, REFERENCE_YEAR - career_start)

    summary = None
    if "summary" not in force_missing and rng.random() < 0.75:
        title = experience[0]["title"] if experience else rng.choice(JOB_TITLES)
        summary = _generate_summary(rng, title, years)

    skills = _generate_skills(rng, spec)
    projects = _generate_projects(rng, force_missing)
    certifications = _generate_certifications(rng, career_start, force_missing)

    # Field order matches the Pydantic schema exactly (canonical serialization).
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "summary": summary,
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certifications,
    }


def _count_hardship_factors(record: dict[str, Any], spec: TemplateSpec) -> int:
    factors = 0
    if spec.style in ("paragraph", "grouped_skills", "fragmented", "noisy"):
        factors += 1
    if spec.date_policy == "mixed":
        factors += 1
    missing_optional = sum(
        1 for key in ("email", "phone", "location", "summary", "projects", "certifications")
        if not record.get(key)
    )
    if missing_optional >= 3:
        factors += 1
    if len(record.get("skills") or []) >= 10:
        factors += 1
    total_resps = sum(len(e.get("responsibilities") or []) for e in record.get("experience") or [])
    if total_resps >= 6:
        factors += 1
    if len(record.get("experience") or []) >= 3:
        factors += 1
    return factors


def _assign_difficulty(base: str, factors: int) -> str:
    if base == "easy":
        if factors >= 3:
            return "hard"
        if factors >= 2:
            return "medium"
        return "easy"
    if base == "medium":
        return "hard" if factors >= 2 else "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def make_example(global_index: int, split: str, spec: TemplateSpec,
                 seed: int) -> dict[str, Any]:
    """Generate one dataset example, fully deterministically."""
    rng = random.Random(derive_seed(seed, split, spec.id, global_index))
    record = generate_record(rng, spec)
    text = render_resume(record, spec.id, rng)
    factors = _count_hardship_factors(record, spec)
    difficulty = _assign_difficulty(spec.base_difficulty, factors)
    return {
        "id": f"resume_{global_index:06d}",
        "text": text,
        "target": record,
        "metadata": {
            "template_id": spec.id,
            "template_name": spec.name,
            "difficulty": difficulty,
            "split": split,
            "seed": seed,
            "schema_version": SCHEMA_VERSION,
            "dataset_version": DATASET_VERSION,
        },
    }


def _specs_for_split(split: str) -> list[TemplateSpec]:
    return [spec for spec in TEMPLATES_BY_ID.values() if spec.split == split]


def generate_dataset(seed: int = 42, counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Generate the full dataset: train first, then validation, then test.

    IDs are sequential across splits (resume_000001...) and generation is
    order-independent per example.
    """
    counts = {**DEFAULT_COUNTS, **(counts or {})}
    examples: list[dict[str, Any]] = []
    index = 1
    for split in SPLITS:
        specs = _specs_for_split(split)
        if not specs:
            raise ValueError(f"No rendering templates assigned to split '{split}'")
        for i in range(counts[split]):
            spec = specs[i % len(specs)]
            examples.append(make_example(index, split, spec, seed))
            index += 1
    return examples


def dataset_manifest(examples: list[dict[str, Any]], seed: int,
                     file_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    counts = {split: 0 for split in SPLITS}
    templates: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    difficulties: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for ex in examples:
        split = ex["metadata"]["split"]
        counts[split] += 1
        tid = ex["metadata"]["template_id"]
        templates[split][tid] = templates[split].get(tid, 0) + 1
        diff = ex["metadata"]["difficulty"]
        difficulties[split][diff] = difficulties[split].get(diff, 0) + 1
    manifest = {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator": "docutune.data.generator (deterministic, local, no external APIs)",
        "seed": seed,
        "counts": counts,
        "total": len(examples),
        "template_split_map": {t.id: t.split for t in TEMPLATES_BY_ID.values()},
        "template_distribution": templates,
        "difficulty_distribution": difficulties,
        "reference_year": REFERENCE_YEAR,
        "data_license": (
            "Synthetic data generated by this repository; "
            "same license as project code."
        ),
        "contains_real_personal_data": False,
    }
    if file_hashes:
        manifest["file_sha256"] = file_hashes
    return manifest


def serialize_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)

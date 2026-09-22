"""Deterministic resume text rendering.

12 distinct rendering templates turn a canonical structured record into
messy, varied resume text. The canonical record is the gold label; the
renderer only changes *how* it is written, never *what* is written.

Split assignment (see docs/DATASET.md):
  templates 01-08  -> train
  templates 09-10  -> validation
  templates 11-12  -> test (held-out rendering styles)
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from docutune.data.pools import MONTHS_FULL

SECTION_KEYS = ["summary", "skills", "experience", "education", "projects", "certifications"]

HEADING_SYNONYMS = {
    "summary": ["Summary", "Professional Summary", "Profile", "About Me", "Objective"],
    "experience": ["Experience", "Professional Experience", "Work History",
                   "Employment History", "Career History"],
    "education": ["Education", "Academic Background", "Academic History",
                  "Educational Qualifications"],
    "skills": ["Skills", "Technical Skills", "Core Skills", "Key Skills", "Skill Set"],
    "projects": ["Projects", "Selected Projects", "Personal Projects", "Key Projects"],
    "certifications": ["Certifications", "Certificates", "Licenses & Certifications"],
}

PRESENT_WORDS = ["Present", "Current", "Now", "Till Date"]


@dataclass(frozen=True)
class TemplateSpec:
    """Full declarative description of one rendering template."""

    id: str
    name: str
    split: str
    base_difficulty: str
    style: str                     # structural family
    date_policy: str               # full | abbr | slash | iso | mixed | year_only
    section_order: tuple[str, ...]
    headings: Mapping[str, str]
    bullet: str
    range_sep: str
    present_word: str
    skills_joiner: str
    contact_style: int             # 0..4, -1 = omit contacts entirely
    edu_style: int
    project_style: int
    cert_style: int
    force_missing: tuple[str, ...] = field(default=())
    uppercase_name: bool = False
    inline_headings: bool = False  # heading glued to first line of section

    def heading_for(self, section: str, rng: random.Random | None = None) -> str:
        base = self.headings.get(section)
        if base is not None:
            return base
        pool = HEADING_SYNONYMS[section]
        if rng is not None:
            return rng.choice(pool)
        return pool[0]


def _spec(
    idx: int,
    name: str,
    split: str,
    base_difficulty: str,
    style: str,
    date_policy: str,
    section_order: tuple[str, ...],
    headings: Mapping[str, str],
    bullet: str,
    range_sep: str,
    present_word: str,
    skills_joiner: str,
    contact_style: int,
    edu_style: int = 0,
    project_style: int = 0,
    cert_style: int = 0,
    force_missing: tuple[str, ...] = (),
    uppercase_name: bool = False,
    inline_headings: bool = False,
) -> TemplateSpec:
    return TemplateSpec(
        id=f"template_{idx:02d}",
        name=name,
        split=split,
        base_difficulty=base_difficulty,
        style=style,
        date_policy=date_policy,
        section_order=section_order,
        headings=headings,
        bullet=bullet,
        range_sep=range_sep,
        present_word=present_word,
        skills_joiner=skills_joiner,
        contact_style=contact_style,
        edu_style=edu_style,
        project_style=project_style,
        cert_style=cert_style,
        force_missing=force_missing,
        uppercase_name=uppercase_name,
        inline_headings=inline_headings,
    )


TEMPLATES: list[TemplateSpec] = [
    _spec(1, "Traditional resume", "train", "easy", "classic", "full",
          ("summary", "experience", "education", "skills", "projects", "certifications"),
          {"summary": "Summary", "experience": "Experience", "education": "Education",
           "skills": "Skills", "projects": "Projects", "certifications": "Certifications"},
          bullet="-", range_sep=" - ", present_word="Present", skills_joiner=", ",
          contact_style=0),
    _spec(2, "Compact resume", "train", "easy", "compact", "abbr",
          ("summary", "skills", "experience", "education", "projects", "certifications"),
          {"summary": "SUMMARY:", "skills": "SKILLS:", "experience": "EXPERIENCE:",
           "education": "EDUCATION:", "projects": "PROJECTS:",
           "certifications": "CERTIFICATIONS:"},
          bullet="-", range_sep=" - ", present_word="Present", skills_joiner=", ",
          contact_style=1, inline_headings=True),
    _spec(3, "Paragraph-heavy resume", "train", "medium", "paragraph", "full",
          ("summary", "experience", "education", "skills"),
          {"summary": "Professional Summary", "experience": "Professional Experience",
           "education": "Academic Background", "skills": "Technical Skills"},
          bullet="-", range_sep=" - ", present_word="Present", skills_joiner=", ",
          contact_style=3, force_missing=("projects", "certifications")),
    _spec(4, "Bullet-heavy resume", "train", "easy", "bullets", "slash",
          ("summary", "experience", "skills", "education", "projects", "certifications"),
          {"summary": "Summary", "experience": "Experience", "skills": "Core Skills",
           "education": "Education", "projects": "Projects",
           "certifications": "Certifications"},
          bullet="•", range_sep=" – ", present_word="Present", skills_joiner=", ",
          contact_style=2),
    _spec(5, "Minimal resume", "train", "easy", "minimal", "abbr",
          ("skills", "experience"),
          {},
          bullet="-", range_sep=" - ", present_word="Present", skills_joiner=", ",
          contact_style=-1,
          force_missing=("email", "phone", "summary", "projects", "certifications")),
    _spec(6, "Modern resume", "train", "medium", "classic", "abbr",
          ("summary", "skills", "experience", "education", "projects", "certifications"),
          {"summary": "PROFILE", "skills": "TECHNICAL SKILLS",
           "experience": "PROFESSIONAL EXPERIENCE", "education": "ACADEMIC BACKGROUND",
           "projects": "KEY PROJECTS", "certifications": "CERTIFICATIONS"},
          bullet="‣", range_sep=" – ", present_word="Present", skills_joiner=" | ",
          contact_style=1, edu_style=1, project_style=1, uppercase_name=True),
    _spec(7, "Experience-first resume", "train", "medium", "classic", "mixed",
          ("experience", "skills", "projects", "certifications", "summary"),
          {"experience": "Work History", "skills": "Core Skills",
           "projects": "Selected Projects", "certifications": "Certifications",
           "summary": "About Me"},
          bullet="-", range_sep=" - ", present_word="Current", skills_joiner=", ",
          contact_style=2, edu_style=2),
    _spec(8, "Education-first resume", "train", "medium", "classic", "iso",
          ("summary", "education", "experience", "skills", "projects", "certifications"),
          {"summary": "Objective", "education": "Academic History",
           "experience": "Employment History", "skills": "Technical Skills",
           "projects": "Projects", "certifications": "Certifications"},
          bullet="-", range_sep=" to ", present_word="Present", skills_joiner=", ",
          contact_style=0, edu_style=3),
    _spec(9, "Projects-first resume", "validation", "medium", "classic", "full",
          ("projects", "experience", "education", "skills", "certifications"),
          {"projects": "Selected Projects", "experience": "Professional Experience",
           "education": "Academic Background", "skills": "Core Skills",
           "certifications": "Licenses & Certifications"},
          bullet="•", range_sep=" to ", present_word="Present", skills_joiner=", ",
          contact_style=3, project_style=2),
    _spec(10, "Skills-heavy resume", "validation", "medium", "grouped_skills", "abbr",
          ("summary", "skills", "experience", "education", "projects", "certifications"),
          {"summary": "Profile", "skills": "Technical Skills",
           "experience": "Professional Experience", "education": "Education",
           "projects": "Projects", "certifications": "Certifications"},
          bullet="-", range_sep=" - ", present_word="Present", skills_joiner=", ",
          contact_style=1),
    _spec(11, "Fragmented resume", "test", "hard", "fragmented", "mixed",
          ("summary", "experience", "education", "skills", "projects", "certifications"),
          {},
          bullet="-", range_sep=" - ", present_word="Present", skills_joiner=" ; ",
          contact_style=4, edu_style=1),
    _spec(12, "Noisy formatting resume", "test", "hard", "noisy", "mixed",
          ("experience", "education", "skills", "projects", "certifications", "summary"),
          {"experience": "### WORK HISTORY ###", "education": "### EDUCATION ###",
           "skills": "### SKILL SET ###", "projects": "### PROJECTS ###",
           "certifications": "### CERTS ###", "summary": "### SUMMARY ###"},
          bullet="*", range_sep=" - ", present_word="Present", skills_joiner=" ; ",
          contact_style=3, uppercase_name=True, edu_style=2),
]

TEMPLATES_BY_ID: dict[str, TemplateSpec] = {t.id: t for t in TEMPLATES}

TEMPLATE_SPLIT_MAP: dict[str, str] = {t.id: t.split for t in TEMPLATES}


# ---------------------------------------------------------------------------
# Date helpers - the gold label stores normalized dates ("2022-06" or "2022");
# the renderer formats them back into surface forms the model must learn to
# normalize again.
# ---------------------------------------------------------------------------
def parse_gold_date(value: str | None) -> tuple[int, int | None]:
    if not value:
        return 0, None
    parts = str(value).split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else None
    return year, month


def render_date(gold: str | None, policy: str, rng: random.Random) -> str:
    year, month = parse_gold_date(gold)
    if month is None:
        return str(year)
    fmt = policy
    if policy == "mixed":
        fmt = rng.choice(["full", "abbr", "slash", "iso"])
    if fmt == "full":
        return f"{MONTHS_FULL[month - 1]} {year}"
    if fmt == "abbr":
        return f"{MONTHS_FULL[month - 1][:3]} {year}"
    if fmt == "slash":
        return f"{month:02d}/{year}"
    if fmt == "iso":
        return f"{year}-{month:02d}"
    return str(year)


def render_range(
    start_gold: str | None,
    end_gold: str | None,
    current: bool,
    spec: TemplateSpec,
    rng: random.Random,
) -> str:
    sep = spec.range_sep
    if spec.style in ("fragmented", "noisy"):
        sep = rng.choice([" - ", " – ", " to ", "-"])
    end_text = spec.present_word
    if spec.style == "fragmented":
        end_text = rng.choice(PRESENT_WORDS)
    if not current and end_gold:
        end_text = render_date(end_gold, spec.date_policy, rng)
    return f"{render_date(start_gold, spec.date_policy, rng)}{sep}{end_text}"


# ---------------------------------------------------------------------------
# Section renderers (all None-safe: absent gold fields are simply not written)
# ---------------------------------------------------------------------------
def _heading_block(spec: TemplateSpec, section: str, rng: random.Random) -> list[str]:
    heading = spec.heading_for(section, rng)
    if spec.style == "fragmented":
        heading = rng.choice([heading.upper(), heading.lower(), heading])
    return [heading.upper() if spec.headings.get(section, "").isupper() else heading]


def _render_contact(record: dict[str, Any], style: int) -> list[str]:
    email, phone = record.get("email"), record.get("phone")
    parts: list[str] = []
    if style == 0:
        if email:
            parts.append(f"Email: {email}")
        if phone:
            parts.append(f"Phone: {phone}")
        return [" | ".join(parts)] if parts else []
    if style == 1:
        if email:
            parts.append(email)
        if phone:
            parts.append(phone)
        return [" | ".join(parts)] if parts else []
    if style == 2:
        if email:
            parts.append(email)
        if phone:
            parts.append(phone)
        return [" · ".join(parts)] if parts else []
    if style == 3:
        if email:
            parts.append(f"Email: {email}")
        if phone:
            parts.append(f"Phone: {phone}")
        return parts
    # style 4
    if phone:
        parts.append(phone)
    if email:
        parts.append(email)
    return [" · ".join(parts)] if parts else []


def _render_header(record: dict[str, Any], spec: TemplateSpec, rng: random.Random) -> list[str]:
    lines: list[str] = []
    name = record.get("name")
    if name:
        lines.append(name.upper() if spec.uppercase_name else name)
    experience = record.get("experience") or []
    if experience and spec.style in ("classic", "compact", "noisy") and rng.random() < 0.6:
        lines.append(experience[0].get("title") or "")
        lines[-1] = lines[-1].strip()
        if not lines[-1]:
            lines.pop()
    location = record.get("location")
    if location and rng.random() < 0.85:
        prefix = "Location: " if spec.style in ("compact", "noisy") else ""
        lines.append(f"{prefix}{location}")
    lines.extend(_render_contact(record, spec.contact_style))
    return [ln for ln in lines if ln]


def _render_summary(record: dict[str, Any], spec: TemplateSpec, rng: random.Random) -> list[str]:
    summary = record.get("summary")
    if not summary:
        return []
    lines = _heading_block(spec, "summary", rng)
    lines.append(summary)
    return lines


def _skills_csv(skills: list[str], joiner: str) -> str:
    if joiner == "and":
        if len(skills) <= 1:
            return ", ".join(skills)
        return ", ".join(skills[:-1]) + " and " + skills[-1]
    return joiner.join(skills)


def _render_skills(record: dict[str, Any], spec: TemplateSpec, rng: random.Random) -> list[str]:
    skills = record.get("skills") or []
    if not skills:
        return []
    if spec.style == "grouped_skills":
        # Skills embedded in labelled paragraphs - boundaries require parsing.
        from docutune.data.pools import SKILL_CATEGORIES

        by_cat: dict[str, list[str]] = {}
        for skill in skills:
            for cat, cat_skills in SKILL_CATEGORIES.items():
                if skill in cat_skills:
                    by_cat.setdefault(cat, []).append(skill)
                    break
            else:
                by_cat.setdefault("Other", []).append(skill)
        lines = _heading_block(spec, "skills", rng)
        for cat, cat_skills in by_cat.items():
            lines.append(f"{cat}: {_skills_csv(cat_skills, ', ')}")
        return lines
    if spec.style == "minimal":
        return [f"Skills: {_skills_csv(skills, ', ')}"]
    if spec.style == "bullets":
        lines = _heading_block(spec, "skills", rng)
        lines.extend(f"{spec.bullet} {s}" for s in skills)
        return lines
    lines = _heading_block(spec, "skills", rng)
    lines.append(_skills_csv(skills, spec.skills_joiner))
    return lines


def _exp_headline(exp: dict[str, Any], rng: random.Random, spec: TemplateSpec) -> str:
    title, company = exp.get("title"), exp.get("company")
    variants = [
        f"{title} at {company}",
        f"{title}, {company}",
        f"{title} - {company}",
        f"{title} | {company}",
        f"{title} ({company})",
    ]
    if spec.style == "noisy":
        return rng.choice(variants[:4])
    if spec.style == "fragmented":
        return rng.choice(variants[:3])
    return variants[1] if spec.style in ("classic", "compact") else rng.choice(variants[:4])


def _render_experience(record: dict[str, Any], spec: TemplateSpec, rng: random.Random) -> list[str]:
    experience = record.get("experience") or []
    if not experience:
        return []
    lines = _heading_block(spec, "experience", rng)
    if spec.style == "minimal":
        for exp in experience:
            date_range = render_range(exp.get("start_date"), exp.get("end_date"),
                                      exp.get("current", False), spec, rng)
            lines.append(f"{exp.get('title')} at {exp.get('company')} ({date_range})")
        return lines
    if spec.style == "paragraph":
        for exp in experience:
            headline = _exp_headline(exp, rng, spec)
            date_range = render_range(exp.get("start_date"), exp.get("end_date"),
                                      exp.get("current", False), spec, rng)
            loc = exp.get("location")
            loc_part = f", {loc}" if loc else ""
            lines.append(f"{headline}{loc_part}, {date_range}.")
            for resp in exp.get("responsibilities") or []:
                lines.append(resp.rstrip(".") + ".")
        return lines
    for exp in experience:
        headline = _exp_headline(exp, rng, spec)
        date_range = render_range(exp.get("start_date"), exp.get("end_date"),
                                  exp.get("current", False), spec, rng)
        loc = exp.get("location")
        if loc and rng.random() < 0.6:
            headline = f"{headline}, {loc}"
        if spec.style in ("compact", "noisy"):
            lines.append(f"{headline} ({date_range})")
        else:
            lines.append(headline)
            lines.append(date_range)
        for resp in exp.get("responsibilities") or []:
            lines.append(f"{spec.bullet} {resp}")
    return lines


def _render_education(record: dict[str, Any], spec: TemplateSpec, rng: random.Random) -> list[str]:
    education = record.get("education") or []
    if not education:
        return []
    lines = _heading_block(spec, "education", rng)
    style = spec.edu_style if spec.style != "minimal" else 1
    for edu in education:
        degree, field_ = edu.get("degree"), edu.get("field")
        inst = edu.get("institution")
        sy, ey = edu.get("start_year"), edu.get("end_year")
        grade = edu.get("grade")
        if style == 0:
            line = f"{degree} in {field_}" if field_ else degree
            lines.append(str(line))
            years = f"{sy} - {ey}" if sy and ey else str(sy or ey or "")
            lines.append(f"{inst} ({years})" if inst else years)
        elif style == 1:
            lines.append(f"{degree} {field_}, {inst}, {sy}-{ey}")
        elif style == 2:
            lines.append(f"{degree} ({field_}) - {inst}")
            lines.append(f"{sy} to {ey}")
        else:
            degree_str = str(degree or "")
            grad_word = ("Postgraduate" if degree_str.startswith(("M", "P"))
                         else "Undergraduate")
            lines.append(f"{grad_word}: {degree} in {field_} from {inst} ({sy} - {ey})")
        if grade:
            if "%" in grade:
                grade_label = "Grade: " if spec.style == "compact" else "Percentage: "
                lines.append(f"{grade_label}{grade}")
            else:
                lines.append(f"CGPA: {grade}/10")
    return lines


def _render_projects(record: dict[str, Any], spec: TemplateSpec, rng: random.Random) -> list[str]:
    projects = record.get("projects") or []
    if not projects:
        return []
    lines = _heading_block(spec, "projects", rng)
    for proj in projects:
        name = proj.get("name")
        tech = proj.get("technologies") or []
        desc = proj.get("description")
        if spec.project_style == 1:
            lines.append(f"{name} [{', '.join(tech)}]" if tech else str(name))
            if desc:
                lines.append(desc)
        elif spec.project_style == 2:
            lines.append(f"Project: {name}")
            if tech:
                lines.append(f"Tech Stack: {', '.join(tech)}")
            if desc:
                lines.append(desc)
        else:
            head = f"{spec.bullet} {name}"
            if tech:
                head += f" - Technologies: {', '.join(tech)}"
            lines.append(head)
            if desc:
                lines.append(f"  {desc}")
    return lines


def _render_certifications(
    record: dict[str, Any], spec: TemplateSpec, rng: random.Random
) -> list[str]:
    certs = record.get("certifications") or []
    if not certs:
        return []
    lines = _heading_block(spec, "certifications", rng)
    for cert in certs:
        name, issuer, year = cert.get("name"), cert.get("issuer"), cert.get("year")
        if spec.cert_style == 1:
            pieces = [str(name)]
            if issuer:
                pieces.append(str(issuer))
            if year:
                pieces.append(str(year))
            lines.append(", ".join(pieces))
        elif spec.cert_style == 2:
            suffix = f" ({year})" if year else ""
            lines.append(f"{name}{suffix}")
        else:
            issuer_part = f" - {issuer}" if issuer else ""
            year_part = f" ({year})" if year else ""
            lines.append(f"{spec.bullet} {name}{issuer_part}{year_part}")
    return lines


# ---------------------------------------------------------------------------
# Style post-processing
# ---------------------------------------------------------------------------
def _fragmentize(sections: list[list[str]], rng: random.Random) -> list[str]:
    """Fragmented style: split labelled lines, irregular blank lines."""
    out: list[str] = []
    for i, section in enumerate(sections):
        if not section:
            continue
        for j, line in enumerate(section):
            out.append(line)
            if j > 0 and rng.random() < 0.35:
                out.append("")  # irregular gaps inside entries
        if i < len(sections) - 1 and rng.random() < 0.7:
            out.append("")
    return out


def _apply_noise(lines: list[str], rng: random.Random) -> list[str]:
    noisy: list[str] = []
    for line in lines:
        if line and rng.random() < 0.18:
            line = line + "   "  # trailing whitespace
        if line and rng.random() < 0.10:
            parts = line.split(" ")
            k = rng.randrange(len(parts))
            parts[k] = parts[k] + " "  # doubled space
            line = " ".join(parts)
        noisy.append(line)
        if rng.random() < 0.12:
            noisy.append("")  # random blank lines
    return noisy


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render_resume(record: dict[str, Any], template_id: str, rng: random.Random) -> str:
    """Render a canonical record into resume text with the given template."""
    spec = TEMPLATES_BY_ID[template_id]

    section_map = {
        "summary": lambda: _render_summary(record, spec, rng),
        "skills": lambda: _render_skills(record, spec, rng),
        "experience": lambda: _render_experience(record, spec, rng),
        "education": lambda: _render_education(record, spec, rng),
        "projects": lambda: _render_projects(record, spec, rng),
        "certifications": lambda: _render_certifications(record, spec, rng),
    }

    rendered: dict[str, list[str]] = {key: section_map[key]() for key in SECTION_KEYS}
    order = list(spec.section_order)
    if spec.style == "fragmented":
        order = [key for key in order if rendered[key]]
        rng.shuffle(order)

    header = _render_header(record, spec, rng)
    sections = [rendered[key] for key in order]

    if spec.style == "fragmented":
        body = _fragmentize(sections, rng)
    else:
        body: list[str] = []
        gap = "\n" if spec.style == "compact" else "\n\n"
        rendered_sections = ["\n".join(s) for s in sections if s]
        body = gap.join(rendered_sections).split("\n")

    if header:
        head_block = "\n".join(header)
    else:
        head_block = ""

    if spec.style == "compact":
        text = head_block + "\n" + "\n".join(body) if head_block else "\n".join(body)
    elif spec.style == "noisy":
        everything = _apply_noise([*header, "", *body], rng)
        text = "\n".join(everything)
        text = "=" * 30 + "\n" + text + "\n" + "=" * 30
    else:
        text = head_block + "\n\n" + "\n".join(body) if head_block else "\n".join(body)

    # Normalize trailing whitespace but keep interior structure.
    lines = [ln.rstrip() if not (spec.style == "noisy") else ln for ln in text.split("\n")]
    if spec.style != "noisy":
        lines = [ln for ln in lines]
    return "\n".join(lines).strip() + "\n"

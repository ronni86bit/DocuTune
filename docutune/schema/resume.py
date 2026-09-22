"""Pydantic schema for structured resume extraction.

This schema is the contract between:
- the synthetic data generator (canonical gold records)
- the prompt (the model is asked to produce exactly this JSON)
- the evaluator (strict validation feeds the schema-validity metric)
- the API responses

SCHEMA_VERSION is bumped whenever the contract changes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docutune.config import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "EducationEntry",
    "ExperienceEntry",
    "ProjectEntry",
    "CertificationEntry",
    "ResumeExtraction",
    "parse_resume_payload",
    "schema_error_message",
]

# ``extra="forbid"``: outputs carrying arbitrary extra fields fail strict
# validation. This is intentional - schema validity is a benchmark metric and
# base instruction models frequently add explanatory keys.


class EducationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degree: str | None = None
    institution: str | None = None
    field: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    grade: str | None = None


class ExperienceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    company: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    current: bool = False
    responsibilities: list[str] = Field(default_factory=list)


class ProjectEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    technologies: list[str] = Field(default_factory=list)
    description: str | None = None


class CertificationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    issuer: str | None = None
    year: int | None = None


class ResumeExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[CertificationEntry] = Field(default_factory=list)


def parse_resume_payload(payload: object) -> ResumeExtraction:
    """Strictly validate a parsed JSON object against the schema.

    Raises pydantic.ValidationError on any violation (extra fields, wrong
    types). Callers decide whether that counts as a schema failure.
    """
    return ResumeExtraction.model_validate(payload)


def schema_error_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:5]:
            loc = ".".join(str(p) for p in err.get("loc", [])) or "<root>"
            parts.append(f"{loc}: {err.get('msg', 'invalid')}")
        more = len(exc.errors()) - 5
        suffix = f" (+{more} more)" if more > 0 else ""
        return "; ".join(parts) + suffix
    return str(exc)

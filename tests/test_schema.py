"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from docutune.schema.resume import (
    SCHEMA_VERSION,
    EducationEntry,
    ExperienceEntry,
    ResumeExtraction,
)


def test_minimal_valid_record():
    record = ResumeExtraction()
    assert record.name is None
    assert record.skills == []
    assert record.education == []
    assert record.experience == []
    assert record.projects == []
    assert record.certifications == []


def test_full_valid_record():
    record = ResumeExtraction.model_validate({
        "name": "John Doe",
        "email": "j@d.com",
        "phone": "+91-9876543210",
        "location": "Hyderabad, Telangana",
        "summary": "ML engineer.",
        "skills": ["Python"],
        "education": [{"degree": "B.Tech", "start_year": 2018, "end_year": 2022}],
        "experience": [{
            "title": "MLE", "company": "Nova", "current": True,
            "responsibilities": ["Built pipelines"],
        }],
        "projects": [{"name": "X", "technologies": ["Python"]}],
        "certifications": [{"name": "AWS", "year": 2023}],
    })
    assert record.experience[0].current is True
    assert record.education[0].grade is None


def test_extra_top_level_fields_forbidden():
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate({"name": "A", "confidence": 0.9})


def test_extra_entry_fields_forbidden():
    with pytest.raises(ValidationError):
        EducationEntry.model_validate({"degree": "B.Tech", "cgpa": 8.5})


def test_wrong_types_rejected():
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate({"skills": "Python"})
    with pytest.raises(ValidationError):
        ExperienceEntry.model_validate({"current": "yes-please"})


def test_nulls_and_defaults():
    record = ResumeExtraction.model_validate({
        "name": None,
        "experience": [{"title": "X"}],
    })
    assert record.name is None
    assert record.experience[0].current is False
    assert record.experience[0].responsibilities == []


def test_schema_version_constant():
    assert SCHEMA_VERSION == "1.0"


def test_round_trip_dump():
    data = {"name": "A", "skills": ["x"], "experience": [{"title": "T", "current": True}]}
    dumped = ResumeExtraction.model_validate(data).model_dump()
    assert list(dumped.keys()) == [
        "name", "email", "phone", "location", "summary", "skills",
        "education", "experience", "projects", "certifications",
    ]
    assert dumped["experience"][0]["current"] is True

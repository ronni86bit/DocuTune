"""Conservative normalization rules (R1-R12 in docs/EVALUATION.md)."""

from docutune.evaluation.normalization import (
    PRESENT,
    normalize_bool,
    normalize_date,
    normalize_grade,
    normalize_phone,
    normalize_record,
    normalize_scalar,
    normalize_year,
    records_equal,
)


def test_r1_whitespace_collapse():
    assert normalize_scalar("  John   Doe \n") == "John Doe"


def test_r2_null_tokens():
    for token in ("", "  ", "n/a", "N/A", "none", "None", "null", "-"):
        assert normalize_scalar(token) is None


def test_r3_case_insensitive_fields():
    assert normalize_scalar("PYTHON", "skills") == normalize_scalar("python", "skills")
    # names stay case-sensitive
    assert normalize_scalar("JOHN DOE") != normalize_scalar("John Doe")


def test_r4_phone_normalization():
    assert normalize_phone("+91-98765 43210") == normalize_phone("+919876543210")
    assert normalize_phone("(555) 123-4567") == "5551234567"


def test_r6_date_formats():
    assert normalize_date("June 2022") == "2022-06"
    assert normalize_date("Jun 2022") == "2022-06"
    assert normalize_date("06/2022") == "2022-06"
    assert normalize_date("2022-06") == "2022-06"
    assert normalize_date("2022") == "2022"
    assert normalize_date("Present") == PRESENT
    assert normalize_date("Till Date") == PRESENT
    assert normalize_date(None) is None


def test_r7_present_end_date_with_current():
    record = {
        "experience": [{
            "title": "MLE", "company": "Nova", "current": True,
            "end_date": "Present", "responsibilities": [],
        }],
    }
    normalized = normalize_record(record)
    assert normalized["experience"][0]["end_date"] is None


def test_r8_list_order_irrelevant():
    a = normalize_record({"skills": ["Python", "SQL"]})
    b = normalize_record({"skills": ["SQL", "Python"]})
    assert a["skills"] == b["skills"]
    assert records_equal(a, b)


def test_r9_grades():
    assert normalize_grade("CGPA: 8.5/10") == "8.5"
    assert normalize_grade("8.5") == "8.5"
    assert normalize_grade("85%") == "85%"


def test_r10_years():
    assert normalize_year("2018") == 2018
    assert normalize_year(2018) == 2018
    assert normalize_year("20x8") is None


def test_r11_unknown_keys_dropped():
    record = {"name": "A", "mood": "happy"}
    normalized = normalize_record(record)
    assert "mood" not in normalized


def test_r12_wrong_types_not_coerced():
    normalized = normalize_record({"skills": "Python"})
    assert normalized["skills"] is None  # type-failure marker, not ["Python"]
    assert normalize_bool("true") is True
    assert normalize_bool("maybe") is None


def test_exact_match_ignores_casefoldable_fields():
    gold = normalize_record({"email": "John.Doe@Gmail.com", "skills": ["PyTorch"]})
    pred = normalize_record({"email": "john.doe@gmail.com", "skills": ["pytorch"]})
    assert records_equal(pred, gold)
